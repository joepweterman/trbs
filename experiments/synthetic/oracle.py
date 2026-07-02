"""
Ground-truth oracles for the synthetic cases — Phase 1 (convex) + Phases 2/3
(nonconvex regimes), per the ground-truth strategy in the methodology plan.

The oracle turns "method X scored higher" into an ABSOLUTE optimality gap and
recovery rate: every generated case gets a ground-truth block in its manifest.

Solvers, dispatched from the manifest by ``certify_case``:

* convex / ``linear`` — the objective is affine on the capped simplex per
  scenario, so the optimum sits at a vertex. Oracle = enumerate the k corner
  allocations plus zero spend through the REAL pipeline (k+1 evaluations,
  closed-form at any k). Cross-check: the analytic per-scenario gradient
  g_i(s) = sum_j W_j * 100 * fac_js * c_ji / (end_j - start_j) — with W_j
  vlinder's own theme-normalised weights and fac the scenario-dispersion
  factors (1 when absent) — must pick the same corner.

* convex / ``sinusoidal`` — concave program: ANY KKT point is the global
  optimum. Oracle = tight multi-start SLSQP (ftol=1e-12) + a numeric
  KKT-residual certificate.

* smooth_nonconvex / nonsmooth, k <= 6 — ``solve_grid_polish``: a dense
  deterministic grid over the CAPPED SIMPLEX (interior optima are feasible in
  these regimes) + tight-SLSQP polish of the top nodes and extra multistarts.
  Verified ground truth at grid resolution; also reports a basin-count
  estimate from the polished endpoints.

* smooth_nonconvex / nonsmooth, k > 6 — ``solve_multistart_proxy``: best of
  many tight multistarts, explicitly labelled ``verified: false`` (NO
  optimality guarantee — anchors runtime scaling and relative comparison only;
  absolute recovery at high k is anchored on the convex regime).

Run (certify the standard suite):
  & "C:\\Users\\joepw\\.virtualenvs\\tRBS-DclBJWVi-python.exe\\Scripts\\python.exe" `
    C:\\Users\\joepw\\tRBS\\experiments\\synthetic\\oracle.py

Acceptance sweep (convex variants, k = 2..15, 3 seeds, SLSQP-vs-oracle gaps):
  ... oracle.py --sweep
"""

# pylint: disable=invalid-name,protected-access,too-many-locals,cell-var-from-loop,too-many-arguments
# pylint: disable=too-many-positional-arguments
# (math notation B/x*/f*; Optimize internals are the documented experiment surface;
#  the per-scenario objective closure is consumed within its own iteration)

from __future__ import annotations

import copy
import io
import contextlib
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from case_factory import (
    DEFAULT_ROOT,
    standard_cases,
    GENERATOR_VERSION,
    SyntheticCaseFactory,
    SyntheticCaseParams,
    read_manifest,
    validate_case,
)

from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.evaluate import Evaluate
from vlinder.optimize import Optimize, evaluate_allocation
from vlinder.appreciate import Appreciate


ORACLE_DMO = "Oracle"


def _build(name: str, root: Path):
    """Import a generated case through the real pipeline and register the oracle DMO."""
    sim = TheResponsibleBusinessSimulator(name, file_path=Path(root), file_extension="csv")
    with contextlib.redirect_stdout(io.StringIO()):
        sim.build()
        sim.evaluate()
        sim.appreciate()
    opt = Optimize(sim.input_dict, sim.output_dict)
    with contextlib.redirect_stdout(io.StringIO()):
        opt._prepare_input_dict(ORACLE_DMO, sim.input_dict["decision_makers_option_value"][0].copy())
    return sim, opt


def _parse_coefficients(input_dict) -> np.ndarray:
    """Recover the lever->KO coefficient matrix c[j, i] from the imported fixed inputs."""
    n_ko = len(input_dict["key_outputs"])
    k = len(input_dict["internal_variable_inputs"])
    coef = np.zeros((n_ko, k))
    for fi_name, fi_value in zip(input_dict["fixed_inputs"], input_dict["fixed_input_value"]):
        if str(fi_name).startswith("c_"):
            _, j, i = str(fi_name).split("_")
            coef[int(j) - 1, int(i) - 1] = float(fi_value)
    return coef


def _scenario_factors(input_dict, n_ko: int) -> np.ndarray:
    """Per-scenario multiplicative KO factors (the 'Fac j' EVIs); ones when absent."""
    evis = [str(e) for e in input_dict["external_variable_inputs"]]
    sv = np.asarray(input_dict["scenario_value"], dtype=float)
    fac = np.ones((sv.shape[0], n_ko))
    for j in range(n_ko):
        evi_name = f"Fac {j + 1:02d}"
        if evi_name in evis:
            fac[:, j] = sv[:, evis.index(evi_name)]
    return fac


def _tight_slsqp(objective, x0, B: float, k: int):
    """One tight SLSQP solve (ftol 1e-12) of max objective on the capped simplex."""
    return minimize(
        lambda x: -objective(x),
        x0,
        method="SLSQP",
        bounds=[(0.0, B)] * k,
        constraints=({"type": "ineq", "fun": lambda x: float(B - np.sum(x))},),
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )


def solve_linear(name: str, root: Path = DEFAULT_ROOT, budget: float = None) -> dict:
    """Vertex-enumeration oracle for the affine (appreciation='linear') variant.

    Scenario-dispersion aware: the analytic gradient is computed per scenario
    with the 'Fac j' factors, so dispersed cases get per-scenario vertices.
    """
    manifest = read_manifest(name, root)
    if budget is None:
        budget = float(manifest["params"]["budget"])
    sim, opt = _build(name, root)
    B, k = float(budget), opt._k

    coef = _parse_coefficients(opt.input_dict)
    start = np.asarray(opt.input_dict["key_output_start"], dtype=float)
    end = np.asarray(opt.input_dict["key_output_end"], dtype=float)
    weights = np.asarray(Appreciate(opt.input_dict, sim.output_dict)._calculate_weights(), dtype=float)
    fac = _scenario_factors(opt.input_dict, len(coef))

    vertices = [B * np.eye(k)[i] for i in range(k)] + [np.zeros(k)]
    per_scenario = {}
    for s_idx, scenario in enumerate([str(s) for s in sim.input_dict["scenarios"]]):
        gradient = ((weights * fac[s_idx])[:, None] * 100.0 * coef / (end - start)[:, None]).sum(axis=0)
        analytic_corner = int(np.argmax(gradient))
        gradient_sorted = np.sort(gradient)
        tie_gap = float(gradient_sorted[-1] - gradient_sorted[-2]) if k > 1 else float("inf")

        f_vals = [evaluate_allocation(opt.input_dict, x, scenario, ORACLE_DMO) for x in vertices]
        best = int(np.argmax(f_vals))
        assert best < k, "zero spend beat every corner despite a positive gradient — generator bug"
        assert best == analytic_corner, (
            f"vertex enumeration ({best}) and analytic gradient ({analytic_corner}) disagree "
            f"in {scenario} — affinity assumption violated"
        )
        per_scenario[scenario] = {
            "x_star": [round(v, 10) for v in vertices[best]],
            "f_star": float(f_vals[best]),
            "certificate": {
                "basis": "affine objective per scenario (linear STB=0 appreciation of affine KOs, no clipping) "
                "=> vertex optimum",
                "vertex_index": best,
                "gradient_argmax_agrees": True,
                "gradient_tie_gap": round(tie_gap, 10),
                "vertex_values": [round(float(f), 10) for f in f_vals],
            },
        }
    return {
        "method": "vertex_enumeration",
        "verified": True,
        "generator_version": GENERATOR_VERSION,
        "per_scenario": per_scenario,
    }


def _kkt_residual(objective, x_star, budget, k, h_rel=1e-6):
    """Numeric KKT residual for max f(x) s.t. sum x <= B, 0 <= x <= B.

    Central-difference gradient; active set = {i: x_i > tol}. If the budget
    constraint binds, its multiplier is the mean gradient over the active set;
    stationarity requires g_i = lambda on the active set and g_i <= lambda on
    the inactive set (mu_i = lambda - g_i >= 0).
    """
    h = h_rel * budget
    grad = np.zeros(k)
    for i in range(k):
        e = np.zeros(k)
        e[i] = h
        grad[i] = (objective(x_star + e) - objective(x_star - e)) / (2 * h)
    active = x_star > 1e-8 * budget
    budget_active = budget - x_star.sum() < 1e-6 * budget
    lam = float(grad[active].mean()) if (budget_active and active.any()) else 0.0
    stationarity = float(np.abs(grad[active] - lam).max()) if active.any() else 0.0
    sign_condition = float(np.maximum(grad[~active] - lam, 0.0).max()) if (~active).any() else 0.0
    return {
        "gradient": [round(float(g), 8) for g in grad],
        "lambda_budget": round(lam, 8),
        "stationarity_residual": round(stationarity, 8),
        "sign_condition_residual": round(sign_condition, 8),
        "kkt_residual": round(max(stationarity, sign_condition), 8),
    }


def solve_curved(name: str, root: Path = DEFAULT_ROOT, budget: float = None, n_starts: int = 8, seed: int = 0) -> dict:
    """Tight-SLSQP + KKT oracle for the concave (appreciation='sinusoidal') variant.

    By concavity any KKT point is globally optimal, so the certificate carries
    the guarantee; the multi-start is belt-and-braces against solver failures.
    """
    manifest = read_manifest(name, root)
    if budget is None:
        budget = float(manifest["params"]["budget"])
    sim, opt = _build(name, root)
    B, k = float(budget), opt._k

    rng = np.random.default_rng(seed)
    starts = list(rng.dirichlet(np.ones(k), size=n_starts) * B) + [np.full(k, B / k)]

    per_scenario = {}
    for scenario in [str(s) for s in sim.input_dict["scenarios"]]:

        def objective(x, _scenario=scenario):
            return evaluate_allocation(opt.input_dict, x, _scenario, ORACLE_DMO)

        best_x, best_f, n_ok = None, -np.inf, 0
        for x0 in starts:
            res = _tight_slsqp(objective, x0, B, k)
            if res.success:
                n_ok += 1
                if -res.fun > best_f:
                    best_f, best_x = -res.fun, np.clip(res.x, 0.0, B)
        assert best_x is not None, f"no tight-SLSQP start converged in {scenario}"

        kkt = _kkt_residual(objective, best_x, B, k)
        per_scenario[scenario] = {
            "x_star": [round(float(v), 10) for v in best_x],
            "f_star": float(best_f),
            "certificate": {
                "basis": "concave objective (sinusoidal STB=0 appreciation is concave increasing on the bracket, "
                "affine KOs, no clipping) => any KKT point is the global optimum",
                "n_starts_converged": f"{n_ok}/{len(starts)}",
                **kkt,
            },
        }
    return {
        "method": "tight_slsqp_kkt",
        "verified": True,
        "generator_version": GENERATOR_VERSION,
        "per_scenario": per_scenario,
    }


# ---- nonconvex ground truth (smooth_nonconvex / nonsmooth) -----------------
def _compositions(total: int, parts: int):
    """All non-negative integer compositions of ``total`` into ``parts`` entries."""
    for dividers in combinations(range(total + parts - 1), parts - 1):
        prev, comp = -1, []
        for dv in dividers:
            comp.append(dv - prev - 1)
            prev = dv
        comp.append(total + parts - 1 - prev - 1)
        yield comp


def _capped_grid(k: int, budget: float, max_points: int = 4000):
    """Deterministic dense grid on the CAPPED simplex {x >= 0, sum x <= B}:
    compositions of R over k+1 parts (last part = budget slack), with R the
    finest resolution whose node count C(R+k, k) fits ``max_points``."""
    R = 1
    while comb(R + 1 + k, k) <= max_points:
        R += 1
    nodes = np.array(list(_compositions(R, k + 1)), dtype=float)[:, :k] * (budget / R)
    return nodes, R


def _fast_batch_objective(input_dict, output_dict, X, scenario: str, dmo_name: str) -> np.ndarray:
    """Objective at many allocations with ONE deepcopy (grid-oracle fast path).

    Requires frozen boundaries (``_prepare_input_dict`` sets automatic=0), so
    ``Appreciate`` reads the appreciation curves from the input dict and the
    output dict is irrelevant to the boundary computation. Cross-checked
    against ``evaluate_allocation`` by the caller.
    """
    d = copy.deepcopy(input_dict)
    idx = int(np.where(d["decision_makers_options"] == dmo_name)[0][0])
    ev = Evaluate(d)
    app = Appreciate(d, output_dict)
    vals = np.empty(len(X))
    for r, x in enumerate(X):
        d["decision_makers_option_value"][idx] = np.asarray(x, dtype=float)
        value_dict = {"key_outputs": ev.evaluate_all_dependencies(scenario, dmo_name)["key_outputs"]}
        app.appreciate_single_decision_maker_option(value_dict)
        vals[r] = float(value_dict["decision_makers_option_appreciation"])
    return vals


def _mixed_starts(rng, k: int, budget: float, n: int):
    """Multistart points: half on the budget face, half interior (capped simplex)."""
    face = rng.dirichlet(np.ones(k), size=n // 2) * budget
    interior = rng.dirichlet(np.ones(k + 1), size=n - n // 2)[:, :k] * budget
    return list(face) + list(interior)


def _polish_and_summarise(objective, starts, B, k, grid_best_x, grid_best_f):
    """Tight-polish all starts; return the incumbent + a basin-count estimate."""
    best_x, best_f = np.asarray(grid_best_x, dtype=float), float(grid_best_f)
    endpoints_f, endpoints_x = [], []
    for x0 in starts:
        res = _tight_slsqp(objective, x0, B, k)
        if not res.success:
            continue
        f_val = float(-res.fun)
        x_val = np.clip(res.x, 0.0, B)
        endpoints_f.append(f_val)
        endpoints_x.append(x_val)
        if f_val > best_f:
            best_f, best_x = f_val, x_val
    n_basins_f = len({round(f, 2) for f in endpoints_f})
    n_basins_x = len({tuple(np.round(x / B * 50).astype(int)) for x in endpoints_x})
    return best_x, best_f, n_basins_f, n_basins_x, len(endpoints_f)


def solve_grid_polish(
    name: str,
    root: Path = DEFAULT_ROOT,
    budget: float = None,
    max_points: int = 4000,
    n_polish: int = 12,
    n_extra_starts: int = 16,
    seed: int = 0,
) -> dict:
    """Dense-grid + polish oracle for nonconvex regimes at low k (<= ~6).

    The deterministic grid covers the capped simplex globally (interior optima
    included — STB=1 KOs reward under-spending); the top nodes plus mixed
    multistarts are polished with tight SLSQP. Ground truth verified at grid
    resolution; the polished endpoints give a basin-count estimate.
    """
    manifest = read_manifest(name, root)
    if budget is None:
        budget = float(manifest["params"]["budget"])
    sim, opt = _build(name, root)
    B, k = float(budget), opt._k
    nodes, R = _capped_grid(k, B, max_points)
    rng = np.random.default_rng(seed)

    per_scenario = {}
    for scenario in [str(s) for s in sim.input_dict["scenarios"]]:

        def objective(x, _scenario=scenario):
            return evaluate_allocation(opt.input_dict, x, _scenario, ORACLE_DMO)

        vals = _fast_batch_objective(opt.input_dict, sim.output_dict, nodes, scenario, ORACLE_DMO)
        for x_chk, v_chk in zip(nodes[:3], vals[:3]):
            v_ref = objective(x_chk)
            assert abs(v_chk - v_ref) <= 1e-9 * max(
                1.0, abs(v_ref)
            ), "fast objective path disagrees with evaluate_allocation"

        top = np.argsort(vals)[-n_polish:]
        grid_best = int(top[-1])
        starts = [nodes[i] for i in top] + _mixed_starts(rng, k, B, n_extra_starts)
        best_x, best_f, n_basins_f, n_basins_x, n_ok = _polish_and_summarise(
            objective, starts, B, k, nodes[grid_best], vals[grid_best]
        )
        per_scenario[scenario] = {
            "x_star": [round(float(v), 10) for v in best_x],
            "f_star": float(best_f),
            "certificate": {
                "basis": "dense capped-simplex grid + tight-SLSQP polish (nonconvex regime, low k)",
                "grid_resolution": int(R),
                "n_grid_nodes": int(len(nodes)),
                "grid_best_f": round(float(vals[grid_best]), 8),
                "polish_gain": round(float(best_f - vals[grid_best]), 8),
                "n_polished_converged": n_ok,
                "n_basins_f_estimate": n_basins_f,
                "n_basins_x_estimate": n_basins_x,
            },
        }
    return {
        "method": "dense_grid_polish",
        "verified": True,
        "generator_version": GENERATOR_VERSION,
        "per_scenario": per_scenario,
    }


def solve_multistart_proxy(
    name: str, root: Path = DEFAULT_ROOT, budget: float = None, n_starts: int = 64, seed: int = 0
) -> dict:
    """Best-of-multistart PROXY for nonconvex regimes at high k (> ~6).

    NO optimality guarantee — ``verified`` is false and downstream analysis
    must treat gaps against this value as relative, not absolute (the plan's
    ground-truth strategy: absolute recovery at high k is anchored on the
    convex regime, this proxy anchors runtime scaling and relative ranking).
    """
    manifest = read_manifest(name, root)
    if budget is None:
        budget = float(manifest["params"]["budget"])
    sim, opt = _build(name, root)
    B, k = float(budget), opt._k
    rng = np.random.default_rng(seed)
    starts = (
        _mixed_starts(rng, k, B, max(0, n_starts - k - 1)) + [B * np.eye(k)[i] for i in range(k)] + [np.full(k, B / k)]
    )

    per_scenario = {}
    for scenario in [str(s) for s in sim.input_dict["scenarios"]]:

        def objective(x, _scenario=scenario):
            return evaluate_allocation(opt.input_dict, x, _scenario, ORACLE_DMO)

        f0 = objective(np.full(k, B / k))
        best_x, best_f, n_basins_f, n_basins_x, n_ok = _polish_and_summarise(
            objective, starts, B, k, np.full(k, B / k), f0
        )
        per_scenario[scenario] = {
            "x_star": [round(float(v), 10) for v in best_x],
            "f_star": float(best_f),
            "certificate": {
                "basis": "best of tight multistarts — PROXY ONLY, no optimality guarantee (nonconvex, high k)",
                "n_starts": len(starts),
                "n_polished_converged": n_ok,
                "n_basins_f_estimate": n_basins_f,
                "n_basins_x_estimate": n_basins_x,
            },
        }
    return {
        "method": "best_of_multistart_proxy",
        "verified": False,
        "generator_version": GENERATOR_VERSION,
        "per_scenario": per_scenario,
    }


# ---- dispatch ---------------------------------------------------------------
def certify_case(name: str, root: Path = DEFAULT_ROOT) -> dict:
    """Solve the case's ground truth (solver chosen from the manifest regime)
    and patch it into ``manifest.json``, including a scenario-dispersion metric
    (max pairwise L1 distance between per-scenario optima, scaled by B)."""
    manifest = read_manifest(name, root)
    assert manifest is not None, f"no manifest for {name} — regenerate with the current factory"
    regime = manifest.get("regime", "convex")
    B = float(manifest["params"]["budget"])
    k = int(manifest["params"]["k"])

    if regime == "convex":
        solver = solve_linear if manifest["appreciation"] == "linear" else solve_curved
    else:
        solver = solve_grid_polish if k <= 6 else solve_multistart_proxy
    oracle = solver(name, root, budget=B)

    xs = [np.asarray(s["x_star"], dtype=float) for s in oracle["per_scenario"].values()]
    dispersion = 0.0
    for a_idx, x_a in enumerate(xs):
        for x_b in xs[a_idx + 1:]:
            dispersion = max(dispersion, float(np.abs(x_a - x_b).sum()) / B)
    oracle["scenario_optimum_dispersion"] = round(dispersion, 6)

    manifest["oracle"] = oracle
    (Path(root) / name / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return oracle


def acceptance_sweep(ks=(2, 3, 4, 6, 9, 12, 15), seeds=(0, 1, 2), root: Path = DEFAULT_ROOT / "sweep") -> list:
    """Generate + validate + certify both convex variants over the k grid;
    report the production-SLSQP-vs-oracle gap per case (first scenario)."""
    rows = []
    for appreciation in ("linear", "sinusoidal"):
        for k in ks:
            for seed in seeds:
                name = f"Sweep_{appreciation}_k{k:02d}_s{seed}"
                params = SyntheticCaseParams(
                    name=name,
                    k=k,
                    n_key_outputs=max(2, min(5, (k + 1) // 2)),
                    appreciation=appreciation,
                    seed=seed,
                )
                SyntheticCaseFactory(params).write(root)
                validate_case(name, root, budget=params.budget, n_samples=100)
                oracle = certify_case(name, root)

                sim, opt = _build(name, root)
                scenario = str(sim.input_dict["scenarios"][0])
                with contextlib.redirect_stdout(io.StringIO()):
                    slsqp = opt.optimize_slsqp(scenario, params.budget, dmo_name=ORACLE_DMO, n_starts=20, seed=1)
                f_star = oracle["per_scenario"][scenario]["f_star"]
                gap = f_star - float(slsqp.appreciation)
                kkt = oracle["per_scenario"][scenario]["certificate"].get("kkt_residual")
                rows.append(
                    {
                        "name": name,
                        "k": k,
                        "seed": seed,
                        "variant": appreciation,
                        "f_star": round(f_star, 6),
                        "gap": round(gap, 8),
                        "kkt_residual": kkt,
                    }
                )
                print(f"{name}: f*={f_star:.6f}  gap={gap:.2e}" + (f"  kkt={kkt:.2e}" if kkt is not None else ""))
    return rows


def main():
    """Certify the standard suite, or run the acceptance sweep with --sweep."""
    if "--sweep" in sys.argv:
        rows = acceptance_sweep()
        worst_lin = max(abs(r["gap"]) for r in rows if r["variant"] == "linear")
        worst_cur = max(abs(r["gap"]) for r in rows if r["variant"] == "sinusoidal")
        worst_kkt = max(r["kkt_residual"] for r in rows if r["kkt_residual"] is not None)
        print(f"\nworst |gap| linear: {worst_lin:.2e}   worst |gap| curved: {worst_cur:.2e}")
        print(f"worst KKT residual: {worst_kkt:.2e}")
        return
    for params in standard_cases():
        oracle = certify_case(params.name)
        first = next(iter(oracle["per_scenario"].values()))
        print(
            f"{params.name}: method={oracle['method']} verified={oracle['verified']} "
            f"f*[scenario 1]={first['f_star']:.6f} dispersion={oracle['scenario_optimum_dispersion']}"
        )


if __name__ == "__main__":
    main()
