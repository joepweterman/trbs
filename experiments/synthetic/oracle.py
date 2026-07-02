"""
Ground-truth oracles for synthetic convex-regime cases — Phase 1.

The oracle turns "method X scored higher" into an ABSOLUTE optimality gap and
recovery rate: every generated convex case gets a certified global optimum in
its manifest, at any k (the property that lets the synthetic suite carry the
scaling claim where dense grid search is hopeless).

Two variants, matching ``SyntheticCaseParams.appreciation``:

* ``linear`` — the objective is affine on the capped simplex per scenario
  (linear STB=0 appreciation of affine KOs, no clipping by the NO-CLIP
  invariant), so the optimum sits at a vertex. Oracle = enumerate the k corner
  allocations (all B on one lever) plus zero spend through the REAL pipeline
  (``evaluate_allocation``): k+1 evaluations, closed-form at any k.
  Cross-check: the analytic gradient g_i = sum_j W_j * 100 * c_ji /
  (end_j - start_j), with W_j vlinder's own theme-normalised weights, must
  pick the same corner.

* ``sinusoidal`` — all-sinusoidal STB=0 appreciation is concave increasing on
  the bracket, so the objective is a concave program and ANY KKT point is the
  global optimum. Oracle = tight multi-start SLSQP (ftol=1e-12) + a numeric
  KKT-residual certificate (stationarity on the active set, sign condition on
  the inactive set, for max f s.t. sum x <= B, x >= 0).

``certify_case(name, root)`` picks the right solver from the manifest, solves
every scenario, and patches the result into ``manifest.json`` under "oracle".

Run (certify the standard suite):
  & "C:\\Users\\joepw\\.virtualenvs\\tRBS-DclBJWVi-python.exe\\Scripts\\python.exe" `
    C:\\Users\\joepw\\tRBS\\experiments\\synthetic\\oracle.py

Acceptance sweep (both variants, k = 2..15, 3 seeds, SLSQP-vs-oracle gaps):
  ... oracle.py --sweep
"""

# pylint: disable=invalid-name,protected-access,too-many-locals,cell-var-from-loop
# (math notation B/x*/f*; Optimize internals are the documented experiment surface;
#  the per-scenario objective closure is consumed within its own iteration)

from __future__ import annotations

import io
import contextlib
import json
import sys
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


def solve_linear(name: str, root: Path = DEFAULT_ROOT, budget: float = None) -> dict:
    """Vertex-enumeration oracle for the affine (appreciation='linear') variant."""
    manifest = read_manifest(name, root)
    if budget is None:
        budget = float(manifest["params"]["budget"])
    sim, opt = _build(name, root)
    B, k = float(budget), opt._k

    # Analytic gradient (scenario-independent: the external variable enters
    # additively). Reuses vlinder's own weight normalisation.
    coef = _parse_coefficients(opt.input_dict)
    start = np.asarray(opt.input_dict["key_output_start"], dtype=float)
    end = np.asarray(opt.input_dict["key_output_end"], dtype=float)
    weights = np.asarray(Appreciate(opt.input_dict, sim.output_dict)._calculate_weights(), dtype=float)
    gradient = (weights[:, None] * 100.0 * coef / (end - start)[:, None]).sum(axis=0)
    analytic_corner = int(np.argmax(gradient))
    gradient_sorted = np.sort(gradient)
    tie_gap = float(gradient_sorted[-1] - gradient_sorted[-2]) if k > 1 else float("inf")

    vertices = [B * np.eye(k)[i] for i in range(k)] + [np.zeros(k)]
    per_scenario = {}
    for scenario in [str(s) for s in sim.input_dict["scenarios"]]:
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
                "basis": "affine objective (linear STB=0 appreciation of affine KOs, no clipping) => vertex optimum",
                "vertex_index": best,
                "gradient_argmax_agrees": True,
                "gradient_tie_gap": round(tie_gap, 10),
                "vertex_values": [round(float(f), 10) for f in f_vals],
            },
        }
    return {"method": "vertex_enumeration", "generator_version": GENERATOR_VERSION, "per_scenario": per_scenario}


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
    constraints = ({"type": "ineq", "fun": lambda x: float(B - np.sum(x))},)
    bounds = [(0.0, B)] * k

    per_scenario = {}
    for scenario in [str(s) for s in sim.input_dict["scenarios"]]:

        def objective(x, _scenario=scenario):
            return evaluate_allocation(opt.input_dict, x, _scenario, ORACLE_DMO)

        best_x, best_f, n_ok = None, -np.inf, 0
        for x0 in starts:
            res = minimize(
                lambda x: -objective(x),
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500, "disp": False},
            )
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
    return {"method": "tight_slsqp_kkt", "generator_version": GENERATOR_VERSION, "per_scenario": per_scenario}


def certify_case(name: str, root: Path = DEFAULT_ROOT) -> dict:
    """Solve the case's ground truth and patch it into ``manifest.json``."""
    manifest = read_manifest(name, root)
    assert manifest is not None, f"no manifest for {name} — regenerate with the current factory"
    solver = solve_linear if manifest["appreciation"] == "linear" else solve_curved
    oracle = solver(name, root, budget=float(manifest["params"]["budget"]))
    manifest["oracle"] = oracle
    (Path(root) / name / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return oracle


def acceptance_sweep(ks=(2, 3, 4, 6, 9, 12, 15), seeds=(0, 1, 2), root: Path = DEFAULT_ROOT / "sweep") -> list:
    """Generate + validate + certify both variants over the k grid; report the
    production-SLSQP-vs-oracle gap per case (first scenario)."""
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
        print(f"{params.name}: method={oracle['method']}  f*[scenario 1]={first['f_star']:.6f}")


if __name__ == "__main__":
    main()
