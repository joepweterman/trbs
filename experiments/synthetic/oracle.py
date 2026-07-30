"""
This file contains the Oracle class that computes the certified ground-truth
optimum of a generated synthetic case.

The oracle is what turns "method A scored higher than method B" into an
absolute optimality gap: every generated case gets a ground-truth block in its
manifest, and a method's gap is measured against it.

How much the oracle can guarantee depends on the regime of the case:

  * convex + linear appreciation: the objective is affine on the capped simplex,
    so the optimum sits at a corner. Enumerating the k corners plus zero spend
    is exact at any k, and the analytic gradient must pick the same corner.
  * convex + sinusoidal appreciation: the objective is concave, so any KKT point
    is the global optimum. Tight multi-start SLSQP plus a numeric KKT residual
    certifies it.
  * non-convex, k <= 6: a dense grid over the whole capped simplex plus a local
    polish of the best nodes. Verified at grid resolution, not analytically, and
    it also reports how many separate optima the landscape appears to have.
  * non-convex, k > 6: best of many local solves. This is a PROXY and is
    labelled ``verified: false``. It supports runtime scaling and relative
    ranking only; the thesis anchors absolute recovery on the convex regime.

Run (certify the standard suite):
  python experiments/synthetic/oracle.py
"""

# pylint: disable=invalid-name,protected-access
# (math notation B/x*/f*; Optimize internals are the documented experiment surface)

from __future__ import annotations

import copy
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from case_factory import (
    DEFAULT_ROOT,
    GENERATOR_VERSION,
    build_case,
    parse_coefficients,
    read_manifest,
    sample_capped_simplex,
    standard_cases,
)

from vlinder.evaluate import Evaluate
from vlinder.optimize import evaluate_allocation
from vlinder.appreciate import Appreciate

ORACLE_DMO = "Oracle"

#: Solver settings. Frozen: the certified optima these produce are the reference
#: values the study's gaps are measured against, so they are not call-site knobs.
CURVED_N_STARTS = 8
GRID_MAX_POINTS = 4000
GRID_N_POLISH = 12
GRID_N_EXTRA_STARTS = 16
PROXY_N_STARTS = 64
SOLVER_SEED = 0


def _tight_slsqp(objective, x0, B: float, k: int):
    """
    This function runs one tight SLSQP solve (ftol 1e-12) of max objective over
    the capped simplex.
    :param objective: the function to maximise
    :param x0: the starting allocation
    :param B: the budget
    :param k: the number of internal variables
    :return: the scipy OptimizeResult
    """
    return minimize(
        lambda x: -objective(x),
        x0,
        method="SLSQP",
        bounds=[(0.0, B)] * k,
        constraints=({"type": "ineq", "fun": lambda x: float(B - np.sum(x))},),
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )


def _compositions(total: int, parts: int):
    """
    This function yields all non-negative integer compositions of ``total`` into
    ``parts`` entries (the stars-and-bars enumeration).
    :param total: the sum each composition must reach
    :param parts: the number of entries per composition
    :return: a generator of integer arrays
    """
    for dividers in combinations(range(total + parts - 1), parts - 1):
        yield np.diff((-1, *dividers, total + parts - 1)) - 1


def _capped_grid(k: int, budget: float):
    """
    This function builds a deterministic dense grid on the capped simplex
    {x >= 0, sum x <= B}: compositions of R over k+1 parts, where the last part
    is the unspent budget and R is the finest resolution that still fits in
    ``GRID_MAX_POINTS`` nodes.
    :param k: the number of internal variables
    :param budget: the budget
    :return: a tuple of the grid nodes and the resolution R
    """
    R = 1
    while comb(R + 1 + k, k) <= GRID_MAX_POINTS:
        R += 1
    nodes = np.array(list(_compositions(R, k + 1)), dtype=float)[:, :k] * (budget / R)
    return nodes, R


def _mixed_starts(rng, k: int, budget: float, n: int):
    """
    This function draws multistart points, half on the budget face and half in
    the interior of the capped simplex.
    :param rng: the numpy random generator to draw from
    :param k: the number of internal variables
    :param budget: the budget
    :param n: the total number of starts
    :return: a list of allocations
    """
    face = sample_capped_simplex(rng, n // 2, k, budget, face=True)
    interior = sample_capped_simplex(rng, n - n // 2, k, budget)
    return list(face) + list(interior)


def _kkt_residual(objective, x_star, budget, k, h_rel=1e-6):
    """
    This function computes a numeric KKT residual for max f(x) s.t. sum x <= B,
    0 <= x <= B. The gradient is a central difference and the active set is
    {i: x_i > tol}. If the budget constraint binds, its multiplier is the mean
    gradient over the active set; stationarity then requires g_i = lambda on the
    active set and g_i <= lambda off it.
    :param objective: the function that was maximised
    :param x_star: the candidate optimum
    :param budget: the budget
    :param k: the number of internal variables
    :param h_rel: the finite-difference step, relative to the budget
    :return: a dictionary of gradient, multiplier and residuals
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


class Oracle:
    """This class computes the certified ground-truth optimum of one generated case."""

    def __init__(self, name: str, root: Path = DEFAULT_ROOT):
        self.name = name
        self.root = Path(root)
        self.manifest = read_manifest(name, root)
        self.budget = float(self.manifest["params"]["budget"])
        self.k = int(self.manifest["params"]["k"])
        self.sim, self.opt = build_case(name, root, ORACLE_DMO)
        self.scenarios = [str(s) for s in self.sim.input_dict["scenarios"]]

    def _objective(self, scenario: str):
        """
        This function returns the objective function for one scenario.
        :param scenario: the scenario to evaluate in
        :return: a callable mapping an allocation to its weighted appreciation
        """
        return lambda x: evaluate_allocation(self.opt.input_dict, x, scenario, ORACLE_DMO)

    def _scenario_factors(self) -> np.ndarray:
        """
        This function reads the per-scenario multiplicative key-output factors
        (the 'Fac j' external variables), which are ones when dispersion is off.
        :return: an (n_scenarios x n_key_outputs) factor matrix
        """
        n_ko = len(self.opt.input_dict["key_outputs"])
        evis = [str(e) for e in self.opt.input_dict["external_variable_inputs"]]
        sv = np.asarray(self.opt.input_dict["scenario_value"], dtype=float)
        fac = np.ones((sv.shape[0], n_ko))
        for j in range(n_ko):
            if f"Fac {j + 1:02d}" in evis:
                fac[:, j] = sv[:, evis.index(f"Fac {j + 1:02d}")]
        return fac

    # pylint: disable=too-many-locals
    def solve_linear(self) -> dict:
        """
        This function certifies the affine variant by enumerating the corners of
        the capped simplex. The objective is affine per scenario, so the optimum
        must sit at a corner; the analytic per-scenario gradient must select the
        same corner, which cross-checks the affinity assumption.
        :return: the oracle block for the manifest
        """
        B, k = self.budget, self.k
        coef = parse_coefficients(self.opt.input_dict)
        start = np.asarray(self.opt.input_dict["key_output_start"], dtype=float)
        end = np.asarray(self.opt.input_dict["key_output_end"], dtype=float)
        weights = np.asarray(Appreciate(self.opt.input_dict, self.sim.output_dict)._calculate_weights(), dtype=float)
        fac = self._scenario_factors()

        vertices = [B * np.eye(k)[i] for i in range(k)] + [np.zeros(k)]
        per_scenario = {}
        for s_idx, scenario in enumerate(self.scenarios):
            gradient = ((weights * fac[s_idx])[:, None] * 100.0 * coef / (end - start)[:, None]).sum(axis=0)
            analytic_corner = int(np.argmax(gradient))
            gradient_sorted = np.sort(gradient)
            tie_gap = float(gradient_sorted[-1] - gradient_sorted[-2]) if k > 1 else float("inf")

            objective = self._objective(scenario)
            f_vals = [objective(x) for x in vertices]
            best = int(np.argmax(f_vals))
            assert best < k, "zero spend beat every corner despite a positive gradient - generator bug"
            assert best == analytic_corner, (
                f"vertex enumeration ({best}) and analytic gradient ({analytic_corner}) disagree "
                f"in {scenario} - affinity assumption violated"
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
        return {"method": "vertex_enumeration", "verified": True, "per_scenario": per_scenario}

    def solve_curved(self) -> dict:
        """
        This function certifies the concave variant with tight multi-start SLSQP
        plus a KKT certificate. By concavity any KKT point is globally optimal,
        so the certificate carries the guarantee and the multi-start is only
        insurance against solver failures.
        :return: the oracle block for the manifest
        """
        B, k = self.budget, self.k
        rng = np.random.default_rng(SOLVER_SEED)
        starts = list(sample_capped_simplex(rng, CURVED_N_STARTS, k, B, face=True)) + [np.full(k, B / k)]

        per_scenario = {}
        for scenario in self.scenarios:
            objective = self._objective(scenario)
            best_x, best_f, n_ok = None, -np.inf, 0
            for x0 in starts:
                res = _tight_slsqp(objective, x0, B, k)
                if res.success:
                    n_ok += 1
                    if -res.fun > best_f:
                        best_f, best_x = -res.fun, np.clip(res.x, 0.0, B)
            assert best_x is not None, f"no tight-SLSQP start converged in {scenario}"

            per_scenario[scenario] = {
                "x_star": [round(float(v), 10) for v in best_x],
                "f_star": float(best_f),
                "certificate": {
                    "basis": "concave objective (sinusoidal STB=0 appreciation is concave increasing on the bracket, "
                    "affine KOs, no clipping) => any KKT point is the global optimum",
                    "n_starts_converged": f"{n_ok}/{len(starts)}",
                    **_kkt_residual(objective, best_x, B, k),
                },
            }
        return {"method": "tight_slsqp_kkt", "verified": True, "per_scenario": per_scenario}

    def _batch_objective(self, X, scenario: str) -> np.ndarray:
        """
        This function evaluates many allocations with a single deepcopy, the
        fast path the grid oracle needs. It relies on the boundaries being
        frozen by ``build_case``, so Appreciate reads its curves from the input
        dict; ``solve_grid_polish`` cross-checks it against evaluate_allocation.
        :param X: the allocations to evaluate
        :param scenario: the scenario to evaluate in
        :return: an array of appreciation values
        """
        d = copy.deepcopy(self.opt.input_dict)
        idx = int(np.where(d["decision_makers_options"] == ORACLE_DMO)[0][0])
        ev = Evaluate(d)
        app = Appreciate(d, self.sim.output_dict)
        vals = np.empty(len(X))
        for r, x in enumerate(X):
            d["decision_makers_option_value"][idx] = np.asarray(x, dtype=float)
            value_dict = {"key_outputs": ev.evaluate_all_dependencies(scenario, ORACLE_DMO)["key_outputs"]}
            app.appreciate_single_decision_maker_option(value_dict)
            vals[r] = float(value_dict["decision_makers_option_appreciation"])
        return vals

    # pylint: disable=too-many-locals
    def _polish(self, objective, starts, incumbent_x, incumbent_f):
        """
        This function tight-polishes every start and returns the best point plus
        an estimate of how many separate optima the landscape has.
        :param objective: the function to maximise
        :param starts: the starting allocations
        :param incumbent_x: the best allocation found so far
        :param incumbent_f: the value at ``incumbent_x``
        :return: best x, best f, basin count in f, basin count in x, n converged
        """
        B, k = self.budget, self.k
        best_x, best_f = np.asarray(incumbent_x, dtype=float), float(incumbent_f)
        endpoints_f, endpoints_x = [], []
        for x0 in starts:
            res = _tight_slsqp(objective, x0, B, k)
            if not res.success:
                continue
            f_val, x_val = float(-res.fun), np.clip(res.x, 0.0, B)
            endpoints_f.append(f_val)
            endpoints_x.append(x_val)
            if f_val > best_f:
                best_f, best_x = f_val, x_val
        n_basins_f = len({round(f, 2) for f in endpoints_f})
        n_basins_x = len({tuple(np.round(x / B * 50).astype(int)) for x in endpoints_x})
        return best_x, best_f, n_basins_f, n_basins_x, len(endpoints_f)

    # pylint: disable=too-many-locals
    def solve_grid_polish(self) -> dict:
        """
        This function certifies a non-convex case at low k with a dense grid over
        the capped simplex plus a local polish of the best nodes. Interior optima
        are covered because STB=1 key outputs reward under-spending. The ground
        truth is verified at grid resolution, and the polished endpoints give the
        basin-count estimate.
        :return: the oracle block for the manifest
        """
        B, k = self.budget, self.k
        nodes, R = _capped_grid(k, B)
        rng = np.random.default_rng(SOLVER_SEED)

        per_scenario = {}
        for scenario in self.scenarios:
            objective = self._objective(scenario)
            vals = self._batch_objective(nodes, scenario)
            for x_chk, v_chk in zip(nodes[:3], vals[:3]):
                v_ref = objective(x_chk)
                assert abs(v_chk - v_ref) <= 1e-9 * max(
                    1.0, abs(v_ref)
                ), "fast objective path disagrees with evaluate_allocation"

            top = np.argsort(vals)[-GRID_N_POLISH:]
            grid_best = int(top[-1])
            starts = [nodes[i] for i in top] + _mixed_starts(rng, k, B, GRID_N_EXTRA_STARTS)
            best_x, best_f, n_basins_f, n_basins_x, n_ok = self._polish(
                objective, starts, nodes[grid_best], vals[grid_best]
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
        return {"method": "dense_grid_polish", "verified": True, "per_scenario": per_scenario}

    def solve_multistart_proxy(self, n_starts: int = PROXY_N_STARTS) -> dict:
        """
        This function returns a best-of-multistart PROXY for a non-convex case
        above the grid limit. It carries NO optimality guarantee: ``verified`` is
        false and gaps against it are relative, not absolute.
        :param n_starts: the number of multistart points
        :return: the oracle block for the manifest
        """
        B, k = self.budget, self.k
        rng = np.random.default_rng(SOLVER_SEED)
        starts = (
            _mixed_starts(rng, k, B, max(0, n_starts - k - 1))
            + [B * np.eye(k)[i] for i in range(k)]
            + [np.full(k, B / k)]
        )

        per_scenario = {}
        for scenario in self.scenarios:
            objective = self._objective(scenario)
            uniform = np.full(k, B / k)
            best_x, best_f, n_basins_f, n_basins_x, n_ok = self._polish(objective, starts, uniform, objective(uniform))
            per_scenario[scenario] = {
                "x_star": [round(float(v), 10) for v in best_x],
                "f_star": float(best_f),
                "certificate": {
                    "basis": "best of tight multistarts - PROXY ONLY, no optimality guarantee (nonconvex, high k)",
                    "n_starts": len(starts),
                    "n_polished_converged": n_ok,
                    "n_basins_f_estimate": n_basins_f,
                    "n_basins_x_estimate": n_basins_x,
                },
            }
        return {"method": "best_of_multistart_proxy", "verified": False, "per_scenario": per_scenario}

    def certify(self) -> dict:
        """
        This function selects the solver that fits the case's regime, runs it and
        patches the result into the case manifest. It also records how far the
        per-scenario optima lie apart (max pairwise L1 distance, scaled by B).
        :return: the oracle block that was written to the manifest
        """
        if self.manifest["regime"] == "convex":
            oracle = self.solve_linear() if self.manifest["appreciation"] == "linear" else self.solve_curved()
        else:
            oracle = self.solve_grid_polish() if self.k <= 6 else self.solve_multistart_proxy()

        xs = [np.asarray(s["x_star"], dtype=float) for s in oracle["per_scenario"].values()]
        pairs = [float(np.abs(a - b).sum()) / self.budget for a, b in combinations(xs, 2)]
        oracle["generator_version"] = GENERATOR_VERSION
        oracle["scenario_optimum_dispersion"] = round(max(pairs, default=0.0), 6)

        self.manifest["oracle"] = oracle
        (self.root / self.name / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return oracle


def certify_case(name: str, root: Path = DEFAULT_ROOT) -> dict:
    """
    This function computes a case's ground truth and patches it into its
    manifest.
    :param name: the case name
    :param root: the directory the case was written to
    :return: the oracle block that was written to the manifest
    """
    return Oracle(name, root).certify()


def main():
    """Certify the standard suite and print one line per case."""
    for params in standard_cases():
        oracle = certify_case(params.name)
        first = next(iter(oracle["per_scenario"].values()))
        print(
            f"{params.name}: method={oracle['method']} verified={oracle['verified']} "
            f"f*[scenario 1]={first['f_star']:.6f} dispersion={oracle['scenario_optimum_dispersion']}"
        )


if __name__ == "__main__":
    main()
