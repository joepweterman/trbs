# pylint: disable=W0212,R0913,R0917,R0914

"""
Continuous-space optimization for tRBS — W2 thesis scaffold.

This module is the sibling of ``vlinder.optimize``: where the legacy
``Optimize`` class performs a combinatorial grid search over the simplex
{x : Σx_i = B, x_i ≥ 0}, ``ContinuousOptimize`` treats the same allocation
problem as a continuous nonlinear program and solves it with derivative-free
local methods (SLSQP in this iteration; basin-hopping and GA in W3+).

Reference: docs/thesis/w1_optimization_methods.md §2.1 — academic justification
for SLSQP under simplex + bounds, including the KKT-multiplier interpretation
of the equality constraint as the marginal value of capital.

Public API (mirroring ``Optimize``)::

    optimizer = ContinuousOptimize(input_dict, output_dict)
    result = optimizer.optimize(scenario, method="slsqp", n_starts=100)
    # result.best_x, result.best_appreciation, result.per_start_results, ...

The high-level wrapper on the tRBS class is
``TheResponsibleBusinessSimulator.optimize_continuous(scenario, method=..., **kwargs)``.
"""

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from vlinder.appreciate import Appreciate
from vlinder.optimize import evaluate_allocation


@dataclass
class ContinuousOptimizationResult:
    """Structured result of a continuous-optimizer run.

    Holding more than just (x, f) makes the W2 empirical report (deliverable
    ``docs/thesis/w2_slsqp_results.md``) trivial — every diagnostic the
    methodology chapter wants is already captured here.
    """

    best_x: np.ndarray
    best_appreciation: float
    n_starts: int
    n_converged: int
    n_function_evals: int
    wall_time_s: float
    method: str
    per_start_results: List[dict] = field(default_factory=list)


class ContinuousOptimize:
    """Continuous-relaxation optimizer for tRBS decision-maker option allocations."""

    SUPPORTED_METHODS = ("slsqp",)  # basin_hopping (W3), genetic_algorithm (W4)

    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self._k = len(input_dict["internal_variable_inputs"])

    # ------------------------------------------------------------------
    # Phase-A setup — mirrors ``Optimize.grid_search`` lines 122-133.
    # Idempotent: safe to call multiple times with the same dmo_name.
    # ------------------------------------------------------------------
    def _prepare_input_dict(self, dmo_name: str, reference_allocation: np.ndarray) -> None:
        """Register the optimizer's DMO + freeze appreciation boundaries.

        ``evaluate_allocation`` requires:
          1. ``dmo_name`` to be in ``decision_makers_options``
          2. A matching row in ``decision_makers_option_value`` (any feasible
             allocation; it will be overwritten per objective call)
          3. ``key_output_automatic`` / ``key_output_start`` / ``key_output_end``
             set, so Appreciate fixes the appreciation curve consistently
             across all objective evaluations

        :param dmo_name: name to register (added if absent)
        :param reference_allocation: feasible allocation used to seed the row
        """
        if dmo_name not in self.input_dict["decision_makers_options"]:
            self.input_dict["decision_makers_options"] = np.array(
                np.append(self.input_dict["decision_makers_options"], dmo_name), dtype=object
            )
            self.input_dict["decision_makers_option_value"] = np.vstack(
                [self.input_dict["decision_makers_option_value"], np.asarray(reference_allocation)]
            )

        boundaries = Appreciate(self.input_dict, self.output_dict)._get_start_and_end_points()
        self.input_dict["key_output_automatic"] = np.zeros(len(self.input_dict["key_output_automatic"]), dtype=int)
        self.input_dict["key_output_start"] = np.array([v[0] for v in boundaries.values()])
        self.input_dict["key_output_end"] = np.array([v[1] for v in boundaries.values()])

    # ------------------------------------------------------------------
    # Objective: scipy.optimize MINIMIZES, so we negate appreciation.
    # The eval_counter mutable list is a cheap, picklable counter that
    # survives the inner loop — joblib parallelization (W3+) will use a
    # multiprocessing.Value instead.
    # ------------------------------------------------------------------
    def _objective(self, x: np.ndarray, scenario: str, dmo_name: str, eval_counter: list) -> float:
        eval_counter[0] += 1
        return -evaluate_allocation(self.input_dict, x, scenario, dmo_name)

    # ------------------------------------------------------------------
    # Starting points: uniform sample on the simplex via Dirichlet(1,...,1).
    # ``Dirichlet(1,...,1) * B`` is feasible by construction (sums exactly
    # to B, all positive), so SLSQP has no initial-feasibility work.
    # ------------------------------------------------------------------
    def _dirichlet_starts(self, n_starts: int, budget: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.dirichlet(np.ones(self._k), size=n_starts) * budget

    # ------------------------------------------------------------------
    # Single SLSQP solve from a given start. Constraint: Σx_i = B; bounds
    # [(0, B)]^k. No simplex transformation — see methodology doc §2.1.
    # ------------------------------------------------------------------
    def _slsqp_from_start(
        self,
        x0: np.ndarray,
        scenario: str,
        dmo_name: str,
        budget: float,
        eval_counter: list,
    ) -> OptimizeResult:
        constraints = ({"type": "eq", "fun": lambda x: float(np.sum(x) - budget)},)
        bounds = [(0.0, float(budget))] * self._k
        return minimize(
            self._objective,
            x0,
            args=(scenario, dmo_name, eval_counter),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 100, "disp": False},
        )

    # ------------------------------------------------------------------
    # Top-level SLSQP + Dirichlet multi-start.
    # ------------------------------------------------------------------
    def optimize_slsqp(
        self,
        scenario: str,
        budget: float,
        dmo_name: str = "Optimized (SLSQP)",
        reference_allocation=None,
        n_starts: int = 100,
        seed=None,
    ) -> ContinuousOptimizationResult:
        """Multi-start SLSQP on the simplex-constrained appreciation objective.

        :param scenario: scenario name (must be in input_dict["scenarios"])
        :param budget: total allocation budget (Σx_i = budget)
        :param dmo_name: name under which the winning allocation is written back
            to ``input_dict``. Will be registered if not already present.
        :param reference_allocation: feasible allocation used to seed the new
            DMO row (only used when registering). Defaults to the first
            existing DMO's allocation (any feasible point works).
        :param n_starts: number of Dirichlet multi-starts (default 100).
        :param seed: RNG seed for reproducible starts.
        :return: :class:`ContinuousOptimizationResult` with the winner +
            per-start diagnostics for the W2 empirical report.
        """
        if reference_allocation is None:
            reference_allocation = self.input_dict["decision_makers_option_value"][0].copy()
        self._prepare_input_dict(dmo_name, reference_allocation)

        starts = self._dirichlet_starts(n_starts, budget, seed=seed)
        eval_counter = [0]
        t0 = time.perf_counter()

        per_start: List[dict] = []
        best_x = None
        best_neg_f = np.inf

        for i, x0 in enumerate(starts):
            res = self._slsqp_from_start(x0, scenario, dmo_name, budget, eval_counter)
            per_start.append(
                {
                    "i": i,
                    "x0": np.asarray(x0),
                    "x": np.asarray(res.x),
                    "appreciation": float(-res.fun),
                    "success": bool(res.success),
                    "nit": int(res.nit),
                    "message": str(res.message),
                }
            )
            if res.success and res.fun < best_neg_f:
                best_neg_f = res.fun
                best_x = np.asarray(res.x)

        # Write the winning allocation back to input_dict so downstream
        # consumers (visuals, reports) see it as a regular DMO.
        if best_x is not None:
            idx = np.where(self.input_dict["decision_makers_options"] == dmo_name)[0][0]
            self.input_dict["decision_makers_option_value"][idx] = best_x

        return ContinuousOptimizationResult(
            best_x=best_x if best_x is not None else np.full(self._k, np.nan),
            best_appreciation=-best_neg_f if best_x is not None else float("nan"),
            n_starts=n_starts,
            n_converged=sum(1 for r in per_start if r["success"]),
            n_function_evals=eval_counter[0],
            wall_time_s=time.perf_counter() - t0,
            method="slsqp",
            per_start_results=per_start,
        )

    # ------------------------------------------------------------------
    # Public dispatch entry — mirrors Optimize.optimize_single_scenario.
    # ------------------------------------------------------------------
    def optimize(self, scenario: str, method: str = "slsqp", **kwargs) -> ContinuousOptimizationResult:
        """Dispatch to the chosen continuous-optimization method.

        Currently only ``method="slsqp"`` is implemented; ``"basin_hopping"``
        and ``"genetic_algorithm"`` raise :class:`NotImplementedError` and are
        planned for W3+ per the thesis methods shortlist
        (docs/thesis/w1_optimization_methods.md §2).

        If ``budget`` is not passed, it is inferred from the sum of the first
        existing DMO's allocation — appropriate for tRBS cases where every DMO
        spends the same total budget across internal variables.
        """
        if method not in self.SUPPORTED_METHODS:
            raise NotImplementedError(
                f"method={method!r} not implemented yet. "
                f"Supported in W2: {self.SUPPORTED_METHODS}. "
                f"basin_hopping/genetic_algorithm scheduled for W3+."
            )
        if "budget" not in kwargs:
            kwargs["budget"] = float(np.sum(self.input_dict["decision_makers_option_value"][0]))
        return self.optimize_slsqp(scenario, **kwargs)
