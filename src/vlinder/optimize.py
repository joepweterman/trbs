# pylint: disable=W0212,R0913,R0917,R0914

"""
This module is the single home for tRBS optimization. The :class:`Optimize`
class exposes one entry point, :meth:`Optimize.run`, which dispatches to a named
optimization method (``"grid"`` combinatorial search or ``"slsqp"`` continuous
multi-start) — or to several at once, printing each method's result and returning
the best.

It also exposes a module-level pure function ``evaluate_allocation`` shared by all
methods. The function takes a deep copy of the caller's ``input_dict`` so that the
hundreds of objective evaluations a continuous optimizer makes do not corrupt
shared state.
"""

import copy
import math
import time
from dataclasses import dataclass, field
from math import comb
from itertools import combinations_with_replacement, permutations
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from vlinder.appreciate import Appreciate
from vlinder.evaluate import Evaluate
from vlinder.utils import suppress_print


def evaluate_allocation(input_dict, x, scenario, dmo_name):
    """
    Pure-function evaluator: weighted appreciation of allocation ``x`` for a
    given DMO under a given scenario, without mutating ``input_dict``.

    Preconditions (caller's responsibility — typically done once per optimizer run):
      - ``dmo_name`` must already exist in ``input_dict["decision_makers_options"]``
      - ``input_dict["decision_makers_option_value"]`` must already have a row for
        ``dmo_name`` (any feasible allocation is fine; it will be overwritten by ``x``)
      - ``input_dict["key_output_automatic"]``, ``key_output_start``, ``key_output_end``
        must be initialised (Appreciate uses these to fix the boundary points)

    :param input_dict: tRBS case input dictionary (NOT mutated)
    :param x: 1-D array-like of length ``len(input_dict["internal_variable_inputs"])``,
              the allocation to evaluate
    :param scenario: scenario name (str) — must be in ``input_dict["scenarios"]``
    :param dmo_name: decision-maker option name (str) the allocation belongs to
    :return: float — the value of ``output["decision_makers_option_appreciation"]``
    """
    local = copy.deepcopy(input_dict)
    idx = np.where(local["decision_makers_options"] == dmo_name)[0][0]
    local["decision_makers_option_value"][idx] = np.asarray(x)

    output = Evaluate(local).evaluate_selected_scenario(scenario)[dmo_name]
    Appreciate(local, output).appreciate_single_decision_maker_option(output)
    return float(output["decision_makers_option_appreciation"])


@dataclass
class OptimizationResult:
    """Unified result of an optimization run, for any method.

    ``grid`` fills ``method``/``dmo_name``/``allocation``/``appreciation`` and
    leaves the solver diagnostics ``None``; ``slsqp`` (and future continuous
    methods) fill everything. ``best_x`` / ``best_appreciation`` are read-only
    aliases kept for backward compatibility with the W2 experiment harness and
    the existing tests.
    """

    method: str
    dmo_name: str
    allocation: np.ndarray
    appreciation: float
    n_starts: Optional[int] = None
    n_converged: Optional[int] = None
    n_function_evals: Optional[int] = None
    wall_time_s: Optional[float] = None
    per_start_results: List[dict] = field(default_factory=list)

    @property
    def best_x(self):
        """Alias for :attr:`allocation` (kept for back-compat)."""
        return self.allocation

    @property
    def best_appreciation(self):
        """Alias for :attr:`appreciation` (kept for back-compat)."""
        return self.appreciation


class Optimize:
    """
    The Optimize class performs grid search optimization to find the optimal distribution of internal input values
    that maximizes the appreciation value of decision-maker options.
    """

    # String name → adapter method name. Add basin_hopping / genetic_algorithm here (W3+).
    METHOD_REGISTRY = {"grid": "_run_grid", "slsqp": "_run_slsqp"}
    DEFAULT_DMO_NAME = {"grid": "Optimized (grid)", "slsqp": "Optimized (SLSQP)"}

    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.boundaries = None
        self._k = len(input_dict["internal_variable_inputs"])

    def find_dict_values(self, scenario):
        """
        This function retrieves values based on the input and output dictionaries.
        """
        # Identify the decision-maker's option (DMO) with the highest appreciation in the given scenario
        dmo_name = self.output_dict[scenario]["highest_weighted_dmo"]

        # Identify the highest appreciation of that DMO
        max_appreciated_value = self.output_dict[scenario][dmo_name]["decision_makers_option_appreciation"]

        # Identify the distibution of that DMO
        decision_maker_options = self.input_dict["decision_makers_option_value"][
            np.where(self.input_dict["decision_makers_options"] == dmo_name)[0][0]
        ]

        best_dmo_data = {
            "dmo_name": dmo_name,
            "decision_maker_options": decision_maker_options,
            "max_appreciated_value": max_appreciated_value,
        }

        # Sum the values for this DMO (this represents the total investment)
        max_investment = sum(decision_maker_options)

        return best_dmo_data, max_investment

    @staticmethod
    def scale_max_investment(max_investment):
        """
        This function scales down the maximum investment value to make it more manageable for combinatorial purposes.
        It rounds the investment down to the nearest hundred, taking into account the order of magnitude.
        """
        # Determine the order of magnitude of the investment (in thousands)
        order_of_magnitude = math.floor(math.log10(abs(max_investment))) - 3

        # Normalize the value to thousands
        normalized_max_investment = max_investment / (10**order_of_magnitude)

        # Round the value to the nearest hundred
        scaled_max_investment = math.floor(round(normalized_max_investment, 1) / 100) * 100

        return scaled_max_investment

    @staticmethod
    def calculate_step_size(max_investment, scaled_max_investment, num_internal_inputs, max_combinations):
        """
        This function calculates the optimal step size to reduce the number of combinations.
        The goal is to stay under the maximum allowable number of combinations for efficiency.
        """
        step_size_tmp = 1

        while True:
            # Calculate the number of units with the current step size
            units = scaled_max_investment // step_size_tmp

            # Calculate the number of combinations using binomial coefficient
            combinations = comb(units + num_internal_inputs - 1, num_internal_inputs - 1)

            if combinations <= max_combinations and scaled_max_investment % step_size_tmp == 0:
                # If the number of combinations is within the constraints, use this step size
                break

            # Increase the step size if the number of combinations exceeds the limit
            step_size_tmp += 1

        # Scale the step size based on the original max investment
        step_size = max_investment / (scaled_max_investment / step_size_tmp)

        return step_size

    @staticmethod
    def generate_combinations(max_investment, step_size, num_internal_inputs):
        """
        This function generates all valid combinations of internal input values whose sum equals max_investment.
        """
        base_combinations = np.arange(0, max_investment + step_size, step_size)
        valid_combinations = []

        # Generate combinations and filter those that sum to the max investment
        for combination in combinations_with_replacement(base_combinations, num_internal_inputs):
            if sum(combination) == max_investment:
                # Add all unique permutations of the combination
                for perm in set(permutations(combination)):
                    valid_combinations.append(perm)

        return valid_combinations

    @suppress_print
    def grid_search(self, scenario, combinations, opt_dmo_name, best_dmo_data):
        """
        Performs a grid search over all possible combinations of internal input values.
        The function evaluates each combination, calculates the appreciation value, and returns the best one.
        """
        # Get minimum and maximum values for the key outputs across all scenarios
        self.boundaries = Appreciate(self.input_dict, self.output_dict)._get_start_and_end_points()

        # Initialize the grid search decision-maker option
        self.input_dict["decision_makers_options"] = np.array(
            np.append(self.input_dict["decision_makers_options"], opt_dmo_name), dtype=object
        )
        self.input_dict["decision_makers_option_value"] = np.vstack(
            [self.input_dict["decision_makers_option_value"], best_dmo_data["decision_maker_options"]]
        )
        self.input_dict["key_output_automatic"] = np.zeros(len(self.input_dict["key_output_automatic"]), dtype=int)
        self.input_dict["key_output_start"] = np.array([value[0] for value in self.boundaries.values()])
        self.input_dict["key_output_end"] = np.array([value[1] for value in self.boundaries.values()])

        # Arrays to store results
        appreciated_values = []
        tmp_opt_decision_maker_options = None
        tmp_opt_max_appreciated_value = -np.inf

        # Evaluate each combination
        for index, combination in enumerate(combinations):
            comb_array = np.array(combination)

            # Ensure the combination length matches the number of internal inputs
            if len(comb_array) == len(self.input_dict["internal_variable_inputs"]):
                # Pure-function eval (W2 refactor): no per-iter mutation of self.input_dict.
                # The final winning allocation is written back to self.input_dict in the
                # post-loop block below, so external observers see the same end-state.
                appreciated_value = evaluate_allocation(self.input_dict, comb_array, scenario, opt_dmo_name)
                appreciated_values.append((index, comb_array, appreciated_value))

                # Update the best combination if this is the highest appreciation value
                if appreciated_value > tmp_opt_max_appreciated_value:
                    tmp_opt_max_appreciated_value = appreciated_value
                    tmp_opt_decision_maker_options = comb_array

        if tmp_opt_max_appreciated_value > best_dmo_data["max_appreciated_value"]:
            self.input_dict["decision_makers_option_value"][
                np.where(self.input_dict["decision_makers_options"] == opt_dmo_name)[0][0]
            ] = tmp_opt_decision_maker_options
            best_dmo = opt_dmo_name
            best_appreciated_value = tmp_opt_max_appreciated_value
        else:
            self.input_dict["decision_makers_option_value"][
                np.where(self.input_dict["decision_makers_options"] == opt_dmo_name)[0][0]
            ] = best_dmo_data["decision_maker_options"]
            best_dmo = best_dmo_data["dmo_name"]
            best_appreciated_value = best_dmo_data["max_appreciated_value"]

        return best_dmo, best_appreciated_value

    def optimize_single_scenario(self, scenario, tmp_opt_dmo_name, max_combinations):
        """
        Wrapper function that performs the full grid search optimization process.
        It retrieves values, calculates the step size, generates valid combinations,
        and finds the best distribution of internal inputs to maximize appreciation.
        """
        if tmp_opt_dmo_name in self.input_dict["decision_makers_options"]:
            print("This DMO name already exits, please choose another")
            return self.input_dict

        # Step 1: Retrieve values and setup boundaries
        best_dmo_data, max_investment = self.find_dict_values(scenario)

        # Step 2: Scale down the maximum investment for more efficient combinatorial calculations
        scaled_max_investment = self.scale_max_investment(max_investment)

        # TO DO: max_combinations based on CPU
        # Step 3: Find the optimal step size for generating combinations
        step_size = self.calculate_step_size(
            max_investment, scaled_max_investment, len(self.input_dict["internal_variable_inputs"]), max_combinations
        )

        # Step 4: Generate all valid combinations of internal input values
        combinations = self.generate_combinations(
            max_investment, step_size, len(self.input_dict["internal_variable_inputs"])
        )

        # Step 5: Perform grid search over the generated combinations and fill in input_dict
        best_dmo, best_appreciated_value = self.grid_search(scenario, combinations, tmp_opt_dmo_name, best_dmo_data)

        # Print the results
        print("For scenario: ", scenario)
        print("------------------------------------")
        print(
            "Initial best appreciation:",
            round(best_dmo_data["max_appreciated_value"], 2),
            "for DMO:",
            best_dmo_data["dmo_name"],
        )
        print(
            "With the following internal variable distribution: ",
            "["
            + ", ".join(
                str(int(num)) if num.is_integer() else str(num) for num in best_dmo_data["decision_maker_options"]
            )
            + "]",
        )
        print("------------------------------------")
        print("Optimized appreciation:", round(best_appreciated_value, 2), "for DMO:", best_dmo)
        print(
            "With the following internal variable distribution: ",
            "["
            + ", ".join(
                str(int(num)) if num.is_integer() else str(num)
                for num in self.input_dict["decision_makers_option_value"][
                    np.where(self.input_dict["decision_makers_options"] == best_dmo)[0][0]
                ]
            )
            + "]",
        )
        print("------------------------------------")
        print(
            "Total increase appreciated value:",
            round(best_appreciated_value - best_dmo_data["max_appreciated_value"], 2),
        )

        return self.input_dict

    # ==================================================================
    # Continuous optimization (SLSQP) — merged from the former
    # optimize_continuous.ContinuousOptimize (W2 thesis work). Treats the
    # allocation problem {x : Σx_i ≤ B, x_i ≥ 0} (capped simplex; under-spending
    # feasible) as a continuous NLP and solves it with multi-start SLSQP.
    # basin_hopping / GA land here in W3+.
    # ==================================================================
    def _prepare_input_dict(self, dmo_name, reference_allocation):
        """Register the optimizer's DMO + freeze the appreciation boundaries.

        ``evaluate_allocation`` requires ``dmo_name`` to exist with a feasible
        row, and ``key_output_automatic``/``key_output_start``/``key_output_end``
        to be fixed so the appreciation curve is identical across every objective
        evaluation. Idempotent — safe to call repeatedly with the same name.
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

    def _objective(self, x, scenario, dmo_name, eval_counter):
        """Negated appreciation (scipy minimizes); ``eval_counter`` counts calls."""
        eval_counter[0] += 1
        return -evaluate_allocation(self.input_dict, x, scenario, dmo_name)

    def _dirichlet_starts(self, n_starts, budget, seed=None):
        """Multi-start points on the budget face via ``Dirichlet(1,...,1) * budget``.

        These lie on Σx_i = B, which is feasible under the capped-simplex
        constraint Σx_i ≤ B (the face is a subset of the capped simplex). SLSQP
        can still move into the interior when the gradient favours under-spending.
        Interior starts can be added later for cases whose optimum is strictly
        interior (the curvature-heavy synthetic regimes)."""
        rng = np.random.default_rng(seed)
        return rng.dirichlet(np.ones(self._k), size=n_starts) * budget

    def _slsqp_from_start(self, x0, scenario, dmo_name, budget, eval_counter):
        """Single SLSQP solve from ``x0``. Constraint Σx_i ≤ B (capped simplex),
        bounds [0, B]^k.

        The budget is an upper bound, not an equality: under-spending is feasible
        (Vlinder, 2026-06-26). This matters on appreciation surfaces that are
        non-monotone in total spend — e.g. IZZ, where spending the whole budget
        can lower appreciation (exp02). On monotone cases (Beerwiser, Refugee)
        the constraint binds and the solution still spends the full budget, so
        the formulation reduces to the former equality there. scipy reads
        ``ineq`` constraints as ``fun(x) >= 0``, hence ``B - Σx_i >= 0``.
        """
        constraints = ({"type": "ineq", "fun": lambda x: float(budget - np.sum(x))},)
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

    def optimize_slsqp(
        self,
        scenario,
        budget,
        dmo_name="Optimized (SLSQP)",
        reference_allocation=None,
        n_starts=100,
        seed=None,
    ):
        """Multi-start SLSQP on the simplex-constrained appreciation objective.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param budget: total allocation budget (upper bound: Σx_i ≤ budget; under-spending feasible).
        :param dmo_name: name under which the winning allocation is written back to
            ``input_dict`` (registered if absent).
        :param reference_allocation: feasible allocation used to seed the new DMO
            row (only when registering); defaults to the first existing DMO's row.
        :param n_starts: number of Dirichlet multi-starts (default 100).
        :param seed: RNG seed for reproducible starts.
        :return: an :class:`OptimizationResult` with per-start diagnostics.
        """
        if reference_allocation is None:
            reference_allocation = self.input_dict["decision_makers_option_value"][0].copy()
        self._prepare_input_dict(dmo_name, reference_allocation)

        starts = self._dirichlet_starts(n_starts, budget, seed=seed)
        eval_counter = [0]
        t0 = time.perf_counter()

        per_start = []
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

        # Write the winning allocation back so downstream consumers (visuals,
        # reports) see it as a regular DMO.
        if best_x is not None:
            idx = np.where(self.input_dict["decision_makers_options"] == dmo_name)[0][0]
            self.input_dict["decision_makers_option_value"][idx] = best_x

        return OptimizationResult(
            method="slsqp",
            dmo_name=dmo_name,
            allocation=best_x if best_x is not None else np.full(self._k, np.nan),
            appreciation=-best_neg_f if best_x is not None else float("nan"),
            n_starts=n_starts,
            n_converged=sum(1 for r in per_start if r["success"]),
            n_function_evals=eval_counter[0],
            wall_time_s=time.perf_counter() - t0,
            per_start_results=per_start,
        )

    # ==================================================================
    # Unified dispatch — one entry point for every method.
    # ==================================================================
    def _infer_budget(self):
        """Total budget = sum of the first DMO's allocation (every DMO spends the same)."""
        return float(np.sum(self.input_dict["decision_makers_option_value"][0]))

    def _run_grid(  # pylint: disable=unused-argument
        self, scenario, *, dmo_name, budget=None, max_combinations=60000, **_ignored
    ):
        """Grid-search adapter → :class:`OptimizationResult`.

        ``budget`` is accepted for a uniform solver signature but unused — grid
        derives its own ``max_investment`` from the highest-weighted DMO.
        """
        self.optimize_single_scenario(scenario, dmo_name, max_combinations)
        idx = np.where(self.input_dict["decision_makers_options"] == dmo_name)[0][0]
        allocation = np.asarray(self.input_dict["decision_makers_option_value"][idx])
        appreciation = evaluate_allocation(self.input_dict, allocation, scenario, dmo_name)
        return OptimizationResult(method="grid", dmo_name=dmo_name, allocation=allocation, appreciation=appreciation)

    def _run_slsqp(self, scenario, *, dmo_name, budget, **method_kwargs):
        """SLSQP adapter → :class:`OptimizationResult`."""
        return self.optimize_slsqp(scenario, budget, dmo_name=dmo_name, **method_kwargs)

    def run(self, scenario, method="grid", *, dmo_name=None, budget=None, **method_kwargs):
        """Run one or several optimization methods and return the best result.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param method: a single method name (str) or a list of names. Supported:
            ``"grid"``, ``"slsqp"``. Unknown names raise ``NotImplementedError``.
        :param dmo_name: name for the optimizer DMO. Defaults per method
            (:attr:`DEFAULT_DMO_NAME`); for a list, each method uses its own
            default unless an explicit name is given.
        :param budget: total allocation budget; inferred from the first DMO if None.
        :param method_kwargs: forwarded to the chosen solver (grid:
            ``max_combinations``; slsqp: ``n_starts``, ``seed``, ``reference_allocation``).
        :return: a single :class:`OptimizationResult`. For a list of methods,
            every method's appreciation + allocation is printed and the best is
            returned; only the winning allocation is written back as a new DMO.
        """
        if budget is None:
            budget = self._infer_budget()
        methods = [method] if isinstance(method, str) else list(method)
        for name in methods:
            if name not in self.METHOD_REGISTRY:
                raise NotImplementedError(f"method={name!r} not implemented. Supported: {list(self.METHOD_REGISTRY)}")

        # Single method → run on self so the winning DMO + frozen boundaries land here.
        if len(methods) == 1:
            chosen = methods[0]
            name = dmo_name or self.DEFAULT_DMO_NAME[chosen]
            return getattr(self, self.METHOD_REGISTRY[chosen])(scenario, dmo_name=name, budget=budget, **method_kwargs)

        # Multiple methods → isolate each on its own deepcopy so they don't see
        # each other's appended DMOs; print all, then adopt the best run's
        # input_dict (only the winning DMO is written back).
        runs = []
        for chosen in methods:
            sub = Optimize(copy.deepcopy(self.input_dict), self.output_dict)
            name = dmo_name or self.DEFAULT_DMO_NAME[chosen]
            res = getattr(sub, self.METHOD_REGISTRY[chosen])(scenario, dmo_name=name, budget=budget, **method_kwargs)
            print(f"[{chosen}] appreciation={res.appreciation:.6f} allocation={np.asarray(res.allocation).tolist()}")
            runs.append((res, sub))

        best_res, best_sub = max(runs, key=lambda pair: pair[0].appreciation)
        self.input_dict = best_sub.input_dict
        return best_res
