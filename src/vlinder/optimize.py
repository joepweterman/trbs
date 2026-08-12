# pylint: disable=W0212,R0913,R0917,R0914
# pylint: disable=C0302  # single home for ALL optimization methods (Vlinder decision 2026-06-01)

"""
This module contains the Optimize class, which performs grid search optimization to maximize the
appreciation of decision-maker options.

Grid search is no longer the only way to do that, so the module is built up from general to
specific:

  1. ``evaluate_and_appreciate()`` and ``evaluate_allocation()`` are free functions that score a
     single allocation. Every solver calls them and nothing else touches the tRBS engine.
  2. ``OptimizationResult`` is the container every solver returns.
  3. ``BaseSolver`` holds what all solvers share: the budget, registering the optimizer's own
     decision-maker option, writing the winning allocation back, and staying inside the feasible
     set.
  4. The solvers themselves. ``GridSearch`` enumerates the budget face. ``SLSQPSolver`` runs
     multi-start gradient optimization. ``BasinHoppingSolver`` extends it with an escape loop for
     surfaces with more than one optimum. ``GeneticAlgorithmSolver`` is the derivative-free
     alternative. ``MdbhSolver`` is a research method that walks the simplex directly.
  5. ``Optimize`` is the orchestrator. It picks a solver (automatically, by name, or several at
     once) and runs it.

The feasible set for the continuous solvers is the capped simplex, ``{x : x >= 0, sum(x) <= B}``:
spending less than the whole budget is allowed. Grid search is the exception and enumerates the
budget face ``sum(x) = B`` only; see :class:`GridSearch`.
"""

import copy
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import comb
from itertools import combinations_with_replacement, permutations
from typing import List, Optional

import numpy as np
from scipy.optimize import basinhopping, minimize

from vlinder.appreciate import Appreciate
from vlinder.evaluate import Evaluate
from vlinder.method_selection import MethodChoice, diagnose_case, select_method


def evaluate_and_appreciate(input_dict, x, scenario, dmo_name, start_and_end_points=None):
    """
    Evaluate allocation ``x`` and appreciate the result: the full tRBS output for a given
    decision-maker option under a given scenario, without mutating ``input_dict``.

    Preconditions (caller's responsibility, typically done once per optimizer run):
      - ``dmo_name`` must already exist in ``input_dict["decision_makers_options"]``
      - ``input_dict["decision_makers_option_value"]`` must already have a row for
        ``dmo_name`` (any feasible allocation is fine; it will be overwritten by ``x``)
      - ``input_dict["key_output_automatic"]``, ``key_output_start``, ``key_output_end``
        must be initialised (Appreciate uses these to fix the boundary points)

    :param input_dict: tRBS case input dictionary (NOT mutated)
    :param x: 1-D array-like of length ``len(input_dict["internal_variable_inputs"])``,
              the allocation to evaluate
    :param scenario: scenario name (str), must be in ``input_dict["scenarios"]``
    :param dmo_name: decision-maker option name (str) the allocation belongs to
    :param start_and_end_points: precomputed appreciation boundaries (see
        :meth:`BaseSolver._freeze_boundaries`). Optional; derived per call when omitted.
    :return: the output dictionary of this DMO, holding ``key_outputs``,
             ``appreciations``, ``weighted_appreciations`` and
             ``decision_makers_option_appreciation``
    """
    # DMO values that are whole numbers get imported as int64. Writing a float allocation into
    # an int row rounds it silently, which breaks gradient-based solvers. Cast to float to
    # avoid this. Copying that one table (the only one written here) is enough to leave the
    # caller's dictionary untouched, because Evaluate and Appreciate only read the rest.
    values = np.array(input_dict["decision_makers_option_value"], dtype=float)
    idx = np.where(input_dict["decision_makers_options"] == dmo_name)[0][0]
    values[idx] = np.asarray(x, dtype=float)
    local = {**input_dict, "decision_makers_option_value": values}

    # One decision-maker option, not all of them: evaluating the whole scenario would compute
    # every other option as well and throw those results away.
    output = Evaluate(local).evaluate_all_dependencies(scenario, dmo_name)
    Appreciate(local, output, start_and_end_points).appreciate_single_decision_maker_option(output)
    return output


def evaluate_allocation(input_dict, x, scenario, dmo_name, start_and_end_points=None):
    """
    Weighted appreciation of allocation ``x``: :func:`evaluate_and_appreciate` reduced to
    the single number every optimizer maximises. Same preconditions.

    :param input_dict: tRBS case input dictionary (NOT mutated)
    :param x: 1-D array-like of length ``len(input_dict["internal_variable_inputs"])``,
              the allocation to evaluate
    :param scenario: scenario name (str), must be in ``input_dict["scenarios"]``
    :param dmo_name: decision-maker option name (str) the allocation belongs to
    :param start_and_end_points: precomputed appreciation boundaries. Optional.
    :return: float, the value of ``output["decision_makers_option_appreciation"]``
    """
    return float(
        evaluate_and_appreciate(input_dict, x, scenario, dmo_name, start_and_end_points)[
            "decision_makers_option_appreciation"
        ]
    )


@dataclass
class OptimizationResult:  # pylint: disable=too-many-instance-attributes
    """Result of an optimization run, for any method.

    ``method`` through ``timestamp`` are filled by every solver. The solver diagnostics
    below them are filled where they mean something: grid search has no starts to
    converge, the genetic algorithm has no notion of a converged start. ``selection`` is
    filled only when the method was picked by ``method="auto"``, and then holds the
    diagnosis behind that pick.
    """

    method: str
    dmo_name: str
    scenario: str
    allocation: np.ndarray
    appreciation: float
    budget: float
    budget_spent: float
    calculation_time: float
    timestamp: str
    n_starts: Optional[int] = None
    n_converged: Optional[int] = None
    n_function_evals: Optional[int] = None
    per_start_results: List[dict] = field(default_factory=list)
    selection: Optional[MethodChoice] = None

    def summary(self) -> str:
        """A readable one-block summary, for notebooks and logs."""
        spent = f"{self.budget_spent:,.2f} of {self.budget:,.2f}"
        share = f"{self.budget_spent / self.budget:.1%}" if self.budget else "n/a"
        lines = [
            f"method          {self.method}",
            f"scenario        {self.scenario}",
            f"DMO             {self.dmo_name}",
            f"appreciation    {self.appreciation:.4f}",
            f"budget spent    {spent} ({share})",
            f"allocation      {np.round(np.asarray(self.allocation, dtype=float), 2).tolist()}",
            f"calc. time      {self.calculation_time:.2f} s",
            f"evaluations     {self.n_function_evals if self.n_function_evals is not None else 'n/a'}",
            f"run at          {self.timestamp}",
        ]
        if self.selection is not None:
            lines.append(f"chosen because  {self.selection.reason}")
        return "\n".join(lines)


class BaseSolver:
    """What every solver needs: a budget, its own decision-maker option, and a feasible answer.

    A solver is constructed on a case (``input_dict`` + ``output_dict``) and run through
    :meth:`solve`. Solvers mutate ``input_dict`` in exactly two places: registering their
    decision-maker option (:meth:`_prepare_input_dict`) and writing the winning allocation back
    to it (:meth:`_write_back_result`). Everything in between goes through the free functions at
    the top of this module, which never mutate the case.
    """

    #: Name of the decision-maker option this solver writes its answer to.
    default_dmo_name = "Optimized"
    #: Name of the method as reported on :class:`OptimizationResult`.
    method_name = "base"
    #: Short label used when a configured base name is extended with the method.
    method_label = "base"

    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self._k = len(input_dict["internal_variable_inputs"])
        # Appreciation boundaries once frozen; see _freeze_boundaries.
        self._frozen_boundaries = None

    @property
    def budget(self):
        """Total budget available to allocate.

        The largest amount any decision-maker option spends. Taking the first option instead
        would read a "do nothing" option (all zeros, as in the NEMO case) as a zero budget.
        """
        values = np.asarray(self.input_dict["decision_makers_option_value"], dtype=float)
        return float(max(np.sum(row) for row in values))

    def solve(self, scenario, dmo_name, budget, **kwargs):
        """Run the solver and return an :class:`OptimizationResult`. Implemented per solver."""
        raise NotImplementedError(f"{type(self).__name__} does not implement solve()")

    def _prepare_input_dict(self, dmo_name, reference_allocation=None):
        """Register the solver's decision-maker option and fix the appreciation boundaries.

        The free evaluation functions need ``dmo_name`` to exist with a feasible row, and the
        appreciation boundaries to be fixed, so that the appreciation curve is identical across
        every evaluation instead of drifting with whichever allocation is on the table.
        Idempotent, so it is safe to call repeatedly with the same name.
        """
        if reference_allocation is None:
            reference_allocation = self.input_dict["decision_makers_option_value"][0].copy()
        if dmo_name not in self.input_dict["decision_makers_options"]:
            self.input_dict["decision_makers_options"] = np.array(
                np.append(self.input_dict["decision_makers_options"], dmo_name), dtype=object
            )
            self.input_dict["decision_makers_option_value"] = np.vstack(
                [self.input_dict["decision_makers_option_value"], np.asarray(reference_allocation)]
            )
        # Float dtype so the winning (generally non-integer) allocation is written back exactly.
        self.input_dict["decision_makers_option_value"] = np.asarray(
            self.input_dict["decision_makers_option_value"], dtype=float
        )

        boundaries = Appreciate(self.input_dict, self.output_dict)._get_start_and_end_points()
        self.input_dict["key_output_automatic"] = np.zeros(len(self.input_dict["key_output_automatic"]), dtype=int)
        self.input_dict["key_output_start"] = np.array([v[0] for v in boundaries.values()])
        self.input_dict["key_output_end"] = np.array([v[1] for v in boundaries.values()])
        self._freeze_boundaries()

    def _freeze_boundaries(self):
        """Cache the appreciation boundaries for the evaluations to come.

        Once ``key_output_automatic`` is zero everywhere the boundaries no longer depend on the
        allocation, yet Appreciate would rebuild them (through a pandas DataFrame) on every
        single evaluation. Recomputed once here, on the already-frozen dictionary, so it is by
        construction the same result every evaluation would have derived.
        """
        self._frozen_boundaries = Appreciate(self.input_dict, self.output_dict)._get_start_and_end_points()

    def _write_back_result(self, dmo_name, allocation):
        """Store the winning allocation on its decision-maker option.

        Downstream consumers (visuals, reports, the front end) read the case, not the result
        object, so the answer has to land in ``input_dict`` as a regular option.
        """
        idx = np.where(self.input_dict["decision_makers_options"] == dmo_name)[0][0]
        self.input_dict["decision_makers_option_value"][idx] = np.asarray(allocation, dtype=float)

    def _build_result(self, dmo_name, scenario, allocation, appreciation, budget, started_at, **diagnostics):
        """Assemble an :class:`OptimizationResult` with the fields every solver reports."""
        allocation = np.asarray(allocation, dtype=float)
        return OptimizationResult(
            method=self.method_name,
            dmo_name=dmo_name,
            scenario=scenario,
            allocation=allocation,
            appreciation=float(appreciation),
            budget=float(budget),
            budget_spent=float(np.sum(allocation)) if np.all(np.isfinite(allocation)) else float("nan"),
            calculation_time=time.perf_counter() - started_at,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **diagnostics,
        )

    def _objective(self, x, scenario, dmo_name, eval_counter):
        """Negated appreciation (scipy minimizes); ``eval_counter`` counts calls."""
        eval_counter[0] += 1
        return -evaluate_allocation(self.input_dict, x, scenario, dmo_name, self._frozen_boundaries)

    @staticmethod
    def _project_capped_simplex(x, budget):
        """Pull a point back into ``{x >= 0, sum(x) <= budget}``.

        Solvers satisfy their constraints only to their own tolerance, while everything
        downstream assumes a strictly feasible allocation. Clip negatives and rescale an
        over-budget sum; on a converged solution this is a correction of order 1e-8.
        """
        x = np.clip(np.asarray(x, dtype=float), 0.0, None)
        total = float(np.sum(x))
        if total > budget:
            x = x * (budget / total)
        return x


class GridSearch(BaseSolver):
    """Enumerate the budget face and keep the best combination.

    Grid search is the original tRBS optimizer and the baseline the continuous solvers are
    compared against. Unlike them it works on the ordinary simplex ``sum(x) = B`` rather than
    the capped simplex ``sum(x) <= B``: it only builds combinations that spend the budget
    exactly, so an optimum that leaves part of the budget unspent is outside its search space.

    The step size is coarsened until the number of combinations stays under
    ``max_combinations``. The true cost is higher than that number, because every combination is
    expanded into all its distinct permutations.
    """

    default_dmo_name = "Optimized (grid)"
    method_name = "grid"
    method_label = "grid"

    def solve(self, scenario, dmo_name, budget=None, *, max_combinations=60000, **_ignored):
        """Enumerate the budget face for ``scenario`` and write back the best allocation.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param dmo_name: name under which the winning allocation is written back.
        :param budget: accepted for a uniform solver signature but unused. Grid search derives
            its own total from the best-performing decision-maker option.
        :param max_combinations: upper bound on the number of combinations to build.
        :return: an :class:`OptimizationResult`.
        """
        started_at = time.perf_counter()
        best_dmo_data, max_investment = self.find_dict_values(scenario)
        self._prepare_input_dict(dmo_name, best_dmo_data["decision_maker_options"])

        scaled_max_investment = self.scale_max_investment(max_investment)
        step_size = self.calculate_step_size(max_investment, scaled_max_investment, self._k, max_combinations)
        combinations = self.generate_combinations(max_investment, step_size, self._k)

        # Evaluate and appreciate without changing self.input_dict during the loop. Only the
        # best result is written back after the loop finishes.
        best_allocation = None
        best_appreciation = -np.inf
        for combination in combinations:
            allocation = np.array(combination)
            if len(allocation) != self._k:
                continue
            appreciation = evaluate_allocation(
                self.input_dict, allocation, scenario, dmo_name, self._frozen_boundaries
            )
            if appreciation > best_appreciation:
                best_appreciation = appreciation
                best_allocation = allocation

        # An existing option may already beat everything on the grid; then that option is the
        # answer and the optimizer's own option mirrors it.
        if best_appreciation > best_dmo_data["max_appreciated_value"]:
            winning_dmo = dmo_name
        else:
            best_allocation = best_dmo_data["decision_maker_options"]
            best_appreciation = best_dmo_data["max_appreciated_value"]
            winning_dmo = best_dmo_data["dmo_name"]
        self._write_back_result(dmo_name, best_allocation)

        return self._build_result(
            dmo_name=winning_dmo,
            scenario=scenario,
            allocation=best_allocation,
            appreciation=best_appreciation,
            budget=max_investment,
            started_at=started_at,
            n_function_evals=len(combinations),
        )

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


class SLSQPSolver(BaseSolver):
    """Multi-start sequential least-squares programming on the capped simplex.

    Treats the allocation problem as a continuous nonlinear program and solves it from several
    random starting points, keeping the best answer. Fast and accurate when the appreciation
    surface has a single optimum; when it has more than one, a start can settle in the wrong one,
    which is what :class:`BasinHoppingSolver` addresses.

    The budget is an upper bound, not an equality: under-spending is feasible. That matters on
    surfaces where spending the whole budget lowers appreciation. Where it does not, the
    constraint binds and the solution spends everything anyway.
    """

    default_dmo_name = "Optimized (SLSQP)"
    method_name = "slsqp"
    method_label = "SLSQP"

    def solve(self, scenario, dmo_name, budget, *, reference_allocation=None, n_starts=100, seed=None, **_ignored):
        """Run ``n_starts`` local solves and write back the best allocation.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param budget: total allocation budget (upper bound: sum(x) <= budget).
        :param dmo_name: name under which the winning allocation is written back.
        :param reference_allocation: feasible allocation used to seed the new option row
            (only when registering); defaults to the first existing option's row.
        :param n_starts: number of random multi-starts.
        :param seed: RNG seed for reproducible starts.
        :return: an :class:`OptimizationResult` with per-start diagnostics.
        """
        self._prepare_input_dict(dmo_name, reference_allocation)
        starts = self._dirichlet_starts(n_starts, budget, seed=seed)
        eval_counter = [0]
        started_at = time.perf_counter()

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

        if best_x is not None:
            self._write_back_result(dmo_name, best_x)

        return self._build_result(
            dmo_name=dmo_name,
            scenario=scenario,
            allocation=best_x if best_x is not None else np.full(self._k, np.nan),
            appreciation=-best_neg_f if best_x is not None else float("nan"),
            budget=budget,
            started_at=started_at,
            n_starts=n_starts,
            n_converged=sum(1 for r in per_start if r["success"]),
            n_function_evals=eval_counter[0],
            per_start_results=per_start,
        )

    def _dirichlet_starts(self, n_starts, budget, seed=None):
        """Random starting points on the budget face, ``Dirichlet(1, ..., 1) * budget``.

        These lie on sum(x) = B, which is feasible under sum(x) <= B. A solve can still move
        into the interior when the gradient favours under-spending.
        """
        rng = np.random.default_rng(seed)
        return rng.dirichlet(np.ones(self._k), size=n_starts) * budget

    def _objective_z(self, z, scenario, dmo_name, eval_counter, budget):
        """:meth:`BaseSolver._objective` in budget-normalised coordinates, ``x = budget * z``."""
        return self._objective(np.asarray(z, dtype=float) * budget, scenario, dmo_name, eval_counter)

    def _slsqp_minimizer_kwargs(self, scenario, dmo_name, eval_counter, budget):
        """The SLSQP configuration shared by a single solve and by every hop of a hopping run.

        The solve runs in normalised coordinates, ``x = B * z``, on the unit capped simplex.
        SLSQP starts from an identity Hessian approximation and uses absolute tolerances, so on
        raw budgets in the millions its first trial step is microscopic relative to the variable
        scale and the improvement test aborts at the start point: the solver "converges" without
        moving. Normalising makes solver behaviour independent of the budget scale.

        scipy reads ``ineq`` constraints as ``fun(z) >= 0``, hence ``1 - sum(z) >= 0``.
        """
        return {
            "method": "SLSQP",
            "bounds": [(0.0, 1.0)] * self._k,
            "constraints": ({"type": "ineq", "fun": lambda z: float(1.0 - np.sum(z))},),
            "args": (scenario, dmo_name, eval_counter, float(budget)),
            "options": {"ftol": 1e-6, "maxiter": 100, "disp": False, "eps": 1e-6},
        }

    def _slsqp_from_start(self, x0, scenario, dmo_name, budget, eval_counter):
        """Single SLSQP solve from ``x0``; ``res.x`` is mapped back to allocation units."""
        minimizer_kwargs = self._slsqp_minimizer_kwargs(scenario, dmo_name, eval_counter, budget)
        res = minimize(
            self._objective_z,
            np.asarray(x0, dtype=float) / float(budget),
            **minimizer_kwargs,
        )
        res.x = self._project_capped_simplex(res.x * float(budget), float(budget))
        return res


class BasinHoppingSolver(SLSQPSolver):
    """SLSQP plus an escape loop, for surfaces with more than one optimum.

    Basin-hopping (Wales & Doye, 1997) alternates a local solve with a random jump: it solves,
    kicks the solution somewhere else in the feasible set, solves again, and accepts or rejects
    the new optimum by the Metropolis rule. Because the local solve is SLSQP, this is really
    SLSQP-hopping: same model, same feasible set, wrapped in a search for other optima.

    That is what it buys and what it costs. On a surface with one optimum it returns what SLSQP
    already found, for many times the work. On a surface with several it finds optima that
    multi-start SLSQP does not reach.
    """

    default_dmo_name = "Optimized (basin-hopping)"
    method_name = "basin_hopping"
    method_label = "basin-hopping"

    class _RandomFeasibleHop:  # pylint: disable=too-few-public-methods
        """The jump between two local solves, landing inside the feasible set.

        ``scipy.optimize.basinhopping`` calls ``take_step(x)`` between local solves. Its default
        jump ignores the feasible set; this one adds a Gaussian kick and then projects back onto
        the capped simplex, so every proposed start is feasible. Stateful (its own RNG, so
        restarts reproduce) and picklable; the ``stepsize`` attribute lets basin-hopping's
        adaptive step-size control tune the kick towards its target acceptance rate.
        """

        def __init__(self, budget, k, rng, step_frac=0.3):
            self.budget = float(budget)
            self.k = int(k)
            self.rng = rng
            self.stepsize = step_frac * float(budget)

        def __call__(self, x):
            x_new = np.asarray(x, dtype=float) + self.rng.normal(scale=self.stepsize, size=self.k)
            x_new = np.clip(x_new, 0.0, self.budget)
            total = x_new.sum()
            if total > self.budget:
                x_new = x_new * (self.budget / total)
            return x_new

    def solve(
        self,
        scenario,
        dmo_name,
        budget,
        *,
        reference_allocation=None,
        n_hops=100,
        n_starts=1,
        temperature=1.0,
        step_frac=0.3,
        seed=None,
        **_ignored,
    ):
        """Run ``n_starts`` hopping chains of ``n_hops`` hops and write back the best allocation.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param budget: allocation budget (upper bound: sum(x) <= budget).
        :param dmo_name: name under which the winning allocation is written back.
        :param reference_allocation: feasible allocation seeding the new option row.
        :param n_hops: hops per chain.
        :param n_starts: number of independent chains, each from its own random start.
        :param temperature: Metropolis acceptance temperature.
        :param step_frac: jump size as a fraction of the budget.
        :param seed: RNG seed for reproducible starts, jumps and acceptance.
        :return: an :class:`OptimizationResult` with per-chain diagnostics.
        """
        self._prepare_input_dict(dmo_name, reference_allocation)
        starts = self._dirichlet_starts(n_starts, budget, seed=seed)
        eval_counter = [0]
        started_at = time.perf_counter()

        # Hopping runs in the same normalised coordinates as the local solve, which also makes
        # step_frac independent of the case's budget scale.
        minimizer_kwargs = self._slsqp_minimizer_kwargs(scenario, dmo_name, eval_counter, budget)

        per_start = []
        best_x = None
        best_neg_f = np.inf

        for i, x0 in enumerate(starts):
            if seed is None:
                step_rng, hop_rng = np.random.default_rng(), np.random.default_rng()
            else:
                step_rng, hop_rng = np.random.default_rng([seed, i, 0]), np.random.default_rng([seed, i, 1])
            take_step = self._RandomFeasibleHop(1.0, self._k, step_rng, step_frac=step_frac)

            res = basinhopping(
                self._objective_z,
                np.asarray(x0, dtype=float) / float(budget),
                niter=n_hops,
                T=temperature,
                minimizer_kwargs=minimizer_kwargs,
                take_step=take_step,
                rng=hop_rng,
            )
            x_best_start = self._project_capped_simplex(np.asarray(res.x) * float(budget), float(budget))
            per_start.append(
                {
                    "i": i,
                    "x0": np.asarray(x0),
                    "x": x_best_start,
                    "appreciation": float(-res.fun),
                    "success": bool(res.lowest_optimization_result.success),
                    "nit": int(res.nit),
                    "message": str(res.message[0]) if isinstance(res.message, (list, tuple)) else str(res.message),
                }
            )
            if res.fun < best_neg_f:
                best_neg_f = res.fun
                best_x = x_best_start

        if best_x is not None:
            self._write_back_result(dmo_name, best_x)

        return self._build_result(
            dmo_name=dmo_name,
            scenario=scenario,
            allocation=best_x if best_x is not None else np.full(self._k, np.nan),
            appreciation=-best_neg_f if best_x is not None else float("nan"),
            budget=budget,
            started_at=started_at,
            n_starts=n_starts,
            n_converged=sum(1 for r in per_start if r["success"]),
            n_function_evals=eval_counter[0],
            per_start_results=per_start,
        )


class GeneticAlgorithmSolver(BaseSolver):
    """Evolutionary search: no gradients, only comparisons between candidate allocations.

    A population of allocations is improved generation by generation. Pairs of parents are
    picked by tournament, combined into children by simulated binary crossover, nudged by
    polynomial mutation (Deb & Agrawal, 1995), and the best individual always survives. Every
    child is projected onto the capped simplex, so the whole population stays feasible.

    Useful where gradients are unreliable, for example on appreciation surfaces with kinks or
    flat stretches. The price is that it converges slowly towards an exact optimum.
    """

    default_dmo_name = "Optimized (GA)"
    method_name = "genetic_algorithm"
    method_label = "GA"

    def solve(
        self,
        scenario,
        dmo_name,
        budget,
        *,
        reference_allocation=None,
        population_size=50,
        n_generations=60,
        crossover_prob=0.9,
        eta_crossover=15.0,
        eta_mutation=20.0,
        seed=None,
        **_ignored,
    ):
        """Evolve ``population_size`` allocations over ``n_generations`` generations.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param budget: allocation budget (upper bound: sum(x) <= budget).
        :param dmo_name: name under which the winning allocation is written back.
        :param reference_allocation: feasible allocation seeding the new option row.
        :param population_size: individuals per generation.
        :param n_generations: number of generations.
        :param crossover_prob: probability that a parent pair is combined rather than copied.
        :param eta_crossover: crossover distribution index (larger = children closer to parents).
        :param eta_mutation: mutation distribution index.
        :param seed: RNG seed; the full run is deterministic given the seed.
        :return: an :class:`OptimizationResult`; ``per_start_results`` holds the per-generation
            best and mean fitness.
        """
        self._prepare_input_dict(dmo_name, reference_allocation)
        rng = np.random.default_rng(seed)
        eval_counter = [0]
        started_at = time.perf_counter()

        def fitness(z):
            eval_counter[0] += 1
            return evaluate_allocation(self.input_dict, z * float(budget), scenario, dmo_name, self._frozen_boundaries)

        # Uniform initial population over the capped simplex: a Dirichlet draw on k+1
        # coordinates with the slack coordinate dropped.
        population = rng.dirichlet(np.ones(self._k + 1), size=population_size)[:, : self._k]
        scores = np.array([fitness(z) for z in population])
        trace = []

        for generation in range(n_generations):
            children = []
            while len(children) < population_size:
                # Binary tournaments pick the two parents.
                idx = rng.integers(0, population_size, size=4)
                parent_a = population[idx[0]] if scores[idx[0]] >= scores[idx[1]] else population[idx[1]]
                parent_b = population[idx[2]] if scores[idx[2]] >= scores[idx[3]] else population[idx[3]]
                child_a, child_b = self._sbx_pair(parent_a, parent_b, eta_crossover, crossover_prob, rng)
                for child in (child_a, child_b):
                    child = self._polynomial_mutation(child, eta_mutation, rng)
                    children.append(self._project_capped_simplex(child, 1.0))
            children = np.array(children[:population_size])
            child_scores = np.array([fitness(z) for z in children])

            # The incumbent best survives unless a child beats it.
            elite_idx = int(np.argmax(scores))
            worst_child = int(np.argmin(child_scores))
            if scores[elite_idx] > child_scores.max():
                children[worst_child] = population[elite_idx]
                child_scores[worst_child] = scores[elite_idx]
            population, scores = children, child_scores
            trace.append({"generation": generation, "best": float(scores.max()), "mean": float(scores.mean())})

        best_idx = int(np.argmax(scores))
        best_x = population[best_idx] * float(budget)
        self._write_back_result(dmo_name, best_x)

        return self._build_result(
            dmo_name=dmo_name,
            scenario=scenario,
            allocation=best_x,
            appreciation=float(scores[best_idx]),
            budget=budget,
            started_at=started_at,
            n_starts=population_size,
            n_function_evals=eval_counter[0],
            per_start_results=trace,
        )

    @staticmethod
    def _sbx_pair(parent_a, parent_b, eta, crossover_prob, rng):
        """Simulated binary crossover (Deb & Agrawal, 1995) on one parent pair."""
        if rng.random() > crossover_prob:
            return parent_a.copy(), parent_b.copy()
        u = rng.random(parent_a.shape)
        beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
        child_a = 0.5 * ((1.0 + beta) * parent_a + (1.0 - beta) * parent_b)
        child_b = 0.5 * ((1.0 - beta) * parent_a + (1.0 + beta) * parent_b)
        return np.clip(child_a, 0.0, 1.0), np.clip(child_b, 0.0, 1.0)

    @staticmethod
    def _polynomial_mutation(child, eta, rng):
        """Polynomial mutation (Deb & Agrawal, 1995), applied to a gene with probability 1/k."""
        k = child.size
        mutate = rng.random(k) < (1.0 / k)
        u = rng.random(k)
        delta = np.where(
            u < 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0, 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
        )
        return np.clip(np.where(mutate, child + delta, child), 0.0, 1.0)


class MdbhSolver(BaseSolver):
    """Mirror descent inside a basin-hopping loop: a research method, not a default.

    Where SLSQP solves a quadratic subproblem at every step, entropic mirror descent (Beck &
    Teboulle, 2003) multiplies each coordinate by the exponential of its gradient, which keeps
    the iterate on the simplex by construction and costs one pass over the coordinates. Around
    that local engine sits the same escape loop as basin-hopping, with a jump that scales each
    coordinate rather than shifting it.

    The capped simplex is turned into an ordinary simplex by adding a slack coordinate: the
    method walks on k+1 weights, the first k become the allocation and the last absorbs unspent
    budget. The objective never reads the slack, so its derivative is zero and a gradient costs
    k evaluations rather than k+1.
    """

    default_dmo_name = "Optimized (MDBH)"
    method_name = "mdbh"
    method_label = "MDBH"

    def solve(
        self,
        scenario,
        dmo_name,
        budget,
        *,
        reference_allocation=None,
        n_starts=5,
        n_hops=10,
        n_local_steps=50,
        eta=1.0,
        sigma=1.5,
        temperature=1.0,
        seed=None,
        **_ignored,
    ):
        """Run ``n_starts`` chains of mirror-descent solves separated by multiplicative jumps.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param budget: allocation budget (upper bound: sum(x) <= budget).
        :param dmo_name: name under which the winning allocation is written back.
        :param reference_allocation: feasible allocation seeding the new option row.
        :param n_starts: independent chains; the first starts at the centre of the simplex.
        :param n_hops: jumps per chain.
        :param n_local_steps: mirror-descent step budget per local solve.
        :param eta: initial mirror-descent step size (decays as eta/sqrt(t)).
        :param sigma: jump size on the simplex.
        :param temperature: Metropolis acceptance temperature.
        :param seed: RNG seed; the full run is deterministic given the seed.
        :return: an :class:`OptimizationResult` with per-chain diagnostics.
        """
        self._prepare_input_dict(dmo_name, reference_allocation)
        rng = np.random.default_rng(seed)
        eval_counter = [0]
        started_at = time.perf_counter()
        dim = self._k + 1

        per_start = []
        best_w, best_f = None, -np.inf
        for start in range(n_starts):
            w = np.full(dim, 1.0 / dim) if start == 0 else rng.dirichlet(np.ones(dim))
            w, f_w = self._mirror_descent(w, scenario, dmo_name, eval_counter, budget, n_local_steps, eta)
            chain_best_w, chain_best_f = w.copy(), f_w
            accepted = 0
            for _ in range(n_hops):
                kick = rng.normal(size=dim)
                kick -= kick.mean()  # a direction along the simplex: the shifts cancel out
                w_kicked = np.maximum(w * np.exp(sigma * kick), 1e-12)
                w_kicked = w_kicked / w_kicked.sum()
                w_new, f_new = self._mirror_descent(
                    w_kicked, scenario, dmo_name, eval_counter, budget, n_local_steps, eta
                )
                if f_new > f_w or rng.random() < math.exp(min(0.0, f_new - f_w) / temperature):
                    w, f_w = w_new, f_new
                    accepted += 1
                if f_new > chain_best_f:
                    chain_best_w, chain_best_f = w_new.copy(), f_new
            per_start.append(
                {"i": start, "appreciation": float(chain_best_f), "n_hops": n_hops, "n_accepted": accepted}
            )
            if chain_best_f > best_f:
                best_w, best_f = chain_best_w, chain_best_f

        best_x = self._project_capped_simplex(best_w[: self._k] * float(budget), float(budget))
        self._write_back_result(dmo_name, best_x)

        return self._build_result(
            dmo_name=dmo_name,
            scenario=scenario,
            allocation=best_x,
            appreciation=float(best_f),
            budget=budget,
            started_at=started_at,
            n_starts=n_starts,
            n_function_evals=eval_counter[0],
            per_start_results=per_start,
        )

    def _appreciation_w(self, w, scenario, dmo_name, eval_counter, budget):
        """Appreciation of a point on the slack-extended simplex."""
        allocation = np.asarray(w[: self._k], dtype=float) * float(budget)
        return -self._objective(allocation, scenario, dmo_name, eval_counter)

    def _fd_gradient_w(self, w, f_w, scenario, dmo_name, eval_counter, budget, h=1e-6):
        """Finite-difference gradient of the appreciation. The slack coordinate is skipped."""
        grad = np.zeros(w.size)
        for i in range(self._k):
            w_step = np.asarray(w, dtype=float).copy()
            w_step[i] += h
            grad[i] = (self._appreciation_w(w_step, scenario, dmo_name, eval_counter, budget) - f_w) / h
        return grad

    def _mirror_descent(self, w, scenario, dmo_name, eval_counter, budget, n_steps, eta, patience=8):
        """Entropic mirror descent on the slack-extended simplex (maximisation).

        Each coordinate is multiplied by the exponential of its gradient, with the gradient
        normalised so the step size alone controls how far the iterate moves, and a step size
        that decays as eta/sqrt(t). Coordinates are floored just above zero before renormalising:
        the update can drive a coordinate towards zero, which is what a sparse optimum wants, but
        it must never reach zero or it could never recover. The method is not monotone, so the
        best iterate seen is returned and the loop stops after ``patience`` steps without
        improvement.
        """
        best_w = np.asarray(w, dtype=float).copy()
        best_f = self._appreciation_w(best_w, scenario, dmo_name, eval_counter, budget)
        current_w, current_f, stale = best_w.copy(), best_f, 0
        for step in range(1, n_steps + 1):
            grad = self._fd_gradient_w(current_w, current_f, scenario, dmo_name, eval_counter, budget)
            grad_scale = float(np.max(np.abs(grad)))
            if grad_scale <= 0.0:
                break  # locally flat in every coordinate: no direction to follow
            current_w = current_w * np.exp((eta / math.sqrt(step)) * grad / grad_scale)
            current_w = np.maximum(current_w, 1e-12)
            current_w = current_w / current_w.sum()
            current_f = self._appreciation_w(current_w, scenario, dmo_name, eval_counter, budget)
            if current_f > best_f + 1e-12:
                best_w, best_f, stale = current_w.copy(), current_f, 0
            else:
                stale += 1
                if stale >= patience:
                    break
        return best_w, best_f


class Optimize:
    """
    The Optimize class performs grid search optimization to find the optimal distribution of internal input values
    that maximizes the appreciation value of decision-maker options.

    Grid search is the default baseline but no longer the only option: this class chooses a
    solver and runs it. ``method="auto"`` probes the case and lets the decision tree in
    :mod:`vlinder.method_selection` pick the solver that fits the shape of its appreciation
    surface. A name runs that solver; a list of names runs each and returns the best.
    """

    #: Method name to solver class.
    METHOD_REGISTRY = {
        "grid": GridSearch,
        "slsqp": SLSQPSolver,
        "basin_hopping": BasinHoppingSolver,
        "genetic_algorithm": GeneticAlgorithmSolver,
        "mdbh": MdbhSolver,
    }
    # Decision-maker option the method-selection probe writes its allocations to; it only ever
    # exists on the copy the probe runs on.
    PROBE_DMO_NAME = "Method selection probe"

    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self._k = len(input_dict["internal_variable_inputs"])

    @property
    def budget(self):
        """Total budget available to allocate; see :attr:`BaseSolver.budget`."""
        return BaseSolver(self.input_dict, self.output_dict).budget

    def run(self, scenario, method="auto", *, dmo_name=None, budget=None, method_kwargs=None, **shared_kwargs):
        """Run one or several optimization methods and return the best result.

        Solver settings can be given in two ways. Keyword arguments go to every method that
        runs, which is the convenient form for a single method or for a setting they share such
        as ``seed``. ``method_kwargs`` addresses one method by name, which is what you need when
        two methods have different parameters::

            case.optimize("Base case", method="slsqp", n_starts=50)
            case.optimize(
                "Base case",
                method=["grid", "slsqp"],
                seed=1,
                method_kwargs={"grid": {"max_combinations": 1000}, "slsqp": {"n_starts": 50}},
            )

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param method: ``"auto"`` (default), a single method name (str) or a list of names.
            Supported: ``"grid"``, ``"slsqp"``, ``"basin_hopping"``, ``"genetic_algorithm"``,
            ``"mdbh"``. Unknown names raise ``NotImplementedError``. ``"auto"`` probes the case
            and picks one of those methods together with a solver budget for it; any setting you
            pass overrides that budget. It cannot be combined in a list.
        :param dmo_name: name for the optimizer's decision-maker option. Defaults per solver;
            for a list, each method uses its own default unless an explicit name is given.
        :param budget: total allocation budget; inferred from the case if None.
        :param method_kwargs: settings per method, ``{method name: {setting: value}}``. These win
            over the shared keyword arguments. An entry for a method that is not running raises
            ``NotImplementedError``, so a typo cannot pass unnoticed.
        :param shared_kwargs: settings passed to every method that runs.
        :return: a single :class:`OptimizationResult`, with ``selection`` filled when the method
            came from ``"auto"``. For a list of methods, every method's appreciation and
            allocation is printed and the best is returned; only the winning allocation is
            written back.
        """
        if budget is None:
            budget = self.budget

        selection = None
        if method == "auto":
            selection = self.select_method_for(scenario, budget)
            print(self._describe_selection(selection))
            method = selection.method
            shared_kwargs = {**selection.method_kwargs, **shared_kwargs}

        methods = [method] if isinstance(method, str) else list(method)
        if "auto" in methods:
            raise NotImplementedError('method="auto" picks one method and cannot be an entry in a list of methods')
        for name in methods:
            if name not in self.METHOD_REGISTRY:
                raise NotImplementedError(f"method={name!r} not implemented. Supported: {list(self.METHOD_REGISTRY)}")
        for name in method_kwargs or {}:
            if name not in methods:
                raise NotImplementedError(
                    f"method_kwargs holds settings for {name!r}, which is not among the methods being run: {methods}"
                )

        if len(methods) == 1:
            # Single method runs on this case, so the winning option and the frozen boundaries
            # land here.
            solver = self.METHOD_REGISTRY[methods[0]](self.input_dict, self.output_dict)
            name = self._dmo_name_for(dmo_name, solver, scenario)
            settings = self._settings_for(methods[0], shared_kwargs, method_kwargs)
            result = solver.solve(scenario, dmo_name=name, budget=budget, **settings)
            self.input_dict = solver.input_dict
        else:
            result = self._run_several(methods, scenario, dmo_name, budget, shared_kwargs, method_kwargs)

        result.selection = selection
        return result

    @staticmethod
    def _settings_for(method, shared_kwargs, method_kwargs):
        """The settings one method runs with: the shared ones, overridden by its own."""
        return {**shared_kwargs, **(method_kwargs or {}).get(method, {})}

    def select_method_for(self, scenario, budget=None):
        """Diagnose the case and return the method the decision tree picks for it.

        The probe runs on a copy, so the diagnosis leaves no decision-maker option behind in
        this case. It costs a few dozen evaluations, two orders of magnitude less than any of
        the solvers it chooses between.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param budget: total allocation budget; inferred from the case if None.
        :return: a :class:`vlinder.method_selection.MethodChoice`.
        """
        if budget is None:
            budget = self.budget
        probe = BaseSolver(copy.deepcopy(self.input_dict), self.output_dict)
        probe._prepare_input_dict(self.PROBE_DMO_NAME, self.input_dict["decision_makers_option_value"][0].copy())
        diagnosis = diagnose_case(evaluate_and_appreciate, probe.input_dict, scenario, self.PROBE_DMO_NAME, budget)
        return select_method(diagnosis)

    @staticmethod
    def _configured_dmo_name(input_dict):
        """The optimizer name configured on the case, or None.

        A case can carry an ``Optimize_DMO_name`` entry in its configuration sheet. When the
        caller does not pass a name, that entry is the base of the option name. A missing sheet,
        a missing entry or an empty (NaN) value all mean no configured name.
        """
        configurations = input_dict.get("configurations")
        if configurations is None:
            return None
        matches = np.where(np.asarray(configurations) == "Optimize_DMO_name")[0]
        if not matches.size:
            return None
        value = input_dict["configuration_value"][matches[0]]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return str(value)

    def _dmo_name_for(self, dmo_name, solver, scenario):
        """Name the optimizer's decision-maker option.

        The name always records the scenario it was optimized for: an allocation is only optimal
        for that scenario, so optimizing the same case for two scenarios has to produce two
        options rather than overwrite one. A name passed by the caller is used as given, plus the
        scenario. Without one, the name configured on the case is the base, extended with the
        method and the scenario, so a configured "Show me what you got" becomes
        "Show me what you got (grid) (Base case)". Without either, the solver's default (which
        already names the method) plus the scenario.
        """
        if dmo_name:
            return f"{dmo_name} ({scenario})"
        configured = self._configured_dmo_name(self.input_dict)
        if configured:
            return f"{configured} ({solver.method_label}) ({scenario})"
        return f"{solver.default_dmo_name} ({scenario})"

    @staticmethod
    def _describe_selection(selection):
        """A readable version of the automatic choice, for the notebook and the console."""
        readable = {
            "grid": "grid search",
            "slsqp": "multi-start SLSQP",
            "basin_hopping": "basin-hopping",
            "genetic_algorithm": "the genetic algorithm",
            "mdbh": "mirror-descent basin-hopping",
        }
        diagnosis = selection.diagnosis
        lines = [f"Automatic method selection chose {readable.get(selection.method, selection.method)}."]
        if diagnosis is not None:
            shape = (
                "one optimum (the surface is provably concave)"
                if diagnosis.is_provably_concave
                else "possibly more than one optimum"
            )
            lines.append(
                f"  The case has {diagnosis.k} internal variable inputs and a budget of {diagnosis.budget:,.2f}."
            )
            lines.append(f"  Its appreciation surface has {shape}.")
            lines.append(f"  Diagnosed in {diagnosis.n_probe_evaluations} evaluations.")
        lines.append(f"  Why: {selection.reason}.")
        return "\n".join(lines)

    def _run_several(self, methods, scenario, dmo_name, budget, shared_kwargs, method_kwargs):
        """Run every method in ``methods`` and return the best result.

        Each method runs on its own copy of the case so they do not see each other's added
        decision-maker options. All are printed, then the winner's case is adopted.
        """
        runs = []
        for chosen in methods:
            solver = self.METHOD_REGISTRY[chosen](copy.deepcopy(self.input_dict), self.output_dict)
            name = self._dmo_name_for(dmo_name, solver, scenario)
            settings = self._settings_for(chosen, shared_kwargs, method_kwargs)
            res = solver.solve(scenario, dmo_name=name, budget=budget, **settings)
            print(f"[{chosen}] appreciation={res.appreciation:.6f} allocation={np.asarray(res.allocation).tolist()}")
            runs.append((res, solver))

        best_res, best_solver = max(runs, key=lambda pair: pair[0].appreciation)
        self.input_dict = best_solver.input_dict
        return best_res
