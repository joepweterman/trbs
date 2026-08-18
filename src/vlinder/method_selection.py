"""
This module contains the automatic optimization-method selection behind
``Optimize.run(scenario, method="auto")``, which is the default.

Instead of asking the user which solver suits their case, ``auto`` probes the
case with a few dozen objective evaluations and walks a decision tree whose
splits are the conditions under which the appreciation surface is provably
concave on the feasible set ``{x : x >= 0, sum(x) <= B}``:

  1. every key output is an affine function of the allocation,
  2. every appreciation term is concave in its key output, and
  3. no appreciation term is clipped at its floor anywhere on the probe set.

Under those three conditions every local optimum is a global optimum, so the
cheap local solver (multi-start SLSQP) is enough. If a condition fails the
surface can be multimodal and the global-escape method (basin-hopping) is
selected. If the probe also finds flat regions where the objective carries no
gradient information, the derivative-free method (the GA) is selected instead.

Why exactly those three conditions:

  * a concave function of an affine map is concave, so affine key outputs
    preserve concavity while a nonlinear dependency (a ``min``/``max`` operator,
    a product of two internal variable inputs) can introduce a second basin;
  * a linear appreciation term is affine in its key output, and a sinusoidal
    term with ``smaller_the_better = 0`` is ``100*sin(0.5*pi*u)``, concave and
    increasing. A sinusoidal term with ``smaller_the_better = 1`` is
    ``100*(1 - sin(0.5*pi*u))``, which is *convex*, so it can carry a basin;
  * clipping at the cap (appreciation exactly 100) is ``min(concave, 100)`` and
    stays concave, while clipping at the floor (appreciation exactly 0) is
    ``max(concave, 0)`` and does not. That asymmetry is why the clipped optimum
    of a case like Beerwiser sits on a kink.

The tree is theory-driven: the splits come from the conditions above, not from a
model fitted on benchmark results. The solver budgets attached to each leaf are
the empirical part, and their sources are named at each constant below.

The tree selects grid search only when no positive budget can be inferred, which
is the one situation where the continuous methods have nothing to allocate. Grid
search enumerates only the budget face ``sum(x) = B``, at a step size coarsened
to keep the number of multisets under ``max_combinations``, and its true cost
grows with the permutation expansion (multisets times k factorial), so wherever a
continuous method can run it matches or beats grid search at a fraction of the
cost. Grid search of course stays available as an explicit ``method="grid"``.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Probe design. The seed is fixed on purpose: the same case must always yield
# the same diagnosis, otherwise the default solver could change between runs.
PROBE_SEED = 0
N_INTERIOR_PROBES = 8
N_PLATEAU_PROBES = 8
PLATEAU_STEP_FRAC = 1e-3
# Flatness only decides the method when it covers most of where the solvers
# start: an isolated flat point costs a multi-start solver one start, not the run.
PLATEAU_MIN_FRACTION = 0.75

# A key output counts as varying (and a residual as nonzero) above these
# tolerances, both relative to the key output's own scale.
ACTIVE_TOL = 1e-12
AFFINE_TOL = 1e-6
PLATEAU_TOL = 1e-9
# How far a floor-clipped probe point is pulled towards the interior to tell an
# active clip apart from a point that merely touches its boundary.
CLIP_INWARD_STEP = 1e-3

# Solver budgets per leaf.
#   * concave leaf: one start suffices in theory; the probe is a finite check
#     rather than a proof, so a handful of starts guards against a wrong verdict
#     at negligible cost.
#   * basin-hopping leaf: on the bundled cases a single hopping chain can stay
#     in a secondary basin (Refugee, 12.4 appreciation points low) while ten
#     chains always escape it, and 25 hops per chain is enough to find the
#     clipped optimum of Beerwiser.
CONCAVE_N_STARTS = 10
GLOBAL_N_STARTS = 10
GLOBAL_N_HOPS = 25

FLOOR_APPRECIATION = 0.0


@dataclass
class CaseDiagnosis:
    """What the probe measured about the shape of a case's appreciation surface."""

    k: int
    budget: float
    deps_affine: bool
    max_affine_residual: float
    convex_carriers: Tuple[str, ...]
    floor_clip_fraction: float
    plateau_fraction: float
    n_probe_evaluations: int

    @property
    def is_provably_concave(self) -> bool:
        """True when all three concavity conditions hold on the probe set."""
        return self.deps_affine and not self.convex_carriers and self.floor_clip_fraction == 0.0


@dataclass
class MethodChoice:
    """The selected method, the solver budget for it, and why it was selected."""

    method: str
    method_kwargs: dict
    reason: str
    diagnosis: Optional[CaseDiagnosis] = None


def _probe_allocations(k, budget):
    """
    This function builds the deterministic probe set: zero spend, every single-input
    corner, and interior points of the capped simplex.
    :param k: number of internal variable inputs
    :param budget: total allocation budget
    :return: an array of shape (1 + k + N_INTERIOR_PROBES, k)
    """
    rng = np.random.default_rng(PROBE_SEED)
    corners = np.eye(k) * budget
    directions = rng.dirichlet(np.ones(k), size=N_INTERIOR_PROBES)
    factors = rng.uniform(0.3, 1.0, size=(N_INTERIOR_PROBES, 1))
    return np.vstack([np.zeros((1, k)), corners, directions * budget * factors])


def _probe_case(probe, key_outputs, allocations):
    """
    This function evaluates the probe set once and collects everything the diagnosis needs.
    :param probe: callable mapping an allocation to a tRBS output dictionary
    :param key_outputs: the case's key output names, in input_dict order
    :param allocations: the probe allocations
    :return: (key output matrix, appreciation matrix, objective values)
    """
    key_output_rows, appreciation_rows, objective_values = [], [], []
    for allocation in allocations:
        output = probe(allocation)
        key_output_rows.append([output["key_outputs"][name] for name in key_outputs])
        appreciation_rows.append([output["appreciations"][name] for name in key_outputs])
        objective_values.append(output["decision_makers_option_appreciation"])

    return (
        np.asarray(key_output_rows, dtype=float),
        np.asarray(appreciation_rows, dtype=float),
        np.asarray(objective_values, dtype=float),
    )


def _appreciation_probe(probe, key_outputs):
    """
    This function wraps a probe so it returns the per key output appreciations as an array.
    :param probe: callable mapping an allocation to a tRBS output dictionary
    :param key_outputs: the case's key output names, in input_dict order
    :return: a callable mapping an allocation to its appreciation vector
    """

    def measure(allocation):
        output = probe(allocation)
        return np.asarray([output["appreciations"][name] for name in key_outputs], dtype=float)

    return measure


def _active_key_outputs(input_dict, key_output_values):
    """
    This function marks the key outputs that can influence the optimum: they carry weight
    and they move with the allocation. A key output that is constant in x shifts the
    objective by a constant and changes neither its concavity nor its argmax.
    :param input_dict: tRBS case input dictionary
    :param key_output_values: key output matrix from the probe
    :return: a boolean mask over the key outputs
    """
    spread = key_output_values.max(axis=0) - key_output_values.min(axis=0)
    scale = np.maximum(np.abs(key_output_values).max(axis=0), 1.0)
    weighted = np.asarray(input_dict["key_output_weight"], dtype=float) > 0
    return weighted & (spread > ACTIVE_TOL * scale)


def _max_affine_residual(allocations, key_output_values, budget, active):
    """
    This function measures how far the key outputs deviate from the affine model fitted on
    zero spend and the single-input corners, relative to each key output's own spread.
    A residual of order 1 means the dependency is genuinely nonlinear.
    :param allocations: the probe allocations
    :param key_output_values: key output matrix from the probe
    :param budget: total allocation budget
    :param active: boolean mask of the key outputs that matter
    :return: the largest relative residual over the interior probe points
    """
    if not active.any():
        return 0.0
    first_interior = 1 + allocations.shape[1]  # zero spend, then one corner per input
    base, corners = key_output_values[0], key_output_values[1:first_interior]
    interior_allocations, interior_values = allocations[first_interior:], key_output_values[first_interior:]

    predicted = base + (interior_allocations / budget) @ (corners - base)
    spread = key_output_values.max(axis=0) - key_output_values.min(axis=0)
    residual = np.abs(interior_values - predicted) / np.maximum(spread, ACTIVE_TOL)
    return float(residual[:, active].max())


def _convex_carriers(input_dict, active):
    """
    This function lists the active key outputs whose appreciation term is convex, i.e.
    sinusoidal with smaller-the-better = 1. Every other combination is affine or concave.
    :param input_dict: tRBS case input dictionary
    :param active: boolean mask of the key outputs that matter
    :return: a tuple of key output names
    """
    linear = np.asarray(input_dict["key_output_linear"], dtype=int)
    smaller_the_better = np.asarray(input_dict["key_output_smaller_the_better"], dtype=int)
    carriers = active & (linear == 0) & (smaller_the_better == 1)
    return tuple(np.asarray(input_dict["key_outputs"])[carriers].tolist())


def _floor_clip_fraction(appreciation_vector, allocations, appreciations, active):
    """
    This function measures how often an active appreciation term is genuinely clipped at
    its floor. The engine returns exactly 0 for a key output outside its boundaries on the
    worse side, so equality is the exact test for a clipped term, not an approximation.

    A term that merely *touches* its boundary does not break concavity: on an exactly
    bracketed case the zero-spend vertex sits on the lower boundary of every key output,
    yet max(g, 0) equals g everywhere on the feasible set. Only a clip that stays active
    over a region does, so each flagged point is re-evaluated a small step towards the
    interior and only counts if the same term is still on its floor there.

    :param appreciation_vector: callable mapping an allocation to its per key output appreciations
    :param allocations: the probe allocations
    :param appreciations: per key output appreciation matrix from the probe
    :param active: boolean mask of the key outputs that matter
    :return: the fraction of probe points carrying an active floor clip
    """
    if not active.any():
        return 0.0
    on_floor = appreciations[:, active] == FLOOR_APPRECIATION
    inward = allocations.mean(axis=0)  # a strictly interior point of the capped simplex

    clipped = 0
    for index in np.flatnonzero(on_floor.any(axis=1)):
        pulled = allocations[index] + CLIP_INWARD_STEP * (inward - allocations[index])
        still_on_floor = (appreciation_vector(pulled) == FLOOR_APPRECIATION)[active]
        clipped += bool((on_floor[index] & still_on_floor).any())
    return clipped / len(allocations)


def _plateau_fraction(objective, k, budget, best_objective):
    """
    This function measures how often the objective is flat around a start point that is
    not the best one seen. On such a point a gradient-based solver has nothing to follow,
    while a population-based solver still explores.

    The points are drawn the way the solvers draw their own starts, Dirichlet over the
    budget face, because flatness only matters where a solver actually begins: the deep
    interior of a case like Refugee is flat (every appreciation term clipped) while its
    whole start distribution is not. On the face one input can only grow at another's
    expense, so flatness is tested along the transfer directions the solver can take.

    :param objective: callable mapping an allocation to the weighted appreciation
    :param k: number of internal variable inputs
    :param budget: total allocation budget
    :param best_objective: the best objective value observed on the probe set
    :return: the fraction of start points that are flat in every feasible direction
    """
    if k < 2:
        return 0.0
    rng = np.random.default_rng(PROBE_SEED + 1)
    points = rng.dirichlet(np.ones(k), size=N_PLATEAU_PROBES) * budget
    values = [objective(point) for point in points]
    reference = max([best_objective, *values])

    flat = 0
    for point, value in zip(points, values):
        if value >= reference - PLATEAU_TOL:
            continue
        # Move budget from the largest internal variable input into each of the others.
        largest = int(np.argmax(point))
        step = min(PLATEAU_STEP_FRAC * budget, point[largest])
        transfers = (np.eye(k) - np.eye(k)[largest]) * step
        deltas = [abs(objective(point + transfers[i]) - value) for i in range(k) if i != largest]
        if max(deltas) < PLATEAU_TOL:
            flat += 1
    return flat / N_PLATEAU_PROBES


def diagnose_case(evaluate_output, input_dict, scenario, dmo_name, budget) -> CaseDiagnosis:
    """
    This function probes a case and reports the structural properties the selection tree
    splits on. The appreciation boundaries must already be frozen (as
    ``Optimize._prepare_input_dict`` does), so that the probe sees the same objective the
    solver will see.
    :param evaluate_output: callable (input_dict, x, scenario, dmo_name) -> output dictionary
    :param input_dict: tRBS case input dictionary (not mutated)
    :param scenario: scenario name (must be in input_dict["scenarios"])
    :param dmo_name: decision-maker option the probe allocations are written to
    :param budget: total allocation budget
    :return: a :class:`CaseDiagnosis`
    """
    counter = [0]
    key_outputs = list(input_dict["key_outputs"])

    def probe(allocation):
        counter[0] += 1
        return evaluate_output(input_dict, allocation, scenario, dmo_name)

    allocations = _probe_allocations(len(input_dict["internal_variable_inputs"]), budget)
    key_output_values, appreciations, objective_values = _probe_case(probe, key_outputs, allocations)
    active = _active_key_outputs(input_dict, key_output_values)
    residual = _max_affine_residual(allocations, key_output_values, budget, active)

    return CaseDiagnosis(
        k=allocations.shape[1],
        budget=float(budget),
        deps_affine=residual <= AFFINE_TOL,
        max_affine_residual=residual,
        convex_carriers=_convex_carriers(input_dict, active),
        floor_clip_fraction=_floor_clip_fraction(
            _appreciation_probe(probe, key_outputs), allocations, appreciations, active
        ),
        plateau_fraction=_plateau_fraction(
            lambda allocation: probe(allocation)["decision_makers_option_appreciation"],
            allocations.shape[1],
            budget,
            float(objective_values.max()),
        ),
        n_probe_evaluations=counter[0],
    )


def _non_concavity_reasons(diagnosis):
    """
    This function spells out which concavity conditions the probe found violated.
    :param diagnosis: the :class:`CaseDiagnosis` of the case
    :return: a list of human-readable reasons
    """
    reasons = []
    if not diagnosis.deps_affine:
        reasons.append(f"key outputs are not affine in the allocation (residual {diagnosis.max_affine_residual:.1e})")
    if diagnosis.convex_carriers:
        reasons.append(f"convex appreciation term(s): {', '.join(diagnosis.convex_carriers)}")
    if diagnosis.floor_clip_fraction:
        reasons.append(f"floor clipping on {diagnosis.floor_clip_fraction:.0%} of probe points")
    return reasons


def select_method(diagnosis) -> MethodChoice:
    """
    This function walks the decision tree and returns the method to run, with the solver
    budget for it and the reason it was selected.
    :param diagnosis: the :class:`CaseDiagnosis` of the case
    :return: a :class:`MethodChoice`
    """
    if diagnosis.budget <= 0:
        return MethodChoice(
            method="grid",
            method_kwargs={},
            reason=(
                "no positive budget could be inferred from the first decision-maker option, so the continuous "
                "methods have nothing to allocate and only the enumerative baseline applies"
            ),
            diagnosis=diagnosis,
        )

    if diagnosis.is_provably_concave:
        return MethodChoice(
            method="slsqp",
            method_kwargs={"n_starts": CONCAVE_N_STARTS},
            reason=(
                "affine key outputs, concave appreciation terms and no floor clipping, so every local "
                "optimum is global and a local solver suffices"
            ),
            diagnosis=diagnosis,
        )

    reasons = "; ".join(_non_concavity_reasons(diagnosis))
    if diagnosis.plateau_fraction >= PLATEAU_MIN_FRACTION:
        return MethodChoice(
            method="genetic_algorithm",
            method_kwargs={},
            reason=f"{reasons}; the objective is flat away from the optimum, so gradients carry no information",
            diagnosis=diagnosis,
        )

    return MethodChoice(
        method="basin_hopping",
        method_kwargs={"n_starts": GLOBAL_N_STARTS, "n_hops": GLOBAL_N_HOPS},
        reason=f"{reasons}, so the surface can be multimodal and a local solver can settle in the wrong basin",
        diagnosis=diagnosis,
    )
