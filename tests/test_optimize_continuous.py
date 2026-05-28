# pylint: disable=W0212

"""
Tests for the W2 SLSQP scaffold in vlinder.optimize_continuous.

Coverage:
 1. Purity: _objective + multiple calls do not mutate input_dict
 2. SLSQP single-start beats the uniform feasible start
 3. End-to-end via TheResponsibleBusinessSimulator.optimize_continuous —
    multi-start SLSQP matches or beats the legacy grid baseline on Beerwiser
 4. Unsupported methods raise NotImplementedError (NOT silent acceptance)
"""

import copy

import numpy as np
import pytest

from vlinder.optimize import evaluate_allocation
from vlinder.optimize_continuous import ContinuousOptimize, ContinuousOptimizationResult
from vlinder.trbs import TheResponsibleBusinessSimulator
from .params import INPUT_DICT_BEERWISER, OUTPUT_DICT_BEERWISER


@pytest.fixture(name="continuous_beerwiser")
def fixture_continuous_beerwiser():
    """A ContinuousOptimize bound to Beerwiser params, with Phase-A setup done."""
    optimizer = ContinuousOptimize(copy.deepcopy(INPUT_DICT_BEERWISER), copy.deepcopy(OUTPUT_DICT_BEERWISER))
    # Reference allocation = Equal spread DMO ([150000, 150000]); budget 300000
    optimizer._prepare_input_dict("Test DMO", np.array([150000, 150000]))
    return optimizer


@pytest.fixture(name="beerwiser_appreciated")
def fixture_beerwiser_appreciated():
    """A real TheResponsibleBusinessSimulator instance through build+evaluate+appreciate."""
    case = TheResponsibleBusinessSimulator("Beerwiser")
    case.build()
    case.evaluate()
    case.appreciate()
    return case


def test_objective_does_not_mutate_input_dict(continuous_beerwiser):
    """Multiple _objective calls must leave input_dict byte-identical."""
    snapshot = copy.deepcopy(continuous_beerwiser.input_dict)
    counter = [0]
    val_a = continuous_beerwiser._objective(np.array([100000, 200000]), "Base case", "Test DMO", counter)
    val_b = continuous_beerwiser._objective(np.array([200000, 100000]), "Base case", "Test DMO", counter)
    val_c = continuous_beerwiser._objective(np.array([150000, 150000]), "Base case", "Test DMO", counter)

    assert counter[0] == 3
    assert val_a != val_b  # different allocations → different objectives
    # SLSQP minimizes, so we negated; appreciation is positive; objective should be negative
    assert val_a < 0 and val_b < 0 and val_c < 0

    # Byte-identical purity check
    for key in snapshot:
        if isinstance(snapshot[key], np.ndarray):
            np.testing.assert_array_equal(
                continuous_beerwiser.input_dict[key], snapshot[key], err_msg=f"input_dict['{key}'] was mutated"
            )
        else:
            assert continuous_beerwiser.input_dict[key] == snapshot[key], f"input_dict['{key}'] mutated"


def test_slsqp_single_start_beats_uniform(continuous_beerwiser):
    """SLSQP from a uniform start should converge to ≥ uniform's appreciation."""
    budget = 300000.0
    uniform_x = np.array([budget / 2, budget / 2])
    uniform_app = evaluate_allocation(continuous_beerwiser.input_dict, uniform_x, "Base case", "Test DMO")

    eval_counter = [0]
    res = continuous_beerwiser._slsqp_from_start(uniform_x, "Base case", "Test DMO", budget, eval_counter)
    slsqp_app = -float(res.fun)

    assert res.success, f"SLSQP did not converge: {res.message}"
    # Cannot do worse than the start within numerical noise
    assert slsqp_app >= uniform_app - 1e-3
    # Budget feasibility
    assert abs(float(np.sum(res.x)) - budget) < 1e-3
    # Non-negativity
    assert (res.x >= -1e-6).all()


def test_multistart_slsqp_returns_structured_result(continuous_beerwiser):
    """optimize_slsqp returns a ContinuousOptimizationResult with sane fields."""
    budget = 300000.0
    result = continuous_beerwiser.optimize_slsqp(
        scenario="Base case",
        budget=budget,
        dmo_name="Test DMO",  # already prepared by fixture
        n_starts=20,
        seed=42,
    )
    assert isinstance(result, ContinuousOptimizationResult)
    assert result.n_starts == 20
    assert result.n_converged >= 10  # most starts on a 2D simplex should converge
    assert len(result.per_start_results) == 20
    assert result.n_function_evals > 20  # at least 1 per start, usually many more
    assert result.wall_time_s > 0
    assert result.method == "slsqp"
    # Best allocation is feasible
    assert abs(float(np.sum(result.best_x)) - budget) < 1e-3
    assert (result.best_x >= -1e-6).all()
    # Best appreciation should beat the uniform start
    uniform_app = evaluate_allocation(
        continuous_beerwiser.input_dict, np.array([budget / 2, budget / 2]), "Base case", "Test DMO"
    )
    assert result.best_appreciation >= uniform_app - 1e-3


def test_optimize_continuous_end_to_end_beerwiser(beerwiser_appreciated):
    """End-to-end: TheResponsibleBusinessSimulator.optimize_continuous → result matches or beats grid baseline."""
    # Grid baseline via legacy optimize
    case_grid = beerwiser_appreciated.copy()
    case_grid.optimize("Base case", new_dmo_name="Grid Baseline", max_combinations=60000)
    grid_idx = np.where(case_grid.input_dict["decision_makers_options"] == "Grid Baseline")[0][0]
    grid_alloc = case_grid.input_dict["decision_makers_option_value"][grid_idx]
    grid_app = evaluate_allocation(case_grid.input_dict, grid_alloc, "Base case", "Grid Baseline")

    # SLSQP via the new high-level API
    case_slsqp = beerwiser_appreciated.copy()
    result = case_slsqp.optimize_continuous("Base case", method="slsqp", n_starts=30, seed=42, dmo_name="SLSQP DMO")

    # Status moved to "optimize"
    assert 3 in case_slsqp.status

    # SLSQP matches or beats the grid within a small tolerance (continuous should
    # in principle do strictly better; allow 0.5 appreciation-point slack for noise)
    assert (
        result.best_appreciation >= grid_app - 0.5
    ), f"SLSQP appreciation {result.best_appreciation:.4f} is worse than grid {grid_app:.4f} by more than 0.5"

    # The winning DMO is in input_dict
    assert "SLSQP DMO" in case_slsqp.input_dict["decision_makers_options"]


def test_unsupported_method_raises(continuous_beerwiser):
    """basin_hopping / genetic_algorithm are W3+ work — must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="basin_hopping"):
        continuous_beerwiser.optimize("Base case", method="basin_hopping")
    with pytest.raises(NotImplementedError, match="genetic_algorithm"):
        continuous_beerwiser.optimize("Base case", method="genetic_algorithm")
