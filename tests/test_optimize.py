# pylint: disable=W0212

"""
This module contains all tests for the Optimize() class
"""

import copy

import pytest
import numpy as np
from vlinder.appreciate import Appreciate
from vlinder.optimize import Optimize, OptimizationResult, evaluate_allocation
from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.utils import get_values_from_target, suppress_print
from .params import INPUT_DICT_BEERWISER, OUTPUT_DICT_BEERWISER


@pytest.fixture(name="optimize_beerwiser")
def fixture_optimize_beerwiser():
    """
    This fixture initialises a Beerwiser case.
    :return: an Optimize class for Beerwiser
    """
    input_dict = INPUT_DICT_BEERWISER.copy()
    output_dict = OUTPUT_DICT_BEERWISER.copy()
    return Optimize(input_dict, output_dict)


@suppress_print
def test_optimize_name(optimize_beerwiser):
    """
    This function tests the wrapper function .optimize() that performs the full optimization process,
    in case when the optimized DMO name is not already in the input dict.
    :param optimize_beerwiser: an Optimize() class for Beerwiser
    """
    optimize_beerwiser.optimize_single_scenario("Base case", "Equal spread", max_combinations=60000)
    result = optimize_beerwiser.input_dict

    count_dictionaries = get_values_from_target(result, "decision_makers_options")[0].size
    expected_count = 3
    assert count_dictionaries == expected_count


def test_find_dict_values(optimize_beerwiser):
    """
    This function tests find_dict_values() to return the right values.
    :param optimize_beerwiser: an Optimize() class for Beerwiser
    """
    result = optimize_beerwiser.find_dict_values("Base case")
    expected_result_1 = {
        "dmo_name": "Equal spread",
        "decision_maker_options": [150000, 150000],
        "max_appreciated_value": 65.51984611881377,
    }
    expected_result_2 = 300000

    assert result[0]["dmo_name"] == expected_result_1["dmo_name"]
    assert np.array_equal(result[0]["decision_maker_options"], expected_result_1["decision_maker_options"])
    assert result[0]["max_appreciated_value"] == expected_result_1["max_appreciated_value"]
    assert result[1] == expected_result_2


@pytest.mark.parametrize(
    "max_investment, expected_result",
    [
        (1000, 1000),
        (15700, 1500),
        (999999, 10000),
        (1234567, 1200),
        (499999, 5000),
    ],
)
def test_scale_max_investment(max_investment, expected_result):
    """
    This function tests .scale_max_investment() return the right values.
    """
    scaled_max_investment = Optimize.scale_max_investment(max_investment)
    assert scaled_max_investment == expected_result


@pytest.mark.parametrize(
    "max_investment, scaled_max_investment, num_internal_inputs, max_combinations, expected_result",
    [
        (1000, 1000, 2, 60000, 1),
        (1000, 1000, 3, 60000, 4),
        (1000, 1000, 2, 100, 20),
        (15700, 15000, 2, 60000, 1.0466666666666666),
    ],
)
def test_calculate_step_size(
    max_investment, scaled_max_investment, num_internal_inputs, max_combinations, expected_result
):
    """
    This function tests calculate_step_size() to return the right values.
    """
    step_size = Optimize.calculate_step_size(
        max_investment, scaled_max_investment, num_internal_inputs, max_combinations
    )
    assert step_size == expected_result


@pytest.mark.parametrize(
    "max_investment, step_size, num_internal_inputs, expected_result",
    [
        (10, 5, 2, [(5, 5), (10, 0), (0, 10)]),
        (10, 10, 2, [(0, 10), (10, 0)]),
        (10, 5, 3, [(0, 5, 5), (5, 5, 0), (5, 0, 5), (10, 0, 0), (0, 10, 0), (0, 0, 10)]),
    ],
)
def test_generate_combinations(max_investment, step_size, num_internal_inputs, expected_result):
    """
    This function tests generate_combinations() returns the right values
    and if the sum is equal to the max investment
    """
    combinations = Optimize.generate_combinations(max_investment, step_size, num_internal_inputs)
    assert sorted(combinations) == sorted(expected_result)

    for combination in combinations:
        assert sum(combination) == max_investment


def test_grid_search(optimize_beerwiser):
    """
    This function tests grid_search() to return the right values.
    :param optimize_beerwiser: an Optimize() class for Beerwiser
    """
    result = optimize_beerwiser.grid_search(
        "Base case",
        [
            (25000, 275000),
            (20000, 280000),
            (30000, 270000),
        ],
        "Optimized DMO",
        {
            "dmo_name": "Equal spread",
            "decision_maker_options": np.array([150000, 150000]),
            "max_appreciated_value": 65.51984611881377,
        },
    )
    expected_result = ("Optimized DMO", 65.7115899862911)
    assert result == expected_result


def test_evaluate_allocation_purity_and_correctness(optimize_beerwiser):
    """
    Regression test for the W2 pure-function refactor (evaluate_allocation):
      1. Returns the correct appreciation for a known allocation
         (matches Equal spread's pre-computed max_appreciated_value).
      2. Does NOT mutate the caller's input_dict.

    Tests the function in isolation, without going through grid_search,
    to lock down the contract for the SLSQP solver's use.
    """
    # Phase-A setup (mirrors grid_search lines 122-133) — boundaries + reset key_output_automatic.
    # The "Equal spread" DMO already exists in INPUT_DICT_BEERWISER, so no need to register it.
    boundaries = Appreciate(optimize_beerwiser.input_dict, optimize_beerwiser.output_dict)._get_start_and_end_points()
    optimize_beerwiser.input_dict["key_output_automatic"] = np.zeros(
        len(optimize_beerwiser.input_dict["key_output_automatic"]), dtype=int
    )
    optimize_beerwiser.input_dict["key_output_start"] = np.array([v[0] for v in boundaries.values()])
    optimize_beerwiser.input_dict["key_output_end"] = np.array([v[1] for v in boundaries.values()])

    # Snapshot input_dict for the purity check
    snapshot = copy.deepcopy(optimize_beerwiser.input_dict)

    # 1. Correctness: Equal spread allocation [150000, 150000] should give exactly 65.51984611881377
    #    (this is the max_appreciated_value asserted in test_find_dict_values for the same DMO/scenario)
    appreciation = evaluate_allocation(
        optimize_beerwiser.input_dict, np.array([150000, 150000]), "Base case", "Equal spread"
    )
    assert appreciation == pytest.approx(65.51984611881377, abs=1e-9)

    # 2. Call again with a different allocation; result should be different
    appreciation_alt = evaluate_allocation(
        optimize_beerwiser.input_dict, np.array([100000, 200000]), "Base case", "Equal spread"
    )
    assert appreciation_alt != pytest.approx(appreciation, abs=1e-9)

    # 3. Purity: input_dict must be byte-identical to the snapshot after both calls
    for key in snapshot:
        if isinstance(snapshot[key], np.ndarray):
            np.testing.assert_array_equal(
                optimize_beerwiser.input_dict[key], snapshot[key], err_msg=f"input_dict['{key}'] was mutated"
            )
        else:
            assert optimize_beerwiser.input_dict[key] == snapshot[key], f"input_dict['{key}'] was mutated"


@suppress_print
def test_optimize(optimize_beerwiser):
    """
    This function tests the wrapper function .optimize() that performs the full optimization process
    :param optimize_beerwiser: an Optimize() class for Beerwiser
    """
    optimize_beerwiser.optimize_single_scenario("Base case", "Optimized DMO", max_combinations=60000)
    result = optimize_beerwiser.input_dict

    count_dictionaries = get_values_from_target(result, "decision_makers_options")[0].size
    expected_count = 4
    assert count_dictionaries == expected_count


@pytest.fixture(name="optimize_beerwiser_aleady_optimal")
def fixture_optimize_beerwiser_aleady_optimal():
    """
    This fixture initialises a dummy Beerwiser case
    :return: an Optimize class for Beerwiser
    """
    input_dict = INPUT_DICT_BEERWISER.copy()
    output_dict = OUTPUT_DICT_BEERWISER.copy()
    output_dict["Base case"]["highest_weighted_dmo"] = "Equal spread"
    output_dict["Base case"]["Equal spread"][
        "decision_makers_option_appreciation"
    ] = 100  # Give a DMO a weighted apprecation of 100
    return Optimize(input_dict, output_dict)


def test_optimizer_already_optimal(optimize_beerwiser_aleady_optimal):
    """
    This function tests the wrapper function .optimize() that performs the full optimization process,
    in case when the optimizer does not find a better DMO than the one already present in the input_dict.
    To put in a metaphore: when the Optimizer says the Vaalserberg is the highest,
    but you have already found the Mt. Everest.
    :param optimize_beerwiser_aleady_optimal: an Optimize() class for Beerwiser
    """
    optimize_beerwiser_aleady_optimal.optimize_single_scenario("Base case", "Optimized DMO", max_combinations=60000)
    result = optimize_beerwiser_aleady_optimal.input_dict

    result_best_dmo = result["decision_makers_option_value"][
        np.where(result["decision_makers_options"] == "Optimized DMO")[0][0]
    ]
    expected_best_dmo = [
        150000,
        150000,
    ]  # The best DMO is the 'Equal spread', as it has a fictional weighted apprecation of 100
    assert np.array_equal(result_best_dmo, expected_best_dmo)


# ======================================================================
# Continuous optimization (SLSQP) — merged here when ContinuousOptimize
# was folded into Optimize. Plus dispatch/isolation tests for the unified
# run() / TheResponsibleBusinessSimulator.optimize() entry point.
# ======================================================================
@pytest.fixture(name="continuous_beerwiser")
def fixture_continuous_beerwiser():
    """An Optimize bound to Beerwiser params, with Phase-A SLSQP setup done."""
    optimizer = Optimize(copy.deepcopy(INPUT_DICT_BEERWISER), copy.deepcopy(OUTPUT_DICT_BEERWISER))
    # Reference allocation = Equal spread DMO ([150000, 150000]); budget 300000
    optimizer._prepare_input_dict("Test DMO", np.array([150000, 150000]))
    return optimizer


@pytest.fixture(name="beerwiser_appreciated")
def fixture_beerwiser_appreciated():
    """A real TheResponsibleBusinessSimulator through build + evaluate + appreciate."""
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

    for key in snapshot:
        if isinstance(snapshot[key], np.ndarray):
            np.testing.assert_array_equal(
                continuous_beerwiser.input_dict[key], snapshot[key], err_msg=f"input_dict['{key}'] was mutated"
            )
        else:
            assert continuous_beerwiser.input_dict[key] == snapshot[key], f"input_dict['{key}'] mutated"


def test_slsqp_single_start_beats_uniform(continuous_beerwiser):
    """SLSQP from a uniform start should converge to >= uniform's appreciation."""
    budget = 300000.0
    uniform_x = np.array([budget / 2, budget / 2])
    uniform_app = evaluate_allocation(continuous_beerwiser.input_dict, uniform_x, "Base case", "Test DMO")

    eval_counter = [0]
    res = continuous_beerwiser._slsqp_from_start(uniform_x, "Base case", "Test DMO", budget, eval_counter)
    slsqp_app = -float(res.fun)

    assert res.success, f"SLSQP did not converge: {res.message}"
    assert slsqp_app >= uniform_app - 1e-3
    # Capped-simplex formulation: allocation must be feasible (Σx ≤ B). Beerwiser
    # is monotone in spend (exp02), so the budget binds → the solution still spends B.
    assert float(np.sum(res.x)) <= budget + 1e-3  # feasible under Σx ≤ B
    assert abs(float(np.sum(res.x)) - budget) < 1e-3  # binds (Beerwiser monotone)
    assert (res.x >= -1e-6).all()


def test_multistart_slsqp_returns_structured_result(continuous_beerwiser):
    """optimize_slsqp returns an OptimizationResult with sane fields."""
    budget = 300000.0
    result = continuous_beerwiser.optimize_slsqp(
        scenario="Base case",
        budget=budget,
        dmo_name="Test DMO",  # already prepared by the fixture
        n_starts=20,
        seed=42,
    )
    assert isinstance(result, OptimizationResult)
    assert result.method == "slsqp"
    assert result.n_starts == 20
    assert result.n_converged >= 10  # most starts on a 2D simplex should converge
    assert len(result.per_start_results) == 20
    assert result.n_function_evals > 20  # at least 1 per start, usually many more
    assert result.wall_time_s > 0
    # Best allocation is feasible under Σx ≤ B; Beerwiser binds (monotone, exp02)
    assert float(np.sum(result.best_x)) <= budget + 1e-3  # feasible
    assert abs(float(np.sum(result.best_x)) - budget) < 1e-3  # binds
    assert (result.best_x >= -1e-6).all()
    # Best appreciation beats the uniform start (alias best_appreciation → appreciation)
    uniform_app = evaluate_allocation(
        continuous_beerwiser.input_dict, np.array([budget / 2, budget / 2]), "Base case", "Test DMO"
    )
    assert result.best_appreciation >= uniform_app - 1e-3


def test_unsupported_method_raises(continuous_beerwiser):
    """basin_hopping / genetic_algorithm are W3+ work — must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="basin_hopping"):
        continuous_beerwiser.run("Base case", method="basin_hopping")
    with pytest.raises(NotImplementedError, match="genetic_algorithm"):
        continuous_beerwiser.run("Base case", method="genetic_algorithm")


def test_optimize_end_to_end_slsqp_beats_grid(beerwiser_appreciated):
    """End-to-end via the unified .optimize(): SLSQP matches or beats the grid baseline."""
    # Grid baseline
    case_grid = beerwiser_appreciated.copy()
    case_grid.optimize("Base case", method="grid", new_dmo_name="Grid Baseline", max_combinations=60000)
    grid_idx = np.where(case_grid.input_dict["decision_makers_options"] == "Grid Baseline")[0][0]
    grid_alloc = case_grid.input_dict["decision_makers_option_value"][grid_idx]
    grid_app = evaluate_allocation(case_grid.input_dict, grid_alloc, "Base case", "Grid Baseline")

    # SLSQP via the unified API
    case_slsqp = beerwiser_appreciated.copy()
    result = case_slsqp.optimize("Base case", method="slsqp", n_starts=30, seed=42, dmo_name="SLSQP DMO")

    # Status moved to "optimize"
    assert 3 in case_slsqp.status
    # Continuous should in principle do >= grid; allow 0.5 appreciation-point slack for noise
    assert (
        result.best_appreciation >= grid_app - 0.5
    ), f"SLSQP appreciation {result.best_appreciation:.4f} is worse than grid {grid_app:.4f} by more than 0.5"
    assert "SLSQP DMO" in case_slsqp.input_dict["decision_makers_options"]


def test_optimize_single_grid_returns_result(beerwiser_appreciated):
    """A single-method .optimize() returns one OptimizationResult and sets status 3."""
    result = beerwiser_appreciated.optimize("Base case", method="grid")
    assert isinstance(result, OptimizationResult)
    assert result.method == "grid"
    assert result.appreciation > 0
    assert 3 in beerwiser_appreciated.status


def test_optimize_list_prints_all_and_returns_best(beerwiser_appreciated, capsys):
    """A list of methods prints each result and returns the best; only the winner is written back."""
    n_before = beerwiser_appreciated.input_dict["decision_makers_options"].size

    result = beerwiser_appreciated.optimize("Base case", method=["grid", "slsqp"], n_starts=20, seed=42)
    printed = capsys.readouterr().out

    # One comparison line per method
    assert "[grid]" in printed
    assert "[slsqp]" in printed

    # Returns a single OptimizationResult — the better-scoring method
    assert isinstance(result, OptimizationResult)
    assert result.method in ("grid", "slsqp")

    # Only ONE new DMO written back (the winner), not one per method
    assert beerwiser_appreciated.input_dict["decision_makers_options"].size == n_before + 1

    grid_name = Optimize.DEFAULT_DMO_NAME["grid"]
    slsqp_name = Optimize.DEFAULT_DMO_NAME["slsqp"]
    names = list(beerwiser_appreciated.input_dict["decision_makers_options"])
    present = [nm for nm in (grid_name, slsqp_name) if nm in names]
    assert len(present) == 1  # the loser's DMO never reaches the real input_dict
    assert result.dmo_name == present[0]
