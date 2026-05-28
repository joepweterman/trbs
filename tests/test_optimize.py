# pylint: disable=W0212

"""
This module contains all tests for the Optimize() class
"""

import copy

import pytest
import numpy as np
from vlinder.appreciate import Appreciate
from vlinder.optimize import Optimize, evaluate_allocation
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
    to lock down the contract for ContinuousOptimize's use.
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
