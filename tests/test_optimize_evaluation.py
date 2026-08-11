# pylint: disable=W0212

"""
Tests for the free evaluation functions in vlinder.optimize and for the result container.

Every solver scores allocations through these two functions, so their contract is the one thing
the whole module rests on: the right number out, and the caller's case left untouched.
"""

import copy

import numpy as np
import pytest

from vlinder.optimize import OptimizationResult, evaluate_allocation, evaluate_and_appreciate
from .conftest import BEERWISER_BUDGET, BEERWISER_EQUAL_SPREAD


def assert_unchanged(input_dict, snapshot):
    """Assert that no table in ``input_dict`` moved relative to ``snapshot``."""
    for key, expected in snapshot.items():
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(input_dict[key], expected, err_msg=f"input_dict['{key}'] was mutated")
        else:
            assert input_dict[key] == expected, f"input_dict['{key}'] was mutated"


def test_evaluate_allocation_returns_the_known_appreciation(prepared_solver):
    """The "Equal spread" allocation must score its documented appreciation."""
    appreciation = evaluate_allocation(
        prepared_solver.input_dict, np.array([150000, 150000]), "Base case", "Equal spread"
    )

    assert appreciation == pytest.approx(BEERWISER_EQUAL_SPREAD, abs=1e-9)


def test_evaluate_allocation_responds_to_the_allocation(prepared_solver):
    """A different allocation must give a different appreciation."""
    equal = evaluate_allocation(prepared_solver.input_dict, np.array([150000, 150000]), "Base case", "Equal spread")
    skewed = evaluate_allocation(prepared_solver.input_dict, np.array([100000, 200000]), "Base case", "Equal spread")

    assert skewed != pytest.approx(equal, abs=1e-9)


def test_evaluate_allocation_does_not_mutate_the_case(prepared_solver):
    """The caller's case must survive repeated evaluation untouched.

    Solvers call this hundreds of thousands of times against one shared dictionary, so any
    leak would compound silently into every result that follows.
    """
    snapshot = copy.deepcopy(prepared_solver.input_dict)

    evaluate_allocation(prepared_solver.input_dict, np.array([150000, 150000]), "Base case", "Equal spread")
    evaluate_allocation(prepared_solver.input_dict, np.array([100000, 200000]), "Base case", "Equal spread")

    assert_unchanged(prepared_solver.input_dict, snapshot)


def test_evaluate_allocation_accepts_a_float_allocation(prepared_solver):
    """A fractional allocation must not be silently rounded.

    Beerwiser's option values import as whole numbers. Writing a float into that table without
    casting rounds it, which flattens the small perturbations a gradient-based solver relies on
    and leaves it stuck at its starting point.
    """
    base = evaluate_allocation(prepared_solver.input_dict, np.array([150000.0, 150000.0]), "Base case", "Test DMO")
    nudged = evaluate_allocation(prepared_solver.input_dict, np.array([150000.5, 149999.5]), "Base case", "Test DMO")

    assert base != nudged


def test_evaluate_and_appreciate_returns_the_full_output(prepared_solver):
    """The full evaluation carries the key outputs and their appreciations, not just the total."""
    output = evaluate_and_appreciate(
        prepared_solver.input_dict, np.array([150000, 150000]), "Base case", "Equal spread"
    )

    assert set(output) >= {
        "key_outputs",
        "appreciations",
        "weighted_appreciations",
        "decision_makers_option_appreciation",
    }
    assert output["decision_makers_option_appreciation"] == pytest.approx(BEERWISER_EQUAL_SPREAD, abs=1e-9)
    assert set(output["appreciations"]) == set(prepared_solver.input_dict["key_outputs"])


def test_precomputed_boundaries_do_not_change_the_answer(prepared_solver):
    """Passing the frozen boundaries in must give exactly what deriving them per call gives.

    Solvers pass them to avoid rebuilding the same boundaries on every evaluation, so the two
    paths have to agree to the last digit.
    """
    allocation = np.array([123456.0, 176544.0])
    derived = evaluate_allocation(prepared_solver.input_dict, allocation, "Base case", "Test DMO")
    passed_in = evaluate_allocation(
        prepared_solver.input_dict, allocation, "Base case", "Test DMO", prepared_solver._frozen_boundaries
    )

    assert derived == passed_in


def test_optimization_result_reports_budget_and_summary():
    """The result container carries the run's context and renders a readable summary."""
    result = OptimizationResult(
        method="slsqp",
        dmo_name="Optimized (SLSQP) (Base case)",
        scenario="Base case",
        allocation=np.array([25000.0, 275000.0]),
        appreciation=65.7116,
        budget=BEERWISER_BUDGET,
        budget_spent=300000.0,
        calculation_time=1.25,
        timestamp="2026-08-10T12:00:00+00:00",
    )
    summary = result.summary()

    assert result.budget_spent == BEERWISER_BUDGET
    assert "Base case" in summary
    assert "slsqp" in summary
    assert "65.7116" in summary
