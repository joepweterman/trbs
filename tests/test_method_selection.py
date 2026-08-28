# pylint: disable=W0212

"""
This module contains all tests for the automatic method selection (method="auto")
"""

import copy

import pytest
import numpy as np
from vlinder.method_selection import CaseDiagnosis, diagnose_case, select_method
from vlinder.optimize import BaseSolver, Optimize, evaluate_and_appreciate
from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.utils import suppress_print
from .params import INPUT_DICT_BEERWISER, OUTPUT_DICT_BEERWISER


def make_diagnosis(**overrides):
    """
    This function builds a diagnosis describing a provably concave case, with individual
    properties overridden per test.
    :param overrides: the CaseDiagnosis fields to change
    :return: a CaseDiagnosis
    """
    fields = {
        "k": 3,
        "budget": 100.0,
        "deps_affine": True,
        "max_affine_residual": 0.0,
        "convex_carriers": (),
        "floor_clip_fraction": 0.0,
        "plateau_fraction": 0.0,
        "n_probe_evaluations": 30,
    }
    fields.update(overrides)
    return CaseDiagnosis(**fields)


@pytest.fixture(name="optimize_beerwiser")
def fixture_optimize_beerwiser():
    """
    This fixture initialises a Beerwiser case.
    :return: an Optimize class for Beerwiser
    """
    return Optimize(copy.deepcopy(INPUT_DICT_BEERWISER), copy.deepcopy(OUTPUT_DICT_BEERWISER))


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, "slsqp"),
        ({"deps_affine": False, "max_affine_residual": 0.3}, "basin_hopping"),
        ({"convex_carriers": ("Increase in production capacity",)}, "basin_hopping"),
        ({"floor_clip_fraction": 0.5}, "basin_hopping"),
        ({"floor_clip_fraction": 0.5, "plateau_fraction": 1.0}, "genetic_algorithm"),
        ({"floor_clip_fraction": 0.5, "plateau_fraction": 0.5}, "basin_hopping"),
        ({"budget": 0.0}, "grid"),
    ],
)
def test_select_method_branches(overrides, expected):
    """
    This function tests that every branch of the decision tree returns its method.
    :param overrides: the CaseDiagnosis fields that steer the case down a branch
    :param expected: the method that branch should select
    """
    choice = select_method(make_diagnosis(**overrides))

    assert choice.method == expected
    assert choice.reason  # every choice explains itself


def test_select_method_concave_lowers_the_start_budget():
    """
    This function tests that a provably concave case gets the cheap local solver budget:
    with every local optimum global there is nothing for extra starts to find.
    """
    concave = select_method(make_diagnosis())
    multimodal = select_method(make_diagnosis(convex_carriers=("KO",)))

    assert (
        concave.method_kwargs["n_starts"] < multimodal.method_kwargs["n_starts"] * multimodal.method_kwargs["n_hops"]
    )


def test_diagnose_beerwiser(optimize_beerwiser):
    """
    This function tests the probe on Beerwiser, whose dependencies contain min() operators
    (so they are not affine) and whose appreciation terms are clipped at their floor over a
    region. Its key outputs are all bigger-the-better, so it carries no convex term.
    :param optimize_beerwiser: an Optimize() class for Beerwiser
    """
    solver = BaseSolver(optimize_beerwiser.input_dict, optimize_beerwiser.output_dict)
    solver._prepare_input_dict("Test DMO", np.array([150000, 150000]))
    diagnosis = diagnose_case(evaluate_and_appreciate, solver.input_dict, "Base case", "Test DMO", 300000.0)

    assert diagnosis.k == 2
    assert not diagnosis.deps_affine
    assert diagnosis.max_affine_residual > 0
    assert not diagnosis.convex_carriers
    assert diagnosis.floor_clip_fraction > 0
    assert not diagnosis.is_provably_concave
    # The probe must stay cheap relative to the solvers it chooses between.
    assert diagnosis.n_probe_evaluations < 100


def test_select_method_for_leaves_no_dmo_behind(optimize_beerwiser):
    """
    This function tests that diagnosing a case does not change it: the probe runs on a
    deepcopy, so no probe DMO is left in the case that is about to be optimized.
    :param optimize_beerwiser: an Optimize() class for Beerwiser
    """
    before = list(optimize_beerwiser.input_dict["decision_makers_options"])
    choice = optimize_beerwiser.select_method_for("Base case")

    assert choice.method == "basin_hopping"
    assert list(optimize_beerwiser.input_dict["decision_makers_options"]) == before
    assert Optimize.PROBE_DMO_NAME not in before


@suppress_print
def test_optimize_auto_runs_the_selected_method():
    """
    This function tests the "auto" entry point end to end: the case is probed, the chosen
    method runs, and the result reports which method was chosen and why. Keyword arguments
    override the solver budget the tree attached to its choice, which keeps this test cheap.
    """
    case = TheResponsibleBusinessSimulator("Beerwiser")
    case.build()
    case.evaluate()
    case.appreciate()
    case.optimize("Base case", method="auto", n_starts=1, n_hops=2, seed=1)
    result = case.optimization_result

    assert result.method == "basin_hopping"
    assert result.selection.method == "basin_hopping"
    assert result.selection.diagnosis.floor_clip_fraction > 0
    assert result.n_starts == 1  # the override won, not the tree's budget
    assert result.appreciation > 0


@suppress_print
def test_optimize_explicit_method_overrides_selection(optimize_beerwiser):
    """
    This function tests that naming a method skips the tree entirely.
    :param optimize_beerwiser: an Optimize() class for Beerwiser
    """
    result = optimize_beerwiser.run("Base case", method="slsqp", n_starts=2, seed=1)

    assert result.method == "slsqp"
    assert result.selection is None


def test_auto_cannot_be_combined_in_a_list(optimize_beerwiser):
    """
    This function tests that "auto" is rejected inside a list of methods: it picks one
    method, so it cannot itself be one of several to compare.
    :param optimize_beerwiser: an Optimize() class for Beerwiser
    """
    with pytest.raises(NotImplementedError) as error:
        optimize_beerwiser.run("Base case", method=["auto", "slsqp"])

    assert "auto" in str(error.value)
