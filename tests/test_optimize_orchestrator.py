# pylint: disable=W0212

"""
Tests for choosing and dispatching a solver: the Optimize orchestrator and the optimize()
entry point on a case.

These are the paths a user actually takes, so they check the things that only show up once a
solver runs inside a real case: which method gets picked, what ends up in the case afterwards,
and what the caller gets back.
"""

import numpy as np
import pytest

from vlinder.optimize import GridSearch, Optimize, SLSQPSolver, score_allocation
from vlinder.utils import suppress_print
from .conftest import BEERWISER_BUDGET, BEERWISER_OPTIMUM


@pytest.fixture(name="orchestrator")
def fixture_orchestrator(beerwiser_dicts):
    """An Optimize bound to a private copy of Beerwiser."""
    return Optimize(*beerwiser_dicts)


def test_budget_is_inferred_from_the_case(orchestrator):
    """The orchestrator reads the budget off the case when the caller does not supply one."""
    assert orchestrator.budget == BEERWISER_BUDGET


def test_unsupported_method_raises(orchestrator):
    """An unknown method name is refused, and the message names the ones that exist."""
    with pytest.raises(NotImplementedError, match="tabu_search"):
        orchestrator.run("Base case", method="tabu_search")


def test_auto_cannot_be_one_of_several_methods(orchestrator):
    """ "auto" picks a method, so it cannot itself be one of the methods to compare."""
    with pytest.raises(NotImplementedError) as error:
        orchestrator.run("Base case", method=["auto", "slsqp"])

    assert "auto" in str(error.value)


@suppress_print
def test_named_method_skips_the_selection(orchestrator):
    """Naming a method runs it and records no automatic choice."""
    result = orchestrator.run("Base case", method="slsqp", n_starts=2, seed=1)

    assert result.method == "slsqp"
    assert result.selection is None


@suppress_print
def test_the_default_method_is_basin_hopping(orchestrator):
    """Without a method the orchestrator runs basin-hopping, without an automatic choice."""
    result = orchestrator.run("Base case", n_starts=1, n_hops=2, seed=1)

    assert result.method == "basin_hopping"
    assert result.selection is None
    assert "grid_capped" in Optimize.METHOD_REGISTRY


def test_auto_picks_a_method_and_explains_itself(orchestrator, capsys):
    """The automatic choice runs a real solver and says, readably, why it chose it."""
    result = orchestrator.run("Base case", method="auto", n_starts=1, n_hops=2, seed=1)
    printed = capsys.readouterr().out

    assert result.method == "basin_hopping"
    assert result.selection.method == "basin_hopping"
    assert result.n_starts == 1, "an explicit setting must win over the budget the tree attached"
    assert "Automatic method selection chose basin-hopping" in printed
    assert "internal variable inputs" in printed


def test_selection_leaves_no_probe_option_behind(orchestrator):
    """Diagnosing a case must not change it: the probe runs on a copy."""
    before = list(orchestrator.input_dict["decision_makers_options"])
    orchestrator.select_method_for("Base case")

    assert list(orchestrator.input_dict["decision_makers_options"]) == before
    assert Optimize.PROBE_DMO_NAME not in before


@suppress_print
def test_the_option_name_records_the_scenario(orchestrator):
    """An allocation is only optimal for the scenario it was optimized under, so the name says so."""
    result = orchestrator.run("Base case", method="slsqp", n_starts=2, seed=1, dmo_name="My DMO")

    assert result.dmo_name == "My DMO (Base case)"
    assert "My DMO (Base case)" in orchestrator.input_dict["decision_makers_options"]


@suppress_print
def test_optimizing_two_scenarios_keeps_them_apart(orchestrator):
    """Optimizing the same case twice must produce two options, not overwrite one."""
    orchestrator.run("Base case", method="slsqp", n_starts=2, seed=1)
    orchestrator.run("Optimistic", method="slsqp", n_starts=2, seed=1)

    names = list(orchestrator.input_dict["decision_makers_options"])
    assert SLSQPSolver.default_dmo_name + " (Base case)" in names
    assert SLSQPSolver.default_dmo_name + " (Optimistic)" in names


def test_a_list_of_methods_prints_each_and_keeps_the_best(orchestrator, capsys):
    """Every method reports, one winner is written back."""
    n_before = orchestrator.input_dict["decision_makers_options"].size

    result = orchestrator.run("Base case", method=["grid", "slsqp"], seed=42)
    printed = capsys.readouterr().out

    assert "[grid]" in printed
    assert "[slsqp]" in printed
    assert result.method in ("grid", "slsqp")

    # Only the winner reaches the case, not one option per method.
    assert orchestrator.input_dict["decision_makers_options"].size == n_before + 1
    names = list(orchestrator.input_dict["decision_makers_options"])
    candidates = [
        f"{GridSearch.default_dmo_name} (Base case)",
        f"{SLSQPSolver.default_dmo_name} (Base case)",
    ]
    present = [name for name in candidates if name in names]
    assert len(present) == 1


@suppress_print
def test_shared_settings_reach_every_method(orchestrator):
    """A setting passed directly applies to each method that runs.

    Both solvers report the number of starts they used, and both have a different default
    (100 for SLSQP, 1 for basin-hopping), so whichever one wins proves the setting arrived.
    """
    result = orchestrator.run("Base case", method=["slsqp", "basin_hopping"], n_starts=3, n_hops=2, seed=1)

    assert result.n_starts == 3


def test_method_kwargs_give_each_method_its_own_settings(orchestrator, capsys):
    """Two methods with parameters the other one does not have can run in one call."""
    orchestrator.run(
        "Base case",
        method=["grid", "slsqp"],
        method_kwargs={"grid": {"max_combinations": 1000}, "slsqp": {"n_starts": 4, "seed": 1}},
    )
    printed = capsys.readouterr().out

    assert "[grid]" in printed
    assert "[slsqp]" in printed


@suppress_print
def test_method_kwargs_win_over_the_shared_settings(orchestrator):
    """A per-method setting overrides the same setting passed for everyone."""
    result = orchestrator.run(
        "Base case", method="slsqp", n_starts=25, seed=1, method_kwargs={"slsqp": {"n_starts": 3}}
    )

    assert result.n_starts == 3
    assert len(result.per_start_results) == 3


def test_method_kwargs_for_a_method_that_is_not_running_raises(orchestrator):
    """Settings addressed to a method that is not running are a typo, not a silent no-op."""
    with pytest.raises(NotImplementedError, match="genetic_algorithm"):
        orchestrator.run("Base case", method="slsqp", method_kwargs={"genetic_algorithm": {"seed": 1}})


@suppress_print
def test_without_a_configured_name_the_solver_default_applies(orchestrator):
    """The test fixtures carry no Optimize_DMO_name, so the solver default names the option."""
    result = orchestrator.run("Base case", method="slsqp", n_starts=2, seed=1)

    assert result.dmo_name == f"{SLSQPSolver.default_dmo_name} (Base case)"


# ======================================================================
# Through the case: TheResponsibleBusinessSimulator.optimize()
# ======================================================================
@suppress_print
def test_configured_name_is_extended_with_method_and_scenario(beerwiser_appreciated):
    """A configured Optimize_DMO_name is the base; the method and scenario are appended."""
    beerwiser_appreciated.optimize("Base case", method="grid")
    result = beerwiser_appreciated.optimization_result

    assert result.dmo_name == "Optimized_DMO (grid) (Base case)"
    assert result.dmo_name in beerwiser_appreciated.input_dict["decision_makers_options"]


@suppress_print
def test_an_explicit_name_overrides_the_configured_one(beerwiser_appreciated):
    """A caller-supplied name wins over the configuration sheet and gets the scenario only."""
    beerwiser_appreciated.optimize("Base case", method="grid", dmo_name="My Grid Run")

    names = list(beerwiser_appreciated.input_dict["decision_makers_options"])
    assert "My Grid Run (Base case)" in names
    assert not any(name.startswith("Optimized_DMO (") for name in names)


@suppress_print
def test_case_optimize_returns_the_input_dict(beerwiser_appreciated):
    """optimize() hands back the updated case, which is what the front end reads."""
    returned = beerwiser_appreciated.optimize("Base case", method="grid")

    assert returned is beerwiser_appreciated.input_dict
    assert "decision_makers_options" in returned
    assert 3 in beerwiser_appreciated.status


@suppress_print
def test_case_optimize_records_the_result(beerwiser_appreciated):
    """The full result of the run stays available on the case."""
    beerwiser_appreciated.optimize("Base case", method="grid")
    result = beerwiser_appreciated.optimization_result

    assert result.method == "grid"
    assert result.scenario == "Base case"
    assert result.appreciation == pytest.approx(BEERWISER_OPTIMUM, abs=1e-6)
    assert result.budget_spent <= result.budget + 1e-6


@suppress_print
def test_case_optimize_dispatches_every_method(beerwiser_appreciated):
    """Each method reaches the case through the same entry point."""
    beerwiser_appreciated.optimize("Base case", method="basin_hopping", n_hops=3, n_starts=1, seed=42)
    result = beerwiser_appreciated.optimization_result

    assert result.method == "basin_hopping"
    # The packaged case configures Optimize_DMO_name = "Optimized_DMO", which is
    # the base of the default name, extended with the method and the scenario.
    expected_name = "Optimized_DMO (basin-hopping) (Base case)"
    assert expected_name in beerwiser_appreciated.input_dict["decision_makers_options"]


@suppress_print
def test_continuous_matches_or_beats_the_grid_baseline(beerwiser_appreciated):
    """The whole point of the continuous solvers: at least as good as enumerating the grid."""
    case_grid = beerwiser_appreciated.copy()
    case_grid.optimize("Base case", method="grid", dmo_name="Grid Baseline", max_combinations=60000)
    grid_appreciation = case_grid.optimization_result.appreciation

    case_slsqp = beerwiser_appreciated.copy()
    case_slsqp.optimize("Base case", method="slsqp", n_starts=30, seed=42, dmo_name="SLSQP DMO")
    result = case_slsqp.optimization_result

    assert 3 in case_slsqp.status
    assert (
        result.appreciation >= grid_appreciation - 0.5
    ), f"SLSQP appreciation {result.appreciation:.4f} trails grid {grid_appreciation:.4f} by more than 0.5"
    assert "SLSQP DMO (Base case)" in case_slsqp.input_dict["decision_makers_options"]


@suppress_print
def test_the_written_back_allocation_scores_what_the_result_claims(beerwiser_appreciated):
    """The allocation stored on the case must reproduce the reported appreciation."""
    beerwiser_appreciated.optimize("Base case", method="slsqp", n_starts=5, seed=7, dmo_name="Check")
    result = beerwiser_appreciated.optimization_result

    input_dict = beerwiser_appreciated.input_dict
    idx = np.where(input_dict["decision_makers_options"] == "Check (Base case)")[0][0]
    stored = input_dict["decision_makers_option_value"][idx]
    scored = score_allocation(input_dict, stored, "Base case", "Check (Base case)")

    assert scored == pytest.approx(result.appreciation, abs=1e-9)
