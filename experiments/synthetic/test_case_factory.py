"""Tests for the synthetic case factory + oracle (thesis-side, Phase 0.75/1).

Run:
  python -m pytest experiments/synthetic -q
"""

import numpy as np
import pandas as pd
import pytest

from case_factory import (
    SyntheticCaseFactory,
    SyntheticCaseParams,
    build_case,
    read_manifest,
    validate_case,
)
from oracle import Oracle, certify_case, ORACLE_DMO

from vlinder.utils import suppress_print


def _params(**overrides):
    defaults = {"name": "T_case", "k": 3, "n_key_outputs": 3, "seed": 11}
    defaults.update(overrides)
    return SyntheticCaseParams(**defaults)


def test_roundtrip_and_validator_green(tmp_path):
    """A freshly generated case imports and passes every validator invariant."""
    params = _params()
    SyntheticCaseFactory(params).write(tmp_path)
    report = validate_case(params.name, tmp_path, budget=params.budget, n_samples=60)
    assert report["imported"] and report["all_finite"] and report["no_clip"]
    assert report["non_degenerate"] and report["slsqp_feasible"] and report["slsqp_consensus"]


def test_determinism_byte_identical(tmp_path):
    """The same params + seed reproduce byte-identical CSVs and hashes."""
    params = _params()
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    SyntheticCaseFactory(params).write(root_a)
    SyntheticCaseFactory(params).write(root_b)
    for csv_a in sorted((root_a / params.name / "csv").glob("*.csv")):
        csv_b = root_b / params.name / "csv" / csv_a.name
        assert csv_a.read_bytes() == csv_b.read_bytes(), f"{csv_a.name} not reproducible from the seed"
    assert read_manifest(params.name, root_a)["table_sha256"] == read_manifest(params.name, root_b)["table_sha256"]


def test_subseed_isolation():
    """Ceteris paribus: varying one knob must not re-randomise unrelated components."""
    base = SyntheticCaseFactory(_params(n_scenarios=3))
    more_scenarios = SyntheticCaseFactory(_params(n_scenarios=5))
    np.testing.assert_array_equal(base.coef, more_scenarios.coef)
    np.testing.assert_array_equal(base.ko_weights, more_scenarios.ko_weights)
    np.testing.assert_array_equal(base.theme_weights, more_scenarios.theme_weights)


def test_validator_catches_under_bracketing(tmp_path):
    """Recreate the Phase-0 flaw (exp03): DMOs that under-bracket the feasible
    envelope must fail the NO-CLIP invariant."""
    params = _params()
    SyntheticCaseFactory(params).write(tmp_path)
    dmo_path = tmp_path / params.name / "csv" / "decision_makers_options.csv"
    dmo = pd.read_csv(dmo_path, sep=";")
    # Shrink every DMO toward the equal spread: corners no longer reach the
    # feasible extremes and zero spend no longer reaches the lower envelope.
    dmo["value"] = 0.6 * dmo["value"] + 0.4 * (params.budget / params.k)
    dmo.to_csv(dmo_path, sep=";", index=False)
    with pytest.raises(AssertionError, match="auto-boundaries"):
        validate_case(params.name, tmp_path, budget=params.budget, n_samples=20)


def test_oracle_linear_agreement(tmp_path):
    """Vertex oracle: one corner wins everywhere and production SLSQP recovers it."""
    params = _params(name="T_linear_k4", k=4, n_key_outputs=3, seed=5)
    SyntheticCaseFactory(params).write(tmp_path)
    oracle = certify_case(params.name, tmp_path)
    assert oracle["method"] == "vertex_enumeration"
    assert read_manifest(params.name, tmp_path)["oracle"] is not None

    # The analytic gradient is scenario-independent => one vertex wins everywhere.
    vertex_indices = {s["certificate"]["vertex_index"] for s in oracle["per_scenario"].values()}
    assert len(vertex_indices) == 1

    # Production SLSQP recovers the certified optimum.
    sim, opt = build_case(params.name, tmp_path, ORACLE_DMO)
    scenario = str(sim.input_dict["scenarios"][0])
    slsqp = suppress_print(opt.optimize_slsqp)(scenario, params.budget, dmo_name=ORACLE_DMO, n_starts=20, seed=1)
    assert abs(oracle["per_scenario"][scenario]["f_star"] - float(slsqp.appreciation)) <= 1e-6


def test_oracle_curved_kkt(tmp_path):
    """Curved oracle: tiny KKT residual and production SLSQP matches f*."""
    params = _params(name="T_curved_k3", appreciation="sinusoidal", seed=5)
    SyntheticCaseFactory(params).write(tmp_path)
    oracle = certify_case(params.name, tmp_path)
    assert oracle["method"] == "tight_slsqp_kkt"
    for scen in oracle["per_scenario"].values():
        assert scen["certificate"]["kkt_residual"] <= 1e-2

    sim, opt = build_case(params.name, tmp_path, ORACLE_DMO)
    scenario = str(sim.input_dict["scenarios"][0])
    slsqp = suppress_print(opt.optimize_slsqp)(scenario, params.budget, dmo_name=ORACLE_DMO, n_starts=20, seed=1)
    assert abs(oracle["per_scenario"][scenario]["f_star"] - float(slsqp.appreciation)) <= 1e-3


@pytest.mark.parametrize("appreciation", ["linear", "sinusoidal"])
def test_k1_edge(tmp_path, appreciation):
    """Degenerate single-lever cases stay valid in both variants."""
    params = _params(name=f"T_k1_{appreciation}", k=1, n_key_outputs=2, appreciation=appreciation, seed=2)
    SyntheticCaseFactory(params).write(tmp_path)
    report = validate_case(params.name, tmp_path, budget=params.budget, n_samples=20)
    assert report["no_clip"] and report["slsqp_feasible"]


def test_k15_smoke(tmp_path):
    """Largest planned k imports, brackets exactly and reaches SLSQP consensus."""
    params = _params(name="T_k15", k=15, n_key_outputs=5, seed=8)
    SyntheticCaseFactory(params).write(tmp_path)
    report = validate_case(params.name, tmp_path, budget=params.budget, n_samples=40)
    assert report["no_clip"] and report["slsqp_consensus"]


def test_knob_validation_matrix():
    """Unknown values and regime/mechanism inconsistencies are rejected loudly."""
    with pytest.raises(ValueError):
        SyntheticCaseParams(appreciation="cubic")
    with pytest.raises(ValueError):
        SyntheticCaseParams(regime="chaotic")
    with pytest.raises(ValueError):  # convex must keep all mechanisms neutral
        SyntheticCaseParams(regime="convex", n_stb1=1)
    with pytest.raises(ValueError):  # smooth needs at least one curvature carrier
        SyntheticCaseParams(regime="smooth_nonconvex")
    with pytest.raises(ValueError):  # smooth must stay smooth
        SyntheticCaseParams(regime="smooth_nonconvex", n_stb1=1, n_saturation=1)
    with pytest.raises(ValueError):  # nonsmooth needs a nonsmooth mechanism
        SyntheticCaseParams(regime="nonsmooth")
    with pytest.raises(ValueError):  # bilinear is a smooth mechanism (exclusivity)
        SyntheticCaseParams(regime="nonsmooth", bracketing_factor=0.7, n_bilinear=1)
    with pytest.raises(ValueError):  # a bilinear term is a lever pair
        SyntheticCaseParams(regime="smooth_nonconvex", n_bilinear=1, k=1, n_key_outputs=2)


def test_derived_name_separates_cases_that_differ():
    """A case name left empty is derived from the knobs that make the case what
    it is, so two differently-configured cases cannot land in one folder."""
    configs = [
        SyntheticCaseParams(k=3, seed=1),
        SyntheticCaseParams(k=3, seed=2),
        SyntheticCaseParams(k=6, seed=1),
        SyntheticCaseParams(k=3, seed=1, appreciation="sinusoidal"),
        SyntheticCaseParams(k=3, seed=1, scenario_dispersion=0.6),
        SyntheticCaseParams(k=3, seed=1, regime="nonsmooth", bracketing_factor=0.7),
        SyntheticCaseParams(k=3, seed=1, regime="nonsmooth", n_saturation=2),
        SyntheticCaseParams(k=3, seed=1, regime="smooth_nonconvex", appreciation="sinusoidal", n_stb1=1),
        SyntheticCaseParams(k=3, seed=1, regime="smooth_nonconvex", appreciation="sinusoidal", n_bilinear=1),
    ]
    names = [p.name for p in configs]
    assert len(set(names)) == len(names), f"derived names collide: {names}"
    assert all(n.startswith("Synthetic_") for n in names)
    # The same configuration must always derive the same name, or a case could
    # not be found again after it was written.
    assert SyntheticCaseParams(k=3, seed=1).name == SyntheticCaseParams(k=3, seed=1).name


def test_explicit_name_overrides_the_derived_one():
    """Passing a name wins, which is what the curated suite and the fixtures rely on."""
    assert SyntheticCaseParams(k=3, seed=1, name="My_case").name == "My_case"


def test_smooth_stb_only_keeps_exact_bracketing(tmp_path):
    """STB=1 flips change appreciation, not KO envelopes: NO-CLIP must hold and
    the plain-affine independent cross-check still applies."""
    params = _params(name="T_smooth_stb", regime="smooth_nonconvex", n_stb1=2, seed=3)
    SyntheticCaseFactory(params).write(tmp_path)
    report = validate_case(params.name, tmp_path, budget=params.budget, n_samples=40)
    assert report["regime"] == "smooth_nonconvex" and report["no_clip"]
    assert "n_distinct_terminal_values" in report  # multimodality is reported, not asserted


def test_smooth_bilinear_blend_dmo_brackets(tmp_path):
    """The 'Blend' DMO realises the bilinear KO's edge max: envelope stays exact."""
    params = _params(name="T_smooth_bil", regime="smooth_nonconvex", n_bilinear=1, seed=6)
    factory = SyntheticCaseFactory(params)
    factory.write(tmp_path)
    dmos = set(factory.tables()["decision_makers_options"]["decision_makers_option"])
    assert "Blend 01" in dmos
    report = validate_case(params.name, tmp_path, budget=params.budget, n_samples=40)
    assert report["no_clip"]


def test_nonsmooth_clip_is_intentional_and_measured(tmp_path):
    """bracketing_factor < 1 must under-bracket (the exp03 mechanism as a dial)."""
    params = _params(name="T_clip", regime="nonsmooth", bracketing_factor=0.7, seed=5)
    SyntheticCaseFactory(params).write(tmp_path)
    report = validate_case(params.name, tmp_path, budget=params.budget, n_samples=60)
    assert not report["no_clip"]
    assert report["underbracketing_margin"] > 0
    assert report["clipping_incidence"] > 0


def test_nonsmooth_saturation_kinks_without_clipping(tmp_path):
    """min-cap chains put kinks in the dependencies; the greedy 'Envelope' DMOs
    keep the auto-boundaries exactly on the feasible envelope (no clipping)."""
    params = _params(name="T_sat", regime="nonsmooth", n_saturation=2, saturation_point=0.4, seed=6)
    factory = SyntheticCaseFactory(params)
    factory.write(tmp_path)
    dmos = set(factory.tables()["decision_makers_options"]["decision_makers_option"])
    assert any(dmo_name.startswith("Envelope") for dmo_name in dmos)
    report = validate_case(params.name, tmp_path, budget=params.budget, n_samples=40)
    assert report["no_clip"]


def test_dispersion_moves_scenario_optima(tmp_path):
    """scenario_dispersion > 0 must make per-scenario optima diverge while the
    per-scenario convex geometry (and NO-CLIP) is preserved."""
    params = _params(name="T_disp", scenario_dispersion=0.6, seed=7)
    SyntheticCaseFactory(params).write(tmp_path)
    report = validate_case(params.name, tmp_path, budget=params.budget, n_samples=40)
    assert report["no_clip"] and report["slsqp_consensus"]
    oracle = certify_case(params.name, tmp_path)
    assert oracle["method"] == "vertex_enumeration" and oracle["verified"]
    assert oracle["scenario_optimum_dispersion"] > 0.5


def test_grid_oracle_verifies_nonconvex_low_k(tmp_path):
    """Dense-grid + polish ground truth for a nonconvex case at low k."""
    params = _params(
        name="T_grid",
        regime="smooth_nonconvex",
        appreciation="sinusoidal",
        n_stb1=1,
        n_bilinear=1,
        n_scenarios=1,
        seed=4,
    )
    SyntheticCaseFactory(params).write(tmp_path)
    oracle = certify_case(params.name, tmp_path)
    assert oracle["method"] == "dense_grid_polish" and oracle["verified"]
    scen = next(iter(oracle["per_scenario"].values()))
    cert = scen["certificate"]
    assert cert["n_grid_nodes"] > 500 and cert["polish_gain"] >= 0
    # the certified optimum must not be beaten by production SLSQP
    sim, opt = build_case(params.name, tmp_path, ORACLE_DMO)
    scenario = str(sim.input_dict["scenarios"][0])
    slsqp = suppress_print(opt.optimize_slsqp)(scenario, params.budget, dmo_name=ORACLE_DMO, n_starts=10, seed=1)
    assert scen["f_star"] >= float(slsqp.appreciation) - 1e-6


def test_multistart_proxy_high_k_is_labelled(tmp_path):
    """Above the grid limit the ground truth is an explicitly-unverified proxy."""
    params = _params(name="T_proxy", k=7, n_key_outputs=2, regime="smooth_nonconvex", n_stb1=1, n_scenarios=1, seed=2)
    SyntheticCaseFactory(params).write(tmp_path)
    oracle = Oracle(params.name, tmp_path).solve_multistart_proxy(n_starts=4)
    assert oracle["method"] == "best_of_multistart_proxy"
    assert oracle["verified"] is False
    scen = next(iter(oracle["per_scenario"].values()))
    assert "PROXY ONLY" in scen["certificate"]["basis"]
