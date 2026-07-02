"""Tests for the synthetic case factory + oracle (thesis-side, Phase 0.75/1).

Run:
  & "C:\\Users\\joepw\\.virtualenvs\\tRBS-DclBJWVi-python.exe\\Scripts\\python.exe" `
    -m pytest C:\\Users\\joepw\\tRBS\\experiments\\synthetic -q
"""

# pylint: disable=protected-access
# (oracle._build is the shared import-and-prepare helper under test)

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from case_factory import (
    SyntheticCaseFactory,
    SyntheticCaseParams,
    read_manifest,
    validate_case,
)
from oracle import certify_case, _build, ORACLE_DMO


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
    sim, opt = _build(params.name, tmp_path)
    scenario = str(sim.input_dict["scenarios"][0])
    with contextlib.redirect_stdout(io.StringIO()):
        slsqp = opt.optimize_slsqp(scenario, params.budget, dmo_name=ORACLE_DMO, n_starts=20, seed=1)
    assert abs(oracle["per_scenario"][scenario]["f_star"] - float(slsqp.appreciation)) <= 1e-6


def test_oracle_curved_kkt(tmp_path):
    """Curved oracle: tiny KKT residual and production SLSQP matches f*."""
    params = _params(name="T_curved_k3", appreciation="sinusoidal", seed=5)
    SyntheticCaseFactory(params).write(tmp_path)
    oracle = certify_case(params.name, tmp_path)
    assert oracle["method"] == "tight_slsqp_kkt"
    for scen in oracle["per_scenario"].values():
        assert scen["certificate"]["kkt_residual"] <= 1e-2

    sim, opt = _build(params.name, tmp_path)
    scenario = str(sim.input_dict["scenarios"][0])
    with contextlib.redirect_stdout(io.StringIO()):
        slsqp = opt.optimize_slsqp(scenario, params.budget, dmo_name=ORACLE_DMO, n_starts=20, seed=1)
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


def test_unbuilt_regimes_refuse():
    """Unbuilt regimes and unknown appreciation values are rejected loudly."""
    with pytest.raises(NotImplementedError):
        SyntheticCaseParams(regime="smooth_nonconvex")
    with pytest.raises(ValueError):
        SyntheticCaseParams(appreciation="cubic")
