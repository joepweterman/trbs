"""Tests for the regime-aware harness extension (H3/H4 phases).

Run:
  python -m pytest experiments/synthetic -q
"""

import json

import pytest
from case_factory import read_manifest
from regime_harness import RegimeHarness, RegimeSpec
from study_harness import StudySpec


def test_default_spec_is_the_locked_spec():
    """With no variant knobs the extension must reproduce the locked design.

    This is the guarantee the lock rests on: adding the H3/H4 capability may not
    perturb the grid the confirmatory phases were run on.
    """
    locked, extended = StudySpec(), RegimeSpec()
    assert extended.variants == locked.variants
    assert extended.ks == locked.ks
    assert extended.seeds == locked.seeds
    assert extended.methods == locked.methods
    assert extended.runtime_cap_s == locked.runtime_cap_s
    assert extended.recovery_eps == locked.recovery_eps
    for variant in locked.variants:
        for k in (2, 6, 15):
            assert extended.case_name(variant, k, 7) == locked.case_name(variant, k, 7)
    assert extended.factory_kwargs("linear") == {"appreciation": "linear"}
    assert extended.factory_kwargs("sinusoidal") == {"appreciation": "sinusoidal"}


def test_unregistered_variant_is_refused():
    """A typo in a variant label must fail loudly, not build a convex case."""
    with pytest.raises(KeyError):
        RegimeSpec(variants=("smooth-stb1",), variant_params={"smooth-bil1": {}})


def test_label_survives_the_round_trip():
    """The arm of the design must be recoverable from any results row."""
    spec = RegimeSpec(case_prefix="H34", variants=("smooth-stb1",), variant_params={"smooth-stb1": {}})
    name = spec.case_name("smooth-stb1", 6, 12)
    assert name == "H34_smooth-stb1_k06_s12"
    assert spec.variant_from_case_name(name) == "smooth-stb1"
    with pytest.raises(ValueError):
        spec.variant_from_case_name("Study_linear_k06_s12")


def test_an_underscore_in_a_label_is_refused():
    """Labels travel inside the case name, so they may not contain the
    separator the name is parsed on."""
    with pytest.raises(ValueError):
        RegimeSpec(variants=("smooth_stb1",), variant_params={"smooth_stb1": {}})


@pytest.mark.parametrize(
    "label,knobs,expected_regime",
    [
        (
            "smooth-stb1",
            {"regime": "smooth_nonconvex", "appreciation": "sinusoidal", "n_stb1": 1},
            "smooth_nonconvex",
        ),
        (
            "nonsmooth-clip07",
            {"regime": "nonsmooth", "appreciation": "sinusoidal", "bracketing_factor": 0.7},
            "nonsmooth",
        ),
    ],
)
def test_non_convex_cases_are_built_and_certified(tmp_path, label, knobs, expected_regime):
    """The extension must reach the regimes the locked harness cannot express,
    and their ground truth must still be certified at k <= 6."""
    spec = RegimeSpec(
        variants=(label,),
        ks=(3,),
        seeds=(0,),
        methods=("slsqp",),
        case_prefix="Test",
        variant_params={label: knobs},
        root=str(tmp_path / "regimes"),
    )
    harness = RegimeHarness(spec)
    name, manifest = harness.ensure_case(label, 3, 0)

    assert name == f"Test_{label}_k03_s0"
    assert manifest["regime"] == expected_regime
    oracle = manifest["oracle"]
    assert oracle["method"] == "dense_grid_polish"
    assert oracle["verified"] is True, "at k <= 6 the ground truth must be certified, not a proxy"


def test_method_budget_override_applies_only_where_asked(tmp_path):
    """H3 is stated on single-start SLSQP, so the frozen budget must be
    overridable, and only for the method the spec names."""
    label = "smooth-stb1"
    spec = RegimeSpec(
        variants=(label,),
        ks=(2,),
        seeds=(0,),
        methods=("slsqp", "basin_hopping"),
        case_prefix="Test",
        variant_params={label: {"regime": "smooth_nonconvex", "appreciation": "sinusoidal", "n_stb1": 1}},
        method_overrides={"slsqp": {"n_starts": 1}},
        root=str(tmp_path / "override"),
    )
    harness = RegimeHarness(spec)
    tasks = harness.build_tasks()

    slsqp = [t for t in tasks if t["method"] == "slsqp"]
    hopping = [t for t in tasks if t["method"] == "basin_hopping"]
    assert slsqp and hopping
    assert all(t["method_kwargs"]["n_starts"] == 1 for t in slsqp), "the override did not reach SLSQP"
    assert all(t["method_kwargs"] == {"n_hops": 25, "n_starts": 1} for t in hopping), "an untouched method changed"

    from study_harness import METHOD_DEFAULTS  # pylint: disable=import-outside-toplevel

    assert METHOD_DEFAULTS["slsqp"] == {"n_starts": 100}, "the frozen defaults were mutated in place"


def test_rows_reach_the_arm_they_came_from(tmp_path):
    """A full run must produce rows in the non-convex regime whose design arm
    is recoverable, using the locked results schema unchanged."""
    label = "smooth-bil1"
    spec = RegimeSpec(
        variants=(label,),
        ks=(2,),
        seeds=(0,),
        methods=("slsqp",),
        case_prefix="Test",
        variant_params={label: {"regime": "smooth_nonconvex", "appreciation": "sinusoidal", "n_bilinear": 1}},
        root=str(tmp_path / "labelled"),
    )
    harness = RegimeHarness(spec)
    harness.run(n_workers=1)

    with open(harness.results_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert rows, "the run produced no rows"
    assert {r["regime"] for r in rows} == {"smooth_nonconvex"}
    assert {spec.variant_from_case_name(r["case_name"]) for r in rows} == {label}
    assert read_manifest(rows[0]["case_name"], spec.root)["params"]["n_bilinear"] == 1

    meta = json.loads((spec_root := harness.meta_path).read_text(encoding="utf-8"))
    assert meta["spec"]["variant_params"] == spec.variant_params, f"{spec_root} must record the design"
