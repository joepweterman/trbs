"""Tests for the study harness (thesis-side, plan Phase 2b).

Run:
  python -m pytest experiments/synthetic -q
"""

import json

from study_harness import StudyHarness, StudySpec


def test_small_grid_reproduces_acceptance_gaps_and_resumes(tmp_path):
    """A tiny grid must reproduce the Phase-1 acceptance gaps (0.0 on linear,
    <= ~1e-4 on curved) and find nothing left to do on a second pass."""
    spec = StudySpec(
        variants=("linear", "sinusoidal"),
        ks=(2, 3),
        seeds=(0, 1),
        methods=("slsqp",),
        root=str(tmp_path / "dryrun"),
    )
    harness = StudyHarness(spec)
    harness.run(n_workers=1)

    with open(harness.results_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 24, f"expected 24 rows (8 cases x 3 scenarios x 1 method), got {len(rows)}"

    linear = [r["gap"] for r in rows if r["variant"] == "linear"]
    curved = [r["gap"] for r in rows if r["variant"] == "sinusoidal"]
    assert max(abs(g) for g in linear) <= 1e-6, "linear regime must reproduce gap 0.0 (Phase-1 acceptance)"
    assert max(abs(g) for g in curved) <= 1e-3, "curved regime gaps must stay at Phase-1 acceptance level"

    assert not harness.build_tasks(), "resume failed: tasks re-queued after completion"


def test_method_defaults_are_frozen():
    """The per-method budgets are pre-registered, so a silent edit must fail."""
    assert StudySpec().methods == ("grid", "slsqp", "basin_hopping", "genetic_algorithm")
    from study_harness import METHOD_DEFAULTS  # pylint: disable=import-outside-toplevel

    assert METHOD_DEFAULTS == {
        "grid": {"max_combinations": 60000},
        "slsqp": {"n_starts": 100},
        "basin_hopping": {"n_hops": 25, "n_starts": 1},
        "genetic_algorithm": {"population_size": 50, "n_generations": 60},
    }
