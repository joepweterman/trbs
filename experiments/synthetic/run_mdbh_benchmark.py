"""MDBH benchmark driver: the proposal's Step-3 method on the locked grid.

Amendment A3 (2026-08-05, see PREREGISTRATION.md) freezes this design: the
locked 420-case convex grid, all 3 scenarios, method "mdbh" only, with the
calibrated package defaults pinned explicitly below. Results go to a separate
store (results_mdbh.jsonl), so the locked confirmatory results.jsonl is never
appended to. The existing generated cases and their oracle certificates are
reused unchanged.

Must be a real .py file: Windows spawn cannot serve a stdin ``__main__`` to
worker processes (validated in the Phase-2b dry run).

Run (background, resumable, interrupted runs continue where they stopped):
  python experiments/synthetic/run_mdbh_benchmark.py --workers 6
"""

import argparse

from study_harness import METHOD_DEFAULTS, StudyHarness, StudySpec

# Amendment A3 frozen configuration. Injected here rather than edited into
# study_harness.py, which is part of the locked instrument. The values equal
# the package defaults calibrated in the pre-amendment acceptance runs; they
# are pinned explicitly so this file, not the package, defines the benchmark.
METHOD_DEFAULTS["mdbh"] = {
    "n_starts": 5,
    "n_hops": 10,
    "n_local_steps": 50,
    "eta": 1.0,
    "sigma": 1.5,
    "temperature": 1.0,
}


class MdbhHarness(StudyHarness):
    """StudyHarness writing to the A3 stores, next to (never into) the locked ones."""

    def __init__(self, spec):
        super().__init__(spec)
        self.results_path = self.root / "results_mdbh.jsonl"
        self.meta_path = self.root / "meta_mdbh.json"


def main():
    """Run the A3 MDBH grid with checkpoint/resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6, help="parallel worker processes")
    parser.add_argument("--dry-run", action="store_true", help="list the pending tasks without running anything")
    args = parser.parse_args()

    harness = MdbhHarness(StudySpec(methods=("mdbh",)))
    if args.dry_run:
        tasks = harness.build_tasks()
        print(f"[mdbh] {len(tasks)} tasks pending; kwargs = {METHOD_DEFAULTS['mdbh']}")
        return
    results_path = harness.run(n_workers=args.workers)
    print(f"[mdbh] complete - results at {results_path}")


if __name__ == "__main__":
    main()
