"""Confirmatory H1/H2/H5 study driver — runs the pre-registered grid.

The design is frozen in ``PREREGISTRATION.md`` (committed alongside this file;
that commit hash timestamps the lock). ``StudySpec`` defaults ARE the locked
confirmatory grid: {linear, sinusoidal} x k in {2,3,4,6,9,12,15} x 30 seeds x
{grid, slsqp, basin_hopping, genetic_algorithm}, 3 scenarios per case.

Amendment A1 (2026-07-23, see PREREGISTRATION.md): grid at k = 15 never
terminates — ``vlinder``'s ``generate_combinations`` expands each valid budget
split through ``set(permutations(...))``, 15! per split, ~19 days of CPU per
task extrapolated from k = 12. Those cells are written as censored observations
without execution before the run starts, so the resume logic skips them.

Must be a real .py file: Windows spawn cannot serve a stdin ``__main__`` to
worker processes (validated in the Phase-2b dry run).

Run (background, resumable — interrupted runs continue where they stopped):
  & "C:\\Users\\joepw\\.virtualenvs\\tRBS-DclBJWVi-python.exe\\Scripts\\python.exe" `
    experiments/synthetic/run_confirmatory.py --workers 6
"""

import argparse
import json
from datetime import datetime, timezone

from study_harness import StudyHarness, StudySpec

CENSOR_REASON = (
    "amendment A1 (2026-07-23): grid at k=15 censored without execution - "
    "generate_combinations iterates set(permutations(...)) per budget split "
    "(15! ~ 1.31e12); ~19 days CPU per task extrapolated from the measured "
    "k=12 grid task (623 s, 12!); zero k=15 grid completions ever observed"
)


def write_grid_k15_censored_rows(harness):
    """Amendment A1: pre-write censored rows for every pending grid cell at k = 15."""
    spec = harness.spec
    done = harness.completed_keys()
    n_written = 0
    with open(harness.results_path, "a", encoding="utf-8") as out:
        for variant in spec.variants:
            for seed in spec.seeds:
                name, manifest = harness.ensure_case(variant, 15, seed)
                for scenario, cert in manifest["oracle"]["per_scenario"].items():
                    if harness.task_key(name, scenario, "grid") in done:
                        continue
                    row = {
                        "case_name": name,
                        "regime": manifest["regime"],
                        "variant": manifest["appreciation"],
                        "k": 15,
                        "case_seed": seed,
                        "scenario": scenario,
                        "method": "grid",
                        "method_seed": seed,
                        "f_found": None,
                        "f_oracle": float(cert["f_star"]),
                        "gap": None,
                        "recovered": False,
                        "oracle_verified": bool(manifest["oracle"].get("verified", False)),
                        "allocation": None,
                        "spend_fraction": None,
                        "n_function_evals": None,
                        "wall_time_s": None,
                        "censored": True,
                        "censor_reason": CENSOR_REASON,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                    out.write(json.dumps(row) + "\n")
                    n_written += 1
    print(f"[confirmatory] amendment A1: {n_written} grid k=15 cells written as censored")


def main():
    """Run the locked confirmatory grid with checkpoint/resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6, help="parallel worker processes")
    args = parser.parse_args()
    harness = StudyHarness(StudySpec())
    write_grid_k15_censored_rows(harness)
    csv_path = harness.run(n_workers=args.workers)
    print(f"[confirmatory] complete - results at {csv_path}")


if __name__ == "__main__":
    main()
