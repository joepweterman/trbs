"""Confirmatory H1/H2/H5 study driver — runs the pre-registered grid.

The design is frozen in ``PREREGISTRATION.md`` (committed alongside this file;
that commit hash timestamps the lock). ``StudySpec`` defaults ARE the locked
confirmatory grid: {linear, sinusoidal} x k in {2,3,4,6,9,12,15} x 30 seeds x
{grid, slsqp, basin_hopping, genetic_algorithm}, 3 scenarios per case.

Must be a real .py file: Windows spawn cannot serve a stdin ``__main__`` to
worker processes (validated in the Phase-2b dry run).

Run (background, resumable — interrupted runs continue where they stopped):
  & "C:\\Users\\joepw\\.virtualenvs\\tRBS-DclBJWVi-python.exe\\Scripts\\python.exe" `
    experiments/synthetic/run_confirmatory.py --workers 6
"""

import argparse

from study_harness import StudyHarness, StudySpec


def main():
    """Run the locked confirmatory grid with checkpoint/resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6, help="parallel worker processes")
    args = parser.parse_args()
    csv_path = StudyHarness(StudySpec()).run(n_workers=args.workers)
    print(f"[confirmatory] complete - results at {csv_path}")


if __name__ == "__main__":
    main()
