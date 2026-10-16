"""
Genetic-algorithm budget driver: is the GA's collapse with k a budget artefact?

The locked study reads the GA at 50 individuals for 60 generations, 3,050
evaluations whatever the dimension, and its recovery falls from 1.000 at k = 2 to
0.000 at k = 15. The timed comparison on the packaged cases, where the clock rather
than the generation count ends the run, shows the same algorithm within 1e-4 of the
best known value. This exploratory follow-up (review round of 8 September 2026)
separates the two readings on the locked cases: the same GA at 60, 240 and 1,000
generations, and once more left to run until its best individual has stalled for
100 generations, which makes the budget a function of the case instead of a
constant. Every run reports its evaluations, so the recovery gained can be priced.

The locked cases and their certificates are reused unchanged from generated/study;
results go to results_ga_budget.jsonl next to, never inside, the locked stores.

Run:
  python experiments/synthetic/run_ga_budget.py --dry-run
  python experiments/synthetic/run_ga_budget.py --smoke
  python experiments/synthetic/run_ga_budget.py --workers 6
"""

import argparse

from study_harness import StudySpec
from sweep_harness import SweepHarness, config_label

SEEDS = tuple(range(10))

#: The GA along its budget: three fixed generation counts and the stall rule.
GA_CONFIGS = [
    {"population_size": 50, "n_generations": 60},
    {"population_size": 50, "n_generations": 240},
    {"population_size": 50, "n_generations": 1000},
    {"population_size": 50, "n_generations": 4000, "stall_generations": 100},
]
#: The locked roster ran on the capped simplex and without a clock, before the case-level
#: defaults became the budget face and a 60-second limit (28 August review round); pinned
#: here so the 60-generation row reproduces the locked rows exactly.
ROSTER = {"spend_all": False, "max_calculation_time": None}
CONFIGS = [
    (config_label("genetic_algorithm", kwargs), "genetic_algorithm", {**kwargs, **ROSTER}) for kwargs in GA_CONFIGS
]


def harness():
    """The GA budget harness: the locked grid, ten seeds, the locked root."""
    return SweepHarness(StudySpec(seeds=SEEDS, methods=("genetic_algorithm",)), CONFIGS, "ga_budget")


def main():
    """Run, dry-run or smoke-test the GA budget study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6, help="parallel worker processes")
    parser.add_argument("--dry-run", action="store_true", help="count the pending tasks and stop")
    parser.add_argument("--smoke", action="store_true", help="time one cell per configuration and stop")
    args = parser.parse_args()

    study = harness()
    if args.dry_run:
        print(f"[ga-budget] {len(study.build_tasks())} tasks pending")
        return
    if args.smoke:
        study.smoke()
        return
    print(f"[ga-budget] complete - results at {study.run(n_workers=args.workers)}")


if __name__ == "__main__":
    main()
