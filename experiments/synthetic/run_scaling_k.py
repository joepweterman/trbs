"""
Scaling driver: how do the methods behave when k is blown up well past the locked grid?

The locked study stops at k = 15, where the enumerative baseline already fails on
resolution. This exploratory follow-up (review round of 8 September 2026) takes the
same generator and the same certified oracles to k in {15, 20, 30, 50, 75}, on a
thinner seed slice, and reads every method at its locked configuration. Two grids
are included. The packaged grid cannot run here as implemented: its permutation
expansion is infeasible above k = 12 (amendment A1), so the budget-face grid is run
through the refining lattice under the same 60,000-evaluation ceiling. The capped
grid enumerates directly, as in A4. The genetic algorithm is run twice: at the
locked budget, which the locked study showed to be exhausted from k = 6, and left to
run until its best individual has stalled for 100 generations, which is the budget
rule the review asked for.

Cases live in a root of their own (generated/scaling), so the locked study folder is
not touched. The linear oracle fails its affinity assertion at k = 100 (the
scenario's vertex enumeration and analytic gradient disagree), so 75 is the largest
dimension here.

Run:
  python experiments/synthetic/run_scaling_k.py --dry-run
  python experiments/synthetic/run_scaling_k.py --smoke
  python experiments/synthetic/run_scaling_k.py --workers 4
"""

import argparse
from pathlib import Path

from study_harness import METHOD_DEFAULTS, StudySpec
from sweep_harness import SweepHarness, config_label

ROOT = Path(__file__).resolve().parent / "generated" / "scaling"
KS = (15, 20, 30, 50, 75)
SEEDS = tuple(range(5))

MDBH_LOCKED = {"n_starts": 5, "n_hops": 10, "n_local_steps": 50, "eta": 1.0, "sigma": 1.5, "temperature": 1.0}
GA_STALLED = {"population_size": 50, "n_generations": 4000, "stall_generations": 100}

#: The locked roster ran on the capped simplex and without a clock, before the case-level
#: defaults became the budget face and a 60-second limit (28 August review round), so every
#: continuous configuration pins both; the grids pin the clock off so their ceilings rule.
NO_CLOCK = {"max_calculation_time": None}
ROSTER = {"spend_all": False, **NO_CLOCK}


def config(method, kwargs):
    """One (label, method, kwargs) entry; the label omits the pinned roster settings."""
    return (config_label(method, kwargs), method, {**kwargs, **ROSTER})


#: Every configuration read on every cell. The capped grid pins ``spend_all=False`` explicitly:
#: the case-level default (True) would override the A4 shim's default and send it through the
#: packaged grid's permutation expansion, which never finishes at k = 15.
CONFIGS = [
    ("grid (face, refining)", "grid", {"max_function_evals": 60000, "spend_all": True, **NO_CLOCK}),
    (
        config_label("grid_capped", {"max_combinations": 60000}),
        "grid_capped",
        {"max_combinations": 60000, **ROSTER},
    ),
    config("slsqp", METHOD_DEFAULTS["slsqp"]),
    config("basin_hopping", METHOD_DEFAULTS["basin_hopping"]),
    config("genetic_algorithm", METHOD_DEFAULTS["genetic_algorithm"]),
    config("genetic_algorithm", GA_STALLED),
    config("mdbh", MDBH_LOCKED),
]


def harness():
    """The scaling harness: both concave variants, large k, five seeds, its own root."""
    spec = StudySpec(ks=KS, seeds=SEEDS, methods=tuple(dict.fromkeys(m for _, m, _ in CONFIGS)), root=str(ROOT))
    return SweepHarness(spec, CONFIGS, "scaling")


def main():
    """Run, dry-run or smoke-test the scaling study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="parallel worker processes")
    parser.add_argument("--dry-run", action="store_true", help="count the pending tasks and stop")
    parser.add_argument("--smoke", action="store_true", help="time one cell per configuration and stop")
    parser.add_argument("--smoke-k", type=int, default=None, help="the k to smoke-test at (default: the largest)")
    args = parser.parse_args()

    study = harness()
    if args.dry_run:
        print(f"[scaling] {len(study.build_tasks())} tasks pending")
        return
    if args.smoke:
        study.smoke(args.smoke_k)
        return
    print(f"[scaling] complete - results at {study.run(n_workers=args.workers)}")


if __name__ == "__main__":
    main()
