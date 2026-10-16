# pylint: disable=R0913,R0917  # a sweep cell is described by case, limit, method, seed
"""
This file compares every optimizer on the packaged cases under equal budgets: what
does a user actually get within a given number of seconds, or, in the black-box
convention, within a given number of objective evaluations?

Each method is given the same budget through the two parameters every solver
understands: ``max_calculation_time`` (the seconds axis) or ``max_function_evals``
(the evaluations axis). Grid search evaluates ever finer lattices under either,
halving the step each round, and the continuous methods keep adding units of work
(a start, a hop, a generation) until the budget is spent. Their unit knobs are set
far beyond reach so that the budget, not the knob, is what stops them.

The seconds axis runs from a quarter of a second to two minutes. The first sweep
(15 to 120 seconds) showed every curve flat from its first limit onwards, so the
sub-15-second limits are where the methods actually separate. The evaluations
axis is the machine-independent reading of the same question: it charges every
method for what it asks of the objective and nothing else.

Every method runs in both budget modes: spending at most the budget (the capped
simplex, the continuous default) and spending it exactly (the budget face, the
grid default). The stochastic methods run on several seeds and are read at the
median, because a single seed can flatter or wrong-foot a method; the grids are
deterministic and run once.

Reading the result: the interesting question is not who wins at the largest
budget but where each curve flattens. A method whose appreciation stops
improving has found what it is going to find, and the remaining budget is wasted.

Rows are appended to the CSV of the axis as they are produced, so an interrupted
run keeps everything it has already measured, and a rerun with new limits adds
only the missing cells. ``figure_time_limited.py`` and ``table_time_limited.py``
draw and tabulate either axis.

Run:
  python experiments/time_limited_comparison.py                      # seconds axis
  python experiments/time_limited_comparison.py --axis evals         # evaluations axis
  python experiments/time_limited_comparison.py --cases Beerwiser --limits 0.5 1
  python experiments/time_limited_comparison.py --methods slsqp "slsqp (spend all)"
"""

import argparse
import contextlib
import io
import os
from pathlib import Path

import pandas as pd
import vlinder as vl

from vlinder.trbs import TheResponsibleBusinessSimulator

DATA = Path(os.path.dirname(vl.__file__)) / "data"
OUT = Path(__file__).resolve().parent / "out" / "time_limited"
CASES = ("Beerwiser", "Refugee", "IZZ")
SEEDS = (1, 2, 3, 4, 5)

#: Per axis: the solver parameter that carries the budget, the CSV column that records
#: it, the default sweep, the CSV the rows go to, and settings pinned on every run. The
#: seconds CSV is the one the first sweep wrote (15 to 120 s); the sub-15 s limits are
#: appended to it. The evaluations axis switches the case-level 60-second default off, so
#: that only the evaluation budget ends a run.
AXES = {
    "seconds": {
        "parameter": "max_calculation_time",
        "column": "limit_s",
        "limits": (0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120),
        "csv": OUT / "time_limited_comparison_v2.csv",
        "pinned": {},
    },
    "evals": {
        "parameter": "max_function_evals",
        "column": "limit_evals",
        "limits": (30, 100, 300, 1000, 3000, 10000, 30000),
        "csv": OUT / "eval_limited_comparison.csv",
        "pinned": {"max_calculation_time": None},
    },
}

#: Per continuous method, unit knobs set far beyond what any budget here buys, so that
#: the budget is what ends the run. The grids need nothing: they refine until it is spent.
CEILINGS = {
    "grid": {},
    "slsqp": {"n_starts": 10_000},
    "basin_hopping": {"n_starts": 1, "n_hops": 1_000_000},
    "genetic_algorithm": {"population_size": 50, "n_generations": 1_000_000},
    "mdbh": {"n_starts": 10_000, "n_hops": 10, "n_local_steps": 50},
}

#: Every method in both budget modes; the grid carries the same ``spend_all`` switch as the
#: continuous solvers, so its two rows are one method in two modes.
VARIANTS = [("grid", "grid", {"spend_all": True}), ("grid (capped)", "grid", {"spend_all": False})]
for _method in ("slsqp", "basin_hopping", "genetic_algorithm", "mdbh"):
    # spend_all is explicit in both directions: the case-level default is True now.
    VARIANTS.append((_method, _method, {"spend_all": False}))
    VARIANTS.append((f"{_method} (spend all)", _method, {"spend_all": True}))


def parse_limit(value):
    """
    This function reads one budget from the command line, as an int where it is whole
    so that the CSV keys of the first sweep (``15``, not ``15.0``) still match.
    :param value: the command-line string
    :return: an int or a float
    """
    number = float(value)
    return int(number) if number.is_integer() else number


class BudgetedComparison:
    """One full sweep over cases, budgets on one axis, method variants and seeds."""

    def __init__(self, cases, limits, labels, seeds, out_path, axis="seconds"):
        self.cases = list(cases)
        self.limits = [parse_limit(limit) for limit in limits]
        self.variants = [variant for variant in VARIANTS if variant[0] in set(labels)]
        self.seeds = [int(seed) for seed in seeds]
        self.out_path = Path(out_path)
        self.axis = AXES[axis]

    @staticmethod
    def build(case_name):
        """
        This function builds one bundled case through the unmodified pipeline.

        A fresh build per run is required: the grid path freezes the appreciation
        boundaries on the dictionary it runs on, so reusing an instance across methods
        would change the objective later methods are scored on.
        :param case_name: the bundled case name
        :return: a built, evaluated and appreciated simulator
        """
        sim = TheResponsibleBusinessSimulator(case_name, file_path=DATA, file_extension="csv")
        with contextlib.redirect_stdout(io.StringIO()):
            sim.build()
            sim.evaluate()
            sim.appreciate()
        return sim

    @classmethod
    def run_once(cls, case_name, scenario, method, seed, kwargs):
        """
        This function runs one method once and reports what it achieved.
        :param case_name: the bundled case name
        :param scenario: the scenario to solve
        :param method: the method name
        :param seed: the RNG seed for this run
        :param kwargs: the keyword arguments for this run
        :return: the OptimizationResult
        """
        sim = cls.build(case_name)
        with contextlib.redirect_stdout(io.StringIO()):
            sim.optimize(scenario, method=method, dmo_name="Timed", seed=seed, **kwargs)
        return sim.optimization_result

    def append_record(self, record):
        """
        This function appends one finished run to the CSV, so an interrupted sweep
        keeps everything it has already measured.
        :param record: the row to append
        :return: None
        """
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([record])
        frame.to_csv(self.out_path, mode="a", header=not self.out_path.exists(), index=False)

    def finished(self):
        """The (case, limit, label, seed) keys already in the CSV, so a rerun resumes."""
        if not self.out_path.exists():
            return set()
        frame = pd.read_csv(self.out_path)
        return set(zip(frame["case"], frame[self.axis["column"]], frame["label"], frame["seed"]))

    def run(self):
        """Run the sweep, appending each row as it lands, and print the median pivot."""
        finished = self.finished()
        for case_name in self.cases:
            scenario = list(self.build(case_name).input_dict["scenarios"])[0]
            for limit in self.limits:
                for label, method, extra_kwargs in self.variants:
                    kwargs = {**CEILINGS[method], **extra_kwargs, **self.axis["pinned"], self.axis["parameter"]: limit}
                    # The grids are deterministic: one run tells the story.
                    seeds = [0] if method == "grid" else self.seeds
                    for seed in seeds:
                        if (case_name, limit, label, seed) in finished:
                            continue
                        result = self.run_once(case_name, scenario, method, seed, kwargs)
                        self.append_record(
                            {
                                "case": case_name,
                                "scenario": scenario,
                                self.axis["column"]: limit,
                                "label": label,
                                "method": method,
                                "spend_all": bool(extra_kwargs.get("spend_all", False)),
                                "seed": seed,
                                "appreciation": result.appreciation,
                                "seconds": result.calculation_time,
                                "evaluations": result.n_function_evals,
                                "budget_spent": result.budget_spent,
                                "reported_dmo": result.dmo_name,
                                "settings": dict(kwargs),
                            }
                        )
                        print(
                            f"{case_name:<10} {limit:>7g} {self.axis['column'][6:]:<5} {label:<28} seed {seed}  "
                            f"appr {result.appreciation:>10.6f}  used {result.calculation_time:>7.2f}s  "
                            f"evals {str(result.n_function_evals):>9}",
                            flush=True,
                        )
        self.report()

    def report(self):
        """Print the median appreciation per case, limit and variant from the CSV."""
        frame = pd.read_csv(self.out_path)
        print("\nMedian appreciation per case and limit:")
        for case_name, group in frame.groupby("case"):
            pivot = group.pivot_table(
                index="label", columns=self.axis["column"], values="appreciation", aggfunc="median"
            )
            print(f"\n{case_name}\n{pivot.to_string()}")
        print(f"\n[timed] written to {self.out_path}")


def main():
    """Run the budgeted comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=list(AXES), default="seconds")
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--limits", nargs="*", type=parse_limit, default=None)
    parser.add_argument("--methods", nargs="*", default=[label for label, _, _ in VARIANTS])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--out", default=None, help="CSV to append to; defaults to the axis's CSV")
    args = parser.parse_args()

    axis = AXES[args.axis]
    limits = axis["limits"] if args.limits is None else args.limits
    out = axis["csv"] if args.out is None else args.out
    BudgetedComparison(args.cases, limits, args.methods, args.seeds, out, axis=args.axis).run()


if __name__ == "__main__":
    main()
