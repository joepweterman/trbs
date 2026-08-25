# pylint: disable=R0913,R0917  # a sweep cell is described by case, limit, method, seed
"""
This file compares every optimizer on the packaged cases under equal wall-clock
budgets, which is the comparison Louise asked for: what does a user actually get
in 15, 30, 60 or 120 seconds?

Each method is given the same number of seconds through the one parameter every
solver now understands: ``max_calculation_time``. Grid search evaluates ever
finer lattices under it, halving the step each round, and the continuous methods
keep adding units of work (a start, a hop, a generation) until the time is
spent. Their unit knobs are set far beyond reach so that the clock, not the
knob, is what stops them.

Every method runs in both budget modes: spending at most the budget (the capped
simplex, the continuous default) and spending it exactly (the budget face, the
grid default). The stochastic methods run on several seeds and are read at the
median, because a single seed can flatter or wrong-foot a method; the grids are
deterministic and run once.

Reading the result: the interesting question is not who wins at 120 seconds but
where each curve flattens. A method whose appreciation stops improving has found
what it is going to find, and the remaining seconds are wasted.

Rows are appended to the CSV as they are produced, so an interrupted run keeps
everything it has already measured. ``figure_time_limited.py`` draws the graph.

Run:
  python experiments/time_limited_comparison.py
  python experiments/time_limited_comparison.py --cases Beerwiser --limits 15 30
  python experiments/time_limited_comparison.py --methods slsqp "slsqp (spend all)"
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import pandas as pd
import vlinder as vl

from vlinder.trbs import TheResponsibleBusinessSimulator

sys.path.insert(0, str(Path(__file__).resolve().parent / "synthetic"))
from grid_capped import register  # noqa: E402  # pylint: disable=wrong-import-position,wrong-import-order

register()

DATA = Path(os.path.dirname(vl.__file__)) / "data"
OUT = Path(__file__).resolve().parent / "out" / "time_limited"
CASES = ("Beerwiser", "Refugee", "IZZ")
LIMITS = (15, 30, 60, 120)
SEEDS = (1, 2, 3, 4, 5)

#: Per continuous method, unit knobs set far beyond what any time limit here buys,
#: so that ``max_calculation_time`` is what ends the run. The grids need nothing:
#: they refine until the time is spent.
CEILINGS = {
    "grid": {},
    "grid_capped": {},
    "slsqp": {"n_starts": 10_000},
    "basin_hopping": {"n_starts": 1, "n_hops": 1_000_000},
    "genetic_algorithm": {"population_size": 50, "n_generations": 1_000_000},
    "mdbh": {"n_starts": 10_000, "n_hops": 10, "n_local_steps": 50},
}

#: Every method in both budget modes. ``grid`` spends the budget exactly by construction
#: and ``grid_capped`` is its at-most counterpart, so together they are the two grid rows.
VARIANTS = [("grid", "grid", {}), ("grid (capped)", "grid_capped", {})]
for _method in ("slsqp", "basin_hopping", "genetic_algorithm"):
    VARIANTS.append((_method, _method, {}))
    VARIANTS.append((f"{_method} (spend all)", _method, {"spend_all": True}))
VARIANTS.append(("mdbh", "mdbh", {}))
VARIANTS.append(("mdbh (spend all)", "mdbh", {"spend_all": True}))


class TimedComparison:
    """One full sweep over cases, time limits, method variants and seeds."""

    def __init__(self, cases, limits, labels, seeds, out_path):
        self.cases = list(cases)
        self.limits = [int(limit) for limit in limits]
        self.variants = [variant for variant in VARIANTS if variant[0] in set(labels)]
        self.seeds = [int(seed) for seed in seeds]
        self.out_path = Path(out_path)

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
        return set(zip(frame["case"], frame["limit_s"], frame["label"], frame["seed"]))

    def run(self):
        """Run the sweep, appending each row as it lands, and print the median pivot."""
        finished = self.finished()
        for case_name in self.cases:
            scenario = list(self.build(case_name).input_dict["scenarios"])[0]
            for seconds in self.limits:
                for label, method, extra_kwargs in self.variants:
                    kwargs = {**CEILINGS[method], **extra_kwargs, "max_calculation_time": seconds}
                    # The grids are deterministic: one run tells the story.
                    seeds = [0] if method in ("grid", "grid_capped") else self.seeds
                    for seed in seeds:
                        if (case_name, seconds, label, seed) in finished:
                            continue
                        result = self.run_once(case_name, scenario, method, seed, kwargs)
                        self.append_record(
                            {
                                "case": case_name,
                                "scenario": scenario,
                                "limit_s": seconds,
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
                            f"{case_name:<10} {seconds:>4}s  {label:<28} seed {seed}  "
                            f"appr {result.appreciation:>10.6f}  used {result.calculation_time:>6.1f}s  "
                            f"evals {str(result.n_function_evals):>9}",
                            flush=True,
                        )
        self.report()

    def report(self):
        """Print the median appreciation per case, limit and variant from the CSV."""
        frame = pd.read_csv(self.out_path)
        print("\nMedian appreciation per case and limit:")
        for case_name, group in frame.groupby("case"):
            pivot = group.pivot_table(index="label", columns="limit_s", values="appreciation", aggfunc="median")
            print(f"\n{case_name}\n{pivot.to_string()}")
        print(f"\n[timed] written to {self.out_path}")


def main():
    """Run the timed comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--limits", nargs="*", type=int, default=list(LIMITS))
    parser.add_argument("--methods", nargs="*", default=[label for label, _, _ in VARIANTS])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    # A fresh file: the round-one CSV next to it has no label, mode or seed columns.
    parser.add_argument("--out", default=str(OUT / "time_limited_comparison_v2.csv"))
    args = parser.parse_args()

    TimedComparison(args.cases, args.limits, args.methods, args.seeds, args.out).run()


if __name__ == "__main__":
    main()
