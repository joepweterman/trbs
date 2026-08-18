"""
This file compares every optimizer on the packaged cases under equal wall-clock
budgets, which is the comparison Louise asked for: what does a user actually get
in 15, 30, 60 or 120 seconds?

Each method is given the same number of seconds and its own budget knob is scaled
to fit, rather than every method running at its own frozen default. Grid search
uses the new ``max_calculation_time`` parameter; the continuous methods are
calibrated by timing one unit of their work (one start, one hop, one generation)
and then buying as many units as the budget allows.

Reading the result: the interesting question is not who wins at 120 seconds but
where each curve flattens. A method whose appreciation stops improving has found
what it is going to find, and the remaining seconds are wasted.

Run:
  python experiments/time_limited_comparison.py
  python experiments/time_limited_comparison.py --cases Beerwiser --limits 15 30
"""

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import vlinder as vl

from vlinder.trbs import TheResponsibleBusinessSimulator

sys.path.insert(0, str(Path(__file__).resolve().parent / "synthetic"))
from grid_capped import register  # noqa: E402  (registers grid_capped before any solve)

register()

DATA = Path(os.path.dirname(vl.__file__)) / "data"
OUT = Path(__file__).resolve().parent / "out" / "time_limited"
CASES = ("Beerwiser", "Refugee", "IZZ")
LIMITS = (15, 30, 60, 120)

#: Per method, the knob that buys more search and the unit used to calibrate it.
#: Grid search is absent on purpose: it takes the time budget directly.
KNOBS = {
    "slsqp": ("n_starts", {"n_starts": 1}),
    "basin_hopping": ("n_hops", {"n_hops": 1, "n_starts": 1}),
    "genetic_algorithm": ("n_generations", {"population_size": 50, "n_generations": 1}),
    "mdbh": ("n_starts", {"n_starts": 1, "n_hops": 10, "n_local_steps": 50}),
}
METHODS = ("grid", "grid_capped", "slsqp", "basin_hopping", "genetic_algorithm", "mdbh")


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


def run_once(case_name, scenario, method, kwargs):
    """
    This function runs one method once and reports what it achieved.
    :param case_name: the bundled case name
    :param scenario: the scenario to solve
    :param method: the method name
    :param kwargs: the keyword arguments for this run
    :return: the OptimizationResult
    """
    sim = build(case_name)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.optimize(scenario, method=method, dmo_name="Timed", seed=0, **kwargs)
    return sim.optimization_result


def units_within(case_name, scenario, method, seconds):
    """
    This function calibrates a continuous method's budget knob to a time limit by
    timing a single unit of its work and buying as many as fit.
    :param case_name: the bundled case name
    :param scenario: the scenario to solve
    :param method: the method name
    :param seconds: the time budget
    :return: the keyword arguments for the timed run
    """
    knob, minimal = KNOBS[method]
    result = run_once(case_name, scenario, method, minimal)
    per_unit = max(result.calculation_time, 1e-6)
    kwargs = dict(minimal)
    kwargs[knob] = max(1, int(seconds / per_unit))
    return kwargs


def main():
    """Run the timed comparison and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--limits", nargs="*", type=int, default=list(LIMITS))
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    args = parser.parse_args()

    records = []
    for case_name in args.cases:
        scenario = list(build(case_name).input_dict["scenarios"])[0]
        for seconds in args.limits:
            for method in args.methods:
                if method in ("grid", "grid_capped"):
                    kwargs = {"max_calculation_time": seconds}
                else:
                    kwargs = units_within(case_name, scenario, method, seconds)
                result = run_once(case_name, scenario, method, kwargs)
                records.append(
                    {
                        "case": case_name,
                        "scenario": scenario,
                        "limit_s": seconds,
                        "method": method,
                        "appreciation": result.appreciation,
                        "seconds": result.calculation_time,
                        "evaluations": result.n_function_evals,
                        "budget_spent": result.budget_spent,
                        "reported_dmo": result.dmo_name,
                        "settings": {k: v for k, v in kwargs.items()},
                    }
                )
                print(
                    f"{case_name:<10} {seconds:>4}s  {method:<19} "
                    f"appr {result.appreciation:>10.6f}  used {result.calculation_time:>6.1f}s  "
                    f"evals {str(result.n_function_evals):>9}",
                    flush=True,
                )

    frame = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "time_limited_comparison.csv", index=False)

    print("\nBest appreciation per case and limit:")
    for case_name, group in frame.groupby("case"):
        pivot = group.pivot_table(index="method", columns="limit_s", values="appreciation")
        print(f"\n{case_name}\n{pivot.to_string()}")
    print(f"\n[timed] written to {OUT / 'time_limited_comparison.csv'}")


if __name__ == "__main__":
    main()
