# pylint: disable=too-many-locals,comparison-with-itself,unused-variable,wrong-import-order
# (pre-existing in the A4 measurement script; silenced rather than rewritten)
"""
This file measures the enumeration overhead of the two grid baselines directly.

Amendment A4's third confound is that the baseline's wall-clock charges it for
enumeration bookkeeping rather than for search. A4 specifies estimating that
share indirectly, as the evaluation count times a per-evaluation cost inferred
from the continuous methods, against the total wall-clock. That estimator turned
out to be biased: it yields negative shares, because the continuous solvers spend
slightly less per evaluation than a stand-alone probe of the objective does, so
the inferred unit cost is too high.

This script measures the quantity directly instead. It times the step-size choice
plus the lattice construction on their own, before a single objective evaluation
happens, and reports that against the solve time already recorded in the A4
store. The deviation from A4's stated estimator is deliberate and is disclosed in
the report; both readings are published.

Wall-clock and CPU time are both recorded. A suspended machine inflates the first
and not the second, which is how a sleep-contaminated measurement is caught: one
k=12 run of this script reported 55,016 wall seconds for work that takes minutes.
Rows where wall time far exceeds CPU time are flagged rather than silently used.

Run:
  python experiments/synthetic/measure_enumeration_overhead.py
"""

import contextlib
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import vlinder as vl

from analyze_study import STUDY_ROOT
from run_equal_footing import CappedGridSearch  # noqa: E402  (the A4 shim)

from vlinder.optimize import GridSearch
from vlinder.trbs import TheResponsibleBusinessSimulator

PACKAGED_DATA = Path(os.path.dirname(vl.__file__)) / "data"
CEILING = 60000
OUT = STUDY_ROOT / "analysis_equalfooting" / "enumeration_overhead.csv"

#: The cases to measure. The synthetic ones reach beyond the A4 slice on purpose,
#: because the claim being tested is about how the overhead behaves in k.
CASES = [
    ("Beerwiser", None, "packaged"),
    ("Refugee", None, "packaged"),
    ("IZZ", None, "packaged"),
    ("Study_linear_k02_s0", "Scenario 01", "synthetic"),
    ("Study_linear_k03_s0", "Scenario 01", "synthetic"),
    ("Study_linear_k04_s0", "Scenario 01", "synthetic"),
    ("Study_linear_k06_s0", "Scenario 01", "synthetic"),
    ("Study_linear_k09_s0", "Scenario 01", "synthetic"),
    ("Study_linear_k12_s0", "Scenario 01", "synthetic"),
]


def build(name, family):
    """
    This function builds one case through the unmodified pipeline.
    :param name: the case name
    :param family: packaged or synthetic, deciding where the case is read from
    :return: a built, evaluated and appreciated simulator
    """
    root = PACKAGED_DATA if family == "packaged" else STUDY_ROOT
    sim = TheResponsibleBusinessSimulator(name, file_path=root, file_extension="csv")
    with contextlib.redirect_stdout(io.StringIO()):
        sim.build()
        sim.evaluate()
        sim.appreciate()
    return sim


def solve_seconds(store, case_name, scenario, method):
    """
    This function reads the recorded solve time of one cell from the A4 store.
    :param store: the loaded store
    :param case_name: the case
    :param scenario: the scenario
    :param method: the method name
    :return: the wall-clock seconds, or NaN when the cell is not in the store
    """
    label = f"{method}@max_combinations={CEILING}"
    hit = store[(store["case_name"] == case_name) & (store["scenario"] == scenario) & (store["config_label"] == label)]
    return float(hit["wall_time_s"].iloc[0]) if len(hit) else float("nan")


def main():
    """Time enumeration on its own and write the overhead table."""
    path = STUDY_ROOT / "results_equalfooting.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    store = pd.DataFrame(rows)

    records = []
    for name, scenario_override, family in CASES:
        sim = build(name, family)
        scenario = scenario_override or list(sim.input_dict["scenarios"])[0]
        k = len(sim.input_dict["internal_variable_inputs"])
        for solver_class in (GridSearch, CappedGridSearch):
            solver = solver_class(sim.input_dict, sim.output_dict)
            with contextlib.redirect_stdout(io.StringIO()):
                best, budget = solver.find_dict_values(scenario)
            scaled = solver.scale_max_investment(budget)

            wall_started, cpu_started = time.perf_counter(), time.process_time()
            capped = solver.method_name == "grid_capped"
            step = solver.calculate_step_size(budget, scaled, k, CEILING, spend_all=not capped)
            points = solver.generate_combinations(budget, step, k, spend_all=not capped)
            enumeration = time.perf_counter() - wall_started
            enumeration_cpu = time.process_time() - cpu_started

            total = solve_seconds(store, name, scenario, solver_class.method_name)
            records.append(
                {
                    "case_name": name,
                    "family": family,
                    "k": k,
                    "method": solver_class.method_name,
                    "n_points": len(points),
                    "enumeration_s": enumeration,
                    "enumeration_cpu_s": enumeration_cpu,
                    "suspect_suspend": bool(enumeration > 2 * enumeration_cpu + 5),
                    "solve_s": total,
                    "enumeration_share": enumeration / total if total == total else float("nan"),
                }
            )
            print(
                f"{name[:24]:<26}k={k:<3}{solver_class.method_name:<13}"
                f"enum {enumeration:>8.2f}s (cpu {enumeration_cpu:>8.2f}s)  solve {total:>8.2f}s  "
                f"share {records[-1]['enumeration_share']:>7.1%}  points {len(points):>9,}",
                flush=True,
            )

    frame = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"\n[a4] enumeration overhead written to {OUT}")


if __name__ == "__main__":
    main()
