"""
Equal-footing benchmark driver: amendment A4 (2026-08-15, see PREREGISTRATION.md).

A4 freezes a supplementary study that separates three confounds in the locked
comparison of the enumerative baseline against the continuous methods: the
baseline's smaller feasible set, the entanglement of quality with cost, and the
enumeration overhead charged to its wall-clock. This driver runs it.

Two datasets with different jobs:

  * the synthetic slice (both convex variants, k in {2,3,4,6}, seeds 0-9), where
    the ground truth is certified, carries the methodological claim and is run
    along each method's own budget sweep;
  * the packaged cases (Beerwiser, Refugee, IZZ) at locked defaults carry the
    practical claim, namely how much of the measured gap was the search space.

Results go to results_equalfooting.jsonl, next to and never inside the locked
stores. Every objective evaluation of every method is recorded by the shared
tracer of eval_tracer.py, so cost is read on evaluations and not only on time.

Must be a real .py file: Windows spawn cannot serve a stdin __main__ to worker
processes.

Run:
  python experiments/synthetic/run_equal_footing.py --dry-run
  python experiments/synthetic/run_equal_footing.py --smoke
  python experiments/synthetic/run_equal_footing.py --workers 6
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import vlinder as vl

from eval_tracer import trace_evaluations
from grid_capped import register
from study_harness import METHOD_DEFAULTS, StudyHarness, StudySpec

from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.utils import suppress_print

register()

#: Bundled case data, for the packaged half of the study.
PACKAGED_DATA = Path(os.path.dirname(vl.__file__)) / "data"
PACKAGED_CASES = ("Beerwiser", "Refugee", "IZZ")

#: A4 target ladder, in appreciation points below the reference optimum. 1.0 is
#: decision indifference, 0.1 is the pre-registered recovery epsilon, and 0.01
#: sits below the 0.0102 gap between Beerwiser's two optima, so it is the level
#: at which the metric starts to discriminate basins.
TAUS = (1.0, 0.1, 0.01)

#: Locked defaults, extended with the two entries the locked table does not hold.
#: grid_capped is given the packaged grid's ceiling so the two baselines are read
#: at the same budget; mdbh keeps its A3 configuration.
DEFAULTS = dict(METHOD_DEFAULTS)
DEFAULTS["grid_capped"] = {"max_combinations": 60000}
DEFAULTS["mdbh"] = {
    "n_starts": 5,
    "n_hops": 10,
    "n_local_steps": 50,
    "eta": 1.0,
    "sigma": 1.5,
    "temperature": 1.0,
}

#: A4 budget sweep. Each method is read along its own quality-cost frontier
#: instead of at one frozen point. Frozen before the run.
SWEEP = {
    "grid": [{"max_combinations": c} for c in (1000, 6000, 60000, 600000)],
    "grid_capped": [{"max_combinations": c} for c in (1000, 6000, 60000, 600000)],
    "slsqp": [{"n_starts": n} for n in (1, 5, 25, 100)],
    "basin_hopping": [{"n_hops": h, "n_starts": 1} for h in (1, 5, 25)],
    "genetic_algorithm": [{"population_size": 50, "n_generations": g} for g in (15, 60, 240)],
    "mdbh": [dict(DEFAULTS["mdbh"], n_starts=n) for n in (1, 5, 25)],
}

#: The synthetic half: the certified region of the locked grid, thinned in seeds.
SYNTHETIC_VARIANTS = ("linear", "sinusoidal")
SYNTHETIC_KS = (2, 3, 4, 6)
SYNTHETIC_SEEDS = tuple(range(10))

RUNTIME_CAP_S = 600.0
STAIRCASE_MAX_POINTS = 400


def config_label(method, kwargs):
    """
    This function names one point of a method's budget sweep.
    :param method: the method name
    :param kwargs: the frozen keyword arguments of this sweep point
    :return: a label such as "slsqp@n_starts=25"
    """
    if not kwargs:
        return method
    parts = ",".join(f"{key}={value}" for key, value in sorted(kwargs.items()))
    return f"{method}@{parts}"


def thin_staircase(improvements, max_points=STAIRCASE_MAX_POINTS):
    """
    This function reduces a best-so-far staircase to a plottable size, keeping the
    first and last step and spacing the rest logarithmically in evaluations.

    Thinning never touches the target crossings, which are computed from the full
    staircase before this runs.
    :param improvements: the full list of (evals, seconds, f) steps
    :param max_points: the maximum number of steps to keep
    :return: a list of dicts, at most ``max_points`` long
    """
    if len(improvements) <= max_points:
        kept = improvements
    else:
        last_index = len(improvements) - 1
        picks = np.unique(np.geomspace(1, last_index + 1, max_points).astype(int) - 1)
        kept = [improvements[i] for i in picks]
        if kept[-1] != improvements[-1]:
            kept.append(improvements[-1])
    return [{"evals": n, "seconds": round(t, 6), "f": v} for n, t, v in kept]


@suppress_print
def build_synthetic(root, case_name):
    """
    This function builds one generated case through the unmodified pipeline.
    :param root: the directory holding the generated cases
    :param case_name: the case name
    :return: a built, evaluated and appreciated simulator
    """
    sim = TheResponsibleBusinessSimulator(case_name, file_path=Path(root), file_extension="csv")
    sim.build()
    sim.evaluate()
    sim.appreciate()
    return sim


@suppress_print
def build_packaged(case_name):
    """
    This function builds one bundled case through the unmodified pipeline.

    A fresh build per run is required, not merely tidy: the grid path freezes the
    appreciation boundaries on the dictionary it runs on, so reusing one instance
    across methods would silently change the objective later methods are scored on.
    :param case_name: the bundled case name
    :return: a built, evaluated and appreciated simulator
    """
    sim = TheResponsibleBusinessSimulator(case_name, file_path=PACKAGED_DATA, file_extension="csv")
    sim.build()
    sim.evaluate()
    sim.appreciate()
    return sim


@suppress_print
def solve(sim, task):
    """
    This function runs one solver on one scenario with its frozen configuration.
    :param sim: the built simulator
    :param task: the task dictionary
    :return: None
    """
    sim.optimize(
        task["scenario"],
        method=task["method"],
        dmo_name=f"A4 ({task['method']})",
        seed=task["method_seed"],
        **task["method_kwargs"],
    )


def run_task(task):
    """
    This function runs ONE (case, scenario, method, configuration) cell and traces
    every objective evaluation it makes.

    Module level and picklable, so it can run in worker processes on Windows.
    :param task: the task dictionary
    :return: the results row
    """
    sim = (
        build_synthetic(task["root"], task["case_name"])
        if task["family"] == "synthetic"
        else build_packaged(task["case_name"])
    )

    started_at = time.perf_counter()
    with trace_evaluations() as trace:
        solve(sim, task)
    wall = time.perf_counter() - started_at

    result = sim.optimization_result
    f_found = float(result.appreciation)
    f_oracle = task["f_oracle"]
    gap = None if f_oracle is None else f_oracle - f_found
    budget = float(result.budget) if result.budget else float(task["budget"] or 0.0) or None

    row = {
        "case_name": task["case_name"],
        "family": task["family"],
        "regime": task["regime"],
        "variant": task["variant"],
        "k": task["k"],
        "case_seed": task["case_seed"],
        "scenario": task["scenario"],
        "method": task["method"],
        "config_label": task["config_label"],
        "method_kwargs": task["method_kwargs"],
        "method_seed": task["method_seed"],
        "f_found": f_found,
        "f_oracle": f_oracle,
        "gap": gap,
        "recovered": None if gap is None else bool(gap <= 0.1),
        "oracle_verified": task["oracle_verified"],
        "allocation": np.asarray(result.allocation).tolist(),
        "spend_fraction": None if not budget else float(np.sum(result.allocation) / budget),
        "n_function_evals": result.n_function_evals,
        "n_evals_traced": trace.n_evals,
        "n_improvements": len(trace.improvements),
        "wall_time_s": wall,
        "censored": bool(wall > RUNTIME_CAP_S),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if f_oracle is None:
        # No reference yet: the packaged half is only 54 cells, so the full
        # staircase is kept and the crossings are derived in the analysis, once
        # the best value found by any method is known.
        row["cost_to_target"] = None
        row["staircase"] = [{"evals": n, "seconds": round(t, 6), "f": v} for n, t, v in trace.improvements]
    else:
        row["cost_to_target"] = {f"{tau:g}": trace.cost_to_target(f_oracle, tau) for tau in TAUS}
        row["staircase"] = thin_staircase(trace.improvements)
    return row


class EqualFootingHarness(StudyHarness):
    """Task building and the run loop for A4, writing to its own stores.

    Only case preparation is inherited: ``ensure_case`` generates and certifies a
    synthetic case if it is not on disk yet, and the locked cases already are, so
    nothing is regenerated. Everything below is A4's own, because the locked run
    loop pins ``StudyHarness.run_task`` by name in its worker submission.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.results_path = self.root / "results_equalfooting.jsonl"
        self.meta_path = self.root / "meta_equalfooting.json"

    @staticmethod
    def task_key(name, scenario, method):
        """Resume key of one A4 row; ``method`` carries the configuration label."""
        return f"{name}|{scenario}|{method}"

    def completed_keys(self):
        """Keys already present in the A4 store, so an interrupted run resumes."""
        done = set()
        if self.results_path.exists():
            with open(self.results_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        done.add(self.task_key(row["case_name"], row["scenario"], row["config_label"]))
        return done

    def build_tasks(self):
        """Enumerate the pending synthetic sweep cells and packaged cells."""
        done = self.completed_keys()
        return self._synthetic_tasks(done) + self._packaged_tasks(done)

    def _synthetic_tasks(self, done):
        """The certified half: every sweep point on every case and scenario."""
        tasks = []
        for variant in SYNTHETIC_VARIANTS:
            for k in SYNTHETIC_KS:
                for seed in SYNTHETIC_SEEDS:
                    name, manifest = self.ensure_case(variant, k, seed)
                    oracle = manifest["oracle"]
                    for scenario, cert in oracle["per_scenario"].items():
                        for method, configs in SWEEP.items():
                            for kwargs in configs:
                                label = config_label(method, kwargs)
                                if self.task_key(name, scenario, label) in done:
                                    continue
                                tasks.append(
                                    {
                                        "root": str(self.root),
                                        "family": "synthetic",
                                        "case_name": name,
                                        "regime": manifest["regime"],
                                        "variant": manifest["appreciation"],
                                        "k": int(manifest["params"]["k"]),
                                        "case_seed": int(manifest["seed"]),
                                        "budget": float(manifest["params"]["budget"]),
                                        "scenario": scenario,
                                        "method": method,
                                        "method_kwargs": dict(kwargs),
                                        "config_label": label,
                                        "method_seed": int(manifest["seed"]),
                                        "f_oracle": float(cert["f_star"]),
                                        "oracle_verified": bool(oracle.get("verified", False)),
                                    }
                                )
        return tasks

    def _packaged_tasks(self, done):
        """The external-validity half: locked defaults, both baselines, no sweep."""
        tasks = []
        for case_name in PACKAGED_CASES:
            sim = build_packaged(case_name)
            scenarios = list(sim.input_dict["scenarios"])
            k = len(sim.input_dict["internal_variable_inputs"])
            for scenario in scenarios:
                for method in SWEEP:
                    label = config_label(method, DEFAULTS.get(method, {}))
                    if self.task_key(case_name, scenario, label) in done:
                        continue
                    tasks.append(
                        {
                            "root": str(self.root),
                            "family": "packaged",
                            "case_name": case_name,
                            "regime": "packaged",
                            "variant": "packaged",
                            "k": k,
                            "case_seed": None,
                            "budget": None,
                            "scenario": scenario,
                            "method": method,
                            "method_kwargs": dict(DEFAULTS.get(method, {})),
                            "config_label": label,
                            "method_seed": 0,
                            "f_oracle": None,
                            "oracle_verified": False,
                        }
                    )
        return tasks

    def _finished_rows(self, tasks, n_workers):
        """Run the tasks, yielding each finished row (A4's own ``run_task``)."""
        if n_workers <= 1:
            yield from (run_task(task) for task in tasks)
            return
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(run_task, task) for task in tasks]
            for fut in as_completed(futures):
                yield fut.result()

    def run(self, n_workers=1):
        """
        This function executes the pending tasks, appending each finished row as it
        completes so an interrupted run resumes where it stopped.
        :param n_workers: number of worker processes (1 runs in-process)
        :return: the path of the results file
        """
        self.write_meta()
        tasks = self.build_tasks()
        print(f"[a4] {len(tasks)} tasks to run ({len(self.completed_keys())} already done)")
        with open(self.results_path, "a", encoding="utf-8") as out:
            for i, row in enumerate(self._finished_rows(tasks, n_workers)):
                out.write(json.dumps(row) + "\n")
                out.flush()
                gap = "n/a" if row["gap"] is None else f"{row['gap']:.2e}"
                print(
                    f"[a4] {i + 1}/{len(tasks)} {row['case_name']} {row['scenario']} "
                    f"{row['config_label']}: gap={gap} evals={row['n_evals_traced']} "
                    f"({row['wall_time_s']:.1f}s)"
                )
        return self.results_path


def spec():
    """
    This function returns the StudySpec the A4 harness prepares cases with.
    :return: a StudySpec covering the certified synthetic slice
    """
    return StudySpec(
        variants=SYNTHETIC_VARIANTS,
        ks=SYNTHETIC_KS,
        seeds=SYNTHETIC_SEEDS,
        methods=tuple(SWEEP),
        runtime_cap_s=RUNTIME_CAP_S,
    )


def smoke(harness):
    """
    This function times one cell per method on the cheapest and the most expensive
    sweep point, so the full run is sized on measurement rather than on guesswork.
    :param harness: the A4 harness
    :return: None
    """
    tasks = harness.build_tasks()
    by_label = {}
    for task in tasks:
        if task["family"] == "synthetic" and task["k"] == 6 and task["variant"] == "sinusoidal":
            by_label.setdefault(task["config_label"], task)
    print(f"[a4] smoke: {len(by_label)} configurations, one cell each at k=6, sinusoidal")
    for label, task in sorted(by_label.items()):
        row = run_task(task)
        reached = (
            "-"
            if row["cost_to_target"] is None
            else ",".join(
                f"{tau}:{'-' if hit is None else hit['evals']}" for tau, hit in row["cost_to_target"].items()
            )
        )
        print(
            f"  {label:<40} {row['wall_time_s']:>7.2f}s  evals={row['n_evals_traced']:>7}  "
            f"gap={row['gap']:.2e}  reached[{reached}]"
        )


def main():
    """Run, dry-run or smoke-test the A4 equal-footing benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6, help="parallel worker processes")
    parser.add_argument("--dry-run", action="store_true", help="count the pending tasks and stop")
    parser.add_argument("--smoke", action="store_true", help="time one cell per configuration and stop")
    args = parser.parse_args()

    harness = EqualFootingHarness(spec())
    if args.dry_run:
        tasks = harness.build_tasks()
        families = {}
        for task in tasks:
            families[task["family"]] = families.get(task["family"], 0) + 1
        print(f"[a4] {len(tasks)} tasks pending: {families}")
        return
    if args.smoke:
        smoke(harness)
        return
    results_path = harness.run(n_workers=args.workers)
    print(f"[a4] complete - results at {results_path}")


if __name__ == "__main__":
    main()
