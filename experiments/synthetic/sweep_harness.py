# pylint: disable=too-many-locals
"""
This file contains the SweepHarness class: the locked study harness extended with
several configurations per method, writing to a store of its own.

The locked ``StudyHarness`` reads every method at one frozen configuration and pins
its stores by name. The follow-up studies of the review round of 8 September 2026
(``run_scaling_k.py``, ``run_ga_budget.py``) need the same case preparation and the
same row schema, but a list of (label, method, kwargs) configurations and a separate
results file, exactly as the A4 equal-footing driver did. This module holds that
shared machinery once, so the locked harness stays byte-identical and each driver
is a short spec.

Every row records what the locked rows record, plus the configuration label and two
things the locked rows could not hold. The incumbent rule (added to every solver after
the locked runs) lets a solver answer with an existing decision-maker option when it
found nothing better, so ``f_found`` is what the tool reports and ``f_solver`` is what
the method itself found; ``reported_incumbent`` flags the rows where they differ.
Recovery is recorded on both. The locked study predates the rule, so its rows are
comparable to ``f_solver``, not to ``f_found``. On the generated cases the bracketing
options can sit within a few thousandths of the optimum, so without this split a
method that found nothing at all would still "recover" through the rule.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from study_harness import StudyHarness, StudySpec
import vlinder.optimize as vopt
from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.utils import suppress_print

# The capped grid of the A4 study registers itself under "grid_capped" on import.
import run_equal_footing  # noqa: F401  # pylint: disable=unused-import,wrong-import-order


def config_label(method, kwargs):
    """
    This function names one configuration of a method.
    :param method: the method name
    :param kwargs: the keyword arguments of this configuration
    :return: a label such as "slsqp@n_starts=25"
    """
    if not kwargs:
        return method
    parts = ",".join(f"{key}={value}" for key, value in sorted(kwargs.items()))
    return f"{method}@{parts}"


@suppress_print
def build(root, case_name):
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
def solve(sim, task):
    """
    This function runs one solver on one scenario with its configuration.
    :param sim: the built simulator
    :param task: the task dictionary
    :return: None
    """
    sim.optimize(
        task["scenario"],
        method=task["method"],
        dmo_name=f"Sweep ({task['config_label']})",
        seed=task["method_seed"],
        **task["method_kwargs"],
    )


def solve_and_capture(sim, task):
    """
    This function runs one solver and also captures the answer it handed to the
    incumbent rule, i.e. the method's own best before the rule could replace it.

    Every solver ends its run with one call to ``BaseSolver._settle_incumbent``; the
    patch records that call's arguments and is always undone, so worker processes and
    sequential runs in one process do not leak into one another.
    :param sim: the built simulator
    :param task: the task dictionary
    :return: the solver's own (allocation, appreciation)
    """
    # pylint: disable=protected-access  # patching the rule's hook is the whole point here
    captured = {}
    original = vopt.BaseSolver._settle_incumbent

    def capture(self, dmo_name, scenario, allocation, appreciation):
        captured["allocation"] = None if allocation is None else np.asarray(allocation, dtype=float).tolist()
        captured["appreciation"] = float(appreciation)
        return original(self, dmo_name, scenario, allocation, appreciation)

    vopt.BaseSolver._settle_incumbent = capture
    try:
        solve(sim, task)
    finally:
        vopt.BaseSolver._settle_incumbent = original
    return captured.get("allocation"), captured.get("appreciation", float("nan"))


def run_task(task):
    """
    This function runs ONE (case, scenario, configuration) cell.

    Module level and picklable, so it can run in worker processes on Windows.
    :param task: the task dictionary
    :return: the results row
    """
    sim = build(task["root"], task["case_name"])
    started_at = time.perf_counter()
    _, f_solver = solve_and_capture(sim, task)
    wall = time.perf_counter() - started_at

    result = sim.optimization_result
    f_found = float(result.appreciation)
    gap = task["f_oracle"] - f_found
    gap_solver = task["f_oracle"] - f_solver
    return {
        "case_name": task["case_name"],
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
        "f_oracle": task["f_oracle"],
        "gap": gap,
        "recovered": bool(gap <= task["recovery_eps"]),
        "f_solver": f_solver,
        "gap_solver": gap_solver,
        "recovered_solver": bool(gap_solver <= task["recovery_eps"]),
        "oracle_verified": task["oracle_verified"],
        "allocation": np.asarray(result.allocation).tolist(),
        "spend_fraction": float(np.sum(result.allocation) / task["budget"]),
        "n_function_evals": result.n_function_evals,
        "wall_time_s": wall,
        "censored": bool(wall > task["runtime_cap_s"]),
        "reported_dmo": result.dmo_name,
        "reported_incumbent": not str(result.dmo_name).startswith("Sweep ("),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class SweepHarness(StudyHarness):
    """The locked harness with a configuration list and a store of its own."""

    def __init__(self, spec: StudySpec, configs, store_stem):
        """
        :param spec: the case grid to prepare (variants, ks, seeds, root)
        :param configs: a list of (label, method, kwargs) tuples to run on every cell
        :param store_stem: the results go to ``results_<stem>.jsonl`` next to the cases
        """
        super().__init__(spec)
        self.configs = list(configs)
        self.results_path = self.root / f"results_{store_stem}.jsonl"
        self.meta_path = self.root / f"meta_{store_stem}.json"

    @staticmethod
    def task_key(name, scenario, method):
        """Resume key of one row; ``method`` carries the configuration label."""
        return f"{name}|{scenario}|{method}"

    def completed_keys(self):
        """Keys already present in the store, so an interrupted run resumes."""
        done = set()
        if self.results_path.exists():
            with open(self.results_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        done.add(self.task_key(row["case_name"], row["scenario"], row["config_label"]))
        return done

    def _case_tasks(self, name, manifest, done):
        """Enumerate the not-yet-done (scenario, configuration) tasks of one case."""
        oracle = manifest["oracle"]
        tasks = []
        for scenario, cert in oracle["per_scenario"].items():
            for label, method, kwargs in self.configs:
                if self.task_key(name, scenario, label) in done:
                    continue
                tasks.append(
                    {
                        "root": str(self.root),
                        "case_name": name,
                        "regime": manifest["regime"],
                        "variant": manifest["appreciation"],
                        "k": int(manifest["params"]["k"]),
                        "case_seed": int(manifest["seed"]),
                        "budget": float(manifest["params"]["budget"]),
                        "scenario": scenario,
                        "method": method,
                        "config_label": label,
                        "method_kwargs": dict(kwargs),
                        "method_seed": int(manifest["seed"]),
                        "f_oracle": float(cert["f_star"]),
                        "oracle_verified": bool(oracle.get("verified", False)),
                        "recovery_eps": self.spec.recovery_eps,
                        "runtime_cap_s": self.spec.runtime_cap_s,
                    }
                )
        return tasks

    def write_meta(self):
        """Record the environment, the spec and the configuration list next to the results."""
        super().write_meta()
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        meta["configs"] = [
            {"label": label, "method": method, "kwargs": kwargs} for label, method, kwargs in self.configs
        ]
        meta["spec"] = asdict(self.spec)
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _finished_rows(self, tasks, n_workers):
        """Run the tasks, yielding each finished row (this module's ``run_task``)."""
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
        print(f"[sweep] {len(tasks)} tasks to run ({len(self.completed_keys())} already done)", flush=True)
        with open(self.results_path, "a", encoding="utf-8") as out:
            for i, row in enumerate(self._finished_rows(tasks, n_workers)):
                out.write(json.dumps(row) + "\n")
                out.flush()
                flag = f"  [incumbent; solver gap={row['gap_solver']:.2e}]" if row["reported_incumbent"] else ""
                print(
                    f"[sweep] {i + 1}/{len(tasks)} {row['case_name']} {row['scenario']} {row['config_label']}: "
                    f"gap={row['gap']:.2e} evals={row['n_function_evals']} ({row['wall_time_s']:.1f}s){flag}",
                    flush=True,
                )
        return self.results_path

    def smoke(self, k=None):
        """
        This function times one cell per configuration on the largest (or given) k, so
        a full run is sized on measurement rather than guesswork.
        :param k: the dimension to probe; the largest in the spec when None
        :return: None
        """
        k = max(self.spec.ks) if k is None else k
        by_label = {}
        for task in self.build_tasks():
            if task["k"] == k and task["variant"] == self.spec.variants[-1]:
                by_label.setdefault(task["config_label"], task)
        variant = self.spec.variants[-1]
        print(f"[sweep] smoke: {len(by_label)} configurations, one cell each at k={k}, {variant}")
        for label, task in by_label.items():
            row = run_task(task)
            flag = f"  [incumbent; solver gap={row['gap_solver']:.2e}]" if row["reported_incumbent"] else ""
            print(
                f"  {label:<60} {row['wall_time_s']:>7.1f}s  evals={row['n_function_evals']:>8}  "
                f"gap={row['gap']:.2e}{flag}",
                flush=True,
            )
