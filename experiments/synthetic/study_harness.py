"""
This file contains the StudyHarness class that runs the pre-registered
Monte-Carlo study.

It executes a factorial grid (variant x k x seed x method) of synthetic cases
through the unmodified vlinder pipeline and writes one row per case x scenario x
method to ``results.jsonl``, the single source of truth every downstream table
and figure reads from.

Cases are certified by the oracle before any method runs, and the certificate is
cached in the case manifest so re-runs never re-solve it. Every finished task is
appended immediately, and completed (case, scenario, method) keys are skipped on
resume, so an interrupted study continues where it stopped. The design itself is
frozen in ``PREREGISTRATION.md``.
"""

import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from case_factory import SyntheticCaseFactory, SyntheticCaseParams, manifest_path, read_manifest
from oracle import certify_case

from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.utils import suppress_print

#: Frozen per-method defaults, mirroring pre-registration v1.1 section 2.
METHOD_DEFAULTS = {
    "grid": {"max_combinations": 60000},
    "slsqp": {"n_starts": 100},
    "basin_hopping": {"n_hops": 25, "n_starts": 1},
    "genetic_algorithm": {"population_size": 50, "n_generations": 60},
}


@dataclass
class StudySpec:
    """Factorial design of one study run (defaults = the confirmatory H1/H2 grid)."""

    variants: tuple = ("linear", "sinusoidal")
    ks: tuple = (2, 3, 4, 6, 9, 12, 15)
    seeds: tuple = tuple(range(30))
    methods: tuple = ("grid", "slsqp", "basin_hopping", "genetic_algorithm")
    runtime_cap_s: float = 600.0
    recovery_eps: float = 0.1
    root: str = str(Path(__file__).parent / "generated" / "study")

    @staticmethod
    def n_key_outputs(k):
        """Pre-registration section 2: n_key_outputs = max(2, min(5, (k+1)//2))."""
        return max(2, min(5, (k + 1) // 2))

    def case_name(self, variant, k, seed):
        """Deterministic case naming: Study_<variant>_k<k>_s<seed>."""
        return f"Study_{variant}_k{k:02d}_s{seed}"


class StudyHarness:
    """Generates, certifies and runs the factorial study with checkpoint/resume."""

    def __init__(self, spec: StudySpec):
        self.spec = spec
        self.root = Path(spec.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_path = self.root / "results.jsonl"
        self.meta_path = self.root / "meta.json"

    # ---------------- case preparation ----------------
    def ensure_case(self, variant, k, seed):
        """
        This function generates and oracle-certifies one case if that has not
        happened yet, so a study run is idempotent and resumable.
        :param variant: the appreciation variant
        :param k: the number of internal variables
        :param seed: the case seed
        :return: a tuple of the case name and its manifest
        """
        name = self.spec.case_name(variant, k, seed)
        if not manifest_path(name, self.root).exists():
            params = SyntheticCaseParams(
                name=name,
                k=k,
                n_key_outputs=self.spec.n_key_outputs(k),
                seed=seed,
                appreciation=variant,
            )
            SyntheticCaseFactory(params).write(self.root)
        if read_manifest(name, self.root)["oracle"] is None:
            certify_case(name, self.root)
        return name, read_manifest(name, self.root)

    # ---------------- task list & bookkeeping ----------------
    @staticmethod
    def task_key(name, scenario, method):
        """Resume key of one results row."""
        return f"{name}|{scenario}|{method}"

    def completed_keys(self):
        """Keys already present in results.jsonl (crash-safe resume)."""
        done = set()
        if self.results_path.exists():
            with open(self.results_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        done.add(self.task_key(row["case_name"], row["scenario"], row["method"]))
        return done

    def build_tasks(self):
        """Generate/certify all cases, then enumerate the not-yet-done tasks."""
        done = self.completed_keys()
        tasks = []
        for variant in self.spec.variants:
            for k in self.spec.ks:
                for seed in self.spec.seeds:
                    name, manifest = self.ensure_case(variant, k, seed)
                    tasks.extend(self._case_tasks(name, manifest, done))
        return tasks

    def _case_tasks(self, name, manifest, done):
        """Enumerate the not-yet-done (scenario, method) tasks of one case."""
        k = int(manifest["params"]["k"])
        seed = int(manifest["seed"])
        budget = float(manifest["params"]["budget"])
        oracle = manifest["oracle"]
        tasks = []
        for scenario, cert in oracle["per_scenario"].items():
            for method in self.spec.methods:
                if self.task_key(name, scenario, method) in done:
                    continue
                tasks.append(
                    {
                        "root": str(self.root),
                        "case_name": name,
                        "regime": manifest["regime"],
                        "variant": manifest["appreciation"],
                        "k": k,
                        "case_seed": seed,
                        "scenario": scenario,
                        "method": method,
                        "method_kwargs": dict(METHOD_DEFAULTS.get(method, {})),
                        "method_seed": seed,
                        "budget": budget,
                        "f_oracle": float(cert["f_star"]),
                        "oracle_verified": bool(oracle.get("verified", False)),
                        "recovery_eps": self.spec.recovery_eps,
                        "runtime_cap_s": self.spec.runtime_cap_s,
                    }
                )
        return tasks

    # ---------------- execution ----------------
    @staticmethod
    def run_task(task):
        """Run ONE (case, scenario, method) cell through the real pipeline.

        Static and picklable so it can run in worker processes on Windows.
        """

        @suppress_print
        def _build():
            sim = TheResponsibleBusinessSimulator(
                task["case_name"], file_path=Path(task["root"]), file_extension="csv"
            )
            sim.build()
            sim.evaluate()
            sim.appreciate()
            return sim

        sim = _build()
        t0 = time.perf_counter()
        sim.optimize(
            task["scenario"],
            method=task["method"],
            dmo_name=f"Study ({task['method']})",
            seed=task["method_seed"],
            **task["method_kwargs"],
        )
        # optimize() returns the updated input_dict; the run's diagnostics live here.
        result = sim.optimization_result
        wall = time.perf_counter() - t0
        gap = task["f_oracle"] - float(result.appreciation)
        return {
            "case_name": task["case_name"],
            "regime": task["regime"],
            "variant": task["variant"],
            "k": task["k"],
            "case_seed": task["case_seed"],
            "scenario": task["scenario"],
            "method": task["method"],
            "method_seed": task["method_seed"],
            "f_found": float(result.appreciation),
            "f_oracle": task["f_oracle"],
            "gap": gap,
            "recovered": bool(gap <= task["recovery_eps"]),
            "oracle_verified": task["oracle_verified"],
            "allocation": np.asarray(result.allocation).tolist(),
            "spend_fraction": float(np.sum(result.allocation) / task["budget"]),
            "n_function_evals": result.n_function_evals,
            "wall_time_s": wall,
            "censored": bool(wall > task["runtime_cap_s"]),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def write_meta(self):
        """Record the environment + spec next to the results (pre-reg section 6)."""
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        meta = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "vlinder_commit": commit,
            "method_defaults": METHOD_DEFAULTS,
            "spec": asdict(self.spec),
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _finished_rows(self, tasks, n_workers):
        """
        This function runs the tasks and yields each finished results row.
        :param tasks: the task dictionaries to execute
        :param n_workers: number of worker processes (1 runs in-process)
        :return: a generator of results rows
        """
        if n_workers <= 1:
            yield from (self.run_task(task) for task in tasks)
            return
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(StudyHarness.run_task, t) for t in tasks]
            for fut in as_completed(futures):
                yield fut.result()

    def run(self, n_workers=1):
        """
        This function executes all remaining tasks, appending each finished row
        to results.jsonl as it completes so an interrupted run can resume.
        :param n_workers: number of worker processes (1 runs in-process)
        :return: the path of the results file
        """
        self.write_meta()
        tasks = self.build_tasks()
        print(f"[harness] {len(tasks)} tasks to run ({len(self.completed_keys())} already done)")
        with open(self.results_path, "a", encoding="utf-8") as out:
            for i, row in enumerate(self._finished_rows(tasks, n_workers)):
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(
                    f"[harness] {i + 1}/{len(tasks)} {row['case_name']} {row['scenario']} "
                    f"{row['method']}: gap={row['gap']:.2e} ({row['wall_time_s']:.1f}s)"
                )
        return self.results_path
