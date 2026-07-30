"""
SCIP global-solver benchmark for the synthetic convex regime.

Pre-registration section 5 pre-declares a SCIP (``pyscipopt``) benchmark as the
stretch method, on Paul Bouman's suggestion, to be added by amendment before it
runs. Amendment A2 does that; this module is the instrument.

Why it cannot be a METHOD_REGISTRY entry
----------------------------------------
Every method in the confirmatory roster treats the objective as a black box: it
calls ``evaluate_allocation`` and looks at the number that comes back. SCIP
cannot work that way. It is a branch-and-bound global solver and needs the
objective as an *algebraic model* it can bound and split, so the model has to be
rebuilt in SCIP's own expression language. That is only possible where the
generating process is known in closed form, which is exactly the synthetic
cases and not an arbitrary tRBS case. Registering it beside ``slsqp`` would
promise something it cannot deliver on a real case, so it lives here instead.

What it buys
------------
The other four methods can only be compared against each other, or against an
oracle this project wrote itself. SCIP is independent: it reconstructs the
problem from the case tables and returns a solution with a *proof* of global
optimality (a zero primal-dual gap). Agreement is therefore an external check on
the certified ground truth, not another opinion from the same family.

On the convex regime both variants are globally solvable in principle: the
linear variant is an LP, and the curved variant maximises a concave function
(sin on [0, pi/2]) over a polytope. A disagreement would mean the ground truth,
not SCIP, is wrong.

Run:
    python scip_benchmark.py               # the full pre-registered grid
    python scip_benchmark.py --smoke       # a few cases, for verification
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from case_factory import DEFAULT_ROOT
from oracle import ORACLE_DMO, Oracle
from pyscipopt import Model, quicksum
from pyscipopt import sin as scip_sin
from vlinder.appreciate import Appreciate
from vlinder.evaluate import Evaluate
from vlinder.optimize import evaluate_allocation

STUDY_ROOT = DEFAULT_ROOT / "study"
#: Model fidelity tolerance: the SCIP objective must agree with the real
#: pipeline evaluated at SCIP's own solution, in appreciation points.
FIDELITY_TOL = 1e-6
SCIP_GAP_LIMIT = 0.0
SCIP_TIME_LIMIT_S = 600.0
#: The reconstruction assumes affine key outputs; these govern how hard that is
#: checked at interior points before the model is trusted.
AFFINITY_PROBE_POINTS = 4
AFFINITY_TOL = 1e-7


class ScipBenchmark(Oracle):
    """
    Rebuilds one generated case as an algebraic SCIP model and solves it to
    proven global optimality.

    Inherits the case loading from :class:`Oracle` on purpose: boundaries and
    weights are then read exactly as the certified oracle reads them, so a
    disagreement can only come from the optimisation and never from two
    different readings of the case. The key-output map itself is probed from the
    pipeline rather than inherited, for the reason given on :meth:`affine_map`.

    :param name: the case name
    :param root: the directory the case was written to
    """

    def __init__(self, name: str, root: Path = DEFAULT_ROOT):
        super().__init__(name, root)
        data = self.opt.input_dict
        self.start = np.asarray(data["key_output_start"], dtype=float)
        self.end = np.asarray(data["key_output_end"], dtype=float)
        self.weights = np.asarray(
            Appreciate(data, self.sim.output_dict)._calculate_weights(),
            dtype=float,  # pylint: disable=protected-access
        )
        self.is_linear = np.asarray(data["key_output_linear"], dtype=int)
        self.stb = np.asarray(data["key_output_smaller_the_better"], dtype=int)
        self.n_key_outputs = len(data["key_outputs"])

    def _key_outputs(self, x, scenario: str) -> np.ndarray:
        """
        Run one allocation through the real evaluation pipeline and return the
        key-output vector.
        :param x: the allocation
        :param scenario: the scenario to evaluate in
        :return: key-output values in ``input_dict`` order
        """
        local = copy.deepcopy(self.opt.input_dict)
        local["decision_makers_option_value"] = np.asarray(local["decision_makers_option_value"], dtype=float)
        idx = int(np.where(local["decision_makers_options"] == ORACLE_DMO)[0][0])
        local["decision_makers_option_value"][idx] = np.asarray(x, dtype=float)
        out = Evaluate(local).evaluate_selected_scenario(scenario)[ORACLE_DMO]
        return np.array([float(out["key_outputs"][k]) for k in self.opt.input_dict["key_outputs"]])

    def affine_map(self, scenario: str):
        """
        Reconstruct each key output as ``KO(x) = base + A x``, by probing the
        real pipeline rather than re-deriving the algebra from the case tables.

        Deriving it by hand is what went wrong first: the dependency graph adds a
        constant baseline to a key output that a coefficient matrix alone does
        not carry, and because a constant does not move the argmax, neither the
        gradient cross-check nor the optimum itself reveals the omission. Probing
        the pipeline at zero spend and at each single-lever corner recovers the
        intercept and the slopes exactly, and cannot drift from the pipeline
        because it *is* the pipeline.
        :param scenario: the scenario to reconstruct in
        :return: the intercept vector and the slope matrix
        """
        base = self._key_outputs(np.zeros(self.k), scenario)
        slopes = np.zeros((self.n_key_outputs, self.k))
        for i in range(self.k):
            probe = np.zeros(self.k)
            probe[i] = self.budget
            slopes[:, i] = (self._key_outputs(probe, scenario) - base) / self.budget
        return base, slopes

    def check_affinity(self, scenario: str, base, slopes) -> float:
        """
        Verify the reconstruction on interior points, since the whole model rests
        on the key outputs being affine on the feasible set.
        :param scenario: the scenario to check
        :param base: the intercept vector
        :param slopes: the slope matrix
        :return: the largest absolute deviation found
        """
        rng = np.random.default_rng(int(self.manifest["seed"]))
        worst = 0.0
        for _ in range(AFFINITY_PROBE_POINTS):
            weights = rng.dirichlet(np.ones(self.k + 1))[: self.k]
            x = weights * self.budget
            worst = max(worst, float(np.abs(self._key_outputs(x, scenario) - (base + slopes @ x)).max()))
        if worst > AFFINITY_TOL:
            raise ValueError(f"{self.name}/{scenario}: key outputs are not affine (max deviation {worst:.2e})")
        return worst

    def _appreciation_expr(self, model: Model, core, j: int):
        """
        Express one key output's appreciation, mirroring
        ``Appreciate._appreciate_single_key_output`` term for term.

        The interior branches are the only ones modelled. The generator freezes
        each boundary at the exact feasible envelope of the capped simplex and
        the validator enforces it (the NO-CLIP invariant), so the normalised
        core cannot leave [0, 1] and the saturating branches are unreachable.
        The core variable carries those bounds, and ``solve`` verifies the whole
        model against the real pipeline afterwards rather than trusting this.
        :param model: the SCIP model
        :param core: the normalised key-output variable
        :param j: key-output index
        :return: an expression in appreciation points
        """
        if self.is_linear[j]:
            return 100.0 * core if self.stb[j] == 0 else 100.0 * (1.0 - core)
        curved = scip_sin(0.5 * math.pi * core)
        if self.stb[j] == 0:
            return 100.0 * curved
        # STB = 1 mirrors the curve: (-sin(...) + 1) * 100
        aux = model.addVar(lb=-1.0, ub=1.0, name=f"sin_{j}")
        model.addCons(aux == curved)
        return 100.0 * (1.0 - aux)

    def build_model(self, scenario: str):
        """
        Build the SCIP model of one scenario.
        :param scenario: the scenario to model
        :return: the model and its allocation variables
        """
        base, slopes = self.affine_map(scenario)
        self.check_affinity(scenario, base, slopes)

        model = Model(f"{self.name}|{scenario}")
        model.hideOutput()

        x = [model.addVar(lb=0.0, ub=self.budget, name=f"x_{i}") for i in range(self.k)]
        model.addCons(quicksum(x) <= self.budget)

        terms = []
        for j in range(self.n_key_outputs):
            span = float(self.end[j] - self.start[j])
            if span <= 0:
                # A degenerate key output is indifferent; the pipeline returns 0.
                continue
            key_output = float(base[j]) + quicksum(float(slopes[j, i]) * x[i] for i in range(self.k))
            core = model.addVar(lb=0.0, ub=1.0, name=f"core_{j}")
            model.addCons(core == (key_output - float(self.start[j])) / span)
            terms.append(float(self.weights[j]) * self._appreciation_expr(model, core, j))

        # SCIP takes only a linear objective, so the (possibly nonlinear) sum is
        # tied to an auxiliary variable and that variable is maximised. Its
        # bounds are safe: appreciation is on 0-100 and the key-output weights
        # sum to one.
        total = model.addVar(lb=-1e4, ub=1e4, name="appreciation")
        model.addCons(total == quicksum(terms))
        model.setObjective(total, "maximize")
        model.setParam("limits/gap", SCIP_GAP_LIMIT)
        model.setParam("limits/time", SCIP_TIME_LIMIT_S)
        return model, x

    def solve(self, scenario: str) -> dict:
        """
        Solve one scenario to proven global optimality and verify the model.

        The verification is the point of the whole exercise: SCIP's objective is
        recomputed by running its own solution back through the real
        ``evaluate_allocation`` pipeline. If the two disagree the algebraic model
        is not the case, and the comparison would be meaningless however
        confident SCIP is.
        :param scenario: the scenario to solve
        :return: the result row
        """
        model, x = self.build_model(scenario)
        started = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - started

        status = model.getStatus()
        solved = status == "optimal" and model.getNSols() > 0
        allocation = [float(model.getVal(v)) for v in x] if solved else None
        objective = float(model.getObjVal()) if solved else None

        pipeline_value = None
        fidelity = None
        if allocation is not None:
            # Strip solver dust before the pipeline sees it; SCIP satisfies the
            # budget row to its feasibility tolerance, not exactly.
            arr = np.clip(np.asarray(allocation, dtype=float), 0.0, None)
            if arr.sum() > self.budget:
                arr = arr * (self.budget / arr.sum())
            allocation = [float(v) for v in arr]
            pipeline_value = float(evaluate_allocation(self.opt.input_dict, arr, scenario, ORACLE_DMO))
            fidelity = abs(pipeline_value - objective)

        certified = self.manifest.get("oracle", {}) or {}
        f_star = None
        if certified.get("per_scenario", {}).get(scenario):
            f_star = float(certified["per_scenario"][scenario]["f_star"])

        return {
            "case_name": self.name,
            "scenario": scenario,
            "k": self.k,
            "variant": self.manifest["appreciation"],
            "regime": self.manifest["regime"],
            "case_seed": int(self.manifest["seed"]),
            "status": status,
            "proved_optimal": bool(solved and model.getGap() <= 1e-9),
            "scip_gap": float(model.getGap()) if model.getNSols() > 0 else None,
            "scip_objective": objective,
            "pipeline_value_at_scip_x": pipeline_value,
            "model_fidelity_abs_diff": fidelity,
            "model_faithful": bool(fidelity is not None and fidelity <= FIDELITY_TOL),
            "f_oracle": f_star,
            "gap_vs_oracle": None if (f_star is None or pipeline_value is None) else f_star - pipeline_value,
            "allocation": allocation,
            "wall_time_s": elapsed,
            "n_nodes": int(model.getNNodes()),
        }


def study_cases(root: Path = STUDY_ROOT) -> list:
    """
    The confirmatory grid's case directories, in a deterministic order.
    :param root: the study directory
    :return: case names
    """
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir() and p.name.startswith("Study_"))


def run(cases: list, root: Path = STUDY_ROOT, out_name: str = "scip_results.jsonl") -> Path:
    """
    Solve every scenario of every listed case and append the rows to a store.
    :param cases: case names to run
    :param root: the study directory
    :param out_name: results file name
    :return: the path written
    """
    out = Path(root) / out_name
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done.add((row["case_name"], row["scenario"]))

    with open(out, "a", encoding="utf-8") as handle:
        for index, name in enumerate(cases, start=1):
            bench = ScipBenchmark(name, root)
            for scenario in bench.scenarios:
                if (name, scenario) in done:
                    continue
                row = bench.solve(scenario)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            if index % 25 == 0 or index == len(cases):
                print(f"  {index}/{len(cases)} cases", flush=True)
    return out


def main():
    """Run the SCIP benchmark over the pre-registered grid, or a smoke subset."""
    cases = study_cases()
    if "--smoke" in sys.argv:
        cases = [c for c in cases if c.endswith("_s0") and ("k02" in c or "k06" in c or "k15" in c)]
        print(f"smoke: {len(cases)} cases")
    print(f"[scip] {len(cases)} cases")
    path = run(cases)
    print(f"[scip] complete -> {path}")


if __name__ == "__main__":
    main()
