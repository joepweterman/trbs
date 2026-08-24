"""
exp05: can basin-hopping be configured better than the roster's single chain?

The roster runs basin-hopping at one start, a hundred hops, temperature 1.0 and a
0.3-budget kick, with SLSQP as the inner solve. Louise's question is whether those
choices are the right ones, and this sweep varies them one at a time:

  * restarts against chain length at an equal hop total, because the equal-footing
    evidence says restarts buy reliability where longer chains do not;
  * the Metropolis temperature, including one derived from the case itself: the
    spread of appreciations over random allocations, so acceptance is read on the
    scale the appreciation functions actually produce rather than at a fixed 1.0;
  * the kick width, because a narrow kick cannot change which options get money;
  * the inner minimizer: Powell (derivative-free, so kinks where scores clip do
    not stall it) and COBYLA next to SLSQP.

Every configuration runs on the packaged cases over several seeds and is read at
the median. Rows are appended to the CSV as they land, so an interrupted sweep
keeps what it measured.

Run:
  python experiments/exp05_basin_hopping_tuning.py
  python experiments/exp05_basin_hopping_tuning.py --cases Refugee --configs baseline "restarts 5x20"
"""

# pylint: disable=protected-access,R0913,R0917  # the harness drives solver classes directly

import argparse
import contextlib
import copy
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import vlinder as vl

from vlinder.optimize import BaseSolver, BasinHoppingSolver
from vlinder.trbs import TheResponsibleBusinessSimulator

DATA = Path(os.path.dirname(vl.__file__)) / "data"
OUT = Path(__file__).resolve().parent / "out" / "exp05_basin_hopping_tuning"
CASES = ("Beerwiser", "Refugee", "IZZ")
SEEDS = (1, 2, 3, 4, 5)


class PowellHoppingSolver(BasinHoppingSolver):  # pylint: disable=too-few-public-methods
    """Basin-hopping with Powell as the inner solve.

    Powell is derivative-free, so the clip kinks that stall a gradient step cost it
    nothing. It cannot hold a constraint, only bounds, so the objective projects every
    trial point onto the capped simplex before evaluating: outside the feasible set the
    surface continues flatly from its boundary, which Powell has no reason to climb.
    """

    method_name = "basin_hopping_powell"

    def _objective_z(self, z, scenario, dmo_name, eval_counter, budget):
        z = self._project_capped_simplex(np.asarray(z, dtype=float), 1.0)
        return super()._objective_z(z, scenario, dmo_name, eval_counter, budget)

    def _slsqp_minimizer_kwargs(
        self, scenario, dmo_name, eval_counter, budget, spend_all=False
    ):  # pylint: disable=unused-argument
        return {
            "method": "Powell",
            "bounds": [(0.0, 1.0)] * self._k,
            "args": (scenario, dmo_name, eval_counter, float(budget)),
            "options": {"maxiter": 100, "xtol": 1e-6, "ftol": 1e-6},
        }


class CobylaHoppingSolver(BasinHoppingSolver):  # pylint: disable=too-few-public-methods
    """Basin-hopping with COBYLA as the inner solve: derivative-free, constraint-aware."""

    method_name = "basin_hopping_cobyla"

    def _slsqp_minimizer_kwargs(
        self, scenario, dmo_name, eval_counter, budget, spend_all=False
    ):  # pylint: disable=unused-argument
        return {
            "method": "COBYLA",
            "bounds": [(0.0, 1.0)] * self._k,
            "constraints": ({"type": "ineq", "fun": lambda z: float(1.0 - np.sum(z))},),
            "args": (scenario, dmo_name, eval_counter, float(budget)),
            "options": {"maxiter": 200, "rhobeg": 0.3},
        }


#: Each configuration varies one thing against the roster's baseline.
CONFIGS = {
    "baseline": (BasinHoppingSolver, {"n_starts": 1, "n_hops": 100, "temperature": 1.0, "step_frac": 0.3}),
    "restarts 5x20": (BasinHoppingSolver, {"n_starts": 5, "n_hops": 20, "temperature": 1.0, "step_frac": 0.3}),
    "T=0.1": (BasinHoppingSolver, {"n_starts": 1, "n_hops": 100, "temperature": 0.1, "step_frac": 0.3}),
    "T=5": (BasinHoppingSolver, {"n_starts": 1, "n_hops": 100, "temperature": 5.0, "step_frac": 0.3}),
    "T=auto": (BasinHoppingSolver, {"n_starts": 1, "n_hops": 100, "temperature": "auto", "step_frac": 0.3}),
    "wide kick": (BasinHoppingSolver, {"n_starts": 1, "n_hops": 100, "temperature": 1.0, "step_frac": 1.0}),
    "inner=Powell": (PowellHoppingSolver, {"n_starts": 1, "n_hops": 100, "temperature": 1.0, "step_frac": 0.3}),
    "inner=COBYLA": (CobylaHoppingSolver, {"n_starts": 1, "n_hops": 100, "temperature": 1.0, "step_frac": 0.3}),
}


class TuningSweep:
    """The exp05 sweep: configurations by cases by seeds, read at the median."""

    def __init__(self, cases, config_names, seeds, out_path):
        self.cases = list(cases)
        self.configs = {name: CONFIGS[name] for name in config_names}
        self.seeds = [int(seed) for seed in seeds]
        self.out_path = Path(out_path)

    @staticmethod
    def build(case_name):
        """
        This function builds one bundled case through the unmodified pipeline.
        :param case_name: the bundled case name
        :return: a built, evaluated and appreciated simulator
        """
        sim = TheResponsibleBusinessSimulator(case_name, file_path=DATA, file_extension="csv")
        with contextlib.redirect_stdout(io.StringIO()):
            sim.build()
            sim.evaluate()
            sim.appreciate()
        return sim

    @staticmethod
    def auto_temperature(input_dict, output_dict, scenario, n_probe=32, seed=0):
        """
        This function derives a Metropolis temperature from the case itself.

        The acceptance rule compares appreciation differences to the temperature, so
        the natural scale is how much appreciation varies over the feasible set. The
        standard deviation over random allocations measures exactly that; a fixed 1.0
        is tiny on a case whose optima lie tens of points apart and enormous on one
        whose surface is nearly flat.
        :param input_dict: the case input dictionary
        :param output_dict: the case output dictionary
        :param scenario: the scenario being solved
        :param n_probe: how many random allocations to appreciate
        :param seed: the RNG seed of the probe
        :return: the derived temperature and the number of evaluations spent
        """
        from vlinder.optimize import evaluate_allocation  # pylint: disable=import-outside-toplevel

        probe = BaseSolver(copy.deepcopy(input_dict), output_dict)
        probe._prepare_input_dict("Temperature probe")
        rng = np.random.default_rng(seed)
        draws = rng.dirichlet(np.ones(probe._k), size=n_probe) * probe.budget
        appreciations = [
            evaluate_allocation(probe.input_dict, draw, scenario, "Temperature probe", probe._frozen_boundaries)
            for draw in draws
        ]
        return max(float(np.std(appreciations)), 1e-6), n_probe

    def run_once(self, case_name, scenario, solver_class, settings, seed):
        """
        This function runs one configuration once on a fresh copy of the case.
        :param case_name: the bundled case name
        :param scenario: the scenario to solve
        :param solver_class: the basin-hopping variant to run
        :param settings: the solver settings, temperature possibly still "auto"
        :param seed: the RNG seed of this run
        :return: the OptimizationResult and the settings the run actually used
        """
        sim = self.build(case_name)
        settings = dict(settings)
        extra_evals = 0
        if settings.get("temperature") == "auto":
            settings["temperature"], extra_evals = self.auto_temperature(
                sim.input_dict, sim.output_dict, scenario, seed=seed
            )
        solver = solver_class(copy.deepcopy(sim.input_dict), sim.output_dict)
        with contextlib.redirect_stdout(io.StringIO()):
            result = solver.solve(scenario, "BH tuned", solver.budget, seed=seed, **settings)
        result.n_function_evals += extra_evals
        return result, settings

    def append_record(self, record):
        """
        This function appends one finished run to the CSV.
        :param record: the row to append
        :return: None
        """
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([record]).to_csv(self.out_path, mode="a", header=not self.out_path.exists(), index=False)

    def finished(self):
        """The (case, config, seed) keys already in the CSV, so a rerun resumes."""
        if not self.out_path.exists():
            return set()
        frame = pd.read_csv(self.out_path)
        return set(zip(frame["case"], frame["config"], frame["seed"]))

    def run(self):
        """Run the sweep, appending each row as it lands, and print the median pivot."""
        finished = self.finished()
        for case_name in self.cases:
            scenario = list(self.build(case_name).input_dict["scenarios"])[0]
            for label, (solver_class, settings) in self.configs.items():
                for seed in self.seeds:
                    if (case_name, label, seed) in finished:
                        continue
                    result, used = self.run_once(case_name, scenario, solver_class, settings, seed)
                    self.append_record(
                        {
                            "case": case_name,
                            "scenario": scenario,
                            "config": label,
                            "seed": seed,
                            "appreciation": result.appreciation,
                            "seconds": result.calculation_time,
                            "evaluations": result.n_function_evals,
                            "temperature": used.get("temperature"),
                            "settings": used,
                        }
                    )
                    print(
                        f"{case_name:<10} {label:<15} seed {seed}  appr {result.appreciation:>10.6f}  "
                        f"{result.calculation_time:>6.1f}s  evals {result.n_function_evals:>7}",
                        flush=True,
                    )
        self.report()

    def report(self):
        """Print the median appreciation, cost and time per case and configuration."""
        frame = pd.read_csv(self.out_path)
        for value in ("appreciation", "evaluations", "seconds"):
            print(f"\nMedian {value} per case and configuration:")
            print(frame.pivot_table(index="config", columns="case", values=value, aggfunc="median").to_string())
        print(f"\n[exp05] written to {self.out_path}")


def main():
    """Run the sweep from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--out", default=str(OUT / "exp05_basin_hopping_tuning.csv"))
    args = parser.parse_args()

    TuningSweep(args.cases, args.configs, args.seeds, args.out).run()


if __name__ == "__main__":
    main()
