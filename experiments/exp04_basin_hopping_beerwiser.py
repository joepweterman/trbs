"""exp04: solver-stall diagnosis and the SLSQP vs basin-hopping benchmark on Beerwiser.

Reproduces the 2026-07-06 findings (knowledge base: exp04-solver-stall-and-basin-hopping.md):

1. STALL REGRESSION: two silent failure modes made every earlier continuous
   result on Beerwiser a best-of-random-starts, not an optimization:
   (a) int64 truncation: Beerwiser's DMO values import as int64, so assigning a
       float allocation truncated it and finite-difference perturbations
       flattened to zero;
   (b) budget-scale stall: on raw budgets of 1e5-1e7, SLSQP's identity initial
       Hessian makes the first trial step microscopic and the improvement test
       aborts at the start point.
   Fixed by a float cast in evaluate_allocation and by solving in
   budget-normalised z-space (x = B*z on the unit capped simplex).

2. KINK CHARACTERISATION: Beerwiser's global optimum [25000, 275000]
   (appreciation 65.711590) sits at a nonsmooth clip point: 'Accidents
   reduction' is floor-clipped (rel -0.125) and 'Water use reduction' is
   cap-clipped (rel 1.095) at the optimum. A real-case instance of the exp03
   floor/cap lemma. The landscape has two basins on the budget face: the clip
   kink at x1=25000 (global) and a smooth interior optimum at x1~123000
   (65.7014), only 0.0104 appreciation points apart.

3. METHOD COMPARISON: post-fix, both multi-start SLSQP (100 starts) and
   basin-hopping (25 hops) recover the global kink 12/12 seeds, matching the
   grid value at ~3.3% and ~1.2% of grid's 60,000 evaluations respectively.
"""

# pylint: disable=protected-access  # experiment harness drives Optimize internals directly

import copy
import json
from pathlib import Path

import numpy as np

from vlinder.evaluate import Evaluate
from vlinder.optimize import Optimize, evaluate_allocation
from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.utils import suppress_print


class Exp04BasinHoppingBeerwiser:
    """Re-runnable exp04 harness (class-based per Vlinder house style)."""

    BUDGET = 300000.0
    GRID_OPTIMUM = 65.7115899862911  # grid baseline at [25000, 275000] (exp01)
    KINK_BASIN_MAX_X1 = 60000.0  # allocations left of the inter-basin valley (~50k)

    def __init__(self, n_seeds=12, out_path="experiments/out/exp04_basin_hopping.json"):
        self.n_seeds = n_seeds
        self.out_path = Path(out_path)
        self.case = self._build_case()
        self.results = {}

    @staticmethod
    @suppress_print
    def _build_case():
        case = TheResponsibleBusinessSimulator("Beerwiser")
        case.build()
        case.evaluate()
        case.appreciate()
        return case

    def _fresh_optimizer(self):
        return Optimize(copy.deepcopy(self.case.input_dict), copy.deepcopy(self.case.output_dict))

    def run_stall_regression(self):
        """From [60000, 240000] a working solver must move and improve (>0.5 pts)."""
        opt = self._fresh_optimizer()
        opt._prepare_input_dict("EXP04", opt.input_dict["decision_makers_option_value"][0].copy())
        x0 = np.array([60000.0, 240000.0])
        start_app = evaluate_allocation(opt.input_dict, x0, "Base case", "EXP04")
        eval_counter = [0]
        res = opt._slsqp_from_start(x0, "Base case", "EXP04", self.BUDGET, eval_counter)
        record = {
            "start_x": x0.tolist(),
            "start_appreciation": start_app,
            "final_x": np.asarray(res.x).tolist(),
            "final_appreciation": float(-res.fun),
            "n_evals": eval_counter[0],
            "moved_units": float(abs(res.x[0] - x0[0])),
            "stalled": bool(abs(res.x[0] - x0[0]) < 1000),
        }
        self.results["stall_regression"] = record
        print(
            f"[stall] start {start_app:.4f} -> final {record['final_appreciation']:.4f} "
            f"at x1={res.x[0]:.0f} ({record['n_evals']} evals, stalled={record['stalled']})"
        )

    @staticmethod
    def _clip_status(input_dict, dmo_name, x_opt):
        """Per-KO position in the frozen appreciation bracket at allocation ``x_opt``."""
        local = copy.deepcopy(input_dict)
        local["decision_makers_option_value"] = np.asarray(local["decision_makers_option_value"], dtype=float)
        idx = np.where(local["decision_makers_options"] == dmo_name)[0][0]
        local["decision_makers_option_value"][idx] = np.asarray(x_opt, dtype=float)
        out = Evaluate(local).evaluate_selected_scenario("Base case")[dmo_name]
        start, end = local["key_output_start"], local["key_output_end"]
        rows = []
        for i, key_output in enumerate(local["key_outputs"]):
            val = out["key_outputs"][key_output] if "key_outputs" in out else out[key_output]
            rel = (val - start[i]) / (end[i] - start[i]) if end[i] != start[i] else float("nan")
            rows.append({"key_output": str(key_output), "rel_position": float(rel)})
        return rows

    def run_kink_characterisation(self):
        """Face scan around the kink + KO clip status at the optimum."""
        opt = self._fresh_optimizer()
        opt._prepare_input_dict("EXP04", opt.input_dict["decision_makers_option_value"][0].copy())

        scan = []
        for x1 in range(20000, 32001, 1000):
            x = np.array([float(x1), self.BUDGET - x1])
            scan.append({"x1": x1, "appreciation": evaluate_allocation(opt.input_dict, x, "Base case", "EXP04")})

        clip_status = self._clip_status(opt.input_dict, "EXP04", [25000.0, 275000.0])
        self.results["kink"] = {"face_scan": scan, "clip_status_at_optimum": clip_status}
        print("[kink] clip status at [25000, 275000]:")
        for row in clip_status:
            flag = (
                " (floor-clipped)"
                if row["rel_position"] <= 0
                else " (cap-clipped)" if row["rel_position"] >= 1 else ""
            )
            print(f"        {row['key_output'][:40]:<40} rel={row['rel_position']:+.4f}{flag}")

    def run_method_comparison(self):
        """n_seeds x {multi-start SLSQP, basin-hopping}: basin identity + cost."""
        rows = []
        for seed in range(self.n_seeds):
            for method, kwargs in (
                ("slsqp", {"n_starts": 100}),
                ("basin_hopping", {"n_hops": 25, "n_starts": 1}),
            ):
                opt = self._fresh_optimizer()
                res = opt.run("Base case", method=method, seed=seed, dmo_name="EXP04", **kwargs)
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "appreciation": res.appreciation,
                        "allocation": np.asarray(res.allocation).tolist(),
                        "basin": "kink" if res.allocation[0] < self.KINK_BASIN_MAX_X1 else "interior",
                        "gap_to_grid": self.GRID_OPTIMUM - res.appreciation,
                        "n_function_evals": res.n_function_evals,
                        "wall_time_s": res.wall_time_s,
                    }
                )
        self.results["comparison"] = rows
        for method in ("slsqp", "basin_hopping"):
            sub = [r for r in rows if r["method"] == method]
            recovered = sum(1 for r in sub if r["basin"] == "kink")
            med_evals = int(np.median([r["n_function_evals"] for r in sub]))
            print(
                f"[compare] {method:<13} kink-basin recovery {recovered}/{len(sub)}, "
                f"median evals {med_evals} (grid: 60000)"
            )

    def save(self):
        """Write all recorded results to the output JSON."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as fh:
            json.dump(self.results, fh, indent=2)
        print(f"[saved] {self.out_path}")

    def run(self):
        """Run the full experiment: stall regression, kink probe, method comparison."""
        self.run_stall_regression()
        self.run_kink_characterisation()
        self.run_method_comparison()
        self.save()


if __name__ == "__main__":
    Exp04BasinHoppingBeerwiser().run()
