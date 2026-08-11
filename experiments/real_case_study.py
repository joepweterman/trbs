"""
Real-case study: external validity of the method comparison, plus the
decision-support outputs the thesis proposal promises (shadow prices and
scenario-robustness bands).

Where the synthetic study measures recovery against a *certified* optimum, the
bundled cases have no ground truth. Everything here is therefore reported
against a best-known reference (the best value any method attained on that
case and scenario), which is explicitly not a certificate: a shared miss by all
four methods would be invisible. That is the price of external validity and it
is stated rather than papered over.

This study is descriptive. It tests none of the pre-registered hypotheses
H1-H5, so it needs no amendment to the locked pre-registration.

Run:
    python real_case_study.py

Outputs (under ``experiments/out/real_cases/``):
    benchmark.csv          one row per case x scenario x method
    shadow_prices.csv      budget shadow price per case x scenario
    robustness.csv         cross-scenario regret matrix
    RESULTS.md             human-readable summary
    tables/*.tex           booktabs tables
    figures/*.png|pdf      figures
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vlinder as vl
from vlinder.optimize import BaseSolver, evaluate_allocation
from vlinder.trbs import TheResponsibleBusinessSimulator

# Headless rendering; switching after import keeps every import at the top.
plt.switch_backend("Agg")

DATA = Path(os.path.dirname(vl.__file__)) / "data"
OUT = Path(__file__).resolve().parent / "out" / "real_cases"
CASES = ["Beerwiser", "Refugee", "IZZ"]
METHODS = ["grid", "slsqp", "basin_hopping", "genetic_algorithm"]
METHOD_LABELS = {
    "grid": "Grid (baseline)",
    "slsqp": "SLSQP multi-start",
    "basin_hopping": "Basin-hopping",
    "genetic_algorithm": "Genetic algorithm",
}
METHOD_COLORS = {
    "grid": "#B00020",
    "slsqp": "#0B5FA5",
    "basin_hopping": "#127A52",
    "genetic_algorithm": "#C46A00",
}
BUDGET_DELTA = 0.05
SHADOW_METHOD = "slsqp"
SEED = 0


def build_case(name: str) -> TheResponsibleBusinessSimulator:
    """
    Build one bundled case through the unmodified pipeline.

    A fresh build per run is required, not merely tidy: the grid path freezes the
    appreciation boundaries and clears ``key_output_automatic`` on the dictionary
    it runs on, so reusing one instance across methods would silently change the
    objective that later methods are scored on.
    :param name: bundled case name
    :return: a built, evaluated and appreciated simulator
    """
    sim = TheResponsibleBusinessSimulator(name, file_path=DATA, file_extension="csv")
    with contextlib.redirect_stdout(io.StringIO()):
        sim.build()
        sim.evaluate()
        sim.appreciate()
    return sim


class RealCaseStudy:
    """
    Runs every method on the bundled cases and derives the decision-support
    quantities (shadow prices, scenario-robustness bands) from the results.

    :param out: directory the artefacts are written to
    """

    def __init__(self, out: Path = OUT):
        self.out = Path(out)
        (self.out / "tables").mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        self.benchmark = pd.DataFrame()
        self.shadow = pd.DataFrame()
        self.robustness = pd.DataFrame()
        self.multimodality = pd.DataFrame()

    def load_existing(self):
        """
        Reload previously computed tables so figures and the report can be
        rebuilt without repeating the compute (the baseline alone is minutes per
        cell).
        """
        self.benchmark = pd.read_csv(self.out / "benchmark.csv")
        self.shadow = pd.read_csv(self.out / "shadow_prices.csv")
        self.robustness = pd.read_csv(self.out / "robustness.csv")
        path = self.out / "multimodality.csv"
        if path.exists():
            self.multimodality = pd.read_csv(path)

    # ------------------------------------------------------------------
    @staticmethod
    def case_facts(name: str) -> dict:
        """
        Read a case's structural facts once.
        :param name: bundled case name
        :return: levers, scenarios and budget
        """
        sim = build_case(name)
        data = sim.input_dict
        return {
            "case": name,
            "k": len(data["internal_variable_inputs"]),
            "levers": [str(v) for v in data["internal_variable_inputs"]],
            "scenarios": [str(s) for s in data["scenarios"]],
            "budget": float(sum(data["decision_makers_option_value"][0])),
            "n_key_outputs": len(data["key_outputs"]),
        }

    @staticmethod
    def solve(name: str, scenario: str, method: str, budget: float = None) -> dict:
        """
        Run one method on one case and scenario through the real pipeline.
        :param name: bundled case name
        :param scenario: scenario name
        :param method: registered method name
        :param budget: explicit budget; the case's own budget when None
        :return: the result row
        """
        sim = build_case(name)
        kwargs = {"seed": SEED} if method != "grid" else {}
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            sim.optimize(scenario, method=method, budget=budget, **kwargs)
        result = sim.optimization_result
        elapsed = time.perf_counter() - started
        allocation = np.asarray(result.allocation, dtype=float)
        effective_budget = (
            budget if budget is not None else float(sum(sim.input_dict["decision_makers_option_value"][0]))
        )
        return {
            "case": name,
            "scenario": scenario,
            "method": method,
            "appreciation": float(result.appreciation),
            "allocation": [float(v) for v in allocation],
            "spend": float(allocation.sum()),
            "spend_fraction": float(allocation.sum() / effective_budget) if effective_budget else float("nan"),
            "n_function_evals": result.n_function_evals,
            "wall_time_s": elapsed,
            "budget": effective_budget,
        }

    def run_benchmark(self) -> pd.DataFrame:
        """
        Run every method on every case and scenario, and score each against the
        best value any method reached on that cell.
        :return: the benchmark table
        """
        rows = []
        for name in CASES:
            facts = self.case_facts(name)
            for scenario in facts["scenarios"]:
                for method in METHODS:
                    row = self.solve(name, scenario, method)
                    row["k"] = facts["k"]
                    rows.append(row)
                    print(
                        f"  {name:10s} {scenario:20s} {method:18s} "
                        f"f={row['appreciation']:8.4f} t={row['wall_time_s']:7.2f}s"
                    )
        frame = pd.DataFrame(rows)
        best = frame.groupby(["case", "scenario"]).appreciation.transform("max")
        frame["best_known"] = best
        frame["gap_vs_best_known"] = best - frame.appreciation
        frame["is_best"] = frame.gap_vs_best_known <= 1e-9
        self.benchmark = frame
        return frame

    # ------------------------------------------------------------------
    def run_shadow_prices(self) -> pd.DataFrame:
        """
        Estimate the budget shadow price: the appreciation gained per extra unit
        of capital, i.e. the multiplier on the capped-simplex budget constraint.

        It is estimated by re-solving at a perturbed budget rather than read off
        the solver, because the reported multiplier of a constraint that is not
        active is zero and would hide the distinction this study is about: where
        the constraint is slack, extra capital is genuinely worthless, and the
        one-sided difference shows that directly.
        :return: the shadow-price table
        """
        rows = []
        for name in CASES:
            facts = self.case_facts(name)
            budget = facts["budget"]
            for scenario in facts["scenarios"]:
                base = self.solve(name, scenario, SHADOW_METHOD)
                up = self.solve(name, scenario, SHADOW_METHOD, budget=budget * (1 + BUDGET_DELTA))
                down = self.solve(name, scenario, SHADOW_METHOD, budget=budget * (1 - BUDGET_DELTA))
                d_up = (up["appreciation"] - base["appreciation"]) / (budget * BUDGET_DELTA)
                d_down = (base["appreciation"] - down["appreciation"]) / (budget * BUDGET_DELTA)
                rows.append(
                    {
                        "case": name,
                        "scenario": scenario,
                        "budget": budget,
                        "appreciation": base["appreciation"],
                        "spend_fraction": base["spend_fraction"],
                        "binding": bool(base["spend_fraction"] > 0.999),
                        "shadow_price_up": d_up,
                        "shadow_price_down": d_down,
                        "points_per_1pct_budget_up": d_up * budget / 100.0,
                        "points_per_1pct_budget_down": d_down * budget / 100.0,
                        "appreciation_at_minus_5pct": down["appreciation"],
                        "appreciation_at_plus_5pct": up["appreciation"],
                    }
                )
                print(
                    f"  {name:10s} {scenario:20s} binding={rows[-1]['binding']!s:5s} "
                    f"+1% budget -> {rows[-1]['points_per_1pct_budget_up']:+.4f} pts"
                )
        self.shadow = pd.DataFrame(rows)
        return self.shadow

    def _cross_evaluate_case(self, name: str) -> list:
        """
        Score every scenario's winning allocation under every scenario of one case.
        :param name: bundled case name
        :return: the cross-evaluation rows for this case
        """
        scenarios = self.case_facts(name)["scenarios"]
        best_alloc, best_value = {}, {}
        for scenario in scenarios:
            winner = (
                self.benchmark[(self.benchmark.case == name) & (self.benchmark.scenario == scenario)]
                .sort_values("appreciation", ascending=False)
                .iloc[0]
            )
            best_alloc[scenario] = np.asarray(winner.allocation, dtype=float)
            best_value[scenario] = float(winner.appreciation)

        # One prepared dictionary per case, reused read-only across the
        # cross-evaluations so every cell is scored on identical boundaries.
        sim = build_case(name)
        optimizer = BaseSolver(sim.input_dict, sim.output_dict)
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer._prepare_input_dict(  # pylint: disable=protected-access
                "CrossEval", sim.input_dict["decision_makers_option_value"][0].copy()
            )
        rows = []
        for designed_for in scenarios:
            for realised in scenarios:
                value = float(
                    evaluate_allocation(optimizer.input_dict, best_alloc[designed_for], realised, "CrossEval")
                )
                rows.append(
                    {
                        "case": name,
                        "designed_for": designed_for,
                        "realised": realised,
                        "appreciation": value,
                        "scenario_best": best_value[realised],
                        "regret": best_value[realised] - value,
                    }
                )
        return rows

    # ------------------------------------------------------------------
    def run_robustness(self) -> pd.DataFrame:
        """
        Cross-evaluate each scenario's optimal allocation under every scenario.

        The diagonal is the scenario-specific optimum; off-diagonal entries are
        what that allocation delivers if a different scenario materialises. The
        regret column is the cost of having optimised for the wrong world, which
        is the robustness band the proposal promises.
        :return: the cross-scenario table
        """
        rows = []
        for name in CASES:
            rows.extend(self._cross_evaluate_case(name))
        frame = pd.DataFrame(rows)
        worst = frame.groupby(["case", "designed_for"]).regret.transform("max")
        frame["max_regret_of_allocation"] = worst
        self.robustness = frame
        return frame

    # ------------------------------------------------------------------
    def probe_multimodality(self, n_seeds: int = 12) -> pd.DataFrame:
        """
        Probe how multimodal each case's primary scenario is, by running SLSQP
        from a single random start across many seeds and counting where it stops.

        This exists because the benchmark table is otherwise misleading. The
        roster fixes basin-hopping at one start (pre-registration section 2), so
        a single unlucky seed can leave it far below the other methods on one
        cell and read as a broken method. The probe separates the two
        explanations: if single-start runs cluster at several distinct values,
        the case really is multimodal and the multi-start budget is what buys
        the difference.
        :param n_seeds: number of independent single-start runs per case
        :return: one row per case x seed
        """
        rows = []
        for name in CASES:
            scenario = self.case_facts(name)["scenarios"][0]
            for seed in range(n_seeds):
                sim = build_case(name)
                with contextlib.redirect_stdout(io.StringIO()):
                    sim.optimize(scenario, method="slsqp", seed=seed, n_starts=1)
                result = sim.optimization_result
                rows.append(
                    {
                        "case": name,
                        "scenario": scenario,
                        "seed": seed,
                        "appreciation": float(result.appreciation),
                    }
                )
            values = [r["appreciation"] for r in rows if r["case"] == name]
            print(f"  {name:10s} single-start spread: {min(values):.4f} .. {max(values):.4f}")
        frame = pd.DataFrame(rows)
        # Cluster terminal values at a tolerance far above solver noise but far
        # below a meaningful difference in appreciation.
        frame["basin"] = frame.groupby("case").appreciation.transform(lambda s: s.round(3).rank(method="dense"))
        self.multimodality = frame
        return frame

    # ------------------------------------------------------------------
    def figure_quality_cost(self):
        """Appreciation reached against time spent, per case."""
        frame = self.benchmark
        cases = list(frame.case.unique())
        fig, axes = plt.subplots(1, len(cases), figsize=(3.3 * len(cases), 3.2), layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, cases):
            sub = frame[frame.case == name]
            for method in METHODS:
                cell = sub[sub.method == method]
                if cell.empty:
                    continue
                ax.scatter(
                    cell.wall_time_s,
                    cell.gap_vs_best_known,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    s=26,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                )
            ax.set_xscale("log")
            # Below ~1e-4 points the differences are solver noise, not quality;
            # a finer threshold would spend most of the axis resolving it.
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_ylim(bottom=0.0)
            ax.set_title(f"{name} ($k$={int(sub.k.iloc[0])})")
            ax.set_xlabel("wall-clock time (s)")
        axes[0].set_ylabel("gap vs best known (points)")
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=METHOD_COLORS[m], label=METHOD_LABELS[m])
            for m in METHODS
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
        self._save(fig, "fig4_real_case_quality_vs_cost")

    def figure_robustness(self):
        """Regret heatmaps: allocation designed for one scenario, realised in another."""
        cases = list(self.robustness.case.unique())
        # Each case gets its own colour scale (the regret ranges differ by an
        # order of magnitude), so the panels need room for a colourbar each.
        fig, axes = plt.subplots(1, len(cases), figsize=(3.9 * len(cases), 3.1), layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, cases):
            sub = self.robustness[self.robustness.case == name]
            pivot = sub.pivot(index="designed_for", columns="realised", values="regret")
            image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", vmin=0.0)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([c[:12] for c in pivot.columns], rotation=35, ha="right", fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([c[:12] for c in pivot.index], fontsize=7)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.to_numpy()[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > pivot.to_numpy().max() * 0.6 else "0.1",
                    )
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("scenario realised", fontsize=8)
            ax.grid(False)
            fig.colorbar(image, ax=ax, fraction=0.046, label="regret (points)")
        axes[0].set_ylabel("allocation optimised for", fontsize=8)
        self._save(fig, "fig5_scenario_regret")

    def _save(self, fig, stem: str):
        """Write a figure to png and pdf, guarding against a collapsed layout."""
        width, height = fig.get_size_inches()
        if not (0.5 < width < 20 and 0.5 < height < 20):
            raise ValueError(f"{stem}: implausible figure size {width:.1f}x{height:.1f} in")
        for suffix in ("png", "pdf"):
            fig.savefig(self.out / "figures" / f"{stem}.{suffix}", bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    def table_benchmark(self) -> str:
        """Booktabs table of the per-case method comparison."""
        lines = [
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Case & Method & Appreciation & Gap vs best & Time (s) & Spend \\",
            r"\midrule",
        ]
        for name in CASES:
            sub = self.benchmark[self.benchmark.case == name]
            for i, method in enumerate(METHODS):
                cell = sub[sub.method == method]
                if cell.empty:
                    continue
                head = name if i == 0 else ""
                lines.append(
                    f"{head} & {METHOD_LABELS[method]} & {cell.appreciation.mean():.4f} & "
                    f"{cell.gap_vs_best_known.mean():.2e} & {cell.wall_time_s.mean():.2f} & "
                    f"{cell.spend_fraction.mean():.3f} \\\\"
                )
            lines.append(r"\addlinespace")
        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def table_shadow(self) -> str:
        """Booktabs table of budget shadow prices."""
        lines = [
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Case & Scenario & Spend & Binding & Points per +1\% budget \\",
            r"\midrule",
        ]
        for row in self.shadow.itertuples():
            lines.append(
                f"{row.case} & {row.scenario} & {row.spend_fraction:.3f} & "
                f"{'yes' if row.binding else 'no'} & {row.points_per_1pct_budget_up:+.4f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _md_benchmark(self) -> list:
        """The provenance block, the win count and the quality/cost table."""
        wins = self.benchmark[self.benchmark.is_best].groupby("method", observed=True).size()
        cells = self.benchmark.groupby(["case", "scenario"]).ngroups
        out = [
            "# Real-case study (external validity)",
            "",
            "Generated by `real_case_study.py`; do not edit by hand.",
            "",
            "> **Reference caveat.** The bundled cases have no certified optimum, so every gap here is measured",
            "> against the best value *any* of the four methods reached on that cell. A miss shared by all four",
            "> would be invisible. This is a descriptive study and tests none of the pre-registered hypotheses.",
            "",
            f"- Cases: {', '.join(CASES)}",
            f"- Cells: {cells} (case x scenario), {len(self.benchmark)} runs",
            "",
            "## Which method reached the best known value",
            "",
            "| Method | Cells won | of |",
            "|---|---|---|",
        ]
        for method in METHODS:
            out.append(f"| {METHOD_LABELS[method]} | {int(wins.get(method, 0))} | {cells} |")

        out += [
            "",
            "## Quality and cost by case",
            "",
            "| Case | k | Method | Mean appreciation | Mean gap vs best | Mean time (s) |",
            "|---|---|---|---|---|---|",
        ]
        for name in CASES:
            sub = self.benchmark[self.benchmark.case == name]
            for method in METHODS:
                cell = sub[sub.method == method]
                if cell.empty:
                    continue
                out.append(
                    f"| {name} | {int(cell.k.iloc[0])} | {METHOD_LABELS[method]} | {cell.appreciation.mean():.4f} | "
                    f"{cell.gap_vs_best_known.mean():.2e} | {cell.wall_time_s.mean():.2f} |"
                )
        return out

    def _md_shadow_and_robustness(self) -> list:
        """The shadow-price table and the scenario-regret tables."""
        out = [
            "",
            "## Budget shadow prices",
            "",
            "The value of an extra unit of capital, estimated by re-solving at a perturbed budget.",
            "",
            "| Case | Scenario | Spend fraction | Constraint binds | Points per +1% budget | Points per -1% budget |",
            "|---|---|---|---|---|---|",
        ]
        for row in self.shadow.itertuples():
            out.append(
                f"| {row.case} | {row.scenario} | {row.spend_fraction:.3f} | {'yes' if row.binding else 'no'} | "
                f"{row.points_per_1pct_budget_up:+.4f} | {row.points_per_1pct_budget_down:+.4f} |"
            )

        out += [
            "",
            "## Scenario robustness",
            "",
            "Maximum regret of each scenario-specific allocation, i.e. the most appreciation it gives up if a",
            "different scenario turns out to be the real one.",
            "",
            "| Case | Allocation optimised for | Max regret (points) |",
            "|---|---|---|",
        ]
        worst = self.robustness.groupby(["case", "designed_for"]).regret.max().reset_index()
        for row in worst.itertuples():
            out.append(f"| {row.case} | {row.designed_for} | {row.regret:.4f} |")

        minimax = worst.loc[worst.groupby("case").regret.idxmin()]
        out += [
            "",
            "The allocation with the smallest worst case, per case:",
            "",
            "| Case | Minimax-regret allocation | Max regret |",
            "|---|---|---|",
        ]
        for row in minimax.itertuples():
            out.append(f"| {row.case} | optimised for *{row.designed_for}* | {row.regret:.4f} |")
        return out

    def _md_multimodality(self) -> list:
        """The single-start probe and what it says about the reference value."""
        out = []
        if not self.multimodality.empty:
            out += [
                "",
                "## How multimodal are the real cases?",
                "",
                "Single-start SLSQP from independent random starts on each case's first scenario. The spread is",
                "what the multi-start budget is buying.",
                "",
                "| Case | Distinct optima | Single-start best | Single-start worst | Best known | Runs missing it |",
                "|---|---|---|---|---|---|",
            ]
            for name in CASES:
                sub = self.multimodality[self.multimodality.case == name]
                if sub.empty:
                    continue
                scenario = sub.scenario.iloc[0]
                # Compared against the multi-start benchmark, never against the
                # probe's own maximum: on a case where every single start lands
                # in the same secondary basin, a self-referential comparison
                # would report it as unimodal and hide the miss entirely.
                cell = self.benchmark[(self.benchmark.case == name) & (self.benchmark.scenario == scenario)]
                best_known = float(cell.appreciation.max())
                missing = int((sub.appreciation < best_known - 1e-6).sum())
                # Six decimals: at four, a probe value a whisker below the
                # reference prints as identical to it and the count of runs that
                # miss it reads as a contradiction.
                out.append(
                    f"| {name} | {int(sub.basin.nunique())} | {sub.appreciation.max():.6f} | "
                    f"{sub.appreciation.min():.6f} | {best_known:.6f} | {missing} of {len(sub)} |"
                )
            out += [
                "",
                "> This is why the benchmark table should not be read as a ranking of algorithm quality alone. The",
                "> roster fixes basin-hopping at a single start, so on a case with a strong secondary optimum one",
                "> unlucky seed drops it far below the rest. That is a property of the configured start budget, not",
                "> evidence that the algorithm is weaker.",
                "",
                "> Beerwiser is the sharpest illustration: every single-start run converges to the *same* value, so",
                "> the case looks unimodal from the inside, yet all of them sit below what the 100-start roster",
                "> finds. The two basins are about 0.01 points apart, which is the gap exp04 documented, and no",
                "> amount of restarting from one start would reveal it.",
            ]
            beaten = []
            for name in CASES:
                sub = self.multimodality[self.multimodality.case == name]
                if sub.empty:
                    continue
                cell = self.benchmark[
                    (self.benchmark.case == name) & (self.benchmark.scenario == sub.scenario.iloc[0])
                ]
                margin = float(sub.appreciation.max()) - float(cell.appreciation.max())
                if margin > 1e-6:
                    beaten.append((name, margin))
            if beaten:
                detail = "; ".join(f"{name} by {margin:.2e} points" for name, margin in beaten)
                out += [
                    "",
                    f"> **The best-known reference is not tight.** A diagnostic single-start run exceeded it on: "
                    f"{detail}. The caveat at the top of this file is therefore not hypothetical on these cases:",
                    "> gaps measured against the roster's best understate the true optimality gap by at least this",
                    "> much, and no claim of optimality should be made for the bundled cases.",
                ]
        return out

    def write_report(self):
        """Write the human-readable summary of the real-case study."""
        out = self._md_benchmark() + self._md_shadow_and_robustness() + self._md_multimodality()
        out += [
            "",
            "## Files",
            "",
            "- `benchmark.csv`, `shadow_prices.csv`, `robustness.csv`, `multimodality.csv`",
            "- `figures/fig4_real_case_quality_vs_cost.(png|pdf)`",
            "- `figures/fig5_scenario_regret.(png|pdf)`",
            "- `tables/benchmark.tex`, `tables/shadow_prices.tex`",
            "",
        ]
        (self.out / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")

    def write_artefacts(self):
        """Write every table, figure and report from the current tables."""
        self.benchmark.to_csv(self.out / "benchmark.csv", index=False)
        self.shadow.to_csv(self.out / "shadow_prices.csv", index=False)
        self.robustness.to_csv(self.out / "robustness.csv", index=False)
        if not self.multimodality.empty:
            self.multimodality.to_csv(self.out / "multimodality.csv", index=False)
        (self.out / "tables" / "benchmark.tex").write_text(self.table_benchmark(), encoding="utf-8")
        (self.out / "tables" / "shadow_prices.tex").write_text(self.table_shadow(), encoding="utf-8")
        self.figure_quality_cost()
        self.figure_robustness()
        self.write_report()
        print(f"artefacts -> {self.out}")

    def run(self):
        """Execute the full real-case study and write every artefact."""
        print("[1/4] benchmarking every method on every case and scenario")
        self.run_benchmark()
        print("[2/4] estimating budget shadow prices")
        self.run_shadow_prices()
        print("[3/4] cross-evaluating scenarios")
        self.run_robustness()
        print("[4/4] probing multimodality with single-start runs")
        self.probe_multimodality()
        (self.out / "case_facts.json").write_text(
            json.dumps([self.case_facts(n) for n in CASES], indent=2), encoding="utf-8"
        )
        self.write_artefacts()


def main():
    """Run the real-case study, or rebuild its artefacts from saved tables."""
    study = RealCaseStudy()
    if "--figures-only" in sys.argv:
        study.load_existing()
        study.write_artefacts()
        return
    if "--probe-only" in sys.argv:
        study.load_existing()
        study.probe_multimodality()
        study.write_artefacts()
        return
    study.run()


if __name__ == "__main__":
    main()
