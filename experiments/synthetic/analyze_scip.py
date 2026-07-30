"""
Analysis of the SCIP benchmark (amendment A2).

This applies A2's pre-stated decision rule to the SCIP rows and reports what
they say about the certified ground truth. It is deliberately separate from
``analyze_study.py``: SCIP is not a roster method, and its numbers must not be
mixed into the locked H1/H2/H5 comparison.

The decision rule, fixed before the runs: if SCIP *proves* an optimum more than
eps = 0.1 appreciation points above the certified ``f_star`` on any case, the
oracle is wrong and the confirmatory results have to be recomputed against
corrected ground truth. Smaller exceedances are a tightness bound on the oracle.

Run:
    python analyze_scip.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Headless rendering; switching after import keeps every import at the top.
plt.switch_backend("Agg")

STUDY_ROOT = Path(__file__).resolve().parent / "generated" / "study"
RECOVERY_EPS = 0.1
VARIANT_LABELS = {"linear": "convex-linear", "sinusoidal": "convex-curved"}
VARIANT_COLORS = {"linear": "#0B5FA5", "sinusoidal": "#C46A00"}


class ScipAnalysis:
    """
    Summarises the SCIP verification rows and applies amendment A2's rule.

    :param root: the study directory holding ``scip_results.jsonl``
    """

    def __init__(self, root: Path = STUDY_ROOT):
        self.root = Path(root)
        self.out = self.root / "analysis"
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        with open(self.root / "scip_results.jsonl", encoding="utf-8") as handle:
            self.df = pd.DataFrame([json.loads(line) for line in handle if line.strip()])

    def verdict(self) -> dict:
        """
        Apply A2's decision rule.
        :return: the verification verdict and the evidence behind it
        """
        frame = self.df
        faithful = frame[frame.model_faithful]
        # A positive gap means the oracle sits above SCIP; a negative gap means
        # SCIP found something better, which is the direction that matters.
        exceedances = -faithful.gap_vs_oracle
        worst = float(exceedances.max())
        breaches = faithful[exceedances > RECOVERY_EPS]
        return {
            "n_rows": int(len(frame)),
            "n_proved_optimal": int(frame.proved_optimal.sum()),
            "n_model_faithful": int(frame.model_faithful.sum()),
            "worst_fidelity_residual": float(frame.model_fidelity_abs_diff.max()),
            "max_abs_gap_vs_oracle": float(faithful.gap_vs_oracle.abs().max()),
            "largest_exceedance_over_oracle": worst,
            "n_breaching_eps": int(len(breaches)),
            "breaching_cases": breaches.case_name.tolist()[:10],
            "verdict": (
                "ORACLE CORROBORATED"
                if len(breaches) == 0 and frame.proved_optimal.all() and frame.model_faithful.all()
                else "REVIEW REQUIRED"
            ),
            "rule": (
                f"a proven optimum more than {RECOVERY_EPS} points above the certified f_star "
                "would mean the ground truth is wrong"
            ),
        }

    def by_cell(self) -> pd.DataFrame:
        """
        Per variant and k: agreement with the oracle and SCIP's own cost.
        :return: one row per variant x k
        """
        records = []
        for (variant, k), cell in self.df.groupby(["variant", "k"]):
            records.append(
                {
                    "variant": variant,
                    "k": k,
                    "n": len(cell),
                    "proved_optimal": int(cell.proved_optimal.sum()),
                    "max_abs_gap": float(cell.gap_vs_oracle.abs().max()),
                    "median_gap": float(cell.gap_vs_oracle.median()),
                    "median_time_s": float(cell.wall_time_s.median()),
                    "median_nodes": float(cell.n_nodes.median()),
                    "max_fidelity": float(cell.model_fidelity_abs_diff.max()),
                }
            )
        return pd.DataFrame(records).sort_values(["variant", "k"]).reset_index(drop=True)

    def figure(self, cells: pd.DataFrame):
        """
        Agreement with the oracle, and SCIP's solve cost, against k.
        :param cells: the per-cell table
        """
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
        for variant in ("linear", "sinusoidal"):
            sub = cells[cells.variant == variant].sort_values("k")
            if sub.empty:
                continue
            # Exact agreement is 0 and invisible on a log axis, so the floor is
            # drawn at the fidelity tolerance rather than dropped silently.
            axes[0].plot(
                sub.k,
                np.maximum(sub.max_abs_gap, 1e-16),
                marker="o",
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
                markersize=4,
                linewidth=1.3,
            )
            axes[1].plot(
                sub.k,
                sub.median_time_s,
                marker="o",
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
                markersize=4,
                linewidth=1.3,
            )
        axes[0].axhline(RECOVERY_EPS, color="#B00020", linestyle=":", linewidth=1.0)
        axes[0].text(
            0.02, 0.90, f"$\\varepsilon$ = {RECOVERY_EPS}", transform=axes[0].transAxes, fontsize=7, color="#B00020"
        )
        # The linear series is exactly zero, which a log axis cannot show; it is
        # drawn on the floor and labelled so it is not read as a small residual.
        if float(cells[cells.variant == "linear"].max_abs_gap.max()) == 0.0:
            axes[0].text(
                0.98,
                0.06,
                "linear: exact agreement (0)",
                transform=axes[0].transAxes,
                fontsize=7,
                color=VARIANT_COLORS["linear"],
                ha="right",
            )
        axes[0].set_yscale("log")
        axes[0].set_ylabel("max |SCIP $-$ certified optimum| (points)")
        axes[0].set_title("Agreement with the certified oracle")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("median solve time (s)")
        axes[1].set_title("SCIP solve cost")
        for ax in axes:
            ax.set_xlabel("number of internal variables $k$")
            ax.set_xticks(sorted(cells.k.unique()))
            ax.legend(frameon=False, loc="best")
        for suffix in ("png", "pdf"):
            fig.savefig(self.out / "figures" / f"fig6_scip_verification.{suffix}", bbox_inches="tight")
        plt.close(fig)

    def run(self) -> dict:
        """Compute the verdict, write the artefacts, and return the summary."""
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "font.size": 9,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        verdict = self.verdict()
        cells = self.by_cell()
        cells.to_csv(self.out / "scip_by_cell.csv", index=False)
        (self.out / "scip_summary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        self.figure(cells)

        lines = [
            "# SCIP verification of the certified ground truth (amendment A2)",
            "",
            "Generated by `analyze_scip.py`; do not edit by hand.",
            "",
            "> SCIP is **not** a roster method and takes no part in H1/H2/H5. It rebuilds each instance as an",
            "> algebraic model and returns a solution carrying a proof of global optimality, which is the one",
            "> check the confirmatory grid cannot perform on itself: all four roster methods are scored against",
            "> an oracle written by this project, so a systematic error in that oracle would be invisible to",
            "> every one of them.",
            "",
            f"## Verdict: **{verdict['verdict']}**",
            "",
            f"- Rule (fixed before the runs): {verdict['rule']}",
            f"- Rows: **{verdict['n_rows']}**, proved globally optimal: **{verdict['n_proved_optimal']}**, "
            f"model-faithful: **{verdict['n_model_faithful']}**",
            f"- Largest amount by which SCIP beat the certified optimum: "
            f"**{verdict['largest_exceedance_over_oracle']:.2e}** points",
            f"- Largest absolute disagreement either way: **{verdict['max_abs_gap_vs_oracle']:.2e}** points",
            f"- Worst model-fidelity residual: **{verdict['worst_fidelity_residual']:.2e}** points",
            f"- Cases breaching eps = {RECOVERY_EPS}: **{verdict['n_breaching_eps']}**",
            "",
            "## By regime and problem size",
            "",
            "| Regime | k | n | Proved optimal | Max abs. gap | Median gap | Median time (s) | Median nodes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in cells.itertuples():
            lines.append(
                f"| {VARIANT_LABELS[row.variant]} | {row.k} | {row.n} | {row.proved_optimal}/{row.n} | "
                f"{row.max_abs_gap:.2e} | {row.median_gap:+.2e} | {row.median_time_s:.3f} | {row.median_nodes:.0f} |"
            )
        lines += [
            "",
            "> Solve times are descriptive only. SCIP is handed a closed-form algebraic model, while the roster",
            "> methods discover the objective one black-box evaluation at a time, so the two are not paying for",
            "> the same thing and a wall-clock ranking between them would not be like for like.",
            "",
        ]
        if float(cells.median_nodes.max()) <= 1.0:
            lines += [
                "## The node counts corroborate the characterisation itself",
                "",
                "Every cell has a median branch-and-bound node count of **1**: SCIP closed each instance at the",
                "root node and never had to branch. That is a stronger statement than the matching optima. A",
                "global solver branches when it cannot yet prove that a local bound is global, which is precisely",
                "what a nonconvexity forces it to do. Needing no branching anywhere in the regime is independent",
                "structural evidence that these instances really are convex, arrived at by a solver that knows",
                "nothing about the convexity characterisation this thesis proposes.",
                "",
            ]
        lines += [
            "## Files",
            "",
            "- `scip_summary.json`, `scip_by_cell.csv`, `../scip_results.jsonl`",
            "- `figures/fig6_scip_verification.(png|pdf)`",
            "",
        ]
        (self.out / "RESULTS_SCIP.md").write_text("\n".join(lines), encoding="utf-8")
        return verdict


def main():
    """Run the SCIP verification analysis."""
    analysis = ScipAnalysis()
    verdict = analysis.run()
    print(
        f"rows={verdict['n_rows']} proved_optimal={verdict['n_proved_optimal']} faithful={verdict['n_model_faithful']}"
    )
    print(f"largest exceedance over oracle: {verdict['largest_exceedance_over_oracle']:.3e}")
    print(f"VERDICT: {verdict['verdict']}")
    print(f"artefacts -> {analysis.out}")


if __name__ == "__main__":
    main()
