# pylint: disable=R0914  # one panel per case keeps its plotting state local
"""
This file draws the graph of the budgeted comparison: what does each method deliver
within a given number of seconds, or of objective evaluations, on each packaged case?

One panel per case. The horizontal axis is the budget (seconds or evaluations, both
on a log scale), the vertical axis the appreciation reached within it, read at the
median over the seeds of ``time_limited_comparison.py``. Colour identifies the
method; the line style carries the budget mode, solid where under-spending is
allowed (sum(x) <= B) and dashed where the budget is spent exactly (sum(x) = B),
so the two rows of the same method are read together.

Run (after experiments/time_limited_comparison.py has produced the CSV):
  python experiments/figure_time_limited.py
  python experiments/figure_time_limited.py --axis evals
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from time_limited_comparison import AXES, OUT  # pylint: disable=import-error

sys.path.insert(0, str(Path(__file__).resolve().parent / "synthetic"))
from analyze_study import METHOD_COLORS, _style  # noqa: E402  # pylint: disable=wrong-import-position

#: Per axis: the CSV column with the budget, the axis label, the log base and the file stem.
AXIS_STYLE = {
    "seconds": {"label": "time limit (s)", "base": 2, "stem": "fig_time_limited"},
    "evals": {"label": "evaluation budget", "base": 10, "stem": "fig_eval_limited"},
}

COLORS = dict(METHOD_COLORS, mdbh="#6A3D9A")
BASE_LABELS = {
    "grid": "Grid",
    "slsqp": "SLSQP multi-start",
    "basin_hopping": "Basin-hopping",
    "genetic_algorithm": "Genetic algorithm",
    "mdbh": "MDBH",
}
#: Solid where under-spending is feasible, dashed where the budget is spent exactly.
#: The packaged grid spends exactly by construction; its capped twin is the solid line.
FACE_STYLE, CAPPED_STYLE = (0, (5, 2)), "-"


def spends_exactly(row):
    """
    This function reads whether a row's method spent the budget exactly.

    Read from the label rather than the ``spend_all`` column: the rows of the first
    sweep predate the grid's ``spend_all`` switch and carry ``False`` for the packaged
    grid, which spends exactly by construction.
    :param row: one CSV row
    :return: True on the budget face, False on the capped simplex
    """
    label = str(row["label"])
    return label == "grid" or label.endswith("(spend all)")


def draw(frame, budget_axis="seconds", path_stem=None):
    """
    This function draws one panel per case and writes the figure.
    :param frame: the comparison rows of one axis
    :param budget_axis: "seconds" or "evals"
    :param path_stem: the file name without extension; defaults to the axis's stem
    :return: the written paths
    """
    _style()
    column, style = AXES[budget_axis]["column"], AXIS_STYLE[budget_axis]
    path_stem = path_stem or style["stem"]
    frame = frame.copy().rename(columns={column: "limit"})
    frame["face"] = frame.apply(spends_exactly, axis=1)
    # The first sweep recorded the capped grid under the A4 shim's method name.
    frame["base"] = frame["method"].replace({"grid_capped": "grid"})
    cases = list(dict.fromkeys(frame["case"]))

    fig, axes = plt.subplots(1, len(cases), figsize=(3.1 * len(cases), 2.9), constrained_layout=True)
    axes = [axes] if len(cases) == 1 else list(axes)

    for axis, case_name in zip(axes, cases):
        rows = frame[frame["case"] == case_name]
        for (base, face), group in rows.groupby(["base", "face"]):
            mode = "$= B$" if face else "$\\leq B$"
            by_limit = group.groupby("limit")["appreciation"]
            median = by_limit.median().sort_index()
            axis.plot(
                median.index,
                median.values,
                color=COLORS[base],
                linestyle=FACE_STYLE if face else CAPPED_STYLE,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"{BASE_LABELS[base]} ({mode})",
            )
        axis.set_xscale("log", base=style["base"])
        limits = sorted(rows["limit"].unique())
        axis.set_xticks(limits)
        axis.set_xticklabels([f"{limit:g}" for limit in limits], rotation=45 if len(limits) > 6 else 0)
        axis.minorticks_off()
        axis.set_xlabel(style["label"])
        axis.set_title(case_name)
    axes[0].set_ylabel("appreciation (median over seeds)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.16))

    written = []
    for suffix in ("png", "pdf"):
        target = OUT / f"{path_stem}.{suffix}"
        fig.savefig(target, bbox_inches="tight")
        written.append(target)
    plt.close(fig)
    return written


def main():
    """Draw the comparison figure of one axis from its CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=list(AXES), default="seconds")
    args = parser.parse_args()

    frame = pd.read_csv(AXES[args.axis]["csv"])
    for target in draw(frame, args.axis):
        print(f"[figure] written to {target}")


if __name__ == "__main__":
    main()
