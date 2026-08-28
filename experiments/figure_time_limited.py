# pylint: disable=R0914  # one panel per case keeps its plotting state local
"""
This file draws the graph of the timed comparison: what does each method deliver
within 15, 30, 60 or 120 seconds on each packaged case?

One panel per case. The horizontal axis is the time limit, the vertical axis the
appreciation reached within it, read at the median over the seeds of
``time_limited_comparison.py``. Colour identifies the method; the line style
carries the budget mode, solid where under-spending is allowed (sum(x) <= B) and
dashed where the budget is spent exactly (sum(x) = B), so the two rows of the
same method are read together.

Run (after experiments/time_limited_comparison.py has produced the CSV):
  python experiments/figure_time_limited.py
"""

import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "synthetic"))
from analyze_study import METHOD_COLORS, _style  # noqa: E402  # pylint: disable=wrong-import-position

OUT = Path(__file__).resolve().parent / "out" / "time_limited"
CSV = OUT / "time_limited_comparison_v3.csv"

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
    :param row: one CSV row
    :return: True on the budget face, False on the capped simplex
    """
    return bool(row["spend_all"])


def draw(frame, path_stem="fig_time_limited"):
    """
    This function draws one panel per case and writes the figure.
    :param frame: the timed-comparison rows
    :param path_stem: the file name without extension
    :return: the written paths
    """
    _style()
    frame = frame.copy()
    frame["face"] = frame.apply(spends_exactly, axis=1)
    frame["base"] = frame["method"]
    cases = list(dict.fromkeys(frame["case"]))

    fig, axes = plt.subplots(1, len(cases), figsize=(3.1 * len(cases), 2.9), constrained_layout=True)
    axes = [axes] if len(cases) == 1 else list(axes)

    for axis, case_name in zip(axes, cases):
        rows = frame[frame["case"] == case_name]
        for (base, face), group in rows.groupby(["base", "face"]):
            mode = "$= B$" if face else "$\\leq B$"
            by_limit = group.groupby("limit_s")["appreciation"]
            median = by_limit.median().sort_index()
            # The band shows the full seed spread; deterministic rows collapse it to the line.
            axis.fill_between(
                by_limit.min().sort_index().index,
                by_limit.min().sort_index().values,
                by_limit.max().sort_index().values,
                color=COLORS[base],
                alpha=0.12,
                linewidth=0,
            )
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
        axis.set_xscale("log", base=2)
        limits = sorted(rows["limit_s"].unique())
        axis.set_xticks(limits)
        axis.set_xticklabels([str(limit) for limit in limits])
        axis.set_xlabel("time limit (s)")
        axis.set_title(case_name)
    axes[0].set_ylabel("appreciation (median over seeds; band = min-max)")

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
    """Draw the timed-comparison figure from the CSV."""
    frame = pd.read_csv(CSV)
    for target in draw(frame):
        print(f"[figure] written to {target}")


if __name__ == "__main__":
    main()
