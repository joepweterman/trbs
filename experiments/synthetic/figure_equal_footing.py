"""
This file draws the data profile of the A4 equal-footing study.

A quality-cost frontier is the obvious plot here and it is the wrong one: the
median gap of grid, the capped grid, SLSQP and basin-hopping is exactly zero at
every configuration, so a gap-against-cost picture would be four flat lines on
the axis and would read as "everything is perfect". What separates the methods is
not the median but how many cells reach a target and how dearly.

The figure is therefore a data profile in the sense of More and Wild (2009): for
each target tolerance, the fraction of cells solved as a function of the number
of objective evaluations spent. A curve that flattens below one has cells it
never solves at any budget.

Run:
  python experiments/synthetic/figure_equal_footing.py
"""

import numpy as np
from matplotlib import pyplot as plt

from analyze_equal_footing import DEFAULT_CONFIG, OUT, TAUS, crossing_column, load
from analyze_study import METHOD_COLORS, _style

#: The roster of A4, in reading order, with the two entries the locked study
#: does not carry. The capped baseline shares the baseline's hue because it is
#: the same method on a different feasible set.
METHOD_ORDER = ["grid", "grid_capped", "slsqp", "basin_hopping", "genetic_algorithm", "mdbh"]
COLORS = dict(METHOD_COLORS, grid_capped="#E2725B", mdbh="#6A3D9A")
LABELS = {
    "grid": "Grid (budget face)",
    "grid_capped": "Grid (capped simplex)",
    "slsqp": "SLSQP multi-start",
    "basin_hopping": "Basin-hopping",
    "genetic_algorithm": "Genetic algorithm",
    "mdbh": "MDBH",
}
#: SLSQP and basin-hopping cost the same to within a hop or two on this regime,
#: so their curves coincide. Basin-hopping is drawn dotted on top rather than
#: hidden underneath, which is the honest way to show two lines that agree.
STYLES = {
    "grid": "-",
    "grid_capped": (0, (5, 2)),
    "slsqp": "-",
    "basin_hopping": (0, (1, 1.6)),
    "genetic_algorithm": "-",
    "mdbh": "-",
}


def profile_points(costs, n_cells, grid):
    """
    This function evaluates one empirical data profile on a cost grid.
    :param costs: the evaluations each solved cell needed, unsolved cells absent
    :param n_cells: the number of cells the method was run on
    :param grid: the evaluation counts to read the profile at
    :return: the fraction of cells solved at or below each grid point
    """
    solved = np.sort(np.asarray(costs, dtype=float))
    return np.searchsorted(solved, grid, side="right") / max(1, n_cells)


def draw(frame, path_stem):
    """
    This function draws one panel per target and writes the figure.
    :param frame: the synthetic frame reduced to one configuration per method
    :param path_stem: file name without extension
    :return: the paths written
    """
    _style()
    fig, axes = plt.subplots(1, len(TAUS), figsize=(10.5, 3.4), sharey=True)

    for axis, tau in zip(axes, TAUS):
        costs_by_method = {}
        for method in METHOD_ORDER:
            subset = frame[frame["method"] == method]
            reached = crossing_column(subset, tau).dropna()
            costs_by_method[method] = (reached.to_numpy(), len(subset))

        all_costs = np.concatenate([c for c, _ in costs_by_method.values() if len(c)])
        grid = np.geomspace(max(1.0, all_costs.min()), all_costs.max() * 1.05, 400)

        for method in METHOD_ORDER:
            costs, n_cells = costs_by_method[method]
            if not len(costs):
                continue
            axis.step(
                grid,
                profile_points(costs, n_cells, grid),
                where="post",
                color=COLORS[method],
                linestyle=STYLES[method],
                linewidth=1.8 if method == "basin_hopping" else 1.5,
                label=LABELS[method],
            )
        axis.set_xscale("log")
        axis.set_xlabel("objective evaluations")
        value = float(tau)
        axis.set_title(f"within {value:g} appreciation point" + ("" if value == 1 else "s"))
        axis.set_ylim(-0.02, 1.02)

    axes[0].set_ylabel("fraction of cells solved")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.09))

    out_dir = OUT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        target = out_dir / f"{path_stem}.{suffix}"
        fig.savefig(target, bbox_inches="tight")
        written.append(target)
    plt.close(fig)
    return written


def main():
    """Draw the data profile from the A4 store."""
    synthetic, _ = load()
    defaults = list(DEFAULT_CONFIG.values())
    frame = synthetic[synthetic["config_label"].isin(defaults)]
    for path in draw(frame, "fig_equalfooting_profile"):
        print(f"[a4] {path}")


if __name__ == "__main__":
    main()
