"""W2 thesis experiment: SLSQP multi-start vs. legacy grid search on Beerwiser.

Run::

    python experiments/w2_slsqp_vs_grid.py

Writes a JSON results file to ``experiments/out/w2_slsqp_vs_grid.json`` and
prints a markdown summary table to stdout. The summary table feeds the
canonical report at
``C:\\Users\\joepw\\thesis-knowledge\\04-experiments\\exp01-slsqp-vs-grid-beerwiser.md``.

This script is intentionally self-contained and re-runnable. The thesis
committee should be able to clone the repo, ``pipenv install``, and reproduce
the table by running this file.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from vlinder.optimize import evaluate_allocation
from vlinder.trbs import TheResponsibleBusinessSimulator

# ---------------------------------------------------------------------------
# Experiment configuration — single source of truth for the report.
# ---------------------------------------------------------------------------
SCENARIO = "Base case"
GRID_MAX_COMBINATIONS = 60000  # matches the legacy default
N_STARTS_GRID = (10, 50, 100, 500)
SEEDS = (42, 123, 456)

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "w2_slsqp_vs_grid.json"


def build_case():
    """Build + evaluate + appreciate a fresh Beerwiser instance."""
    case = TheResponsibleBusinessSimulator("Beerwiser")
    case.build()
    case.evaluate()
    case.appreciate()
    return case


def run_grid_baseline():
    """Run the legacy grid optimizer once. Returns (allocation, appreciation, wall_time)."""
    case = build_case()
    t0 = time.perf_counter()
    case.optimize(SCENARIO, new_dmo_name="Grid Baseline", max_combinations=GRID_MAX_COMBINATIONS)
    wall = time.perf_counter() - t0
    idx = np.where(case.input_dict["decision_makers_options"] == "Grid Baseline")[0][0]
    allocation = case.input_dict["decision_makers_option_value"][idx]
    appreciation = evaluate_allocation(case.input_dict, allocation, SCENARIO, "Grid Baseline")
    return {
        "method": "grid",
        "max_combinations": GRID_MAX_COMBINATIONS,
        "wall_time_s": wall,
        "allocation": allocation.tolist(),
        "appreciation": float(appreciation),
    }


def run_slsqp(n_starts: int, seed: int):
    """Run SLSQP multi-start once. Returns the structured result as a dict."""
    case = build_case()
    result = case.optimize_continuous(
        SCENARIO,
        method="slsqp",
        n_starts=n_starts,
        seed=seed,
        dmo_name=f"SLSQP n{n_starts} seed{seed}",
    )
    return {
        "method": "slsqp",
        "n_starts": n_starts,
        "seed": seed,
        "best_x": result.best_x.tolist(),
        "best_appreciation": result.best_appreciation,
        "n_converged": result.n_converged,
        "n_function_evals": result.n_function_evals,
        "wall_time_s": result.wall_time_s,
    }


def main():
    """Run the full SLSQP vs grid comparison, save JSON, print summary table."""
    print(f"# W2 experiment: SLSQP vs grid — Beerwiser, scenario '{SCENARIO}'")
    print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | scipy {scipy.__version__}")
    print(f"Platform: {platform.platform()}")
    print()

    runs = []
    print("Running grid baseline ...")
    grid = run_grid_baseline()
    runs.append(grid)
    print(
        f"  grid: appreciation={grid['appreciation']:.6f}, "
        f"alloc={grid['allocation']}, wall={grid['wall_time_s']:.3f}s"
    )
    print()

    for n_starts in N_STARTS_GRID:
        for seed in SEEDS:
            print(f"Running SLSQP n_starts={n_starts}, seed={seed} ...")
            res = run_slsqp(n_starts, seed)
            runs.append(res)
            print(
                f"  appreciation={res['best_appreciation']:.6f}, "
                f"converged={res['n_converged']}/{res['n_starts']}, "
                f"n_evals={res['n_function_evals']}, "
                f"wall={res['wall_time_s']:.3f}s"
            )
    print()

    # ---- Save raw results JSON for reproducibility -----------------------
    payload = {
        "scenario": SCENARIO,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "grid_baseline": grid,
        "slsqp_runs": [r for r in runs if r["method"] == "slsqp"],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved raw results to {OUT_PATH}")
    print()

    # ---- Markdown summary table -----------------------------------------
    print("## Aggregated table (mean ± std over seeds)")
    print()
    print(
        "| n_starts | best_appreciation (mean ± std) | n_converged (mean) | "
        "n_function_evals (mean) | wall_time_s (mean ± std) |"
    )
    print(
        "|---------:|:------------------------------:|:------------------:|"
        ":-----------------------:|:-------------------------:|"
    )
    for n_starts in N_STARTS_GRID:
        slsqp_for_n = [r for r in runs if r["method"] == "slsqp" and r["n_starts"] == n_starts]
        apps = np.array([r["best_appreciation"] for r in slsqp_for_n])
        convs = np.array([r["n_converged"] for r in slsqp_for_n])
        evals = np.array([r["n_function_evals"] for r in slsqp_for_n])
        walls = np.array([r["wall_time_s"] for r in slsqp_for_n])
        print(
            f"| {n_starts:>8} | {apps.mean():.6f} ± {apps.std():.6f} | {convs.mean():.1f} | "
            f"{evals.mean():.0f} | {walls.mean():.3f} ± {walls.std():.3f} |"
        )
    print()
    print(
        f"**Grid baseline:** appreciation = {grid['appreciation']:.6f}, "
        f"wall = {grid['wall_time_s']:.3f}s, allocation = {grid['allocation']}"
    )


if __name__ == "__main__":
    main()
