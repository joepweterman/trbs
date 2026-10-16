# pylint: disable=too-many-locals
"""
This file analyses a sweep store written by ``sweep_harness.py``: the scaling study
(``run_scaling_k.py``) or the GA budget study (``run_ga_budget.py``).

Per configuration and k it reports recovery pooled over scenarios with a Wilson
interval, the median gap, the median evaluations and wall-clock, and two counts the
locked analysis did not need: censored rows and rows answered by the incumbent
rule. Recovery and gaps are read on the method's own answer (``f_solver``), which
is what the locked rows hold; the tool-level recovery after the incumbent rule
(``f_found``) is reported next to it. The statistical machinery is imported from
``analyze_study.py`` and not reimplemented.

Outputs go next to the store: a markdown summary, a CSV of the cell statistics and
two LaTeX tables, recovery by k and median evaluations by k, in the layout of the
MDBH recovery table of the thesis.

Run:
  python experiments/synthetic/analyze_sweep.py --study scaling
  python experiments/synthetic/analyze_sweep.py --study ga_budget
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_study import STUDY_ROOT, StudyAnalysis

HERE = Path(__file__).resolve().parent
STUDIES = {
    "scaling": HERE / "generated" / "scaling" / "results_scaling.jsonl",
    "ga_budget": STUDY_ROOT / "results_ga_budget.jsonl",
}

#: Configuration label to table row, for the labels the two drivers produce.
PRETTY = {
    "grid (face, refining)": "Grid (budget face, refining, 60k evals)",
    "grid_capped@max_combinations=60000": "Grid (capped simplex, 60k ceiling)",
    "slsqp@n_starts=100": "SLSQP multi-start (100 starts)",
    "basin_hopping@n_hops=25,n_starts=1": "Basin-hopping (25 hops)",
    "genetic_algorithm@n_generations=60,population_size=50": "GA, 60 generations (locked)",
    "genetic_algorithm@n_generations=240,population_size=50": "GA, 240 generations",
    "genetic_algorithm@n_generations=1000,population_size=50": "GA, 1,000 generations",
    "genetic_algorithm@n_generations=4000,population_size=50,stall_generations=100": "GA, until stalled (100 gen.)",
    "mdbh@eta=1.0,n_hops=10,n_local_steps=50,n_starts=5,sigma=1.5,temperature=1.0": "MDBH (locked)",
}


def load(path):
    """
    This function reads a sweep store into a frame.
    :param path: the results file
    :return: a DataFrame with one row per cell
    """
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise SystemExit(f"no rows in {path} yet")
    return mark_suspended(pd.DataFrame(rows))


def mark_suspended(frame):
    """
    This function separates real censoring from a laptop that went to sleep mid-run.

    A row whose wall-clock exceeds the cap while its evaluation count is ordinary and
    every other row of its cell finished in a small fraction of that time did not run
    long; the clock kept counting through a suspend (the A4 overhead measurement met
    the same artefact). Such rows are flagged ``suspect_suspend`` and taken out of the
    ``censored`` count; their result is kept because the solver's answer is unaffected.
    :param frame: the sweep rows
    :return: the frame with ``suspect_suspend`` added and ``censored`` corrected
    """
    frame = frame.copy()
    typical = frame.groupby(["config_label", "k"])["wall_time_s"].transform("median")
    frame["suspect_suspend"] = frame["censored"] & (frame["wall_time_s"] > 20 * typical)
    frame["censored"] = frame["censored"] & ~frame["suspect_suspend"]
    return frame


def cell_statistics(frame):
    """
    This function summarises every (configuration, variant, k) cell, pooled over
    scenarios and seeds.
    :param frame: the sweep rows
    :return: a DataFrame of cell statistics
    """
    records = []
    for (label, variant, k), group in frame.groupby(["config_label", "variant", "k"]):
        clean = group[~group["censored"]]
        successes = int(group["recovered_solver"].sum())
        point, low, high = StudyAnalysis.wilson_ci(successes, len(group))
        records.append(
            {
                "config_label": label,
                "variant": variant,
                "k": int(k),
                "n": len(group),
                "recovered": successes,
                "recovery": point,
                "wilson_low": low,
                "wilson_high": high,
                "recovery_tool": float(group["recovered"].mean()),
                "median_gap": float(np.median(clean["gap_solver"])) if len(clean) else np.nan,
                "max_gap": float(clean["gap_solver"].max()) if len(clean) else np.nan,
                "median_evals": float(np.median(clean["n_function_evals"])) if len(clean) else np.nan,
                "median_wall_s": float(np.median(clean["wall_time_s"])) if len(clean) else np.nan,
                "max_wall_s": float(clean["wall_time_s"].max()) if len(clean) else np.nan,
                "censored": int(group["censored"].sum()),
                "incumbent": int(group["reported_incumbent"].sum()),
            }
        )
    return pd.DataFrame(records).sort_values(["config_label", "variant", "k"])


def pooled_statistics(frame):
    """
    This function pools both variants: recovery and cost per configuration and k.
    :param frame: the sweep rows
    :return: a DataFrame of pooled statistics
    """
    records = []
    for (label, k), group in frame.groupby(["config_label", "k"]):
        clean = group[~group["censored"]]
        successes = int(group["recovered_solver"].sum())
        point, low, _ = StudyAnalysis.wilson_ci(successes, len(group))
        records.append(
            {
                "config_label": label,
                "k": int(k),
                "n": len(group),
                "recovery": point,
                "wilson_low": low,
                "recovery_tool": float(group["recovered"].mean()),
                "median_gap": float(np.median(clean["gap_solver"])) if len(clean) else np.nan,
                "median_evals": float(np.median(clean["n_function_evals"])) if len(clean) else np.nan,
                "median_wall_s": float(np.median(clean["wall_time_s"])) if len(clean) else np.nan,
                "censored": int(group["censored"].sum()),
                "incumbent": int(group["reported_incumbent"].sum()),
            }
        )
    return pd.DataFrame(records).sort_values(["config_label", "k"])


def order(labels):
    """The configuration labels in the order of PRETTY, unknown ones last."""
    known = [label for label in PRETTY if label in set(labels)]
    return known + sorted(set(labels) - set(known))


def latex_table(pooled, fmt, note):
    """
    This function renders one pooled statistic as a configuration-by-k LaTeX table.
    :param pooled: the pooled statistics
    :param fmt: a format callable turning one pooled row into a cell
    :param note: the generator comment at the top of the table
    :return: the table as a string
    """
    ks = sorted(pooled["k"].unique())
    lines = [
        f"% Generated by analyze_sweep.py ({note}) - do not edit by hand.",
        "\\begin{tabular}{l" + "c" * len(ks) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"$k={k}$" for k in ks) + " \\\\",
        "\\midrule",
    ]
    for label in order(pooled["config_label"].unique()):
        rows = pooled[pooled["config_label"] == label].set_index("k")
        cells = [fmt(rows.loc[k]) if k in rows.index else "--" for k in ks]
        lines.append(f"{PRETTY.get(label, label)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def recovery_cell(row):
    """Recovery with its Wilson lower bound beneath, and a dagger where a cell was censored."""
    dagger = "$^\\dagger$" if row["censored"] else ""
    return f"{row['recovery']:.3f}{dagger} \\scriptsize({row['wilson_low']:.3f})"


def evals_cell(row):
    """Median evaluations with a thousands separator."""
    return "--" if np.isnan(row["median_evals"]) else f"{int(round(row['median_evals'])):,}"


def wall_cell(row):
    """Median wall-clock seconds."""
    return "--" if np.isnan(row["median_wall_s"]) else f"{row['median_wall_s']:.1f}"


#: The tolerance ladder of the A4 study: decision indifference, the pre-registered
#: recovery epsilon, and the level below Beerwiser's two-basin gap.
TOLERANCES = (0.1, 0.01, 0.001)


def tolerance_ladder(frame, variant=None):
    """
    This function reads recovery of the method's own answer at each tolerance of the
    ladder, per configuration and k, so that a coarse lattice's "recovery" at 0.1 on a
    flat surface can be told apart from actually resolving the optimum.
    :param frame: the sweep rows
    :param variant: restrict to one appreciation variant, or None for both
    :return: a DataFrame with one row per configuration, k and tolerance
    """
    work = frame if variant is None else frame[frame["variant"] == variant]
    records = []
    for (label, k), group in work.groupby(["config_label", "k"]):
        for eps in TOLERANCES:
            records.append(
                {
                    "config_label": label,
                    "k": int(k),
                    "eps": eps,
                    "n": len(group),
                    "recovery": float((group["gap_solver"] <= eps).mean()),
                }
            )
    return pd.DataFrame(records)


def ladder_table(ladder, note):
    """
    This function renders the tolerance ladder as a LaTeX table: one row per
    configuration and tolerance, one column per k.
    :param ladder: the frame from :func:`tolerance_ladder`
    :param note: the generator comment at the top of the table
    :return: the table as a string
    """
    ks = sorted(ladder["k"].unique())
    lines = [
        f"% Generated by analyze_sweep.py ({note}) - do not edit by hand.",
        "\\begin{tabular}{ll" + "c" * len(ks) + "}",
        "\\toprule",
        "Method & $\\varepsilon$ & " + " & ".join(f"$k={k}$" for k in ks) + " \\\\",
    ]
    for label in order(ladder["config_label"].unique()):
        lines.append("\\midrule")
        for i, eps in enumerate(TOLERANCES):
            rows = ladder[(ladder["config_label"] == label) & (ladder["eps"] == eps)].set_index("k")
            cells = [f"{rows.loc[k, 'recovery']:.3f}" if k in rows.index else "--" for k in ks]
            head = PRETTY.get(label, label) if i == 0 else ""
            lines.append(f"{head} & {eps:g} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _md_table(frame, columns):
    """Render selected columns of a frame as a markdown table."""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False):
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    """Analyse one sweep store and write its summary, CSV and tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=list(STUDIES), required=True)
    args = parser.parse_args()

    store = STUDIES[args.study]
    frame = load(store)
    out = store.parent / f"analysis_{args.study}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    cells = cell_statistics(frame)
    pooled = pooled_statistics(frame)
    cells.to_csv(out / "cells.csv", index=False)
    pooled.to_csv(out / "pooled.csv", index=False)
    (out / "tables" / f"{args.study}_recovery.tex").write_text(
        latex_table(pooled, recovery_cell, f"{args.study}: recovery pooled over variants and scenarios"),
        encoding="utf-8",
    )
    (out / "tables" / f"{args.study}_evals.tex").write_text(
        latex_table(pooled, evals_cell, f"{args.study}: median objective evaluations"),
        encoding="utf-8",
    )
    (out / "tables" / f"{args.study}_wall.tex").write_text(
        latex_table(pooled, wall_cell, f"{args.study}: median wall-clock seconds"),
        encoding="utf-8",
    )
    ladder = tolerance_ladder(frame)
    ladder_curved = tolerance_ladder(frame, "sinusoidal")
    ladder.to_csv(out / "tolerance_ladder.csv", index=False)
    (out / "tables" / f"{args.study}_ladder.tex").write_text(
        ladder_table(ladder, f"{args.study}: recovery at three tolerances, both variants"), encoding="utf-8"
    )
    (out / "tables" / f"{args.study}_ladder_curved.tex").write_text(
        ladder_table(ladder_curved, f"{args.study}: recovery at three tolerances, curved variant"),
        encoding="utf-8",
    )
    ladder_wide = ladder.pivot_table(index=["config_label", "eps"], columns="k", values="recovery").reset_index()
    ladder_wide.columns = [f"k={c}" if isinstance(c, (int, np.integer)) else str(c) for c in ladder_wide.columns]
    ladder_wide["config"] = ladder_wide["config_label"].map(lambda label: PRETTY.get(label, label))
    ladder_columns = ["config", "eps"] + [f"k={k}" for k in sorted(ladder["k"].unique())]

    lines = [
        f"# {args.study}: results",
        "",
        f"Generated by `analyze_sweep.py` from `{store.name}`; do not edit by hand.",
        f"Rows: **{len(frame)}** ({int(frame['censored'].sum())} censored, "
        f"{int(frame['suspect_suspend'].sum())} over the cap through a machine suspend and not counted as "
        f"censored, {int(frame['reported_incumbent'].sum())} answered by the incumbent rule).",
        "",
        "Recovery and gaps are the method's own answer (`f_solver`, comparable to the locked rows);",
        "`recovery_tool` is what the tool reports after the incumbent rule, and `incumbent` counts the rows",
        "where the rule replaced the method's answer by an existing decision-maker option.",
        "",
        "## Recovery and cost pooled over both variants and all scenarios",
        "",
        _md_table(
            pooled.assign(config=pooled["config_label"].map(lambda label: PRETTY.get(label, label))),
            [
                "config",
                "k",
                "n",
                "recovery",
                "wilson_low",
                "recovery_tool",
                "median_gap",
                "median_evals",
                "median_wall_s",
                "censored",
                "incumbent",
            ],
        ),
        "",
        "## Per variant",
        "",
        _md_table(
            cells.assign(config=cells["config_label"].map(lambda label: PRETTY.get(label, label))),
            [
                "config",
                "variant",
                "k",
                "n",
                "recovery",
                "wilson_low",
                "recovery_tool",
                "median_gap",
                "max_gap",
                "median_evals",
                "median_wall_s",
                "max_wall_s",
                "censored",
                "incumbent",
            ],
        ),
        "",
        "## Recovery of the method's own answer at three tolerances (both variants)",
        "",
        _md_table(ladder_wide, ladder_columns),
        "",
        "## Files",
        "",
        "- `cells.csv`, `pooled.csv`, `tolerance_ladder.csv`",
        f"- `tables/{args.study}_recovery.tex`, `tables/{args.study}_evals.tex`, `tables/{args.study}_wall.tex`, "
        f"`tables/{args.study}_ladder.tex`, `tables/{args.study}_ladder_curved.tex`",
    ]
    report = out / f"RESULTS_{args.study.upper()}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(pooled.to_string(index=False))
    print(f"[sweep] report at {report}")


if __name__ == "__main__":
    main()
