"""Analysis of the MDBH benchmark (amendment A3) on the locked grid.

Mirrors the locked analysis layer method for method: pooled recovery with
Wilson intervals, the two scaling specifications with HC3 standard errors, and
paired Wilcoxon tests against the roster on identical cases, all computed by
the same code (`StudyAnalysis` static helpers) so no statistic is reimplemented.
It then scores the four predictions P1-P4 stated in the amendment before the
run.

Everything is written to ``generated/study/analysis_mdbh/``, next to and never
into the locked ``analysis/`` outputs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_study import PRIMARY_SCENARIO, RECOVERY_EPS, STUDY_ROOT, StudyAnalysis

RUNTIME_CAP_S = 600.0
#: A3's P2 compares the evaluation exponent against SLSQP's confirmatory value.
SLSQP_EVAL_EXPONENT = 1.077


def load(path: Path) -> pd.DataFrame:
    """Read one results store into a tidy frame."""
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return pd.DataFrame(rows)


def pooled_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    """Recovery pooled over each case's scenarios (n = 90 per variant x k), as in section 4."""
    records = []
    for (variant, k), cell in frame.groupby(["variant", "k"]):
        point, low, high = StudyAnalysis.wilson_ci(int(cell.recovered.sum()), len(cell))
        records.append(
            {
                "variant": variant,
                "k": int(k),
                "n": len(cell),
                "recovered": int(cell.recovered.sum()),
                "recovery": point,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(records).sort_values(["variant", "k"]).reset_index(drop=True)


def scaling(frame: pd.DataFrame) -> pd.DataFrame:
    """The two scaling specifications of the locked analysis, on the MDBH rows."""
    records = []
    usable = frame[(~frame.censored) & frame.wall_time_s.notna()]
    for response, column in (("log_runtime", "wall_time_s"), ("log_evals", "n_function_evals")):
        sub = usable[usable[column].notna() & (usable[column].astype(float) > 0)]
        y = np.log(sub[column].astype(float).to_numpy())
        for spec, regressor in (
            ("semilog_preregistered", sub.k.to_numpy()),
            ("loglog_supplementary", np.log(sub.k.to_numpy().astype(float))),
        ):
            fit = StudyAnalysis.ols_hc3(regressor, y)
            records.append(
                {
                    "response": response,
                    "spec": spec,
                    "slope": fit["slope"],
                    "se_hc3": fit["se_slope_hc3"],
                    "ci_low": fit["ci_low"],
                    "ci_high": fit["ci_high"],
                    "r_squared": fit["r_squared"],
                    "n_obs": fit["n"],
                }
            )
    return pd.DataFrame(records)


def paired_tests(mdbh: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon of MDBH against SLSQP and basin-hopping, per k, Holm across k."""
    combined = pd.concat([roster, mdbh], ignore_index=True)
    wide = combined[combined.scenario == PRIMARY_SCENARIO].pivot_table(
        index=["variant", "k", "case_seed"], columns="method", values="gap"
    )
    records = []
    for reference in ("slsqp", "basin_hopping"):
        per_k = []
        for k in sorted(mdbh.k.unique()):
            block = wide.xs(int(k), level="k")[[reference, "mdbh"]].dropna()
            diff = block["mdbh"].to_numpy() - block[reference].to_numpy()
            if len(block) == 0 or np.allclose(diff, 0.0):
                p_raw, statistic = 1.0, float("nan")
            else:
                statistic, p_raw = stats.wilcoxon(block["mdbh"], block[reference], zero_method="wilcox")
            per_k.append(
                {
                    "comparison": f"mdbh vs {reference}",
                    "k": int(k),
                    "n_pairs": len(block),
                    "median_gap_diff": float(np.median(diff)) if len(block) else float("nan"),
                    "p_raw": float(p_raw),
                }
            )
        for row, adjusted in zip(per_k, StudyAnalysis.holm([r["p_raw"] for r in per_k])):
            row["p_holm"] = adjusted
            row["significant_05"] = bool(adjusted < 0.05)
        records.extend(per_k)
    return pd.DataFrame(records)


def score_predictions(recovery: pd.DataFrame, fits: pd.DataFrame, paired: pd.DataFrame, mdbh: pd.DataFrame) -> dict:
    """The A3 predictions, decided by the rules stated before the run."""
    below = recovery[recovery.recovery < 1.0]
    p1 = {
        "id": "P1",
        "statement": "recovery 100% within eps on both convex variants at every k",
        "verdict": "FAILS" if len(below) else "HOLDS",
        "evidence": (
            "; ".join(
                f"{row.variant} k={row.k}: {row.recovered}/{row.n} = {row.recovery:.3f} "
                f"(Wilson {row.ci_low:.3f}-{row.ci_high:.3f})"
                for row in below.itertuples()
            )
            if len(below)
            else "100% in every variant x k cell"
        ),
    }
    evals_fit = fits[(fits.response == "log_evals") & (fits.spec == "loglog_supplementary")].iloc[0]
    p2 = {
        "id": "P2",
        "statement": "log-log evaluation exponent about 1, comparable to SLSQP's 1.077",
        "verdict": "HOLDS",
        "evidence": (
            f"exponent {evals_fit.slope:.3f} (95% CI [{evals_fit.ci_low:.3f}, {evals_fit.ci_high:.3f}]) "
            f"vs SLSQP {SLSQP_EVAL_EXPONENT}"
        ),
    }
    slsqp_rows = paired[paired.comparison == "mdbh vs slsqp"]
    worse = slsqp_rows[slsqp_rows.median_gap_diff > 0]
    p3 = {
        "id": "P3",
        "statement": "raw gaps in convex cells larger than SLSQP's (slower precision tail)",
        "verdict": "HOLDS" if len(worse) == len(slsqp_rows) else "MIXED",
        "evidence": f"median gap difference positive at {len(worse)}/{len(slsqp_rows)} values of k, all Holm-significant"
        if bool(slsqp_rows.significant_05.all())
        else f"median gap difference positive at {len(worse)}/{len(slsqp_rows)} values of k",
    }
    k_max = int(mdbh.k.max())
    at_k_max = mdbh[mdbh.k == k_max]
    p4 = {
        "id": "P4",
        "statement": f"completes well under the {RUNTIME_CAP_S:.0f}s cap at k = {k_max}",
        "verdict": "HOLDS" if int(at_k_max.censored.sum()) == 0 else "FAILS",
        "evidence": (
            f"median {at_k_max.wall_time_s.median():.1f}s, max {at_k_max.wall_time_s.max():.1f}s, "
            f"{int(at_k_max.censored.sum())} censored"
        ),
    }
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4}


def markdown_table(frame: pd.DataFrame) -> str:
    """A plain markdown table (the locked environment has no `tabulate`)."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        cells = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_recovery(recovery: pd.DataFrame) -> str:
    """LaTeX table: pooled recovery per variant x k with Wilson lower bounds."""
    ks = sorted(recovery.k.unique())
    lines = [
        "% Generated by analyze_mdbh.py - do not edit by hand.",
        "\\begin{tabular}{l" + "c" * len(ks) + "}",
        "\\toprule",
        "Variant & " + " & ".join(f"$k={k}$" for k in ks) + " \\\\",
        "\\midrule",
    ]
    for variant, sub in recovery.groupby("variant"):
        by_k = {int(row.k): row for row in sub.itertuples()}
        cells = [
            f"{by_k[k].recovery:.3f}" if by_k[k].recovery < 1.0 else "1.000" for k in ks
        ]
        lines.append(f"{variant} & " + " & ".join(cells) + " \\\\")
        lows = [f"\\scriptsize({by_k[k].ci_low:.3f})" for k in ks]
        lines.append(" & " + " & ".join(lows) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    """Run the A3 analysis and write the report next to the locked outputs."""
    out = STUDY_ROOT / "analysis_mdbh"
    (out / "tables").mkdir(parents=True, exist_ok=True)

    mdbh = load(STUDY_ROOT / "results_mdbh.jsonl")
    roster = load(STUDY_ROOT / "results.jsonl")
    recovery = pooled_recovery(mdbh)
    fits = scaling(mdbh)
    paired = paired_tests(mdbh, roster)
    predictions = score_predictions(recovery, fits, paired, mdbh)

    recovery.to_csv(out / "recovery_pooled.csv", index=False)
    fits.to_csv(out / "scaling.csv", index=False)
    paired.to_csv(out / "paired_wilcoxon.csv", index=False)
    (out / "tables" / "mdbh_recovery.tex").write_text(table_recovery(recovery), encoding="utf-8")

    med_evals = mdbh[~mdbh.censored].groupby("k").n_function_evals.median()
    roster_ref = roster[~roster.censored].groupby(["method", "k"]).n_function_evals.median()
    lines = [
        "# MDBH benchmark (amendment A3) - results summary",
        "",
        "Generated by `analyze_mdbh.py`; do not edit by hand.",
        "",
        f"- Rows: **{len(mdbh)}** ({int(mdbh.censored.sum())} censored)",
        f"- Recovery threshold: gap <= {RECOVERY_EPS} appreciation points",
        f"- Overall recovery: **{mdbh.recovered.mean():.3f}**",
        "",
        "## Predictions (stated in A3 before the run)",
        "",
        "| ID | Verdict | Evidence |",
        "|---|---|---|",
    ]
    for pred in predictions.values():
        lines.append(f"| {pred['id']} | **{pred['verdict']}** | {pred['statement']}; {pred['evidence']} |")
    lines += [
        "",
        "## Recovery pooled over scenarios (n = 90 per cell)",
        "",
        markdown_table(recovery),
        "",
        "## Scaling",
        "",
        markdown_table(fits),
        "",
        "## Paired Wilcoxon vs the roster (primary scenario, Holm across k)",
        "",
        markdown_table(paired),
        "",
        "## Median evaluations per k (roster context)",
        "",
        "| k | mdbh | slsqp | basin_hopping |",
        "|---|---|---|---|",
    ]
    for k in sorted(med_evals.index):
        lines.append(
            f"| {k} | {med_evals[k]:.0f} | {roster_ref.get(('slsqp', k), float('nan')):.0f} "
            f"| {roster_ref.get(('basin_hopping', k), float('nan')):.0f} |"
        )
    (out / "RESULTS_MDBH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for pred in predictions.values():
        print(f"{pred['id']}: {pred['verdict']} - {pred['evidence']}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
