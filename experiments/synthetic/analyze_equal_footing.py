"""
This file analyses the A4 equal-footing study (see PREREGISTRATION.md, A4).

It reads results_equalfooting.jsonl and answers the amendment's analysis plan:
the constraint decomposition on the packaged cases, cost to target at the three
frozen tolerances, each method's own quality-cost frontier, and the four
predictions stated before the run.

The statistical machinery is imported from analyze_study.py and not
reimplemented, so the supplement is read with the same Wilson intervals and the
same Holm correction as the locked analysis.

Cost is read on objective evaluations. Wall-clock is reported alongside and
carries no claim, which is the whole point of P3: part of the baseline's runtime
is enumeration bookkeeping rather than search.

Run:
  python experiments/synthetic/analyze_equal_footing.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_study import STUDY_ROOT, StudyAnalysis

#: The frozen target ladder, as the string keys the driver wrote.
TAUS = ("1", "0.1", "0.01")

#: The locked default configuration of each method, the point at which the
#: sweep is read when a single configuration is needed.
DEFAULT_CONFIG = {
    "grid": "grid@max_combinations=60000",
    "grid_capped": "grid_capped@max_combinations=60000",
    "slsqp": "slsqp@n_starts=100",
    "basin_hopping": "basin_hopping@n_hops=25,n_starts=1",
    "genetic_algorithm": "genetic_algorithm@n_generations=60,population_size=50",
    "mdbh": "mdbh@eta=1.0,n_hops=10,n_local_steps=50,n_starts=5,sigma=1.5,temperature=1.0",
}

OUT = STUDY_ROOT / "analysis_equalfooting"


def load(path=None):
    """
    This function reads the A4 store into two frames, one per family.
    :param path: the results store; defaults to the A4 store next to the locked one
    :return: a tuple of the synthetic frame and the packaged frame
    """
    path = Path(path or STUDY_ROOT / "results_equalfooting.jsonl")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    frame = pd.DataFrame(rows)
    return frame[frame["family"] == "synthetic"].copy(), frame[frame["family"] == "packaged"].copy()


def first_crossing(staircase, reference, tau):
    """
    This function reads the cost at which a run first came within ``tau`` of a
    reference, from its stored best-so-far staircase.
    :param staircase: the stored list of steps
    :param reference: the reference optimum
    :param tau: the tolerance in appreciation points
    :return: a dict with evals and seconds, or None if the target was never met
    """
    threshold = reference - tau
    for step in staircase or []:
        if step["f"] >= threshold:
            return {"evals": step["evals"], "seconds": step["seconds"]}
    return None


def crossing_column(frame, tau):
    """
    This function extracts the evaluations to one target as a numeric column,
    with NaN where the target was never reached.
    :param frame: a results frame carrying a cost_to_target column
    :param tau: the target key
    :return: a float Series of evaluation counts
    """
    return frame["cost_to_target"].map(lambda c: np.nan if not c or not c.get(tau) else c[tau]["evals"])


def per_eval_seconds(frame):
    """
    This function estimates the cost of one objective evaluation per case and
    scenario, from the continuous methods only.

    The continuous solvers do almost nothing per evaluation beyond calling the
    objective, so their wall-clock over their evaluation count is a clean estimate
    of what one evaluation costs. Comparing that against the baseline's wall-clock
    is what isolates its enumeration overhead.
    :param frame: a results frame
    :param frame: the frame to estimate from
    :return: a dict keyed by (case_name, scenario)
    """
    continuous = frame[frame["method"].isin(["slsqp", "basin_hopping", "mdbh"])]
    estimate = {}
    for key, group in continuous.groupby(["case_name", "scenario"]):
        usable = group[(group["n_evals_traced"] > 0) & (~group["censored"])]
        if len(usable):
            estimate[key] = float(np.median(usable["wall_time_s"] / usable["n_evals_traced"]))
    return estimate


def overhead_share(frame):
    """
    This function reports, per method and k, the share of wall-clock that was not
    objective evaluation (P3).
    :param frame: a results frame
    :return: a DataFrame of median overhead shares
    """
    cost = per_eval_seconds(frame)
    records = []
    for row in frame.itertuples():
        unit = cost.get((row.case_name, row.scenario))
        if unit is None or row.censored or not row.n_evals_traced:
            continue
        predicted = row.n_evals_traced * unit
        records.append(
            {
                "method": row.method,
                "k": row.k,
                "share_not_evaluation": max(0.0, 1.0 - predicted / row.wall_time_s),
            }
        )
    if not records:
        return pd.DataFrame(columns=["method", "k", "median_share", "n"])
    made = pd.DataFrame(records)
    out = made.groupby(["method", "k"])["share_not_evaluation"].agg(["median", "count"]).reset_index()
    return out.rename(columns={"median": "median_share", "count": "n"})


def frontier(frame):
    """
    This function builds each method's quality-cost frontier: one point per sweep
    configuration, summarised over the cells it ran on.
    :param frame: the synthetic frame
    :return: a DataFrame with one row per method, k and configuration
    """
    clean = frame[~frame["censored"]]
    grouped = clean.groupby(["method", "k", "config_label"]).agg(
        median_evals=("n_evals_traced", "median"),
        median_gap=("gap", "median"),
        median_seconds=("wall_time_s", "median"),
        recovered=("recovered", "mean"),
        n=("gap", "size"),
    )
    return grouped.reset_index().sort_values(["method", "k", "median_evals"])


def data_profile(frame, tau):
    """
    This function builds a data profile: the fraction of cells that reached the
    target, per method and k, with a Wilson interval.
    :param frame: the synthetic frame, already reduced to one configuration per method
    :param tau: the target key
    :return: a DataFrame of reached fractions with confidence bounds
    """
    reached = crossing_column(frame, tau).notna()
    work = frame.assign(reached=reached)
    records = []
    for (method, k), group in work.groupby(["method", "k"]):
        successes = int(group["reached"].sum())
        total = int(len(group))
        point, low, high = StudyAnalysis.wilson_ci(successes, total)
        records.append(
            {
                "tau": tau,
                "method": method,
                "k": k,
                "reached": successes,
                "n": total,
                "fraction": point,
                "wilson_low": low,
                "wilson_high": high,
                "median_evals": float(np.nanmedian(crossing_column(group, tau))) if successes else np.nan,
            }
        )
    return pd.DataFrame(records)


def paired_cost(frame, tau, reference="slsqp"):
    """
    This function compares every method's cost to target against a reference on
    identical cells, with a paired Wilcoxon test and a Holm correction.
    :param frame: the synthetic frame reduced to one configuration per method
    :param tau: the target key
    :param reference: the method to compare against
    :return: a DataFrame of paired comparisons
    """
    work = frame.assign(evals_to_target=crossing_column(frame, tau))
    key = ["case_name", "scenario"]
    base = work[work["method"] == reference].set_index(key)["evals_to_target"]
    records, pvalues = [], []
    for method, group in work.groupby("method"):
        if method == reference:
            continue
        mine = group.set_index(key)["evals_to_target"]
        joined = pd.concat([base.rename("ref"), mine.rename("mine")], axis=1).dropna()
        if len(joined) < 5:
            continue
        statistic, pvalue = stats.wilcoxon(joined["mine"], joined["ref"])
        records.append(
            {
                "tau": tau,
                "method": method,
                "n_pairs": len(joined),
                "median_ratio": float(np.median(joined["mine"] / joined["ref"])),
                "statistic": float(statistic),
                "p_raw": float(pvalue),
            }
        )
        pvalues.append(pvalue)
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records)
    out["p_holm"] = StudyAnalysis.holm(pvalues)
    return out


def packaged_decomposition(packaged):
    """
    This function performs the constraint decomposition on the packaged cases:
    how much of the baseline's gap was the smaller search space rather than the
    method (P1).
    :param packaged: the packaged frame
    :return: a tuple of the decomposition frame and the per-cell reference values
    """
    references = packaged.groupby(["case_name", "scenario"])["f_found"].max().to_dict()
    records = []
    for (case_name, scenario), group in packaged.groupby(["case_name", "scenario"]):
        reference = references[(case_name, scenario)]
        found = group.set_index("method")["f_found"].to_dict()
        if "grid" not in found or "grid_capped" not in found:
            continue
        gap_face = reference - found["grid"]
        gap_capped = reference - found["grid_capped"]
        records.append(
            {
                "case_name": case_name,
                "k": int(group["k"].iloc[0]),
                "scenario": scenario,
                "reference": reference,
                "gap_face": gap_face,
                "gap_capped": gap_capped,
                "closed_by_capping": gap_face - gap_capped,
                "spend_face": float(group[group["method"] == "grid"]["spend_fraction"].iloc[0] or np.nan),
                "spend_capped": float(group[group["method"] == "grid_capped"]["spend_fraction"].iloc[0] or np.nan),
            }
        )
    return pd.DataFrame(records), references


def packaged_crossings(packaged, references):
    """
    This function fills in the packaged half's cost to target, which could not be
    computed during the run because the reference is the best value any method
    found.
    :param packaged: the packaged frame
    :param references: the per-cell reference optima
    :return: the frame with a cost_to_target column
    """
    filled = []
    for row in packaged.itertuples():
        reference = references[(row.case_name, row.scenario)]
        filled.append({tau: first_crossing(row.staircase, reference, float(tau)) for tau in TAUS})
    packaged = packaged.copy()
    packaged["cost_to_target"] = filled
    return packaged


def measured_overhead():
    """
    This function reads the directly timed enumeration overhead, if it has been
    measured, and drops any row whose wall time far exceeds its CPU time.

    That guard is not decorative: one k=12 measurement ran while the machine was
    suspended and reported 55,016 wall seconds for work that takes minutes.
    :return: a DataFrame of overhead rows, empty when the measurement is absent
    """
    path = OUT / "enumeration_overhead.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "suspect_suspend" in frame.columns:
        frame = frame[~frame["suspect_suspend"].astype(bool)]
    return frame


def verdicts(synthetic_default, packaged_decomp, packaged_overhead, profiles):
    """
    This function scores the four predictions frozen in A4.
    :param synthetic_default: the synthetic frame at one configuration per method
    :param packaged_decomp: the packaged constraint decomposition
    :param packaged_overhead: the packaged overhead shares
    :param profiles: the concatenated data profiles
    :return: a dict of verdicts
    """
    out = {}

    has_packaged = "case_name" in packaged_decomp.columns and len(packaged_decomp)
    izz = packaged_decomp[packaged_decomp["case_name"] == "IZZ"] if has_packaged else packaged_decomp
    beer = packaged_decomp[packaged_decomp["case_name"] == "Beerwiser"] if has_packaged else packaged_decomp
    out["P1"] = {
        "izz_median_closed": float(izz["closed_by_capping"].median()) if len(izz) else None,
        "beerwiser_median_closed": float(beer["closed_by_capping"].median()) if len(beer) else None,
        "holds": (
            None
            if not (len(izz) and len(beer))
            else bool(izz["closed_by_capping"].median() > 0 and abs(beer["closed_by_capping"].median()) < 1e-6)
        ),
    }

    grid_profile = profiles[profiles["method"] == "grid"]
    coarse = grid_profile[grid_profile["tau"] == "1"]
    fine = grid_profile[(grid_profile["tau"] == "0.01") & (grid_profile["k"] >= 6)]
    out["P2"] = {
        "grid_reached_1.0_everywhere": bool(len(coarse) and (coarse["fraction"] == 1.0).all()),
        "grid_reached_0.01_at_k6_or_more": float(fine["fraction"].max()) if len(fine) else None,
        "holds": bool(len(coarse) and (coarse["fraction"] == 1.0).all())
        and bool(len(fine) and (fine["fraction"] == 0.0).all()),
    }

    if len(packaged_overhead):
        izz_overhead = packaged_overhead[
            (packaged_overhead["k"] >= 9) & (packaged_overhead["method"].isin(["grid", "grid_capped"]))
        ]
    else:
        izz_overhead = packaged_overhead

    # A4 specifies the indirect estimator, which turned out to be biased: it
    # infers the per-evaluation cost from the continuous solvers, who spend
    # slightly less per evaluation than a stand-alone probe, so the inferred
    # unit is too high and the shares come out negative. Both readings are
    # reported; the verdict is taken on the direct timing.
    inferred = float(izz_overhead["median_share"].median()) if len(izz_overhead) else None
    measured = measured_overhead()
    at_k9 = measured[(measured["k"] == 9) & (measured["method"] == "grid")] if len(measured) else measured
    out["P3"] = {
        "inferred_share_k9plus_A4_estimator": inferred,
        "inferred_estimator_note": "biased, yields negative shares; superseded by direct timing",
        "measured_enumeration_share_k9": float(at_k9["enumeration_share"].median()) if len(at_k9) else None,
        "holds": None if not len(at_k9) else bool(at_k9["enumeration_share"].median() > 0.5),
    }

    ratios = {}
    for tau in TAUS:
        work = synthetic_default.assign(evals=crossing_column(synthetic_default, tau))
        key = ["case_name", "scenario"]
        ref = work[work["method"] == "slsqp"].set_index(key)["evals"]
        mine = work[work["method"] == "mdbh"].set_index(key)["evals"]
        joined = pd.concat([ref.rename("ref"), mine.rename("mine")], axis=1).dropna()
        ratios[tau] = float(np.median(joined["mine"] / joined["ref"])) if len(joined) else None
    out["P4"] = {
        "mdbh_over_slsqp_evals": ratios,
        "holds": bool(
            ratios.get("1") is not None
            and ratios["1"] <= 2.0
            and ratios.get("0.01") is not None
            and ratios["0.01"] > 2.0
        ),
    }
    return out


def _md_table(frame):
    """
    This function renders a frame as a markdown table.

    Hand-rolled on purpose: A4 commits to an unchanged environment, and pandas
    ``to_markdown`` would pull in tabulate as a new dependency.
    :param frame: the frame to render
    :return: the markdown table as a string
    """
    if not len(frame):
        return "_No rows._"
    columns = list(frame.columns)

    def cell(value):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def _verdict(holds):
    """
    This function renders a prediction verdict, distinguishing a missing answer
    from a negative one.
    :param holds: True, False, or None when the data needed is not in yet
    :return: the verdict label
    """
    if holds is None:
        return "PENDING"
    return "HOLDS" if holds else "FAILS"


def write_cost_table(frame, out_dir):
    """
    This function writes the chapter's cost-to-target table: the median number of
    objective evaluations each method needed to come within each frozen tolerance,
    per dimension.
    :param frame: the synthetic frame reduced to one configuration per method
    :param out_dir: the directory to write the table into
    :return: the path written
    """
    ks = sorted(frame["k"].unique())
    order = ["grid", "grid_capped", "slsqp", "basin_hopping", "genetic_algorithm", "mdbh"]
    labels = {
        "grid": "Grid (budget face)",
        "grid_capped": "Grid (capped simplex)",
        "slsqp": "SLSQP multi-start",
        "basin_hopping": "Basin-hopping",
        "genetic_algorithm": "Genetic algorithm",
        "mdbh": "MDBH",
    }
    lines = [
        "% Generated by analyze_equal_footing.py - do not edit by hand.",
        "\\begin{tabular}{ll" + "c" * len(ks) + "}",
        "\\toprule",
        "Target & Method & " + " & ".join(f"$k={k}$" for k in ks) + " \\\\",
        "\\midrule",
    ]
    for tau in TAUS:
        column = crossing_column(frame, tau)
        work = frame.assign(evals=column)
        for i, method in enumerate(order):
            cells = []
            for k in ks:
                sub = work[(work["method"] == method) & (work["k"] == k)]["evals"].dropna()
                reached = len(sub) / max(1, len(work[(work["method"] == method) & (work["k"] == k)]))
                if not len(sub):
                    cells.append("--")
                elif reached < 1.0:
                    cells.append(f"{int(np.median(sub)):,}$^{{{reached:.2f}}}$")
                else:
                    cells.append(f"{int(np.median(sub)):,}")
            head = f"$\\tau={tau}$" if i == 0 else ""
            lines.append(f"{head} & {labels[method]} & " + " & ".join(cells) + " \\\\")
        if tau != TAUS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]

    target = out_dir / "tables"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "equalfooting_cost.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_report(out_dir, profiles, front, paired, decomp, overhead_synth, overhead_pack, scored, counts):
    """
    This function writes the markdown report and the csv tables.
    :param out_dir: the output directory
    :param profiles: the concatenated data profiles
    :param front: the quality-cost frontiers
    :param paired: the paired cost comparisons
    :param decomp: the packaged constraint decomposition
    :param overhead_synth: overhead shares on the synthetic half
    :param overhead_pack: overhead shares on the packaged half
    :param scored: the prediction verdicts
    :param counts: a dict of row counts for the header
    :return: the report path
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(out_dir / "data_profiles.csv", index=False)
    front.to_csv(out_dir / "frontiers.csv", index=False)
    if len(paired):
        paired.to_csv(out_dir / "paired_cost_to_target.csv", index=False)
    decomp.to_csv(out_dir / "constraint_decomposition.csv", index=False)
    overhead_synth.to_csv(out_dir / "overhead_synthetic.csv", index=False)
    overhead_pack.to_csv(out_dir / "overhead_packaged.csv", index=False)

    lines = [
        "# A4 equal-footing study: results",
        "",
        "Exploratory supplement to the locked analysis. It revises no locked verdict.",
        f"Rows read: {counts['synthetic']:,} synthetic and {counts['packaged']} packaged.",
        "",
        "## Predictions",
        "",
        "| prediction | verdict | evidence |",
        "| --- | --- | --- |",
    ]
    evidence = {
        "P1": f"IZZ median closed by capping {scored['P1']['izz_median_closed']}, "
        f"Beerwiser {scored['P1']['beerwiser_median_closed']}",
        "P2": f"grid reached 1.0 everywhere: {scored['P2']['grid_reached_1.0_everywhere']}, "
        f"best fraction reaching 0.01 at k>=6: {scored['P2']['grid_reached_0.01_at_k6_or_more']}",
        "P3": f"measured share of the solve that is lattice construction at k=9: "
        f"{scored['P3']['measured_enumeration_share_k9']}",
        "P4": f"mdbh/slsqp evaluations to target: {scored['P4']['mdbh_over_slsqp_evals']}",
    }
    for name in ("P1", "P2", "P3", "P4"):
        lines.append(f"| {name} | {_verdict(scored[name]['holds'])} | {evidence[name]} |")

    decomp_block = _md_table(decomp) if len(decomp) else "_The packaged half has not run yet._"
    lines += ["", "## Constraint decomposition on the packaged cases", "", decomp_block]
    lines += ["", "## Overhead: the share of wall-clock that is not objective evaluation", ""]
    pack_block = _md_table(overhead_pack) if len(overhead_pack) else "_Not run yet._"
    lines += ["### Packaged", "", pack_block]
    lines += ["", "### Synthetic", "", _md_table(overhead_synth)]
    lines += ["", "## Data profiles at the frozen targets", "", _md_table(profiles)]
    if len(paired):
        lines += ["", "## Paired cost to target against SLSQP", "", _md_table(paired)]

    report = out_dir / "RESULTS_EQUALFOOTING.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    """Analyse the A4 store and write the report."""
    synthetic, packaged = load()
    if packaged.empty and synthetic.empty:
        raise SystemExit("no rows in the A4 store yet")

    defaults = list(DEFAULT_CONFIG.values())
    synthetic_default = synthetic[synthetic["config_label"].isin(defaults)]

    profiles = pd.concat([data_profile(synthetic_default, tau) for tau in TAUS], ignore_index=True)
    front = frontier(synthetic)
    paired = pd.concat([paired_cost(synthetic_default, tau) for tau in TAUS], ignore_index=True)
    overhead_synth = overhead_share(synthetic)

    if packaged.empty:
        decomp, overhead_pack = pd.DataFrame(), pd.DataFrame(columns=["method", "k", "median_share", "n"])
    else:
        decomp, references = packaged_decomposition(packaged)
        packaged = packaged_crossings(packaged, references)
        overhead_pack = overhead_share(packaged)

    scored = verdicts(synthetic_default, decomp, overhead_pack, profiles)
    counts = {"synthetic": len(synthetic), "packaged": len(packaged)}
    write_cost_table(synthetic_default, OUT)
    report = write_report(OUT, profiles, front, paired, decomp, overhead_synth, overhead_pack, scored, counts)
    for name in ("P1", "P2", "P3", "P4"):
        print(f"  {name}: {_verdict(scored[name]['holds'])}  {scored[name]}")
    print(f"[a4] report at {report}")


if __name__ == "__main__":
    main()
