"""
Analysis layer for the confirmatory synthetic study (pre-registration v1.2, §4).

This module turns the frozen results store written by ``run_confirmatory.py``
into every number, table and figure that the results chapter reports. It is the
single source of truth: the tables and the figures are generated from the same
frozen rows in one pass, so a number in the text cannot drift from the figure
beside it.

The analyses are exactly the ones pre-registered in §4, applied to the
hypotheses of §1 that the locked grid covers (H1a, H1b, H2, H5). H3/H4 concern
the smooth_nonconvex and nonsmooth regimes, which §7 defers to a later
amendment, so they are not touched here.

Run:
    python analyze_study.py

Outputs (under ``generated/study/analysis/``):
    RESULTS.md          human-readable summary incl. the hypothesis decisions
    summary.json        the same numbers, machine-readable
    tables/*.tex        booktabs tables for the thesis
    figures/*.png|pdf   thesis-styled figures
"""

from __future__ import annotations

import json
from math import comb, exp, factorial, isfinite, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from study_report import StudyReport
from vlinder.optimize import GridSearch

# Headless rendering; switching after import keeps every import at the top.
plt.switch_backend("Agg")

STUDY_ROOT = Path(__file__).resolve().parent / "generated" / "study"
RECOVERY_EPS = 0.1
RUNTIME_CAP_S = 600.0
PRIMARY_SCENARIO = "Scenario 01"
GRID_MAX_COMBINATIONS = 60000
BUDGET = 100.0
Z_95 = 1.959963984540054

METHOD_LABELS = {
    "grid": "Grid (baseline)",
    "slsqp": "SLSQP multi-start",
    "basin_hopping": "Basin-hopping",
    "genetic_algorithm": "Genetic algorithm",
}
METHOD_ORDER = ["grid", "slsqp", "basin_hopping", "genetic_algorithm"]
METHOD_COLORS = {
    "grid": "#B00020",
    "slsqp": "#0B5FA5",
    "basin_hopping": "#127A52",
    "genetic_algorithm": "#C46A00",
}
METHOD_MARKERS = {"grid": "s", "slsqp": "o", "basin_hopping": "^", "genetic_algorithm": "D"}
VARIANT_LABELS = {"linear": "convex-linear", "sinusoidal": "convex-curved"}


def _style():
    """Apply a restrained, print-friendly figure style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


class StudyAnalysis:
    """
    Computes the pre-registered §4 analyses over the frozen confirmatory results
    and writes the tables, figures and decision log.

    :param root: the study directory holding ``results.jsonl``
    """

    def __init__(self, root: Path = STUDY_ROOT):
        self.root = Path(root)
        self.out = self.root / "analysis"
        (self.out / "tables").mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        self.df = self._load()
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.summary: dict = {}
        # Exposed so the report renders from the analysis rather than
        # re-importing this module's constants.
        self.recovery_eps = RECOVERY_EPS
        self.primary_scenario = PRIMARY_SCENARIO
        self.method_order = METHOD_ORDER
        self.method_labels = METHOD_LABELS

    def _load(self) -> pd.DataFrame:
        """Read the frozen results store into a tidy frame."""
        path = self.root / "results.jsonl"
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        frame = pd.DataFrame(rows)
        frame["method"] = pd.Categorical(frame["method"], categories=METHOD_ORDER, ordered=True)
        return frame

    # ------------------------------------------------------------------
    # Statistics (hand-rolled so the locked environment is not perturbed)
    # ------------------------------------------------------------------
    @staticmethod
    def wilson_ci(successes: int, n: int, z: float = Z_95):
        """
        Wilson score interval for a binomial proportion, the pre-registered
        interval for per-cell recovery rates.
        :param successes: number of recoveries
        :param n: number of trials in the cell
        :param z: normal quantile (default 95%)
        :return: (point estimate, lower bound, upper bound)
        """
        if n == 0:
            return float("nan"), float("nan"), float("nan")
        p = successes / n
        denom = 1.0 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return p, max(0.0, centre - half), min(1.0, centre + half)

    @staticmethod
    def ols_hc3(x: np.ndarray, y: np.ndarray) -> dict:
        """
        Simple OLS of ``y`` on ``[1, x]`` with HC3 heteroskedasticity-consistent
        standard errors (MacKinnon-White), as pre-registered for the scaling
        regressions.
        :param x: regressor (here: k)
        :param y: response (here: log runtime or log evaluations)
        :return: slope, intercept, HC3 SE of the slope, 95% CI, n and R^2
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        design = np.column_stack([np.ones(n), x])
        xtx_inv = np.linalg.inv(design.T @ design)
        beta = xtx_inv @ design.T @ y
        resid = y - design @ beta
        leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
        omega = (resid / (1.0 - leverage)) ** 2
        cov = xtx_inv @ (design.T * omega) @ design @ xtx_inv
        se_slope = float(sqrt(cov[1, 1]))
        total = float(((y - y.mean()) ** 2).sum())
        r_squared = float(1.0 - (resid**2).sum() / total) if total > 0 else float("nan")
        return {
            "intercept": float(beta[0]),
            "slope": float(beta[1]),
            "se_slope_hc3": se_slope,
            "ci_low": float(beta[1] - Z_95 * se_slope),
            "ci_high": float(beta[1] + Z_95 * se_slope),
            "n": int(n),
            "r_squared": r_squared,
        }

    @staticmethod
    def holm(pvalues: list) -> list:
        """
        Holm-Bonferroni step-down adjustment, applied across the k grid as
        pre-registered.
        :param pvalues: raw p-values
        :return: adjusted p-values in the input order
        """
        m = len(pvalues)
        order = sorted(range(m), key=lambda i: pvalues[i])
        adjusted = [0.0] * m
        running = 0.0
        for rank, idx in enumerate(order):
            value = (m - rank) * pvalues[idx]
            running = max(running, value)
            adjusted[idx] = min(1.0, running)
        return adjusted

    # ------------------------------------------------------------------
    # Grid's analytic cost curve
    # ------------------------------------------------------------------
    @staticmethod
    def grid_analytic_cost(k: int, budget: float = BUDGET) -> dict:
        """
        The baseline's enumeration cost at k levers, computed from the shipped
        implementation rather than assumed: ``Optimize`` picks the step size, so
        the resolution and the resulting number of budget-face grid points
        C(units+k-1, k-1) are exact for the study's budget.

        ``generate_combinations`` expands every multiset through
        ``set(permutations(...))``, so the realised iteration count carries an
        additional factorial factor; both are reported (amendment A1).
        :param k: number of internal variables
        :param budget: case budget
        :return: resolution, grid points, and the permutation-expansion bound
        """
        scaled = GridSearch.scale_max_investment(budget)
        step = GridSearch.calculate_step_size(budget, scaled, k, GRID_MAX_COMBINATIONS)
        units = int(round(budget / step))
        points = comb(units + k - 1, k - 1)
        return {
            "k": k,
            "units": units,
            "grid_points": points,
            "permutation_bound": float(points) * float(factorial(k)),
        }

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------
    def recovery(self, primary_only: bool = True) -> pd.DataFrame:
        """
        Per-cell recovery rate with a Wilson 95% interval. Censored cells count
        as non-recovery, exactly as §3 and amendment A1 prescribe.
        :param primary_only: restrict to the primary scenario (§2)
        :return: one row per variant x k x method
        """
        frame = self.df
        if primary_only:
            frame = frame[frame.scenario == PRIMARY_SCENARIO]
        records = []
        for (variant, k, method), cell in frame.groupby(["variant", "k", "method"], observed=True):
            n = len(cell)
            successes = int(cell.recovered.sum())
            point, low, high = self.wilson_ci(successes, n)
            records.append(
                {
                    "variant": variant,
                    "k": k,
                    "method": method,
                    "n": n,
                    "recovered": successes,
                    "censored": int(cell.censored.sum()),
                    "recovery": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        return pd.DataFrame(records).sort_values(["variant", "method", "k"]).reset_index(drop=True)

    def pooled_recovery(self) -> pd.DataFrame:
        """
        Recovery pooled over the scenarios of each case, which is the basis the
        decision rules of §4 are stated on ("pooled recovery Wilson CI").

        The distinction matters and is not cosmetic. The per-cell table uses the
        30 seeds of the primary scenario, and at n = 30 even flawless recovery
        has a Wilson lower bound of 0.886, so a ">= 0.95" rule could never be
        met by any method however good. Pooling a case's three scenarios gives
        n = 90 per variant x k, where perfect recovery clears the threshold at
        0.959, which is the only reading under which the pre-registered rule
        discriminates between methods.
        :return: one row per variant x k x method, pooled over scenarios
        """
        records = []
        for (variant, k, method), cell in self.df.groupby(["variant", "k", "method"], observed=True):
            n = len(cell)
            successes = int(cell.recovered.sum())
            point, low, high = self.wilson_ci(successes, n)
            records.append(
                {
                    "variant": variant,
                    "k": k,
                    "method": method,
                    "n": n,
                    "recovered": successes,
                    "censored": int(cell.censored.sum()),
                    "recovery": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        return pd.DataFrame(records).sort_values(["variant", "method", "k"]).reset_index(drop=True)

    def gaps(self, primary_only: bool = True) -> pd.DataFrame:
        """
        Median and IQR of the optimality gap per cell, over the cells that
        produced a value (censored cells have no gap and are counted, not
        imputed).
        :param primary_only: restrict to the primary scenario
        :return: one row per variant x k x method
        """
        frame = self.df[~self.df.censored]
        if primary_only:
            frame = frame[frame.scenario == PRIMARY_SCENARIO]
        records = []
        for (variant, k, method), cell in frame.groupby(["variant", "k", "method"], observed=True):
            gap = cell.gap.astype(float)
            records.append(
                {
                    "variant": variant,
                    "k": k,
                    "method": method,
                    "n": len(cell),
                    "gap_median": float(gap.median()),
                    "gap_q1": float(gap.quantile(0.25)),
                    "gap_q3": float(gap.quantile(0.75)),
                    "gap_max": float(gap.max()),
                    "runtime_median_s": float(cell.wall_time_s.astype(float).median()),
                }
            )
        return pd.DataFrame(records).sort_values(["variant", "method", "k"]).reset_index(drop=True)

    def scaling(self) -> pd.DataFrame:
        """
        Pre-registered scaling regressions: log(runtime) and log(evaluations) on
        k per method, OLS with HC3 standard errors.

        Censored observations carry no runtime, so they leave the regression.
        For the baseline this is survivorship: the cells that were dropped are
        precisely the most expensive ones, so its fitted growth rate is a lower
        bound. That is flagged in the output rather than corrected.
        :return: one row per method x response
        """
        records = []
        usable = self.df[(~self.df.censored) & self.df.wall_time_s.notna()]
        for method, cell in usable.groupby("method", observed=True):
            for response, column in (("log_runtime", "wall_time_s"), ("log_evals", "n_function_evals")):
                sub = cell[cell[column].notna()]
                sub = sub[sub[column].astype(float) > 0]
                if len(sub) < 10 or sub.k.nunique() < 3:
                    continue
                y = np.log(sub[column].astype(float).to_numpy())
                censored_n = int(self.df[(self.df.method == method) & self.df.censored].shape[0])
                # A response that never varies (the GA spends a fixed evaluation
                # budget by construction) carries no regression information; it is
                # recorded as constant rather than reported as a degenerate fit.
                if float(np.var(y)) == 0.0:
                    records.append(
                        {
                            "method": method,
                            "response": response,
                            "spec": "constant",
                            "slope_per_lever": 0.0,
                            "factor_per_lever": 1.0,
                            "se_hc3": float("nan"),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                            "r_squared": float("nan"),
                            "n_obs": int(len(sub)),
                            "n_censored_excluded": censored_n,
                            "survivorship_biased": censored_n > 0,
                            "note": "fixed by design (constant evaluation budget), not estimated",
                        }
                    )
                    continue
                # Pre-registered spec: log(y) on k. Its slope is a growth rate per
                # added lever. The log-log spec is added as a clearly-labelled
                # supplement because H5's second clause is stated as growth that is
                # "~linear in k", which is a statement about an exponent and cannot
                # be read off a semi-log slope.
                for spec, regressor in (
                    ("semilog_preregistered", sub.k.to_numpy()),
                    ("loglog_supplementary", np.log(sub.k.to_numpy().astype(float))),
                ):
                    fit = self.ols_hc3(regressor, y)
                    records.append(
                        {
                            "method": method,
                            "response": response,
                            "spec": spec,
                            "slope_per_lever": fit["slope"],
                            "factor_per_lever": exp(fit["slope"]) if spec == "semilog_preregistered" else float("nan"),
                            "se_hc3": fit["se_slope_hc3"],
                            "ci_low": fit["ci_low"],
                            "ci_high": fit["ci_high"],
                            "r_squared": fit["r_squared"],
                            "n_obs": fit["n"],
                            "n_censored_excluded": censored_n,
                            "survivorship_biased": censored_n > 0,
                            "note": "",
                        }
                    )
        return pd.DataFrame(records)

    def paired_tests(self, reference: str = "slsqp") -> pd.DataFrame:
        """
        Paired Wilcoxon signed-rank tests of the reference method against each
        other method on identical cases, per k, Holm-corrected across the k grid
        (§4). Pairs where either method is censored are dropped and counted.
        :param reference: the method every other method is compared against
        :return: one row per comparison x k, with Holm-adjusted p-values
        """
        wide = self.df[self.df.scenario == PRIMARY_SCENARIO].pivot_table(
            index=["variant", "k", "case_seed"], columns="method", values="gap", observed=True, dropna=False
        )
        records = []
        for method in METHOD_ORDER:
            if method == reference:
                continue
            per_k = []
            for k in sorted(self.df.k.unique()):
                block = wide.xs(k, level="k")[[reference, method]].dropna()
                n_pairs = len(block)
                dropped = int((wide.xs(k, level="k")[[reference, method]].isna().any(axis=1)).sum())
                diff = block[method].to_numpy() - block[reference].to_numpy()
                if n_pairs == 0 or np.allclose(diff, 0.0):
                    p_raw, statistic = 1.0, float("nan")
                else:
                    statistic, p_raw = stats.wilcoxon(block[method], block[reference], zero_method="wilcox")
                per_k.append(
                    {
                        "comparison": f"{method} vs {reference}",
                        "k": int(k),
                        "n_pairs": n_pairs,
                        "n_dropped_censored": dropped,
                        "median_gap_diff": float(np.median(diff)) if n_pairs else float("nan"),
                        "statistic": float(statistic),
                        "p_raw": float(p_raw),
                    }
                )
            for row, adjusted in zip(per_k, self.holm([r["p_raw"] for r in per_k])):
                row["p_holm"] = adjusted
                row["significant_05"] = bool(adjusted < 0.05)
            records.extend(per_k)
        return pd.DataFrame(records)

    def oracle_tightness(self) -> dict:
        """
        Report how often a method exceeded the certified optimum and by how
        much. A negative gap beyond solver noise bounds how tight the oracle is;
        §3 forbids silently truncating these to zero.
        :return: counts and the largest exceedance
        """
        valid = self.df[~self.df.censored]
        beyond = valid[valid.gap < -1e-6]
        return {
            "n_negative_gaps": int((valid.gap < 0).sum()),
            "n_beyond_noise_1e-6": int(len(beyond)),
            "largest_exceedance": float(-valid.gap.min()),
            "share_beyond_noise": float(len(beyond) / len(valid)),
            "note": (
                "Largest exceedance is far below the recovery threshold "
                f"eps={RECOVERY_EPS}, so no recovery decision is affected; it bounds oracle tightness."
            ),
        }

    @staticmethod
    def _decide_h1(recovery: pd.DataFrame, variant: str) -> dict:
        """
        Decide one of the two H1 cells: SLSQP recovers the certified optimum at
        every k in this regime variant.
        :param recovery: the pooled recovery table
        :param variant: regime variant
        :return: the verdict entry
        """
        cells = recovery[(recovery.variant == variant) & (recovery.method == "slsqp")]
        worst = cells.loc[cells.ci_low.idxmin()]
        return {
            "hypothesis": (
                f"SLSQP recovers the certified optimum in >=99% of {VARIANT_LABELS[variant]} cases at every k"
            ),
            "rule": "supported if the Wilson lower bound is >= 0.95 at every k",
            "verdict": "SUPPORTED" if bool((cells.ci_low >= 0.95).all()) else "NOT SUPPORTED",
            "min_recovery": float(cells.recovery.min()),
            "min_ci_low": float(worst.ci_low),
            "worst_k": int(worst.k),
        }

    @staticmethod
    def _decide_h2(recovery: pd.DataFrame, h1_holds: bool) -> dict:
        """
        Decide H2: the baseline degrades in k while SLSQP holds H1.
        :param recovery: the pooled recovery table
        :param h1_holds: whether both H1 cells were supported
        :return: the verdict entry
        """
        grid = recovery[recovery.method == "grid"]
        fails = grid[grid.ci_high < 0.50]
        censored_k = sorted(int(k) for k in grid[grid.censored > 0].k.unique())
        return {
            "hypothesis": "Grid recovery degrades with k while SLSQP runtime grows at most polynomially",
            "rule": "supported if grid recovery CI upper bound < 0.5 at some k, or grid is censored, while H1 holds",
            "verdict": "SUPPORTED" if bool((len(fails) > 0 or censored_k) and h1_holds) else "NOT SUPPORTED",
            "grid_censored_at_k": censored_k,
            "k_with_ci_high_below_50pct": sorted(int(k) for k in fails.k.unique()),
            "grid_recovery_at_max_k": float(grid[grid.k == grid.k.max()].recovery.mean()),
            "h1_holds": h1_holds,
        }

    @staticmethod
    def _decide_h5(scaling: pd.DataFrame) -> dict:
        """
        Decide H5, which has two clauses: the baseline grows faster than SLSQP
        (pre-registered semi-log slopes), and SLSQP's evaluation count grows
        about linearly in k. The second is a claim about an exponent, so it is
        read off the supplementary log-log fit and its interval must contain 1.
        :param scaling: the scaling regression table
        :return: the verdict entry
        """

        def slope(spec: str, response: str, method: str, column: str = "slope_per_lever") -> float:
            """Pull one fitted statistic, or NaN when that fit does not exist."""
            rows = scaling[(scaling.spec == spec) & (scaling.response == response) & (scaling.method == method)]
            return float(rows.iloc[0][column]) if len(rows) else float("nan")

        grid_slope = slope("semilog_preregistered", "log_runtime", "grid")
        slsqp_slope = slope("semilog_preregistered", "log_runtime", "slsqp")
        exponent = slope("loglog_supplementary", "log_evals", "slsqp")
        ci = (
            slope("loglog_supplementary", "log_evals", "slsqp", "ci_low"),
            slope("loglog_supplementary", "log_evals", "slsqp", "ci_high"),
        )
        clause_1 = bool(isfinite(grid_slope) and isfinite(slsqp_slope) and grid_slope > slsqp_slope)
        clause_2 = bool(isfinite(exponent) and ci[0] <= 1.0 <= ci[1])
        return {
            "hypothesis": "Grid runtime grows super-linearly in k; SLSQP evaluations grow ~linearly at fixed recovery",
            "rule": "clause 1 on the pre-registered semi-log slopes; clause 2 needs an exponent, so it is read off "
            "the supplementary log-log fit (CI must contain 1)",
            "verdict": "SUPPORTED" if (clause_1 and clause_2) else ("PARTIALLY SUPPORTED" if clause_1 else "MIXED"),
            "clause_1_grid_grows_faster": clause_1,
            "clause_2_slsqp_evals_linear_in_k": clause_2,
            "grid_log_runtime_slope": grid_slope,
            "grid_runtime_factor_per_lever": exp(grid_slope) if isfinite(grid_slope) else float("nan"),
            "slsqp_log_runtime_slope": slsqp_slope,
            "slsqp_runtime_factor_per_lever": exp(slsqp_slope) if isfinite(slsqp_slope) else float("nan"),
            "slsqp_evals_loglog_exponent": exponent,
            "slsqp_evals_loglog_ci": list(ci),
            "caveat": "The grid slope is a lower bound: its most expensive cells are censored out of the regression.",
        }

    def decisions(self, recovery: pd.DataFrame, scaling: pd.DataFrame) -> dict:
        """
        Apply the pre-registered decision rules (§4) to the computed cells and
        log a verdict per hypothesis.
        :param recovery: the pooled recovery table the decision rules are stated on
        :param scaling: the scaling regression table
        :return: one entry per hypothesis with verdict and evidence
        """
        verdicts = {
            "H1a": self._decide_h1(recovery, "linear"),
            "H1b": self._decide_h1(recovery, "sinusoidal"),
        }
        h1_holds = all(verdicts[h]["verdict"] == "SUPPORTED" for h in ("H1a", "H1b"))
        verdicts["H2"] = self._decide_h2(recovery, h1_holds)
        verdicts["H5"] = self._decide_h5(scaling)
        return verdicts

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    def figure_recovery(self, recovery: pd.DataFrame):
        """Recovery against k, per method, one panel per regime variant."""
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
        for ax, variant in zip(axes, ["linear", "sinusoidal"]):
            cells = recovery[recovery.variant == variant]
            for position, method in enumerate(METHOD_ORDER):
                sub = cells[cells.method == method].sort_values("k")
                if sub.empty:
                    continue
                # Several methods sit exactly on 1.0, so without a small horizontal
                # dodge the later series simply paint over the earlier ones and the
                # reader cannot tell that they are present at all.
                dodge = (position - 1.5) * 0.12
                # Wilson bounds are not symmetric about the point estimate and the
                # clamp to [0, 1] can round a bound a hair past it, so clip the
                # error-bar lengths at zero.
                lower = np.clip(sub.recovery - sub.ci_low, 0.0, None)
                upper = np.clip(sub.ci_high - sub.recovery, 0.0, None)
                ax.errorbar(
                    sub.k + dodge,
                    sub.recovery,
                    yerr=[lower, upper],
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    markersize=4,
                    linewidth=1.3,
                    capsize=2,
                    elinewidth=0.8,
                )
            ax.set_title(VARIANT_LABELS[variant])
            ax.set_xlabel("number of internal variables $k$")
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks(sorted(cells.k.unique()))
        axes[0].set_ylabel(f"recovery rate (gap $\\leq$ {RECOVERY_EPS})")
        axes[1].legend(loc="lower left", frameon=False)
        self._save(fig, "fig1_recovery_vs_k")

    def figure_scaling(self):
        """Runtime and evaluation scaling in k, with the analytic baseline curve."""
        usable = self.df[(~self.df.censored) & self.df.wall_time_s.notna()]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

        for method in METHOD_ORDER:
            sub = usable[usable.method == method]
            if sub.empty:
                continue
            med = sub.groupby("k", observed=True).wall_time_s.median()
            axes[0].plot(
                med.index,
                med.to_numpy(),
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                markersize=4,
                linewidth=1.3,
            )
        censored_ks = sorted(self.df[(self.df.method == "grid") & self.df.censored].k.unique())
        if censored_ks:
            axes[0].axvspan(
                min(censored_ks) - 0.35,
                max(censored_ks) + 0.35,
                color=METHOD_COLORS["grid"],
                alpha=0.07,
                zorder=0,
            )
            # Annotations are positioned in axes fractions: reading limits back off
            # a log axis mid-build gives a degenerate tight bounding box on save.
            axes[0].text(
                0.98,
                0.03,
                "grid censored",
                transform=axes[0].transAxes,
                color=METHOD_COLORS["grid"],
                fontsize=7,
                va="bottom",
                ha="right",
            )
        axes[0].axhline(RUNTIME_CAP_S, color="0.35", linestyle=":", linewidth=1.0)
        axes[0].text(0.02, 0.93, "600 s cap", transform=axes[0].transAxes, fontsize=7, color="0.35", va="top")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("number of internal variables $k$")
        axes[0].set_ylabel("median wall-clock time (s)")
        axes[0].set_title("Runtime (censored cells excluded)")
        axes[0].set_xticks(sorted(self.df.k.unique()))

        for method in METHOD_ORDER:
            sub = usable[(usable.method == method) & usable.n_function_evals.notna()]
            if sub.empty:
                continue
            med = sub.groupby("k", observed=True).n_function_evals.median()
            # Unlabelled: the methods are identified once, in the shared legend
            # below the figure, so this panel's legend carries only the analytic
            # baseline curve.
            axes[1].plot(
                med.index,
                med.to_numpy(),
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                markersize=4,
                linewidth=1.3,
            )
        # The baseline reports no evaluation count, so its cost is analytic. The
        # honest quantity is the enumeration it actually performs: every budget
        # split is expanded through set(permutations(...)), so the point count is
        # multiplied by up to k!. The point count alone is non-monotone in k and
        # would misrepresent the cost, because the implementation keeps it under
        # max_combinations by coarsening the step size as k grows.
        ks = sorted(self.df.k.unique())
        analytic = [self.grid_analytic_cost(int(k)) for k in ks]
        axes[1].plot(
            ks,
            [a["permutation_bound"] for a in analytic],
            color=METHOD_COLORS["grid"],
            linestyle="--",
            linewidth=1.3,
            marker=METHOD_MARKERS["grid"],
            markersize=3.5,
            markerfacecolor="white",
            label="Grid enumeration (analytic bound)",
        )
        axes[1].set_yscale("log")
        axes[1].set_xlabel("number of internal variables $k$")
        axes[1].set_ylabel("median objective evaluations")
        axes[1].set_title("Cost in evaluations")
        axes[1].set_xticks(ks)
        axes[1].legend(loc="upper left", frameon=False, fontsize=7)
        handles = [
            plt.Line2D(
                [0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], markersize=4, label=METHOD_LABELS[m]
            )
            for m in METHOD_ORDER
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
        self._save(fig, "fig2_scaling_vs_k")

    def figure_gaps(self):
        """Distribution of the optimality gap against k, per method."""
        frame = self.df[(~self.df.censored) & (self.df.scenario == PRIMARY_SCENARIO)]
        ks = sorted(frame.k.unique())
        fig, ax = plt.subplots(figsize=(7.6, 3.2))
        width = 0.18
        for offset, method in enumerate(METHOD_ORDER):
            sub = frame[frame.method == method]
            if sub.empty:
                continue
            data = [np.clip(sub[sub.k == k].gap.astype(float).to_numpy(), 0.0, None) for k in ks]
            positions = [i + (offset - 1.5) * width for i in range(len(ks))]
            box = ax.boxplot(
                data, positions=positions, widths=width * 0.85, patch_artist=True, showfliers=False, manage_ticks=False
            )
            for patch in box["boxes"]:
                patch.set(facecolor=METHOD_COLORS[method], alpha=0.55, linewidth=0.6)
            for line in box["whiskers"] + box["caps"] + box["medians"]:
                line.set(color="0.2", linewidth=0.7)
        ax.axhline(RECOVERY_EPS, color="0.35", linestyle=":", linewidth=1.0)
        ax.text(
            0.008,
            0.97,
            f"recovery threshold $\\varepsilon$ = {RECOVERY_EPS}",
            transform=ax.transAxes,
            fontsize=7,
            color="0.35",
            ha="left",
            va="top",
        )
        # Gaps are clipped at zero for display, so the negative half of a symlog
        # axis would be empty and would waste half the panel.
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_ylim(bottom=0.0)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_xlabel("number of internal variables $k$")
        ax.set_ylabel("optimality gap (points)")
        ax.set_title("Optimality gap by method (negative gaps clipped to zero for display)")
        handles = [
            plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=5, alpha=0.55, label=METHOD_LABELS[m])
            for m in METHOD_ORDER
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
        self._save(fig, "fig3_gap_distribution")

    def _save(self, fig, stem: str):
        """
        Write a figure to png and pdf and close it.

        A degenerate layout can make ``bbox_inches="tight"`` compute an enormous
        canvas, which silently produces a file no reader or LaTeX run can open,
        so the resulting size is checked rather than trusted.
        :param fig: the figure to write
        :param stem: file name without extension
        """
        width, height = fig.get_size_inches()
        if not (0.5 < width < 20 and 0.5 < height < 20):
            raise ValueError(f"{stem}: implausible figure size {width:.1f}x{height:.1f} in; layout collapsed")
        for suffix in ("png", "pdf"):
            fig.savefig(self.out / "figures" / f"{stem}.{suffix}", bbox_inches="tight")
        png = self.out / "figures" / f"{stem}.png"
        if png.stat().st_size > 8_000_000:
            raise ValueError(f"{stem}: png is {png.stat().st_size / 1e6:.0f} MB; the tight bbox has blown up")
        plt.close(fig)

    # ------------------------------------------------------------------
    # LaTeX tables
    # ------------------------------------------------------------------
    def table_recovery(self, recovery: pd.DataFrame) -> str:
        """Booktabs table of recovery with Wilson intervals, per variant."""
        lines = [
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Regime & Method & $k$ & Recovery & 95\% Wilson CI & Censored \\",
            r"\midrule",
        ]
        for variant in ("linear", "sinusoidal"):
            for method in METHOD_ORDER:
                sub = recovery[(recovery.variant == variant) & (recovery.method == method)].sort_values("k")
                for i, row in enumerate(sub.itertuples()):
                    regime = VARIANT_LABELS[variant] if (i == 0 and method == METHOD_ORDER[0]) else ""
                    label = METHOD_LABELS[method] if i == 0 else ""
                    lines.append(
                        f"{regime} & {label} & {row.k} & {row.recovery:.3f} & "
                        f"[{row.ci_low:.3f}, {row.ci_high:.3f}] & {row.censored} \\\\"
                    )
                lines.append(r"\addlinespace")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def table_scaling(self, scaling: pd.DataFrame) -> str:
        """Booktabs table of the pre-registered HC3 scaling regressions."""

        def fmt(value, digits=3):
            """Format a possibly-missing statistic."""
            return "--" if not isfinite(value) else f"{value:.{digits}f}"

        prereg = scaling[scaling.spec.isin(["semilog_preregistered", "constant"])]
        lines = [
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Response & Method & Slope per lever & HC3 SE & 95\% CI & $R^2$ \\",
            r"\midrule",
        ]
        for response, pretty in (("log_runtime", r"$\log$ runtime"), ("log_evals", r"$\log$ evaluations")):
            sub = prereg[prereg.response == response]
            for i, row in enumerate(sub.itertuples()):
                head = pretty if i == 0 else ""
                interval = "--" if not isfinite(row.ci_low) else f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
                lines.append(
                    f"{head} & {METHOD_LABELS[row.method]} & {fmt(row.slope_per_lever)} & {fmt(row.se_hc3)} & "
                    f"{interval} & {fmt(row.r_squared)} \\\\"
                )
            lines.append(r"\addlinespace")
        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def table_paired(self, paired: pd.DataFrame) -> str:
        """Booktabs table of the Holm-corrected paired Wilcoxon comparisons."""
        lines = [
            r"\begin{tabular}{lrrrrl}",
            r"\toprule",
            r"Comparison & $k$ & Pairs & Median gap diff. & $p$ (Holm) & Sig. \\",
            r"\midrule",
        ]
        for comparison, sub in paired.groupby("comparison"):
            for i, row in enumerate(sub.sort_values("k").itertuples()):
                head = comparison.replace("_", r"\_") if i == 0 else ""
                sig = r"$\ast$" if row.significant_05 else ""
                lines.append(
                    f"{head} & {row.k} & {row.n_pairs} & {row.median_gap_diff:.2e} & {row.p_holm:.3g} & {sig} \\\\"
                )
            lines.append(r"\addlinespace")
        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Compute every analysis, write all artefacts, and return the summary."""
        _style()
        recovery = self.recovery()
        pooled = self.pooled_recovery()
        gaps = self.gaps()
        scaling = self.scaling()
        paired = self.paired_tests()
        tightness = self.oracle_tightness()
        verdicts = self.decisions(pooled, scaling)

        recovery.to_csv(self.out / "recovery_primary_scenario.csv", index=False)
        pooled.to_csv(self.out / "recovery_pooled_over_scenarios.csv", index=False)
        gaps.to_csv(self.out / "gaps_and_runtime.csv", index=False)
        scaling.to_csv(self.out / "scaling_regressions.csv", index=False)
        paired.to_csv(self.out / "paired_wilcoxon.csv", index=False)

        (self.out / "tables" / "recovery.tex").write_text(self.table_recovery(recovery), encoding="utf-8")
        (self.out / "tables" / "scaling.tex").write_text(self.table_scaling(scaling), encoding="utf-8")
        (self.out / "tables" / "paired_wilcoxon.tex").write_text(self.table_paired(paired), encoding="utf-8")

        self.figure_recovery(recovery)
        self.figure_scaling()
        self.figure_gaps()

        analytic = [self.grid_analytic_cost(int(k)) for k in sorted(self.df.k.unique())]
        self.summary = {
            "source": str(self.root / "results.jsonl"),
            "n_rows": int(len(self.df)),
            "n_censored": int(self.df.censored.sum()),
            "environment_at_generation": self.meta,
            "recovery_eps": RECOVERY_EPS,
            "primary_scenario": PRIMARY_SCENARIO,
            "hypotheses": verdicts,
            "oracle_tightness": tightness,
            "grid_analytic_cost": analytic,
            "decision_basis": (
                "Decision rules are evaluated on recovery pooled over each case's three scenarios "
                "(n=90 per variant x k). At the per-cell n=30 a >=0.95 Wilson lower bound is "
                "unreachable even under flawless recovery (max 0.886), so the pooled reading is the "
                "only one under which the pre-registered rule discriminates."
            ),
            "pooled_recovery_by_method": {
                method: float(pooled[pooled.method == method].recovery.mean()) for method in METHOD_ORDER
            },
        }
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        report = StudyReport(self).render(pooled, scaling, paired)
        (self.out / "RESULTS.md").write_text(report, encoding="utf-8")
        return self.summary


def main():
    """Run the full analysis and print the hypothesis verdicts."""
    analysis = StudyAnalysis()
    summary = analysis.run()
    print(f"rows={summary['n_rows']} censored={summary['n_censored']}")
    for hid, block in summary["hypotheses"].items():
        print(f"  {hid}: {block['verdict']}")
    print(f"artefacts -> {analysis.out}")


if __name__ == "__main__":
    main()
