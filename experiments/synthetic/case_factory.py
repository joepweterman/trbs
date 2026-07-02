"""
Synthetic tRBS case factory — Phase 0.75 (schema + native-format emitter + validator).

Part of the thesis synthetic-case generator (a known data-generating process for
the convexity-characterisation Monte-Carlo study). Lives THESIS-SIDE in
``experiments/`` — deliberately NOT inside the ``vlinder`` package (Vlinder Q3).

Deliverables:
  * ``SyntheticCaseParams`` — the tunable knob schema (a dataclass).
  * ``SyntheticCaseFactory`` — emits the 11 native tRBS CSV tables for a case so
    it flows through the real ``case_importer -> evaluate -> appreciate -> optimize``
    pipeline unchanged (the optimizer cannot tell synthetic from real), plus a
    ``manifest.json`` (params, seed, convexity claim, table hashes; the oracle
    block is patched in by ``oracle.py``).
  * ``validate_case`` — round-trips a generated case through build/evaluate/appreciate
    and checks: objective finite and non-degenerate over the capped simplex
    {x >= 0, sum x <= B}; the frozen auto-boundaries equal the analytic feasible
    envelope (NO-CLIP invariant, see below); and a short multi-start SLSQP is
    feasible and — in the convex regime — reaches consensus across starts.

Only the ``convex`` regime is implemented, in two variants:
  * ``appreciation="linear"``     — affine objective; vertex optimum.
  * ``appreciation="sinusoidal"`` — all-sinusoidal STB=0 appreciation, which is
    concave increasing on the bracket, so the objective is a concave program
    with a (generically) face-interior optimum. Structurally closer to IZZ.
``smooth_nonconvex`` / ``nonsmooth`` raise NotImplementedError on purpose — the
schema is forward-looking but the code never claims a regime it has not built.

NO-CLIP invariant (Phase-0 audit, exp03, 2026-07-02): vlinder's auto-boundaries
are the min/max of KO values over the DMO set, and appreciation hard-clips to
0/100 outside them. DMOs that under-bracket the feasible KO range therefore put
clipping kinks INSIDE the feasible set; the 0-floor clip is convex and makes
even the linear regime multimodal (k3: two SLSQP basins; k9: >=12 distinct
terminal values). The factory emits bracketing DMOs — every pure corner (all B
on one lever), zero spend, equal spread — so the auto-boundaries equal the exact
feasible envelope and no clipping can occur on the capped simplex.

Budget is the capped simplex sum x <= B (exp02 decision, 2026-06-26): the
bracketing corner DMOs spend the full budget but the optimiser may under-spend.

Reproducibility: all randomness is drawn in ``__init__`` from independent
``SeedSequence`` child streams (coefficients / KO weights / theme weights /
scenario weights), so varying one factorial knob never re-randomises unrelated
components, and ``tables()`` is idempotent.

Run:
  & "C:\\Users\\joepw\\.virtualenvs\\tRBS-DclBJWVi-python.exe\\Scripts\\python.exe" `
    C:\\Users\\joepw\\tRBS\\experiments\\synthetic\\case_factory.py
"""

# pylint: disable=invalid-name,protected-access,too-many-locals,too-many-instance-attributes
# (math notation B/X/f; Optimize internals are the documented experiment surface;
#  the factory pre-draws every random component, hence the attribute count)

from __future__ import annotations

import io
import contextlib
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.optimize import Optimize, evaluate_allocation

DEFAULT_ROOT = Path(__file__).parent / "generated"
GENERATOR_VERSION = "0.75"

# Boilerplate report configuration mirrored from the bundled cases (unused by
# build/evaluate/appreciate, but kept faithful so generated cases also report).
_REPORT_FLAGS = {
    "report_title_page": "True",
    "report_strategic_challenge": "True",
    "report_key_outputs_theme": "True",
    "report_decision_makers_options": "True",
    "report_scenarios": "True",
    "report_fixed_inputs": "True",
    "report_dependencies": "False",
    "report_weighted_appreciations": "True",
}


@dataclass
class SyntheticCaseParams:
    """Tunable knobs for one synthetic tRBS case.

    Phase 0 implements the ``convex`` regime. The remaining knobs that only bite
    in later regimes (ruggedness, saturation_point, scenario_dispersion, ...)
    are intentionally absent until the phase that uses them, to avoid dead
    flexibility.
    """

    name: str = "Synthetic_convex_k3"
    k: int = 3  # number of internal variables (levers)
    budget: float = 100.0  # B (capped simplex: sum x <= B)
    n_key_outputs: int = 3
    themes: Tuple[str, ...] = ("People", "Planet", "Profit")
    n_scenarios: int = 3
    regime: str = "convex"  # convex | smooth_nonconvex | nonsmooth
    appreciation: str = "linear"  # linear (affine, vertex optimum) | sinusoidal (concave, interior optimum)
    seed: int = 0
    coef_low: float = 0.5  # lever->KO coefficient range (positive => monotone)
    coef_high: float = 1.5

    def __post_init__(self):
        if self.regime != "convex":
            raise NotImplementedError(
                f"regime={self.regime!r} not implemented in Phase 0 (only 'convex'). "
                "smooth_nonconvex/nonsmooth arrive in Phase 2/3."
            )
        if self.appreciation not in ("linear", "sinusoidal"):
            raise ValueError(f"appreciation={self.appreciation!r} must be 'linear' or 'sinusoidal'")
        if min(self.k, self.n_key_outputs, self.n_scenarios) < 1:
            raise ValueError("k, n_key_outputs and n_scenarios must all be >= 1")
        if not self.themes:
            raise ValueError("themes must be non-empty")


class SyntheticCaseFactory:
    """Emits the 11 native tRBS CSV tables for a :class:`SyntheticCaseParams`."""

    def __init__(self, params: SyntheticCaseParams):
        self.p = params

        p = params
        # Zero-padded names so the importer's alphabetical pivot order matches 1..k.
        self.levers: List[str] = [f"Lever {i:02d}" for i in range(1, p.k + 1)]
        self.kos: List[str] = [f"KO {j:02d}" for j in range(1, p.n_key_outputs + 1)]
        self.evi = "Ext 01"
        self.scenarios: List[str] = [f"Scenario {s:02d}" for s in range(1, p.n_scenarios + 1)]

        # Round-robin themes over key outputs; only USED themes appear in theme_weights.
        self.ko_theme: List[str] = [p.themes[j % len(p.themes)] for j in range(p.n_key_outputs)]
        self.used_themes: List[str] = list(dict.fromkeys(self.ko_theme))

        # ALL randomness is drawn here, from independent SeedSequence child
        # streams per component. Ceteris paribus for the factorial design:
        # varying one knob (e.g. n_scenarios) must not re-randomise unrelated
        # draws (e.g. coefficients), and tables() stays idempotent.
        coef_ss, ko_w_ss, theme_w_ss, scen_w_ss = np.random.SeedSequence(p.seed).spawn(4)
        # Dense positive coefficients c[j, i] for KO_j += x_i * c[j, i] (affine => convex).
        self.coef = np.random.default_rng(coef_ss).uniform(p.coef_low, p.coef_high, size=(p.n_key_outputs, p.k))
        self.ce = 1.0  # external-variable coefficient (used in KO 01)
        self.ko_weights = np.random.default_rng(ko_w_ss).integers(1, 4, size=len(self.kos))
        self.theme_weights = np.random.default_rng(theme_w_ss).integers(1, 4, size=len(self.used_themes))
        self.scenario_weights = np.random.default_rng(scen_w_ss).integers(1, 4, size=len(self.scenarios))

    # ---- individual tables ------------------------------------------------
    def _key_outputs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "key_output": self.kos,
                "theme": self.ko_theme,
                "monetary": 0,
                "smaller_the_better": 0,  # STB=0: sinusoidal variant stays concave increasing
                "linear": 1 if self.p.appreciation == "linear" else 0,
                "automatic": 1,  # Vlinder Q2: keep auto-boundaries
                "start": np.nan,
                "end": np.nan,
            }
        )

    def _key_output_weights(self) -> pd.DataFrame:
        return pd.DataFrame({"key_output": self.kos, "weight": self.ko_weights})

    def _theme_weights(self) -> pd.DataFrame:
        return pd.DataFrame({"theme": self.used_themes, "weight": self.theme_weights})

    def _decision_makers_options(self) -> pd.DataFrame:
        """Bracketing DMOs: k pure 'Corner' options (all B on one lever), an
        'Equal spread' and a 'Zero spend'.

        Corners + zero realise every KO's feasible extremes, so the auto-
        boundaries equal the exact feasible envelope and appreciation never
        clips inside the capped simplex (the NO-CLIP invariant; under-bracketed
        DMOs made even the linear regime multimodal — exp03). 'Corner 01' sorts
        alphabetically first and spends the full budget, which keeps
        ``Optimize._infer_budget()`` (sum of the first DMO row) correct.
        """
        B, k = self.p.budget, self.p.k
        rows = []
        for i in range(k):
            for i2, lv in enumerate(self.levers):
                rows.append(
                    {
                        "internal_variable_input": lv,
                        "decision_makers_option": f"Corner {i + 1:02d}",
                        "value": B if i2 == i else 0.0,
                    }
                )
        for lv in self.levers:  # equal spread
            rows.append({"internal_variable_input": lv, "decision_makers_option": "Equal spread", "value": B / k})
        for lv in self.levers:  # zero spend (lower envelope)
            rows.append({"internal_variable_input": lv, "decision_makers_option": "Zero spend", "value": 0.0})
        return pd.DataFrame(rows)

    def _scenarios(self) -> pd.DataFrame:
        """One external variable, shifted across scenarios so cases are not scenario-degenerate."""
        ext_vals = 0.1 * self.p.budget * np.linspace(0.8, 1.2, self.p.n_scenarios)
        rows = [
            {"external_variable_input": self.evi, "scenario": sc, "value": v}
            for sc, v in zip(self.scenarios, ext_vals)
        ]
        return pd.DataFrame(rows)

    def _scenario_weights(self) -> pd.DataFrame:
        return pd.DataFrame({"scenario": self.scenarios, "weight": self.scenario_weights})

    def _fixed_inputs(self) -> pd.DataFrame:
        rows = []
        for j in range(self.p.n_key_outputs):
            for i in range(self.p.k):
                rows.append({"fixed_input": f"c_{j + 1:02d}_{i + 1:02d}", "value": float(self.coef[j, i])})
        rows.append({"fixed_input": "ce_01", "value": self.ce})
        return pd.DataFrame(rows)

    def _dependencies(self) -> pd.DataFrame:
        """KO_j = sum_i x_i * c_ji  (+ ext * ce on KO 01). Accumulation handles the sum."""
        rows = []
        for j, ko in enumerate(self.kos):
            for i, lever in enumerate(self.levers):
                rows.append(
                    {
                        "destination": ko,
                        "argument_1": lever,
                        "argument_2": f"c_{j + 1:02d}_{i + 1:02d}",
                        "operator": "*",
                    }
                )
        # use the external variable on the first KO (keeps every EVI used in the graph)
        rows.append({"destination": self.kos[0], "argument_1": self.evi, "argument_2": "ce_01", "operator": "*"})
        return pd.DataFrame(rows)

    def _configurations(self) -> pd.DataFrame:
        rows = [
            {"configuration": "language", "value": "EN"},
            {"configuration": "Optimize_DMO_name", "value": "Optimized_DMO"},
        ]
        rows += [{"configuration": kcfg, "value": v} for kcfg, v in _REPORT_FLAGS.items()]
        return pd.DataFrame(rows)

    def _case_text_elements(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "case_text_element": ["strategic_challenge"],
                "value": [
                    "Synthetic benchmark case generated to study continuous optimisation "
                    "methods on the tRBS appreciation model under a known data-generating process."
                ],
            }
        )

    def _generic_text_elements(self) -> pd.DataFrame:
        # Minimal: build/evaluate/appreciate do not use these; reporting can extend later.
        return pd.DataFrame(
            {
                "generic_text_element": ["title_strategic_challenge", "title_key_outputs", "title_dmo"],
                "value": ["Strategic challenge", "Key outputs", "Options"],
            }
        )

    # ---- write ------------------------------------------------------------
    def tables(self) -> dict:
        """All 11 native tables, keyed by table name (idempotent)."""
        return {
            "key_outputs": self._key_outputs(),
            "key_output_weights": self._key_output_weights(),
            "theme_weights": self._theme_weights(),
            "decision_makers_options": self._decision_makers_options(),
            "scenarios": self._scenarios(),
            "scenario_weights": self._scenario_weights(),
            "fixed_inputs": self._fixed_inputs(),
            "dependencies": self._dependencies(),
            "configurations": self._configurations(),
            "case_text_elements": self._case_text_elements(),
            "generic_text_elements": self._generic_text_elements(),
        }

    def manifest(self) -> dict:
        """Case manifest: params + seed + convexity claim (+ oracle, patched in
        by ``oracle.py``; ``table_sha256`` is filled by :meth:`write`)."""
        basis = (
            "affine dependencies + linear STB=0 appreciation + exact-envelope bracketing DMOs (no clipping)"
            if self.p.appreciation == "linear"
            else "affine dependencies + sinusoidal STB=0 appreciation (concave increasing on the bracket) "
            "+ exact-envelope bracketing DMOs (no clipping)"
        )
        return {
            "generator_version": GENERATOR_VERSION,
            "params": asdict(self.p),
            "regime": self.p.regime,
            "appreciation": self.p.appreciation,
            "seed": self.p.seed,
            "convexity_claim": {
                "geometry": "affine" if self.p.appreciation == "linear" else "concave",
                "basis": basis,
            },
            "oracle": None,
            "table_sha256": {},
        }

    def write(self, root: Path = DEFAULT_ROOT) -> Path:
        """Write ``<root>/<name>/csv/<table>.csv`` + ``<root>/<name>/manifest.json``
        and return the case root (the ``<root>``).

        CSVs are written as bytes with ``\\n`` line endings so the sha256 hashes
        in the manifest are exactly the on-disk content (byte-reproducible from
        the seed on any platform).
        """
        case_dir = Path(root) / self.p.name
        csv_dir = case_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        for table, df in self.tables().items():
            text = df.to_csv(sep=";", index=False)
            (csv_dir / f"{table}.csv").write_bytes(text.encode("utf-8"))
            manifest["table_sha256"][table] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return Path(root)


def _capped_simplex(rng, n, k, budget):
    """Uniform samples on {x >= 0, sum x <= budget} (Dirichlet over k+1, drop slack)."""
    return rng.dirichlet(np.ones(k + 1), size=n)[:, :k] * budget


def read_manifest(name: str, root: Path = DEFAULT_ROOT) -> dict | None:
    """Read ``<root>/<name>/manifest.json`` if present."""
    path = Path(root) / name / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _analytic_envelope(input_dict, budget) -> tuple:
    """Feasible KO envelope for factory-generated convex cases.

    By construction KO_j = sum_i x_i * c_ji (+ ext * ce_01 on KO 01) with all
    c_ji > 0, x on the capped simplex, and boundaries pooled over scenarios —
    so lo_j = 0 (+ ce*min ext) and hi_j = B * max_i c_ji (+ ce*max ext).
    Independent recomputation path: coefficients are parsed back from the
    imported ``fixed_inputs``, not taken from the factory object.
    """
    kos = list(input_dict["key_outputs"])
    k = len(input_dict["internal_variable_inputs"])
    coef = np.zeros((len(kos), k))
    ce = 0.0
    for fi_name, fi_value in zip(input_dict["fixed_inputs"], input_dict["fixed_input_value"]):
        if str(fi_name).startswith("c_"):
            _, j, i = str(fi_name).split("_")
            coef[int(j) - 1, int(i) - 1] = float(fi_value)
        elif str(fi_name) == "ce_01":
            ce = float(fi_value)
    ext = np.asarray(input_dict["scenario_value"], dtype=float).ravel()
    lo = np.zeros(len(kos))
    hi = budget * coef.max(axis=1)
    lo[0] += ce * ext.min()
    hi[0] += ce * ext.max()
    return lo, hi


def validate_case(
    name: str, root: Path = DEFAULT_ROOT, budget: float = None, n_samples: int = 400, seed: int = 1
) -> dict:
    """Round-trip a generated case and check it is well-formed and optimisable.

    Hard invariants (AssertionError on failure):
      * every objective evaluation over the capped simplex is finite;
      * NO-CLIP: the frozen auto-boundaries equal the analytic feasible envelope
        (under-bracketing puts clipping kinks inside the feasible set and makes
        even the linear regime multimodal — exp03);
      * the SLSQP solution is feasible;
      * convex regime: all converged SLSQP starts agree in OBJECTIVE value
        (consensus spread <= 0.1 points, one recovery epsilon). Spread must be
        measured in f, not x: near-tied gradient components (common at large k)
        leave x unstable across starts while f converges. Termination scatter
        at ftol=1e-6 is <= ~0.02 points empirically (k <= 15); the genuine
        clipping-induced basins in exp03 spread 3.3-47 points. A spread beyond
        0.1 on a case claimed concave is a generator bug.
    Soft signals (e.g. degenerate/flat appreciation) are reported.

    ``budget`` should be passed explicitly (or via the manifest); the fallback
    ``Optimize._infer_budget()`` sums the FIRST DMO row and is only correct
    because 'Corner 01' sorts first and spends the full budget.
    """
    manifest = read_manifest(name, root)
    if budget is None and manifest is not None:
        budget = float(manifest["params"]["budget"])

    sim = TheResponsibleBusinessSimulator(name, file_path=Path(root), file_extension="csv")
    with contextlib.redirect_stdout(io.StringIO()):
        sim.build()
        sim.evaluate()
        sim.appreciate()

    opt = Optimize(sim.input_dict, sim.output_dict)
    B = float(budget) if budget is not None else opt._infer_budget()
    k = opt._k
    with contextlib.redirect_stdout(io.StringIO()):
        opt._prepare_input_dict("Probe", sim.input_dict["decision_makers_option_value"][0].copy())

    # NO-CLIP invariant: frozen boundaries == analytic feasible envelope.
    start = np.asarray(opt.input_dict["key_output_start"], dtype=float)
    end = np.asarray(opt.input_dict["key_output_end"], dtype=float)
    lo, hi = _analytic_envelope(opt.input_dict, B)
    scale = max(1.0, float(np.abs(hi).max()))
    no_clip = bool(np.allclose(start, lo, atol=1e-9 * scale) and np.allclose(end, hi, atol=1e-9 * scale))
    assert no_clip, (
        f"auto-boundaries do not bracket the feasible envelope "
        f"(start={start} vs {lo}; end={end} vs {hi}) — appreciation would clip inside the feasible set"
    )

    scenario = str(sim.input_dict["scenarios"][0])
    rng = np.random.default_rng(seed)
    X = _capped_simplex(rng, n_samples, k, B)
    vals = np.array([evaluate_allocation(opt.input_dict, x, scenario, "Probe") for x in X])

    finite = np.isfinite(vals)
    assert finite.all(), f"{(~finite).sum()}/{n_samples} objective evaluations were non-finite"

    with contextlib.redirect_stdout(io.StringIO()):
        slsqp = opt.optimize_slsqp(scenario, B, dmo_name="Probe", n_starts=20, seed=seed)

    spend = float(np.sum(slsqp.best_x))
    slsqp_feasible = bool(spend <= B + 1e-6 and (slsqp.best_x >= -1e-6).all())
    assert slsqp_feasible, f"SLSQP solution infeasible: spend={spend}, x={slsqp.best_x}"

    converged = [float(r["appreciation"]) for r in slsqp.per_start_results if r["success"]]
    consensus_spread = float(max(converged) - min(converged)) if converged else float("nan")
    slsqp_consensus = bool(converged and consensus_spread <= 0.1)
    if manifest is None or manifest.get("regime") == "convex":
        assert slsqp_consensus, (
            f"SLSQP starts disagree (spread={consensus_spread:.6f}) on a case claimed concave — "
            "distinct local optima indicate a generator bug (see exp03)"
        )

    return {
        "name": name,
        "imported": True,
        "k": int(k),
        "budget": float(B),
        "n_key_outputs": int(len(sim.input_dict["key_outputs"])),
        "n_scenarios": int(len(sim.input_dict["scenarios"])),
        "n_dmos": int(len(sim.input_dict["decision_makers_options"])),
        "scenario_tested": scenario,
        "no_clip": no_clip,
        "appreciation_min": round(float(vals.min()), 6),
        "appreciation_max": round(float(vals.max()), 6),
        "appreciation_spread": round(float(vals.max() - vals.min()), 6),
        "non_degenerate": bool(vals.max() - vals.min() > 1e-6),
        "all_finite": True,
        "slsqp_appreciation": round(float(slsqp.appreciation), 6),
        "slsqp_spend": round(spend, 4),
        "slsqp_feasible": slsqp_feasible,
        "slsqp_consensus": slsqp_consensus,
        "consensus_spread": round(consensus_spread, 8),
        "slsqp_converged": f"{slsqp.n_converged}/{slsqp.n_starts}",
    }


def standard_cases() -> list:
    """The standard validation suite. B=100 throughout: budget-scale robustness
    is a pre-registered check, not incidental heterogeneity."""
    return [
        SyntheticCaseParams(name="Synthetic_convex_k2", k=2, n_key_outputs=2, seed=0),
        SyntheticCaseParams(name="Synthetic_convex_k3", k=3, n_key_outputs=3, seed=1),
        SyntheticCaseParams(name="Synthetic_convex_k6", k=6, n_key_outputs=4, seed=2),
        SyntheticCaseParams(name="Synthetic_convex_k9", k=9, n_key_outputs=5, seed=3),
        SyntheticCaseParams(
            name="Synthetic_convex_curved_k3", k=3, n_key_outputs=3, appreciation="sinusoidal", seed=1
        ),
        SyntheticCaseParams(
            name="Synthetic_convex_curved_k9", k=9, n_key_outputs=5, appreciation="sinusoidal", seed=3
        ),
    ]


def main():
    """Regenerate + validate the standard suite; print the JSON report."""
    report = {}
    for params in standard_cases():
        root = SyntheticCaseFactory(params).write()
        report[params.name] = validate_case(params.name, root, budget=params.budget)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
