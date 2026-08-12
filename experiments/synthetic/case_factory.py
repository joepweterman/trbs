"""
This file contains the synthetic tRBS case factory: a data-generating process
that emits cases whose optimum is known in advance.

A case is written as the 11 native tRBS tables, so it flows through the real
import -> evaluate -> appreciate -> optimize pipeline unchanged and the
optimizer cannot tell it from a real case. Each case also gets a
``manifest.json`` recording its parameters, seed, convexity claim, analytic
envelope and table hashes; ``oracle.py`` patches the certified optimum into it.

Three regimes set the geometry of the appreciation landscape: ``convex`` (one
optimum), ``smooth_nonconvex`` (several optima, no kinks) and ``nonsmooth``
(kinks and plateaus). The mechanism-by-mechanism map from parameters to
geometry, and the NO-CLIP invariant the validator enforces, are documented in
``thesis-knowledge/03-methodology/synthetic_case_generator_plan.md``.

Budget is the capped simplex sum x <= B (exp02 decision).

Run:
  python experiments/synthetic/case_factory.py
"""

# pylint: disable=invalid-name,protected-access,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-branches,too-many-statements
# (the knob-validation matrix and the regime-aware validator are inherently branchy)
# (math notation B/X/f; Optimize internals are the documented experiment surface;
#  the factory pre-draws every random component, hence the attribute count)

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.evaluate import Evaluate
from vlinder.optimize import SLSQPSolver, evaluate_allocation
from vlinder.utils import suppress_print

DEFAULT_ROOT = Path(__file__).parent / "generated"
GENERATOR_VERSION = "1.0"

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
    """Tunable knobs for one synthetic tRBS case (see module docstring for the
    regime/mechanism map). Knob validity is enforced in ``__post_init__``."""

    name: str = ""  # left empty, a descriptive name is derived from the knobs
    k: int = 3  # number of internal variable inputs
    budget: float = 100.0  # B (capped simplex: sum x <= B)
    n_key_outputs: int = 3
    themes: Tuple[str, ...] = ("People", "Planet", "Profit")
    n_scenarios: int = 3
    regime: str = "convex"  # convex | smooth_nonconvex | nonsmooth
    appreciation: str = "linear"  # linear (affine, vertex optimum) | sinusoidal (concave, interior optimum)
    seed: int = 0
    coef_low: float = 0.5  # internal-variable-input->KO coefficient range (positive => monotone)
    coef_high: float = 1.5
    # -- smooth_nonconvex mechanisms (convex-curvature carriers = "ruggedness") --
    n_stb1: int = 0  # KOs flipped to smaller_the_better=1 + sinusoidal (convex decreasing)
    n_bilinear: int = 0  # KOs with one bilinear input-pair term (indefinite curvature)
    # -- nonsmooth mechanisms --
    bracketing_factor: float = 1.0  # (0,1]; <1 shrinks bracketing DMOs => clipping inside the feasible set
    n_saturation: int = 0  # internal variable inputs with a min(x/sp, 1) cap chain (kinks, no clipping)
    saturation_point: float = 0.5  # sp = saturation_point * B for the saturated inputs
    # -- orthogonal knobs --
    scenario_dispersion: float = 0.0  # [0,1]; per-KO multiplicative scenario factors 1 +/- delta
    # How the per-scenario factors are drawn (only matters when dispersion > 0):
    #   independent - each Fac_j(s) drawn independently per KO and scenario, so
    #                 one KPI can improve in the very scenario another worsens
    #   coherent    - scenarios are ordered worst to best; each KO draws one
    #                 sensitivity in [0.2, 1.0] and Fac_j(s) = 1 + sensitivity_j
    #                 * dispersion * position_s with positions evenly spaced on
    #                 [-1, +1], floored at Fac >= 0.1 so no KO ever vanishes
    scenario_mode: str = "independent"

    def __post_init__(self):
        if self.regime not in ("convex", "smooth_nonconvex", "nonsmooth"):
            raise ValueError(f"regime={self.regime!r} must be convex | smooth_nonconvex | nonsmooth")
        if self.appreciation not in ("linear", "sinusoidal"):
            raise ValueError(f"appreciation={self.appreciation!r} must be 'linear' or 'sinusoidal'")
        if min(self.k, self.n_key_outputs, self.n_scenarios) < 1:
            raise ValueError("k, n_key_outputs and n_scenarios must all be >= 1")
        if not self.themes:
            raise ValueError("themes must be non-empty")
        if not 0 <= self.n_stb1 <= self.n_key_outputs:
            raise ValueError("n_stb1 must be in [0, n_key_outputs]")
        if not 0 <= self.n_bilinear <= self.n_key_outputs:
            raise ValueError("n_bilinear must be in [0, n_key_outputs]")
        if self.n_bilinear > 0 and self.k < 2:
            raise ValueError("n_bilinear > 0 needs k >= 2 (a bilinear term is a pair of inputs)")
        if not 0 <= self.n_saturation <= self.k:
            raise ValueError("n_saturation must be in [0, k]")
        if not 0.0 < self.bracketing_factor <= 1.0:
            raise ValueError("bracketing_factor must be in (0, 1]")
        if not 0.0 < self.saturation_point <= 1.0:
            raise ValueError("saturation_point must be in (0, 1]")
        if not 0.0 <= self.scenario_dispersion <= 1.0:
            raise ValueError("scenario_dispersion must be in [0, 1]")
        if self.scenario_mode not in ("independent", "coherent"):
            raise ValueError(f"scenario_mode={self.scenario_mode!r} must be 'independent' or 'coherent'")
        # Regime <-> mechanism consistency (mechanism exclusivity keeps the
        # factorial design clean and the analytic envelope exact).
        if self.regime == "convex":
            if self.n_stb1 or self.n_bilinear or self.n_saturation or self.bracketing_factor != 1.0:
                raise ValueError("convex regime requires all nonconvexity mechanisms at neutral")
        elif self.regime == "smooth_nonconvex":
            if self.n_stb1 + self.n_bilinear < 1:
                raise ValueError("smooth_nonconvex needs at least one curvature carrier (n_stb1/n_bilinear)")
            if self.n_saturation or self.bracketing_factor != 1.0:
                raise ValueError("smooth_nonconvex must stay smooth: no saturation, bracketing_factor=1")
        else:  # nonsmooth
            if self.bracketing_factor == 1.0 and self.n_saturation == 0:
                raise ValueError("nonsmooth needs bracketing_factor < 1 and/or n_saturation >= 1")
            if self.n_bilinear:
                raise ValueError("bilinear terms are a smooth_nonconvex mechanism (exclusivity rule)")
        if not self.name:
            self.name = self.derived_name()

    def derived_name(self) -> str:
        """
        Build a case name from the knobs that distinguish one case from another.

        Naming a case by hand is busywork that also invites collisions: the old
        default was a single fixed string, so two differently-configured cases
        written to the same directory would silently overwrite each other. The
        derived name encodes the regime, every active mechanism, k and the seed,
        so two cases collide only when they are genuinely the same case. Pass
        ``name`` explicitly to override this.
        :return: the derived case name
        """
        parts = ["Synthetic"]
        if self.regime == "convex":
            parts.append("convex" if self.appreciation == "linear" else "curved")
        elif self.regime == "smooth_nonconvex":
            parts.append("smooth")
            if self.n_stb1:
                parts.append(f"stb{self.n_stb1}")
            if self.n_bilinear:
                parts.append(f"bil{self.n_bilinear}")
        else:
            parts.append("nonsmooth")
            if self.bracketing_factor < 1.0:
                parts.append(f"clip{self.bracketing_factor:g}".replace(".", ""))
            if self.n_saturation:
                parts.append(f"sat{self.n_saturation}")
        if self.scenario_dispersion > 0:
            parts.append(f"disp{self.scenario_dispersion:g}".replace(".", ""))
            if self.scenario_mode == "coherent":
                parts.append("coh")
        parts.append(f"k{self.k:02d}")
        parts.append(f"s{self.seed}")
        return "_".join(parts)


class SyntheticCaseFactory:
    """Emits the 11 native tRBS CSV tables for a :class:`SyntheticCaseParams`."""

    def __init__(self, params: SyntheticCaseParams):
        self.p = params

        p = params
        # Zero-padded names so the importer's alphabetical pivot order matches 1..k.
        # The generated column labels keep the historical "Lever" word: the
        # locked study's cases and hash records were generated with it, and a
        # data-level rename would break their byte-reproducibility. The
        # documentation-level name is "internal variable inputs" (vlinder).
        self.levers: List[str] = [f"Lever {i:02d}" for i in range(1, p.k + 1)]
        self.kos: List[str] = [f"KO {j:02d}" for j in range(1, p.n_key_outputs + 1)]
        self.evi = "Ext 01"
        self.scenarios: List[str] = [f"Scenario {s:02d}" for s in range(1, p.n_scenarios + 1)]

        # Round-robin themes over key outputs; only USED themes appear in theme_weights.
        self.ko_theme: List[str] = [p.themes[j % len(p.themes)] for j in range(p.n_key_outputs)]
        self.used_themes: List[str] = list(dict.fromkeys(self.ko_theme))

        # ALL randomness is drawn here, from independent SeedSequence child
        # streams per component. Ceteris paribus for the factorial design:
        # varying one knob (e.g. scenario_dispersion) must not re-randomise
        # unrelated draws (e.g. coefficients), and tables() stays idempotent.
        # NOTE: children 0-3 match the Phase-0.75 streams, so pure-convex cases
        # reproduce the previously generated tables byte-identically.
        coef_ss, ko_w_ss, theme_w_ss, scen_w_ss, disp_ss, bil_ss, sens_ss = np.random.SeedSequence(p.seed).spawn(7)
        # Dense positive coefficients c[j, i] for KO_j += x_i * c[j, i] (affine => convex).
        self.coef = np.random.default_rng(coef_ss).uniform(p.coef_low, p.coef_high, size=(p.n_key_outputs, p.k))
        self.ce = 1.0  # external-variable coefficient (used in KO 01)
        self.ko_weights = np.random.default_rng(ko_w_ss).integers(1, 4, size=len(self.kos))
        self.theme_weights = np.random.default_rng(theme_w_ss).integers(1, 4, size=len(self.used_themes))
        self.scenario_weights = np.random.default_rng(scen_w_ss).integers(1, 4, size=len(self.scenarios))

        # Dispersion factors (always drawn, used iff delta > 0). Independent mode
        # draws Fac_j(s) = 1 + delta * u_js per KO and scenario; coherent mode
        # orders the scenarios worst to best and scales one per-KO sensitivity
        # along that axis, floored at 0.1 so an extreme combination flattens a
        # KO's contribution without removing it from the objective.
        u = np.random.default_rng(disp_ss).uniform(-1.0, 1.0, size=(p.n_key_outputs, p.n_scenarios))
        self.sensitivity = np.random.default_rng(sens_ss).uniform(0.2, 1.0, size=p.n_key_outputs)
        if p.scenario_mode == "coherent":
            positions = np.linspace(-1.0, 1.0, p.n_scenarios) if p.n_scenarios > 1 else np.zeros(1)
            self.disp_factors = np.maximum(
                0.1, 1.0 + p.scenario_dispersion * self.sensitivity[:, None] * positions[None, :]
            )
        else:
            self.disp_factors = 1.0 + p.scenario_dispersion * u

        # Bilinear structure (always drawn where possible, used iff n_bilinear > 0):
        # KO j (j < n_bilinear) gets one term q_j * x_a * x_b. q ~ U(coef range)/B
        # keeps the term's magnitude comparable to the affine part (max B^2/4 * q).
        bil_rng = np.random.default_rng(bil_ss)
        if p.k >= 2:
            self.bilinear_pairs = [tuple(sorted(bil_rng.choice(p.k, size=2, replace=False))) for _ in self.kos]
        else:
            self.bilinear_pairs = []
        self.bilinear_q = bil_rng.uniform(p.coef_low, p.coef_high, size=p.n_key_outputs) / p.budget

        # Derived mechanism sets (first-n conventions keep manipulations ceteris paribus).
        self.stb1_kos = list(range(p.n_stb1))
        self.saturated = list(range(p.n_saturation))
        self.sp = p.saturation_point * p.budget

    # ---- analytic structure ------------------------------------------------
    def _base_value(self, j: int, x: np.ndarray) -> float:
        """KO j's base value (before scenario factor and external term) at allocation x."""
        val = 0.0
        for i in range(self.p.k):
            if i in self.saturated:
                val += self.coef[j, i] * self.sp * min(x[i] / self.sp, 1.0)
            else:
                val += self.coef[j, i] * x[i]
        if j < self.p.n_bilinear:
            a, b = self.bilinear_pairs[j]
            val += self.bilinear_q[j] * x[a] * x[b]
        return float(val)

    def _blend_alloc(self, j: int) -> np.ndarray:
        """Analytic argmax of KO j's base value on the edge of its bilinear pair.

        On the {a,b} edge (x_a = t, x_b = B - t) the base value is concave in t,
        so t* is closed-form; interior three-lever stationary points are saddles
        (indefinite bilinear Hessian), so the face max is max(corners, this edge).
        """
        B = self.p.budget
        a, b = self.bilinear_pairs[j]
        c_a, c_b, q = self.coef[j, a], self.coef[j, b], self.bilinear_q[j]
        t = float(np.clip((B + (c_a - c_b) / q) / 2.0, 0.0, B))
        x = np.zeros(self.p.k)
        x[a], x[b] = t, B - t
        return x

    def _greedy_max_alloc(self, j: int) -> np.ndarray:
        """Exact argmax of KO j's base value on the budget face under saturation.

        The base value is separable concave piecewise-linear, so filling levers
        in descending marginal-rate order is exact: saturated levers take at
        most sp (marginal rate drops to 0 beyond), the first unsaturated lever
        takes everything that remains.
        """
        B, k = self.p.budget, self.p.k
        alloc = np.zeros(k)
        remaining = B
        for i in np.argsort(-self.coef[j]):
            take = min(self.sp, remaining) if i in self.saturated else remaining
            alloc[i] += take
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:  # every lever saturated: park the leftover (adds nothing)
            alloc[int(np.argmax(self.coef[j]))] += remaining
        return alloc

    def _ko_hi_base(self, j: int) -> float:
        """Exact feasible max of KO j's base value over the capped simplex.

        All mechanisms are monotone non-decreasing in every lever, so the max is
        on the budget face; candidates cover each mechanism's exact argmax.
        """
        B, k = self.p.budget, self.p.k
        candidates = [B * np.eye(k)[i] for i in range(k)]
        if j < self.p.n_bilinear:
            candidates.append(self._blend_alloc(j))
        if self.saturated:
            candidates.append(self._greedy_max_alloc(j))
        return max(self._base_value(j, x) for x in candidates)

    def _ext_values(self) -> np.ndarray:
        return 0.1 * self.p.budget * np.linspace(0.8, 1.2, self.p.n_scenarios)

    def envelope(self) -> tuple:
        """Analytic per-KO feasible envelope [lo, hi], pooled over scenarios
        (matching vlinder's pooled auto-boundaries).

        KO j = Fac_j(s) * base_j(x) + [j == 0] * ce * ext_s, with base_j >= 0,
        base_j(0) = 0 and Fac >= 0, so lo comes from x = 0 and hi from the
        per-scenario max of factor * base-max.
        """
        n = self.p.n_key_outputs
        fac = self.disp_factors if self.p.scenario_dispersion > 0 else np.ones((n, self.p.n_scenarios))
        ext = self._ext_values()
        lo = np.zeros(n)
        hi = np.zeros(n)
        for j in range(n):
            hi_base = self._ko_hi_base(j)
            ext_term = self.ce * ext if j == 0 else np.zeros_like(ext)
            hi[j] = float(np.max(fac[j] * hi_base + ext_term))
            lo[j] = float(np.min(ext_term))
        return lo, hi

    # ---- individual tables ------------------------------------------------
    def _key_outputs(self) -> pd.DataFrame:
        stb = [1 if j in self.stb1_kos else 0 for j in range(self.p.n_key_outputs)]
        base_linear = 1 if self.p.appreciation == "linear" else 0
        # STB=1 flips must be sinusoidal: linear STB=1 is affine decreasing and
        # would leave the objective concave, with no curvature carrier at all.
        linear = [0 if j in self.stb1_kos else base_linear for j in range(self.p.n_key_outputs)]
        return pd.DataFrame(
            {
                "key_output": self.kos,
                "theme": self.ko_theme,
                "monetary": 0,
                "smaller_the_better": stb,
                "linear": linear,
                "automatic": 1,  # Vlinder Q2: keep auto-boundaries
                "start": np.nan,
                "end": np.nan,
            }
        )

    def _key_output_weights(self) -> pd.DataFrame:
        return pd.DataFrame({"key_output": self.kos, "weight": self.ko_weights})

    def _theme_weights(self) -> pd.DataFrame:
        return pd.DataFrame({"theme": self.used_themes, "weight": self.theme_weights})

    def _bracketing_allocations(self) -> dict:
        """The f=1 bracketing DMO set: {name: allocation}.

        Corners + lowest spend realise the affine extremes; 'Blend' DMOs realise
        each bilinear KO's edge max; greedy 'Envelope' DMOs realise the per-KO
        max under saturation. Together the observed DMO grid equals the exact
        feasible envelope (NO-CLIP at bracketing_factor = 1).
        """
        B, k = self.p.budget, self.p.k
        allocs = {f"Corner {i + 1:02d}": B * np.eye(k)[i] for i in range(k)}
        allocs["Equal spread"] = np.full(k, B / k)
        # Named "Lowest spend" rather than "Zero spend": under a bracketing
        # factor below 1 every DMO, this one included, is pulled toward the
        # equal spread, so the row is the lowest spend the case observes, not
        # necessarily zero. Renamed 2026-08-13 (review), after every locked
        # study run had completed, because the label changes the bytes of
        # newly generated cases.
        allocs["Lowest spend"] = np.zeros(k)
        for j in range(self.p.n_bilinear):
            allocs[f"Blend {j + 1:02d}"] = self._blend_alloc(j)
        if self.saturated:
            for j in range(self.p.n_key_outputs):
                allocs[f"Envelope {j + 1:02d}"] = self._greedy_max_alloc(j)
        return allocs

    def _decision_makers_options(self) -> pd.DataFrame:
        """Bracketing DMOs, shrunk toward the equal spread by ``bracketing_factor``.

        At factor 1 the DMO grid realises every KO's feasible extremes (NO-CLIP).
        At factor f < 1 every DMO moves toward B/k, deliberately under-bracketing
        the envelope: the exp03 clipping mechanism as a dial. All shrunk DMOs
        stay feasible (convex combinations of feasible points); every name that
        sorts alphabetically first ('Blend 01' or 'Corner 01') lies on the
        budget face, which keeps ``Optimize._infer_budget()`` correct.
        """
        f = self.p.bracketing_factor
        equal = np.full(self.p.k, self.p.budget / self.p.k)
        rows = []
        for dmo_name, alloc in self._bracketing_allocations().items():
            shrunk = alloc if dmo_name == "Equal spread" else f * alloc + (1.0 - f) * equal
            for lv, val in zip(self.levers, shrunk):
                rows.append({"internal_variable_input": lv, "decision_makers_option": dmo_name, "value": float(val)})
        return pd.DataFrame(rows)

    def _scenarios(self) -> pd.DataFrame:
        """External variables per scenario: the additive 'Ext 01' shift (always)
        plus one multiplicative 'Fac j' per KO when scenario_dispersion > 0."""
        rows = [
            {"external_variable_input": self.evi, "scenario": sc, "value": v}
            for sc, v in zip(self.scenarios, self._ext_values())
        ]
        if self.p.scenario_dispersion > 0:
            for j in range(self.p.n_key_outputs):
                for s, sc in enumerate(self.scenarios):
                    rows.append(
                        {
                            "external_variable_input": f"Fac {j + 1:02d}",
                            "scenario": sc,
                            "value": float(self.disp_factors[j, s]),
                        }
                    )
        return pd.DataFrame(rows)

    def _scenario_weights(self) -> pd.DataFrame:
        return pd.DataFrame({"scenario": self.scenarios, "weight": self.scenario_weights})

    def _fixed_inputs(self) -> pd.DataFrame:
        rows = []
        for j in range(self.p.n_key_outputs):
            for i in range(self.p.k):
                if i in self.saturated:
                    # saturated contribution weight: c_ji * sp (slope-preserving cap)
                    rows.append(
                        {"fixed_input": f"cs_{j + 1:02d}_{i + 1:02d}", "value": float(self.coef[j, i] * self.sp)}
                    )
                else:
                    rows.append({"fixed_input": f"c_{j + 1:02d}_{i + 1:02d}", "value": float(self.coef[j, i])})
        for j in range(self.p.n_bilinear):
            rows.append({"fixed_input": f"q_{j + 1:02d}", "value": float(self.bilinear_q[j])})
        if self.saturated:
            rows.append({"fixed_input": "sp_01", "value": float(self.sp)})
        rows.append({"fixed_input": "ce_01", "value": self.ce})
        return pd.DataFrame(rows)

    def _dependencies(self) -> pd.DataFrame:
        """Regime-aware dependency graph (accumulation handles all sums).

        Base structure per KO j: affine lever terms, saturated levers routed
        through a min-cap chain, an optional bilinear intermediate, and (when
        scenario_dispersion > 0) an intermediate 'Base j' multiplied by the
        scenario factor 'Fac j'. The additive external shift stays on KO 01.
        """
        p = self.p
        rows = []
        # Shared min-cap chains: Ratio i = Lever i / sp ; Sat i = min(Ratio i, 1).
        for i in self.saturated:
            rows.append(
                {
                    "destination": f"Ratio {i + 1:02d}",
                    "argument_1": self.levers[i],
                    "argument_2": "sp_01",
                    "operator": "/",
                }
            )
            rows.append(
                {
                    "destination": f"Sat {i + 1:02d}",
                    "argument_1": f"Ratio {i + 1:02d}",
                    "argument_2": "1",
                    "operator": "min",
                }
            )
        # Shared bilinear intermediates: Bil j = Lever a * Lever b.
        for j in range(p.n_bilinear):
            a, b = self.bilinear_pairs[j]
            rows.append(
                {
                    "destination": f"Bil {j + 1:02d}",
                    "argument_1": self.levers[a],
                    "argument_2": self.levers[b],
                    "operator": "*",
                }
            )
        # Per-KO accumulation into KO j (or its pre-factor intermediate Base j).
        for j, ko in enumerate(self.kos):
            dest = f"Base {j + 1:02d}" if p.scenario_dispersion > 0 else ko
            for i, lever in enumerate(self.levers):
                if i in self.saturated:
                    rows.append(
                        {
                            "destination": dest,
                            "argument_1": f"Sat {i + 1:02d}",
                            "argument_2": f"cs_{j + 1:02d}_{i + 1:02d}",
                            "operator": "*",
                        }
                    )
                else:
                    rows.append(
                        {
                            "destination": dest,
                            "argument_1": lever,
                            "argument_2": f"c_{j + 1:02d}_{i + 1:02d}",
                            "operator": "*",
                        }
                    )
            if j < p.n_bilinear:
                rows.append(
                    {
                        "destination": dest,
                        "argument_1": f"Bil {j + 1:02d}",
                        "argument_2": f"q_{j + 1:02d}",
                        "operator": "*",
                    }
                )
            if p.scenario_dispersion > 0:
                rows.append({"destination": ko, "argument_1": dest, "argument_2": f"Fac {j + 1:02d}", "operator": "*"})
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
        """Case manifest: params + seed + convexity claim + analytic envelope
        (+ oracle, patched in by ``oracle.py``; ``table_sha256`` by :meth:`write`)."""
        p = self.p
        geometry = {
            "convex": "affine" if p.appreciation == "linear" else "concave",
            "smooth_nonconvex": "smooth nonconvex (indefinite curvature from STB=1 sinusoidal and/or bilinear terms)",
            "nonsmooth": "piecewise smooth (boundary clipping and/or min-cap saturation kinks)",
        }[p.regime]
        lo, hi = self.envelope()
        return {
            "generator_version": GENERATOR_VERSION,
            "params": asdict(p),
            "regime": p.regime,
            "appreciation": p.appreciation,
            "seed": p.seed,
            "convexity_claim": {"geometry": geometry},
            "mechanisms": {
                "n_stb1": p.n_stb1,
                "n_bilinear": p.n_bilinear,
                "bilinear_pairs": [[int(a) + 1, int(b) + 1] for a, b in self.bilinear_pairs[: p.n_bilinear]],
                "n_saturation": p.n_saturation,
                "saturation_point": p.saturation_point,
                "bracketing_factor": p.bracketing_factor,
                "scenario_dispersion": p.scenario_dispersion,
            },
            "envelope": {"lo": [float(v) for v in lo], "hi": [float(v) for v in hi]},
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


def sample_capped_simplex(rng, n, k, budget, face=False):
    """
    This function draws uniform samples from the feasible set.
    :param rng: the numpy random generator to draw from
    :param n: number of samples
    :param k: number of internal variables
    :param budget: the budget B
    :param face: draw on the budget face {sum x = B} instead of the capped simplex
    :return: an (n x k) array of allocations
    """
    if face:
        return rng.dirichlet(np.ones(k), size=n) * budget
    return rng.dirichlet(np.ones(k + 1), size=n)[:, :k] * budget


def manifest_path(name: str, root: Path = DEFAULT_ROOT) -> Path:
    """
    This function returns the manifest location of a case, which also serves as
    the marker for whether that case has been generated yet.
    :param name: the case name
    :param root: the directory the case was written to
    :return: the path of the case manifest
    """
    return Path(root) / name / "manifest.json"


def read_manifest(name: str, root: Path = DEFAULT_ROOT) -> dict:
    """
    This function reads the manifest of a generated case.
    :param name: the case name
    :param root: the directory the case was written to
    :return: the manifest dictionary
    """
    return json.loads(manifest_path(name, root).read_text(encoding="utf-8"))


@suppress_print
def build_case(name: str, root: Path, probe_dmo: str):
    """
    This function imports a generated case through the real tRBS pipeline and
    registers a probe decision makers option to write trial allocations into.
    Registering the probe also freezes the appreciation boundaries, which is
    what makes a trial allocation comparable to the certified optimum.
    :param name: the case name
    :param root: the directory the case was written to
    :param probe_dmo: name of the decision makers option to register
    :return: a tuple of the built simulator and its prepared Optimize instance
    """
    sim = TheResponsibleBusinessSimulator(name, file_path=Path(root), file_extension="csv")
    sim.build()
    sim.evaluate()
    sim.appreciate()
    opt = SLSQPSolver(sim.input_dict, sim.output_dict)
    opt._prepare_input_dict(probe_dmo, sim.input_dict["decision_makers_option_value"][0].copy())
    return sim, opt


def parse_coefficients(input_dict) -> np.ndarray:
    """
    This function recovers the internal-variable-to-key-output coefficient
    matrix c[j, i] from the imported fixed inputs. Reading the coefficients back
    out of the case (instead of off the factory object) is what makes the
    envelope check and the oracle gradient independent cross-checks.
    :param input_dict: the imported case dictionary
    :return: an (n_key_outputs x k) coefficient matrix
    """
    coef = np.zeros((len(input_dict["key_outputs"]), len(input_dict["internal_variable_inputs"])))
    for fi_name, fi_value in zip(input_dict["fixed_inputs"], input_dict["fixed_input_value"]):
        if str(fi_name).startswith("c_"):
            _, j, i = str(fi_name).split("_")
            coef[int(j) - 1, int(i) - 1] = float(fi_value)
    return coef


def _analytic_envelope(input_dict, budget) -> tuple:
    """Independent feasible-envelope recomputation for PLAIN-AFFINE cases
    (no bilinear, no saturation, no dispersion).

    By construction KO_j = sum_i x_i * c_ji (+ ext * ce_01 on KO 01) with all
    c_ji > 0, x on the capped simplex, and boundaries pooled over scenarios,
    so lo_j = 0 (+ ce*min ext) and hi_j = B * max_i c_ji (+ ce*max ext).
    """
    kos = list(input_dict["key_outputs"])
    coef = parse_coefficients(input_dict)
    ce = next(
        (float(v) for n, v in zip(input_dict["fixed_inputs"], input_dict["fixed_input_value"]) if str(n) == "ce_01"),
        0.0,
    )
    evis = [str(e) for e in input_dict["external_variable_inputs"]]
    ext = np.asarray(input_dict["scenario_value"], dtype=float)[:, evis.index("Ext 01")]
    lo = np.zeros(len(kos))
    hi = budget * coef.max(axis=1)
    lo[0] += ce * ext.min()
    hi[0] += ce * ext.max()
    return lo, hi


def _ko_sample_values(input_dict, X, scenario, dmo_name) -> np.ndarray:
    """Per-KO values (n_samples x n_KO) at allocations X, through the real
    Evaluate engine. One deepcopy total; the probe DMO row is mutated in place
    (Evaluate is read-only on the input dict)."""
    d = copy.deepcopy(input_dict)
    idx = int(np.where(d["decision_makers_options"] == dmo_name)[0][0])
    kos = list(d["key_outputs"])
    out = np.empty((len(X), len(kos)))
    ev = Evaluate(d)
    for r, x in enumerate(X):
        d["decision_makers_option_value"][idx] = np.asarray(x, dtype=float)
        res = ev.evaluate_all_dependencies(scenario, dmo_name)["key_outputs"]
        out[r] = [res[ko] for ko in kos]
    return out


def validate_case(
    name: str, root: Path = DEFAULT_ROOT, budget: float = None, n_samples: int = 400, seed: int = 1
) -> dict:
    """Regime-aware round-trip validation of a generated case.

    Hard invariants (AssertionError on failure):
      * every objective evaluation over the capped simplex is finite;
      * every sampled per-KO value lies inside the manifest's analytic envelope
        (the envelope claim is a true bound on the real pipeline);
      * bracketing_factor = 1: NO-CLIP, the frozen auto-boundaries equal the
        analytic envelope (under-bracketing puts clipping kinks inside the
        feasible set and makes even the linear regime multimodal, see exp03).
        For plain-affine cases the envelope is additionally recomputed
        independently from the imported tables (``_analytic_envelope``);
      * bracketing_factor < 1: the under-bracketing is PRESENT and intentional
        (frozen boundaries strictly inside the envelope);
      * the SLSQP solution is feasible;
      * convex regime only: all converged SLSQP starts agree in OBJECTIVE value
        (consensus spread <= 0.1 points, one recovery epsilon; measured in f,
        not x, because near-tied gradients at large k leave x unstable while f
        converges; genuine clipping basins spread 3.3-47 points in exp03).
        Non-convex regimes report multimodality evidence instead of asserting.

    ``budget`` defaults to the value recorded in the manifest.
    """
    manifest = read_manifest(name, root)
    if budget is None:
        budget = float(manifest["params"]["budget"])
    regime = manifest["regime"]
    mech = manifest["mechanisms"]
    f_brack = float(mech["bracketing_factor"])

    sim, opt = build_case(name, root, "Probe")
    B, k = float(budget), opt._k

    start = np.asarray(opt.input_dict["key_output_start"], dtype=float)
    end = np.asarray(opt.input_dict["key_output_end"], dtype=float)
    lo = np.asarray(manifest["envelope"]["lo"], dtype=float)
    hi = np.asarray(manifest["envelope"]["hi"], dtype=float)
    scale = max(1.0, float(np.abs(hi).max()))
    tol = 1e-9 * scale

    plain_affine = mech["n_bilinear"] == 0 and mech["n_saturation"] == 0 and float(mech["scenario_dispersion"]) == 0.0
    if plain_affine:
        lo2, hi2 = _analytic_envelope(opt.input_dict, B)
        assert np.allclose(lo, lo2, atol=tol) and np.allclose(hi, hi2, atol=tol), (
            "manifest envelope disagrees with the independent affine recomputation "
            f"(lo {lo} vs {lo2}; hi {hi} vs {hi2})"
        )

    if f_brack >= 1.0 - 1e-12:
        no_clip = bool(np.allclose(start, lo, atol=tol) and np.allclose(end, hi, atol=tol))
        assert no_clip, (
            f"auto-boundaries do not bracket the feasible envelope "
            f"(start={start} vs {lo}; end={end} vs {hi}): appreciation would clip inside the feasible set"
        )
        underbracketing_margin = 0.0
    else:
        no_clip = False
        assert (start >= lo - tol).all() and (end <= hi + tol).all(), (
            f"frozen boundaries exceed the analytic envelope (start={start}, lo={lo}; end={end}, hi={hi}): "
            "the envelope claim is wrong"
        )
        margins = (start - lo) + (hi - end)
        underbracketing_margin = float(margins.max())
        assert (
            underbracketing_margin > 1e-6 * scale
        ), f"bracketing_factor={f_brack} < 1 but no under-bracketing measured, shrink not applied?"

    scenario = str(sim.input_dict["scenarios"][0])
    rng = np.random.default_rng(seed)
    X = sample_capped_simplex(rng, n_samples, k, B)
    vals = np.array([evaluate_allocation(opt.input_dict, x, scenario, "Probe") for x in X])

    finite = np.isfinite(vals)
    assert finite.all(), f"{(~finite).sum()}/{n_samples} objective evaluations were non-finite"

    # Per-KO containment: the envelope must bound the real pipeline everywhere.
    n_check = min(len(X), 150)
    KOX = _ko_sample_values(opt.input_dict, X[:n_check], scenario, "Probe")
    assert (KOX >= lo - tol).all() and (
        KOX <= hi + tol
    ).all(), "sampled KO values escape the analytic envelope: the envelope math does not match the pipeline"
    # Clipping incidence: fraction of sampled points with >= 1 KO outside the
    # FROZEN boundaries (nonzero only under intentional under-bracketing).
    outside = (KOX < start - tol) | (KOX > end + tol)
    clipping_incidence = float(outside.any(axis=1).mean())

    slsqp = suppress_print(opt.solve)(scenario, dmo_name="Probe", budget=B, n_starts=20, seed=seed)

    spend = float(np.sum(slsqp.allocation))
    slsqp_feasible = bool(spend <= B + 1e-6 and (slsqp.allocation >= -1e-6).all())
    assert slsqp_feasible, f"SLSQP solution infeasible: spend={spend}, x={slsqp.allocation}"

    converged = [float(r["appreciation"]) for r in slsqp.per_start_results if r["success"]]
    consensus_spread = float(max(converged) - min(converged)) if converged else float("nan")
    n_distinct = len({round(v, 3) for v in converged})
    slsqp_consensus = bool(converged and consensus_spread <= 0.1)
    if regime == "convex":
        assert slsqp_consensus, (
            f"SLSQP starts disagree (spread={consensus_spread:.6f}) on a case claimed concave: "
            "distinct local optima indicate a generator bug (see exp03)"
        )

    return {
        "name": name,
        "imported": True,
        "regime": regime,
        "k": int(k),
        "budget": float(B),
        "n_key_outputs": int(len(sim.input_dict["key_outputs"])),
        "n_scenarios": int(len(sim.input_dict["scenarios"])),
        "n_dmos": int(len(sim.input_dict["decision_makers_options"])),
        "scenario_tested": scenario,
        "no_clip": no_clip,
        "underbracketing_margin": round(underbracketing_margin, 6),
        "clipping_incidence": round(clipping_incidence, 4),
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
        "n_distinct_terminal_values": n_distinct,
        "slsqp_converged": f"{slsqp.n_converged}/{slsqp.n_starts}",
    }


def standard_cases() -> list:
    """The standard validation suite: every regime and every mechanism exercised
    at least once. B=100 throughout: budget-scale robustness is a pre-registered
    check, not incidental heterogeneity."""
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
        SyntheticCaseParams(
            name="Synthetic_smooth_k3",
            k=3,
            n_key_outputs=3,
            regime="smooth_nonconvex",
            appreciation="sinusoidal",
            n_stb1=1,
            n_bilinear=1,
            seed=4,
        ),
        SyntheticCaseParams(
            name="Synthetic_nonsmooth_clip_k3", k=3, n_key_outputs=3, regime="nonsmooth", bracketing_factor=0.7, seed=5
        ),
        SyntheticCaseParams(
            name="Synthetic_nonsmooth_sat_k3",
            k=3,
            n_key_outputs=3,
            regime="nonsmooth",
            n_saturation=2,
            saturation_point=0.4,
            seed=6,
        ),
        SyntheticCaseParams(name="Synthetic_dispersed_k3", k=3, n_key_outputs=3, scenario_dispersion=0.6, seed=7),
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
