# Pre-registration — synthetic Monte-Carlo study of optimization-method adequacy on tRBS cases

**Status: LOCKED v1.2 (2026-07-22).** (v1.1: method roster completed — basin-hopping + real-coded GA registered; solver spec frozen incl. z-space normalisation; ε-discrimination note added after exp04. v1.2: lock.)
**Lock record:** Paul Bouman's TMS reply (received by 2026-07-12) on the pre-registration question: "Normaal gesproken doen we dat niet in heel veel detail, het is dus prima om je experimentele opzet te delen." The lock is therefore self-imposed rigor; this document is shared with Paul ter kennisgeving. Frozen copy committed in the tRBS fork (`joepweterman/trbs`, branch `recover-synthetic`) at `experiments/synthetic/PREREGISTRATION.md`; that commit hash timestamps the lock. Environment verified at lock: numpy 2.3.2, scipy 1.17.1, 17/17 instrument tests green.
**Stretch note (pre-declared):** on Paul's suggestion, a SCIP (`pyscipopt`) global-solver benchmark replaces NSGA-II as the stretch method; if it runs, it is added by amendment before those runs — it is NOT part of the confirmatory roster below.
Lock protocol: review with Paul → freeze this file → commit it in the tRBS fork (`joepweterman/trbs`) so the commit hash timestamps the lock → only then run confirmatory experiments. Any post-lock change is an **amendment** appended at the bottom with date + rationale, never an edit in place.
Companion: `synthetic_case_generator_plan.md` (v2, the instrument), `../04-experiments/exp03-convex-regime-clipping-audit.md` (generator verification).

Everything run before the lock (exp01-exp03, validator/oracle acceptance sweeps) is **exploratory / instrument verification**, and is reported as such. The confirmatory study starts after the lock.

---

## 1. Hypotheses (per regime cell)

The refined convexity characterisation (plan §3b): the tRBS objective is concave iff the dependency graph is affine, every active appreciation term is concave in its KO, and no 0-floor clip is active on the feasible set. The study tests whether **regime membership predicts method adequacy**:

| ID | Regime cell | Hypothesis |
|---|---|---|
| H1a | convex-linear | SLSQP-multistart recovers the certified optimum (gap ≤ ε) in ≥ 99% of cases at every k ∈ {2,…,15}. |
| H1b | convex-curved | Same as H1a — curvature without convexity-loss does not degrade SLSQP. |
| H2 | convex (both) | Grid search recovery degrades with k: gap grows and/or runtime exceeds the cap for k ≳ 4-6 (combinatorial blow-up), while SLSQP runtime grows at most polynomially in k. |
| H3 | smooth_nonconvex (Phase 2; added by amendment before those runs) | Single-start SLSQP recovery falls below 50% as ruggedness rises; multistart/basin-hopping restores it; recovery is a decreasing function of the number of convex-curvature carriers (STB=1 KOs, bilinear deps). |
| H4 | nonsmooth (Phase 3; added by amendment) | Recovery of gradient-based methods falls with `bracketing_factor` < 1 (floor-clip plateaus); population/derivative-free methods degrade more slowly. |
| H5 | scaling (Phase 5) | log(runtime) of grid grows super-linearly in k at fixed resolution; SLSQP evaluations grow ~linearly in k at fixed recovery. |

Falsification is a result: if SLSQP recovers reliably in nonsmooth cells, the characterisation does NOT predict adequacy and the thesis reports that.

## 2. Design (factorial, seeded)

- **Instrument:** `experiments/synthetic/case_factory.py` (v0.75) + `experiments/synthetic/oracle.py`, frozen at the lock commit. Cases are emitted in the native tRBS format and consumed through the unmodified `case_importer → evaluate → appreciate → optimize` pipeline.
- **Grid (confirmatory, Phases 1/5):** regime-variant {convex-linear, convex-curved} × k ∈ {2, 3, 4, 6, 9, 12, 15} × seeds {0, 1, …, 29} → 420 cases. Other knobs at defaults: B = 100, n_key_outputs = max(2, min(5, (k+1)//2)), 3 scenarios, 3 themes, coefficients U(0.5, 1.5).
- **Sub-seeded streams** (SeedSequence children per component) guarantee that factorial manipulations are ceteris paribus.
- **Method roster = `vlinder.optimize.METHOD_REGISTRY` at lock time — complete since 2026-07-06 (commits e1c9640, 2044d1c):**
  - `grid` — stars-and-bars on the budget face (documented structural limitation Σx = B), max_combinations = 60,000;
  - `slsqp` — multi-start SLSQP, Dirichlet(1,…,1) starts on the budget face, n_starts = 100, ftol 1e-6, FD step 1e-6 (z-space);
  - `basin_hopping` — SLSQP inner solves, n_hops = 25, 1 start, T = 1.0, Gaussian capped-simplex take-step with step_frac = 0.3;
  - `genetic_algorithm` — real-coded GA (binary tournament, SBX η=15 at p=0.9, polynomial mutation η=20 at p=1/k, one-elite), population 50 × 60 generations, uniform capped-simplex initialisation.
  - **Common solver spec:** all continuous methods optimise in budget-normalised z-space (x = B·z on the unit capped simplex) and returned solutions are projected onto the capped simplex (clip + rescale, relative correction ~1e-8). Rationale: exp04 — raw-budget scaling silently stalls SLSQP; z-space makes behaviour case-independent. All methods are seeded and report `n_function_evals`.
  - **Known budget sensitivity, recorded pre-lock (Phase-1b acceptance probes, k=6):** the GA's vertex-optimum recovery is evaluation-budget-bound (linear-regime gap 1.56 at 3,050 evals → 0.0 at 10,050 evals; curved-regime gaps ≤ 0.024 at default budget). Defaults are kept at comparable budgets across methods (~2–3k evals); H1/H2 conclusions about the GA must therefore be read as "at comparable budget", and the runtime/eval metrics (§3) carry the trade-off explicitly.
- **Scenario:** each case is optimized per scenario; the first scenario is primary, the others are robustness.

## 3. Metrics (all defined before any confirmatory run)

- **Optimality gap** = f*_oracle − f_method, in appreciation points (0-100 scale), per case × scenario × method. f*_oracle is the certified optimum from the case manifest (vertex-enumeration certificate for linear; KKT certificate for curved). Gaps below solver noise are reported as 0 at display precision, never negative-truncated silently.
- **Recovery** = 1{gap ≤ ε}, **ε = 0.1 points**. Defined on the objective, never on allocation distance (near-tied gradient components make x* unstable while f* is exact — exp03). **ε-discrimination caveat (exp04):** distinct local optima can sit closer than ε in objective value (Beerwiser's two basins are 0.0104 points apart), so recovery at ε = 0.1 does not identify *which* basin was found. Where basin identity matters (H3 multimodality analyses), it is reported separately from the allocation, with the basin partition defined per case in the experiment write-up.
- **Runtime** = wall-clock seconds per optimize call (same machine, single process, fixed env recorded below).
- **Function evaluations** = `OptimizationResult.n_function_evals` (implementation-independent cost proxy alongside wall time).
- **Runtime cap** = 600 s per case × method; exceeding the cap is recorded as a censored observation (counts against adequacy, reported explicitly).

## 4. Analysis plan

- Per cell (regime × k × method): recovery rate with Wilson 95% CI over the 30 seeds; median and IQR of gaps; runtime distribution.
- **Scaling:** regress log(runtime) and log(n_function_evals) on k per method (OLS with HC3 SEs over seeds); report fitted exponents with CIs. Grid's combinatorial growth is also reported analytically (C(n+k−1, k−1) grid points) next to the empirical fit.
- **Paired method comparisons** on identical cases: Wilcoxon signed-rank on gaps, α = 0.05, Holm correction across the k grid.
- **Decision rules:** H1a/H1b supported if the pooled recovery Wilson CI lower bound ≥ 95% at every k. H2 supported if grid recovery CI upper bound < 50% or runtime censored at some k ≤ 15 while SLSQP holds H1. Mixed outcomes are reported per cell, not aggregated away.

## 5. Exclusion & failure rules

- A generated case failing a `validate_case` hard invariant (NO-CLIP, finiteness, feasibility, consensus) is a **generator defect**: the run halts, the defect is fixed, the fix is documented as an amendment, and the full grid is regenerated. Cases are never silently swapped for other seeds.
- Solver non-convergence on some starts is not an exclusion; it flows into the metrics (best-of-starts, n_converged reported).
- No case-level cherry-picking: every case in the locked grid is reported.

## 6. Environment (recorded at lock)

Python 3.11.9, numpy 2.3.2, scipy 1.17.1, vlinder @ lock commit, Windows 11, single machine. Seeds and manifests make every case byte-reproducible (`table_sha256`).

## 7. What this pre-registration does NOT cover yet

Phases 2-4 (smooth_nonconvex, nonsmooth, scenario dispersion): their generator knobs exist (built 2026-07-02, plan Phases 2-4), but their grid rows and hypotheses (H3, H4 above are placeholders) are added **by amendment before those phases run**, keeping the lock-before-run discipline per phase.

---

## Amendments

(none)
