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

**A1 — 2026-07-23: grid at k = 15 is recorded as a censored observation without execution.**
- **Finding:** no grid task at k = 15 has ever completed. Mechanism (source-verified, `vlinder/optimize.py::generate_combinations`): every budget split that passes the sum filter is expanded through `set(permutations(combination))`, which iterates all k! index-permutations — 15! ≈ 1.31 × 10¹² per split. Extrapolating from the measured k = 12 grid task (~623 s, 12! ≈ 4.79 × 10⁸): ≈ 19 days of CPU per k = 15 grid task, × 180 tasks in the design. Empirical corroboration: zero k = 15 grid completions across both confirmatory runs; six workers observed at 100% CPU for 40+ minutes each on k = 15 grid tasks with no completion, while k = 15 continuous-method tasks complete in 3–13 s.
- **Treatment:** every (case, scenario, grid) cell at k = 15 is written to the results store as `censored = true`, `recovered = false`, with null result fields and an explanatory `censor_reason` — i.e. the §3 runtime-cap rule applied categorically: the cap (600 s) is exceeded with certainty, by five orders of magnitude. Recovery = 0 for these cells counts against grid adequacy exactly as §3 prescribes. All other cells, including grid at k ≤ 12 and all continuous methods at k = 15, run unchanged.
- **Interpretation:** this strengthens H2 rather than weakening it — the baseline cannot even enumerate its own grid at k = 15. The analytic combinatorial cost curve of §4 still covers grid at k = 15. No result-dependent discretion is involved: the failure is categorical (weeks vs. a 600 s cap), not marginal.

**A2 — 2026-07-31: the pre-declared SCIP benchmark is added, as an independent check on the certified ground truth (not as a fifth confirmatory method).**

- **Basis:** §5 pre-declares a SCIP (`pyscipopt`) global-solver benchmark, on Paul Bouman's suggestion, "added by amendment before those runs". This is that amendment. It is filed before any SCIP result enters the analysis.
- **Scope — convex regime only (the locked 420-case grid), and deliberately not a roster method.** Every method in §2 treats the objective as a black box behind `evaluate_allocation`. SCIP cannot: it is a branch-and-bound global solver and needs the objective as an algebraic model it can bound and branch on. That is reconstructible only where the data-generating process is known, i.e. the synthetic cases. It is therefore **not** registered in `METHOD_REGISTRY`, is **not** run on the bundled real cases, and does **not** join the H1/H2/H5 roster, which is locked and already executed. Adding it there would change a locked comparison after seeing its results.
- **Role:** SCIP answers a question the confirmatory grid cannot answer about itself. The four roster methods can only be compared with each other or against an oracle written by this project; a systematic error in that oracle would be invisible to all of them. SCIP reconstructs each instance independently and returns a solution carrying a *proof* of global optimality (zero primal-dual gap), so agreement is external corroboration of the ground truth rather than another opinion from the same family. Both convex variants are globally solvable in principle: the linear variant is an LP, and the curved variant maximises a concave function (`sin` on [0, π/2]) over a polytope.
- **Instrument (`experiments/synthetic/scip_benchmark.py`):** key outputs are reconstructed as `KO(x) = base + A·x` by probing the real pipeline at zero spend and at each single-lever corner, never by re-deriving the algebra from the case tables; affinity is then verified at random interior points (tolerance 1e-7) and a violation aborts the case. The appreciation terms mirror `Appreciate._appreciate_single_key_output` branch for branch. Every row carries a **model-fidelity check**: SCIP's own solution is pushed back through `evaluate_allocation` and the objective must agree within 1e-6 appreciation points, otherwise the row is not a valid comparison and is reported as unfaithful rather than as a result.
- **Instrument verification, disclosed:** the first version of the model derived the key outputs from the coefficient matrix and omitted a constant baseline that the dependency graph contributes. Because a constant cannot move an argmax, the omission was invisible to the optimum and to the gradient cross-check, and was caught only by the fidelity check (1–3 points of disagreement). This is why the map is now probed from the pipeline. All of this happened before this amendment was filed and is instrument verification, exactly as §9 treats exp01–exp03.
- **Metrics:** `proved_optimal` (SCIP gap ≤ 1e-9), signed gap against the certified `f_star`, wall-clock, branch-and-bound nodes, and the fidelity residual.
- **Decision rule, stated before the runs:** if SCIP *proves* an optimum exceeding the certified `f_star` by more than ε = 0.1 points on any case, the oracle is wrong, and H1/H2/H5 must be recomputed against corrected ground truth. Exceedances below ε are reported as a tightness bound on the oracle, as in §3. Runtime is reported descriptively only: SCIP solves a different (algebraic) problem from the roster methods and a wall-clock ranking against them would not be like for like.
- **Environment:** `pyscipopt` 6.2.1 (SCIP 10.0) added. numpy 2.3.2 and scipy 1.17.1 verified unchanged by the install, so the environment recorded in §6 for every other result still holds.

**A3 — 2026-08-05: MDBH, the proposal's Step-3 novel method, benchmarked on the locked grid — config frozen before any grid run.**

- **Basis:** the thesis proposal (Methods, Step 3) commits to developing a new hybrid method combining simplex-native local search (Frank-Wolfe / entropic mirror descent) with an adaptive escape mechanism, its final design fixed "once the benchmark shows where the off-the-shelf methods fall short". The benchmark evidence is now in (H1/H2/H5 results; real-case study; exploratory acceptance runs below), and the Mirror-Descent + Basin-Hopping hybrid (MDBH, research memo `docs/thesis/research/novel_algorithm_proposals.md` §2) has been implemented. This amendment freezes its benchmark before any of its grid cells run.
- **What the lock can and cannot protect here, stated plainly:** the roster's grid results are known, so this comparison is one-directional by construction. Pre-declaring MDBH's configuration protects against tuning MDBH *after seeing its own grid results*; it cannot un-know the roster's results. MDBH therefore does **not** join the H1/H2/H5 verdicts (locked and reported); it is reported as the thesis's Step-3 method chapter, compared on the same instrument.
- **Method spec (frozen):** `vlinder.optimize` registry entry `"mdbh"`. Entropic mirror descent as the local engine — multiplicative-weights update with max-norm-normalised finite-difference gradients (h = 1e-6, z-space), step schedule η/√t, best-iterate tracking, patience 8, iterate floor 1e-12 — inside a basin-hopping outer loop with a Fisher-Rao geodesic kick `w ∝ w·exp(σξ)` (ξ centred Gaussian) and Metropolis acceptance. The capped simplex Σx ≤ B is handled with a slack coordinate (the walk lives on Δ_{k+1}, x = B·w[:k]); the slack's partial derivative is 0 analytically, so one gradient costs k evaluations. **Configuration: n_starts = 5 (first chain from the centroid, rest Dirichlet(1,…,1)), n_hops = 10, n_local_steps = 50, η = 1.0, σ = 1.5, T = 1.0, method_seed = case seed** (same seeding rule as the roster).
- **Calibration disclosure (exploratory, pre-amendment, none of it on locked grid cells):** defaults were calibrated on acceptance runs on Beerwiser, IZZ (seed 42), Refugee (seeds 0–4), and four generator cases outside the locked grid (different `n_key_outputs`, non-`Study_*` naming, throwaway roots; params: convex-linear k3 s0, convex-curved k9 s2, smooth-bilinear k6 s12, nonsmooth-clip k6 s13). Two findings from the June design memo were falsified there and are disclosed as such: the σ = 0.1 kick guess (multiplicative kicks move in log-scale, so narrow kicks cannot change allocation support — Refugee's sparse global optimum then traps most chains; σ = 1.5 with restarts escaped on every tested seed) and the unconditional "better on multimodal, vertex-near optima" expectation (holds only after that recalibration). The confirmed design prediction: slower precision tail than SLSQP on smooth cells (June memo, failure mode 2).
- **Design:** the locked 420-case grid (both convex variants, k ∈ {2,…,15}, seeds 0–29), all 3 scenarios, method `mdbh` only → 1,260 rows. Same metrics as §3 (gap, recovery at ε = 0.1, wall-clock, `n_function_evals`, 600 s cap → censored), same analysis machinery as §4 (Wilson CIs per cell, log-log eval/runtime exponents, Wilcoxon pairings against the roster on identical cases). Results go to a **separate store** (`generated/study/results_mdbh.jsonl`); the locked `results.jsonl` is not appended to. Driver: `experiments/synthetic/run_mdbh_benchmark.py`.
- **Predictions, stated before the run:** (P1) recovery = 100% within ε on both convex variants at every k, matching SLSQP/basin-hopping; (P2) the log-log exponent of `n_function_evals` in k is ≈ 1 (the O(k) gradient is the dominant cost), comparable to SLSQP's 1.077; (P3) raw gaps in convex cells are *larger* than SLSQP's (slower tail — structural), so adequacy is judged on recovery and cost, not on raw gap; (P4) at k = 15 MDBH completes well under the runtime cap. A miss on P1 or P4 counts against MDBH exactly as §3 prescribes; a miss on P2 falsifies the O(k)-cost story of the method chapter.
- **Real cases:** MDBH results on Beerwiser/Refugee/IZZ (the acceptance numbers above, plus any re-runs) remain exploratory context for the method chapter, reported as such; the confirmatory-style claim of this amendment is the synthetic grid only.
- **Environment:** unchanged from §6 (no new packages; MDBH is numpy-only).
