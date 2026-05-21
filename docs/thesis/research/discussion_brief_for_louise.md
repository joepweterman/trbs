# Discussion brief — research findings on optimization methods

**For:** Louise (university supervisor)
**From:** Joep Weterman
**Date:** 2026-05-21
**Reading time:** ~6 minutes
**Supporting documents (read only if you want depth):**
- [`existing_methods_survey.md`](existing_methods_survey.md) — ~7,000-word systematic survey of 7 algorithm families
- [`novel_algorithm_proposals.md`](novel_algorithm_proposals.md) — ~5,000-word in-depth proposals for three novel algorithms

---

## 1. What I researched

The proposal commits to comparing three methods: SLSQP, basin-hopping, and a genetic algorithm. I surveyed the wider optimization landscape against the **specific** structural properties of the tRBS problem (low-dimensional, simplex-constrained, black-box dependency graph, sinusoidal value functions, multi-scenario, cheap evaluations). Seven algorithm families covered, with 2022–2026 references where relevant:

1. **Local gradient-based** (SQP, trust-region, interior-point) — validates SLSQP as the right local engine; trust-region as robustness check.
2. **Simplex-native methods** ★ (Frank-Wolfe, mirror descent) — **understudied for MCDA; largest gap I found.**
3. **Global stochastic** (basin-hopping, simulated annealing, DIRECT, MLSL) — validates basin-hopping; DIRECT requires awkward reparameterization for simplex.
4. **Population-based / evolutionary** (real-coded GA, DE, **CMA-ES**, PSO) — CMA-ES is the modern gold standard and is conspicuously missing from the proposal; worth a sentence even if we don't implement.
5. **Multi-objective evolutionary** (NSGA-II/III, MOEA/D, SMS-EMOA) — relevant if we pursue an explicit Pareto-front contribution.
6. **Surrogate / Bayesian** (BO, TRIKE) — **defensibly rule out** for tRBS because evaluations are cheap (ms-scale); GP overhead dominates.
7. **Robust / uncertainty-aware** (Bertsimas-Sim, scenario decomposition, CVaR) — methodological backbone for RQ3 (scenario robustness).

The full survey defends each method's fit/misfit against tRBS with citations. The headline outcome: **two genuine gaps** in the literature that the thesis can exploit.

## 2. The gaps that motivate novel algorithms

- **Gap 1 — Simplex-native methods (Frank-Wolfe, mirror descent) are absent from MCDA/decision-support literature.** Both have ~25 years of strong convergence theory and O(k) per-iteration cost. They appear in online learning and modern portfolio research but apparently nowhere in MCDA.
- **Gap 2 — Multi-start strategies for nonconvex NLP are still using naive random restarts.** Bandit theory (UCB, Thompson sampling) has matured substantially over the last 20 years and has been applied to SAT-solver restarts and hyperparameter optimization, but **not to NLP restart allocation**.
- **Gap 3 — Explicit Pareto-front + scenario-robust optimization is rarely applied to MCDA.** NSGA-II for multi-objective and Bertsimas-Sim Γ-budgets for robust optimization both exist; their combination, applied to MCDA, is novel.

## 3. Three proposed novel algorithms — one paragraph each

The full development (algorithmic pseudocode, theoretical analysis, expected empirical advantage, failure modes, publishability) is in [`novel_algorithm_proposals.md`](novel_algorithm_proposals.md). Each is positioned as a **direct improvement on one of the proposal's three methods**, so the comparison story stays clean.

### A. Mirror-Descent + Basin-Hopping Hybrid (MDBH) — improves basin-hopping

Replace the SLSQP inner loop of basin-hopping with **entropic mirror descent** (the canonical algorithm for simplex-constrained optimization). Mirror descent handles the simplex natively via multiplicative-weights updates in O(k) per step, vs. SLSQP's O(k³) QP subproblems. Inherits basin-hopping's global-escape behavior. Theory combines Beck & Teboulle (2003) for the local rate with Wales & Doye (1997) for the global guarantee. **Publishable** at workshop or *Operations Research Letters* level. Implementation effort: ~38 h.

### B. Scenario-Robust NSGA-II with Bertsimas-Sim Budget (SR-NSGA) — extends GA

Treat KPI appreciations as **explicit objectives** (Pareto front, not weighted sum) and replace scenario weights with a **Bertsimas-Sim Γ-budget** on scenario weight perturbation. Output: a Pareto frontier of *"how much expected appreciation do you give up for one additional unit of scenario-robustness?"* — answering RQ3 directly with a visual that PwC stakeholders can act on. **Publishable** at *EJOR* / *Decision Sciences* level; the applied motivation is the sharpest of the three. Implementation effort: ~40 h.

### C. Bandit-Based Adaptive Multi-Start (BAMS) — improves SLSQP multi-start

Instead of N uniform Dirichlet restarts, allocate compute via **multi-armed bandit UCB**: discover basins in an exploration phase, then concentrate restarts on the most promising basins. Theory: bandit regret bounds (Auer et al. 2002; Lai & Robbins 1985) suggest 2-5× compute-efficiency improvement over naive multi-start. **Strongest standalone story** of the three because the technique is application-agnostic; could land at *Math Prog Computation* or *INFORMS J. Computing*. Lowest implementation risk (~32 h, no external dependencies beyond scipy).

## 4. Recommended path forward

Total effort for the three novel algorithms: **~110 h**, vs. the planning budget of ~130 h for W4–W11 (per [the Gantt](file:///C:/Users/joepw/Desktop/Thesis_Planning_tRBS.xlsx)). Within budget with ~15% slack.

If we have to drop one due to time pressure: **drop SR-NSGA**, keep MDBH and BAMS. Reasoning: SR-NSGA has the largest implementation effort and depends on extending to explicit multi-objective (a real scope expansion); MDBH and BAMS each map directly onto a proposal method and are unambiguous improvements.

A dependency to flag: all three (and especially SR-NSGA) benefit from extracting a **pure-function evaluation wrapper** from the existing `optimize.py` (the current code mutates `input_dict` in place — fragile if an optimizer makes hundreds of calls). This is the W4 refactor flagged in [`w1_codebase_audit.md §3.4`](../w1_codebase_audit.md); I'd want your sign-off before touching shared code.

## 5. Questions I'd like to discuss

1. **Of the three (MDBH, SR-NSGA, BAMS), do you agree with the priority order if I have to drop one?**
2. **Is "publishability beyond the thesis itself" something to actively pursue, or out of scope?** The novel algorithms are framed to be publishable, but that adds writing effort.
3. **SR-NSGA's Bertsimas-Sim Γ-budget on scenario weights — right semantics for tRBS, or should robustness be parametrized differently (e.g., uncertainty in KPI weights instead)?**
4. **CMA-ES** is missing from the proposal's three methods but is the modern gold standard for continuous BBO. Worth adding as a fourth baseline, or stay with the proposal's three?
5. **Bayesian optimization** can be defensibly ruled out because tRBS evaluations are cheap (ms-scale, surveyed in §7 of the existing-methods doc). Comfortable with that framing in the thesis?
6. **DDG-PN (JAX rewrite of `Evaluate` → projected Newton)** — rejected from the main three because it requires a major refactor of shared code. Should I keep it as a stretch contribution after the main three are benchmarked, or drop permanently?

I'd suggest we spend ~30 min of the kick-off meeting on §3–§5 of this brief; if you want depth on any single algorithm, the standalone doc covers pseudocode, theory, and failure modes.
