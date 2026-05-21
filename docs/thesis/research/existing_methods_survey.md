# Existing Optimization Methods for tRBS: A Structured Survey

**Author:** Joep Weterman
**Date:** 2026-05-21
**Status:** Draft for kick-off meeting + foundation for the methodology chapter

---

## 0. How to read this document

This document surveys the existing optimization-method landscape against the specific structural properties of the tRBS problem. Each family section follows the same template:

1. **What it is** — one-paragraph algorithmic sketch with the canonical reference.
2. **Convergence properties** — what is theoretically guaranteed and under what assumptions.
3. **Computational complexity** — per-iteration cost, total-iteration cost, scalability in the number of decision variables k.
4. **Software** — production-quality implementations.
5. **Fit for tRBS** — surgical assessment, not generic praise. The whole point of the survey is to find the families where there is a *real* opportunity to improve on the proposal's choice of SLSQP / basin-hopping / GA.
6. **Key references** — classical and 2022–2026 (the cutoff matters: a lot of theoretical progress happened in the last three years, especially on Frank-Wolfe and mirror descent).

The seven families surveyed:

1. Local gradient-based (SQP, trust-region, interior-point) — §2
2. Simplex-native methods (Frank-Wolfe, mirror descent) — §3 ★ understudied for tRBS
3. Global stochastic (basin-hopping, simulated annealing, DIRECT, MLSL) — §4
4. Population-based / evolutionary (real-coded GA, DE, CMA-ES, PSO) — §5
5. Multi-objective evolutionary (NSGA-II/III, MOEA/D, SMS-EMOA) — §6
6. Surrogate / Bayesian (BO, TRIKE, trust-region surrogate) — §7
7. Robust / uncertainty-aware (Bertsimas-Sim, scenario decomposition, CVaR) — §8

§1 restates the tRBS problem structure for reference. §9 synthesizes the gap analysis that motivates the three novel-algorithm proposals in the companion document.

---

## 1. tRBS problem structure (recap)

Formal: maximize a scalar appreciation F : Δ^k → [0, 100], where

$$ F(\mathbf{x}) = \sum_s w_s \sum_i w_i \, v_i(g_i(\mathbf{x}, s)) $$

with $\mathbf{x}$ on the (k−1)-dimensional probability simplex Δ^k = {x ≥ 0 : Σx_i = B}, $w_s$ scenario weights, $w_i$ two-tier KPI weights, $v_i$ either a linear or sinusoidal value function on [s_i, e_i], and $g_i$ the result of traversing the case's dependency graph.

The structural properties that decide method fitness:

| Property | Value | Consequence |
|---|---|---|
| Dimension k | 2–6 typical, ≤15 for PwC case | Low-dimensional regime; methods designed for k ≥ 100 are wasted |
| Feasible set | Probability simplex (scaled to B) | Methods with native simplex handling (FW, MD) avoid projection cost |
| Smoothness | C^∞ on the interior; sinusoidal value functions saturate (kinks) at v=0 and v=100 | Methods that require strict smoothness everywhere can stall at boundaries |
| Convexity | Non-convex; multimodal when g_i is non-affine in x | Pure local methods need globalization |
| Gradient | Not analytic; available via finite differences or JAX rewrite of `Evaluate` | Methods that need exact Hessians require AD investment |
| Evaluation cost | Milliseconds | Permits 10^4–10^5 evaluations; sample-efficient methods (BO) are over-engineered |
| Number of objectives | 1 under current scalarization; k_KPI under explicit multi-objective | Both single- and multi-objective branches need coverage |
| Number of scenarios | 2–3 | Scenario-wise robust optimization is tractable |

These eight properties together rule out a surprising number of "obvious" choices and elevate two understudied families — simplex-native methods (Family 2) and bandit-style restart strategies layered on top of any local method.

---

## 2. Family 1 — Local gradient-based methods

### 2.1 Sequential Quadratic Programming (SQP)

**What it is.** SQP solves the constrained NLP

$$ \min_x f(x) \;\; \text{s.t.} \;\; c_i^E(x) = 0, \; c_j^I(x) \geq 0 $$

by iteratively forming a quadratic model of the Lagrangian, linearizing the constraints, and solving the resulting quadratic program for a search direction. Three production implementations dominate:

- **SLSQP** (Kraft, 1988) — active-set SQP with BFGS Hessian approximation. Available in `scipy.optimize.minimize(method="SLSQP")`. **Joshi et al. (2024)** released [PySLSQP](https://arxiv.org/html/2408.13420v1), a modernized open-source reimplementation with transparency and visualization tooling — useful for the thesis benchmark because Kraft's original Fortran code is opaque to instrumentation.
- **SNOPT** (Gill, Murray & Saunders, 2005, [SIAM Review](https://epubs.siam.org/doi/10.1137/S0036144504446096)) — commercial, BFGS, designed for problems with thousands of constraints but ≤ 2,000 degrees of freedom. Not freely available; mostly cited for benchmarking.
- **IPOPT** (Wächter & Biegler, 2006) — primal-dual interior-point method (not strictly SQP but the de facto industrial NLP solver). Open-source; requires HSL or MUMPS linear solvers.

**Convergence.** Quadratic locally under standard regularity (LICQ, second-order sufficiency, strict complementarity); see Nocedal & Wright (2006, Ch. 18). Globally only when augmented with line-search or trust-region globalization.

**Complexity.** Per-iteration cost is dominated by the QP subproblem: O(k^3) for active-set, O(k^2) per IPM step. Total iterations: 10–100 on tRBS-scale problems.

**Software.** `scipy.optimize.minimize` (SLSQP, trust-constr), `cyipopt` (IPOPT Python bindings), `casadi` (symbolic AD + multiple solvers).

**Fit for tRBS.**
- ✅ Native handling of the simplex (equality Σx_i = B + bounds x_i ≥ 0).
- ✅ KKT multipliers as shadow prices on the budget — directly interpretable for PwC stakeholders.
- ✅ Mature implementations, well-tested on portfolio NLP (Brenndoerfer, 2024).
- ❌ Local convergence only — vulnerable to multimodality from sinusoidal value functions composed with multiplicative dependencies.
- ⚠ Without analytical gradients, finite differences cost k+1 evaluations per gradient — tolerable for k ≤ 15 and ms-scale evaluations.

**Verdict.** This is the right choice as a *local engine*, but on its own insufficient. The proposal's SLSQP + multi-start is sound. The interesting question is whether the multi-start strategy can be improved (see novel algorithm C in the companion document).

### 2.2 Trust-region methods

**What it is.** At each iterate, form a quadratic model m_k(p) ≈ f(x_k + p), restrict ||p|| ≤ Δ_k (the trust-region radius), solve the constrained subproblem exactly or approximately, accept the step if the actual reduction matches the predicted reduction, adjust Δ_k. References: Conn, Gould & Toint (2000) *Trust-Region Methods* (canonical monograph); `scipy.optimize.minimize(method="trust-constr")` for the constrained version (Conn, Gould & Toint, 1988).

**Convergence.** Global to first-order critical points under mild assumptions; quadratic locally with exact Hessian.

**Fit for tRBS.** Similar to SLSQP but more robust on highly non-convex landscapes — the trust-region radius adapts automatically when the quadratic model is a poor fit. Worth including as a robustness check on SLSQP results (sensitivity analysis in the methodology section). **Key opportunity:** the trust-region quadratic model is generic; for tRBS, where value functions are *known* to be sinusoidal in some KPIs, a domain-specific basis (sinusoidal model) could converge faster — see novel algorithm idea A in the long-list (rejected for the main 3 due to convergence-theory complexity).

### 2.3 Interior-point methods

**What it is.** Replace the constrained problem with a sequence of barrier-augmented unconstrained problems: min f(x) − μ Σ log(c_j(x)), with μ → 0. Solve each by Newton's method. IPOPT is the standard implementation (Wächter & Biegler, 2006).

**Fit for tRBS.** Excellent in theory, polynomial-time guarantees for convex problems, but overkill for tRBS-scale (k ≤ 15). IPOPT's setup overhead and dependency on HSL/MUMPS makes it less appealing than SLSQP for the benchmark. Worth a one-paragraph mention in the thesis methodology for completeness; not worth implementing.

### 2.4 Recent developments (2024–2026)

- **OpenSQP** (Joshi et al., [2025 arXiv preprint](https://arxiv.org/html/2512.05392v1)) — reconfigurable open-source SQP in Python; benchmarks against SNOPT, SLSQP, IPOPT, trust-constr on CUTEst. Useful for citing benchmark methodology in the thesis.
- **I-SLSQP / I-SQP** (process-engineering literature, 2024) — interior-SLSQP variants showing 5–20% cost reduction on difficult problems vs. standard SLSQP. Not yet in scipy.

### 2.5 References

- Kraft, D. (1988). *A software package for sequential quadratic programming*. DFVLR-FB 88-28.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Springer.
- Conn, A. R., Gould, N. I. M. & Toint, P. L. (2000). *Trust-Region Methods*. MPS-SIAM Series on Optimization.
- Gill, P. E., Murray, W. & Saunders, M. A. (2005). [SNOPT: An SQP Algorithm for Large-Scale Constrained Optimization](https://epubs.siam.org/doi/10.1137/S0036144504446096). *SIAM Review*, 47(1), 99–131.
- Wächter, A. & Biegler, L. T. (2006). *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming*. *Mathematical Programming*, 106(1), 25–57.
- Joshi, A. et al. (2024). [PySLSQP](https://arxiv.org/html/2408.13420v1).

---

## 3. Family 2 — Simplex-native methods ★

This family is the most consequential for the thesis. The tRBS feasible set is a simplex, yet the proposal's three methods (SLSQP, basin-hopping, GA) all treat the simplex as a generic linear-equality + bounds constraint, paying constraint-handling cost at every step. Two algorithms are *designed* for simplex constraints and have been developed extensively in the last decade in machine learning and online optimization. They are nearly absent from the MCDA / decision-support literature, which is the gap this thesis can exploit.

### 3.1 Frank-Wolfe (conditional gradient method)

**What it is.** At iterate x_k, solve the **linear** subproblem s_k = argmin_{s ∈ C} ⟨∇f(x_k), s⟩, then take x_{k+1} = x_k + γ_k (s_k − x_k) with step size γ_k ∈ [0, 1]. Originally Frank & Wolfe (1956). Key insight: the iterate stays feasible automatically as a convex combination of x_k and s_k; **no projection step is required**. On the simplex, the linear subproblem is trivial — pick the vertex that minimizes the inner product, an O(k) operation.

**Convergence.** O(1/t) for smooth convex f on a compact convex set (Jaggi, 2013 — [Revisiting Frank-Wolfe](https://www.semanticscholar.org/paper/Revisiting-Frank-Wolfe:-Projection-Free-Sparse-Jaggi/961eabeaebd7035cd7668c9917fa9c39462e1113)). **Linear convergence is achievable on polytopes** (which includes the simplex) under strong convexity, with the modification of Lacoste-Julien & Jaggi (2015). Multi-objective Frank-Wolfe variants with improved rates: [Cocchi et al. (2024)](https://arxiv.org/pdf/2406.06457). Non-convex convergence to stationary points: [Lacoste-Julien (2016)](https://arxiv.org/pdf/1607.00345).

**Complexity.** Per iteration: one gradient evaluation + one linear minimization. Linear minimization on the simplex is **O(k)** — far cheaper than the O(k^3) QP subproblem of SLSQP. Memory: O(k).

**Fit for tRBS.**
- ✅ **Projection-free** — no constraint-handling overhead per iteration.
- ✅ Sparse iterates by design — successive vertices visited form a "frontier" of allocations that can be interpreted as natural intermediate solutions.
- ✅ Recent application to [tactical portfolio optimization](https://www.mdpi.com/2227-7390/13/18/3038) (2025) shows direct applicability.
- ❌ O(1/t) convergence is *slower* than SLSQP's locally-quadratic rate on smooth problems.
- ⚠ Non-convex landscapes require globalization (multi-start or basin-hopping wrapping), same as SLSQP.

**Verdict.** Frank-Wolfe should be in the thesis comparison. It is theoretically elegant for simplex problems, has been entirely absent from the MCDA literature, and a properly conducted benchmark would either (a) show it competitive with SLSQP on tRBS — a publishable result — or (b) show where it fails, with a clear mechanism (likely: slow convergence near the optimum on smooth problems).

### 3.2 Mirror descent (entropic gradient)

**What it is.** Generalized gradient descent where the Euclidean inner-product geometry is replaced by a Bregman divergence induced by a strongly convex "mirror map" ψ. On the simplex, the canonical mirror map is the negative entropy ψ(x) = Σ x_i log x_i, giving the **entropic** mirror descent update:

$$ x_{k+1,i} \;\propto\; x_{k,i} \cdot \exp(-\eta \, \nabla_i f(x_k)) $$

with multiplicative normalization to keep Σx_i = 1. Originally Nemirovsky & Yudin (1983); modern treatment Beck & Teboulle (2003, [*Operations Research Letters*](https://www.sciencedirect.com/science/article/abs/pii/S0167637702002316)). This is the **entropic mirror descent algorithm (EMDA)**.

**Convergence.** O(1/√t) for non-smooth convex f, O(1/t) for smooth convex f with constant step size. **Critically**, the convergence constant depends on the dimension only as √(log k) instead of √k — Beck & Teboulle's headline result: *"an efficiency estimate which is almost independent in the dimension of the problem"*. For tRBS with k ≤ 15 this is a marginal win, but the structural property — that the geometry of the iterate matches the geometry of the constraint set — is what matters.

**Complexity.** Per iteration: one gradient evaluation + one element-wise exponential + one renormalization. **All O(k)**. The cheapest per-iteration cost of any method in this survey.

**Fit for tRBS.**
- ✅ Native simplex handling via the exponential update.
- ✅ Iterates remain strictly in the interior — no boundary degeneracy issues.
- ✅ Naturally combines with stochastic gradients (relevant if you sample scenarios rather than enumerate them).
- ⚠ Pure mirror descent is a first-order method; convergence is slow relative to SLSQP on smooth problems.
- ❌ Non-convex multimodal landscapes still require globalization — same as Frank-Wolfe.

**Verdict.** Mirror descent should be the *local engine* inside a basin-hopping wrapper for the tRBS simplex. This is the kernel of novel algorithm A in the companion document. Pure mirror descent on its own loses to SLSQP on speed; combined with basin-hopping it becomes a method that combines simplex-native handling with global search — and that combination does not appear in the literature.

### 3.3 Projected gradient + projection algorithms

**What it is.** Standard gradient descent followed by a projection onto the simplex: x_{k+1} = Π_Δ(x_k − η ∇f(x_k)). Projection onto the simplex is solvable in O(k log k) via sorting (Held, Wolfe & Crowder, 1974) or O(k) via specialized algorithms (Condat, 2016).

**Fit for tRBS.** Standard, well-understood, but inferior to mirror descent on the simplex — projection introduces a discontinuous step that can stall near vertices. Worth one paragraph; not worth implementing as a separate benchmark.

### 3.4 References

- Frank, M. & Wolfe, P. (1956). *An algorithm for quadratic programming*. *Naval Research Logistics Quarterly*, 3(1–2), 95–110.
- Beck, A. & Teboulle, M. (2003). [Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization](https://www.sciencedirect.com/science/article/abs/pii/S0167637702002316). *Operations Research Letters*, 31(3), 167–175.
- Jaggi, M. (2013). [Revisiting Frank-Wolfe: Projection-free Sparse Convex Optimization](https://www.semanticscholar.org/paper/Revisiting-Frank-Wolfe:-Projection-Free-Sparse-Jaggi/961eabeaebd7035cd7668c9917fa9c39462e1113). *ICML 2013*.
- Lacoste-Julien, S. & Jaggi, M. (2015). *On the global linear convergence of Frank-Wolfe optimization variants*. *NeurIPS 2015*.
- Cocchi, G., et al. (2024). [Improved convergence rates for the multiobjective Frank-Wolfe method](https://arxiv.org/pdf/2406.06457).
- Condat, L. (2016). *Fast projection onto the simplex and the ℓ1 ball*. *Mathematical Programming*, 158(1–2), 575–585.

---

## 4. Family 3 — Global stochastic methods

### 4.1 Basin-hopping (Wales & Doye, 1997)

Already covered in [w1_optimization_methods.md §2.2](../w1_optimization_methods.md). Headline: hybrid local-optimization + Metropolis-accepted perturbation; designed for "smooth basins separated by saddle barriers" — a structural match for the tRBS landscape produced by sinusoidal value functions composed with multiplicative dependencies. Recent [benchmark by Sanvito & Cattaneo (2024)](https://arxiv.org/html/2403.05877v1) confirms competitiveness with state-of-the-art metaheuristics on standard multimodal benchmarks.

### 4.2 Simulated Annealing (SA)

**What it is.** A simpler ancestor of basin-hopping: random walk with Metropolis acceptance, no inner local optimization. Originally Kirkpatrick, Gelatt & Vecchi (1983). Convergence to the global optimum is guaranteed under a logarithmic cooling schedule (Hajek, 1988) — slow.

**Fit for tRBS.** Strictly dominated by basin-hopping for the tRBS problem class. Worth a mention as the conceptual parent; not worth implementing.

### 4.3 DIRECT (DIviding RECTangles)

**What it is.** Deterministic global optimization for bound-constrained problems. Recursively partitions the search space into hyperrectangles; in each iteration, selects "potentially optimal" rectangles using a Lipschitz-constant-free criterion and subdivides them. Originally Jones, Perttunen & Stuckman (1993, *JOTA*). Guaranteed convergence to the global optimum if f is continuous near the optimum.

**Complexity.** Per iteration: O(N_rect) where N_rect grows polynomially. Becomes inefficient above k ≈ 10–15 due to the curse of dimensionality on partition refinement (Jones, 2001).

**Fit for tRBS.**
- ✅ Deterministic — same result every run.
- ✅ No hyperparameter tuning (no temperature, no population size).
- ⚠ Native bound constraints only; the simplex (equality Σx_i = B) needs reparameterization. Standard approach: optimize over the (k−1)-dimensional unit cube and back-transform via cumulative sums or barycentric mapping. Reparameterization distorts the metric and can blow up convergence in practice.
- ⚠ Recent improvements: [HALRECT (2022)](https://arxiv.org/pdf/2205.03015) using local Lipschitz estimates; adaptive variants ([2022 arXiv](https://arxiv.org/html/2211.04129)). Worth mentioning, not worth implementing.

**Verdict.** Cite as a deterministic-global alternative to basin-hopping in the thesis. Reparameterization friction makes it less elegant than mirror-descent-based approaches for the simplex. Don't include in primary benchmark unless time permits.

### 4.4 Multi-Level Single Linkage (MLSL)

**What it is.** A multi-start strategy with clustering: sample candidate starting points uniformly, cluster them by proximity, run local optimization only on cluster representatives that haven't been explored. Theoretically guaranteed to find all local minima with probability 1 as samples → ∞ (Rinnooy Kan & Timmer, 1987). Available as `scipy.optimize.shgo` (a closely related variant).

**Fit for tRBS.** Strong candidate as a *multi-start strategy* (one level below the choice of local engine). Worth comparing against naive Dirichlet multi-start in the thesis. The novel algorithm C (bandit-based multi-start) goes further by using bandit theory to allocate compute *adaptively*.

### 4.5 References

- Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P. (1983). *Optimization by Simulated Annealing*. *Science*, 220(4598), 671–680.
- Wales, D. J. & Doye, J. P. K. (1997). *Global Optimization by Basin-Hopping*. *J. Phys. Chem. A*, 101(28), 5111–5116.
- Jones, D. R., Perttunen, C. D. & Stuckman, B. E. (1993). *Lipschitzian optimization without the Lipschitz constant*. *JOTA*, 79(1), 157–181.
- Rinnooy Kan, A. H. G. & Timmer, G. T. (1987). *Stochastic global optimization methods part II: Multi level methods*. *Mathematical Programming*, 39(1), 57–78.
- Sanvito, F. & Cattaneo, M. (2024). [A Performance Analysis of Basin Hopping](https://arxiv.org/html/2403.05877v1).

---

## 5. Family 4 — Population-based / evolutionary methods

### 5.1 Real-coded Genetic Algorithm (GA)

Covered in [w1_optimization_methods.md §2.3](../w1_optimization_methods.md). SBX crossover + polynomial mutation, standard NSGA-II machinery for the single-objective scalarized variant.

### 5.2 Differential Evolution (DE)

**What it is.** Storn & Price (1997). For each individual x_i in the population, form a mutant v_i = x_a + F(x_b − x_c) using three random distinct individuals, then crossover with x_i, then select. F and CR are hyperparameters.

**Convergence.** No strong theoretical guarantees, but extensive empirical success. Often outperforms GA on continuous problems with fewer hyperparameters.

**Fit for tRBS.** Stronger candidate than GA for the proposal's "metaheuristic" leg. `scipy.optimize.differential_evolution` is mature, supports linear constraints. Could **substitute** for GA if early experiments show GA struggling. Mention in the thesis methodology as a robustness check on the GA results.

### 5.3 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**What it is.** Hansen & Ostermeier (2001). Population-based; the covariance matrix of the sampling distribution adapts to capture the local landscape geometry. Standard benchmark for continuous black-box optimization. Recent variants: [CMA-ES-LED (2024)](https://arxiv.org/pdf/2401.15876) for low-effective-dimension problems; matrix-free CMA-ES (2025) reducing O(d^2) → O(d).

**Convergence.** No formal guarantees; empirically robust on multimodal continuous problems. Designed to be invariant under monotonic transformations of f — relevant for tRBS, where appreciation values are bounded [0, 100].

**Complexity.** Per generation: O(λ × k) function evaluations (λ = population size, typically 4 + ⌊3 log k⌋) + O(k^2) covariance update.

**Fit for tRBS.**
- ✅ Industrial standard for continuous black-box optimization; **conspicuously missing from the proposal's three methods**.
- ✅ Invariant to monotonic transformations — robust to the bounded [0, 100] appreciation range.
- ⚠ No native simplex handling — requires either reparameterization or a penalty term.
- ❌ Designed for higher-dimensional problems (k ≥ 10). For tRBS's k = 2–6 the covariance adaptation machinery may be wasted.

**Verdict.** Worth mentioning in the survey, worth a paragraph in the thesis. Risk of including in primary benchmark: it's a strong baseline that might out-perform the proposal's GA, which would then require explaining why the proposal didn't include it. Decision recommendation: include CMA-ES as the "modern gold-standard metaheuristic" reference point, but only run it as a sanity check, not as a primary comparison.

### 5.4 Particle Swarm Optimization (PSO)

**What it is.** Kennedy & Eberhart (1995). Population of particles with velocity; each particle is attracted to its own best position and the swarm's best position.

**Fit for tRBS.** Generally outperformed by CMA-ES and DE on continuous problems (Rios & Sahinidis, 2013). Skip.

### 5.5 References

- Storn, R. & Price, K. (1997). *Differential Evolution – A Simple and Efficient Heuristic*. *Journal of Global Optimization*, 11(4), 341–359.
- Hansen, N. & Ostermeier, A. (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies*. *Evolutionary Computation*, 9(2), 159–195.
- Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial*. arXiv:1604.00772.
- Nomura, M. & Shibata, M. (2024). [cmaes: A Simple yet Practical Python Library for CMA-ES](https://arxiv.org/pdf/2402.01373).
- Rios, L. M. & Sahinidis, N. V. (2013). *Derivative-Free Optimization: A Review*. *Journal of Global Optimization*, 56(3), 1247–1293.

---

## 6. Family 5 — Multi-objective evolutionary methods

If the thesis stays scalar (weighted-sum aggregation of KPI appreciations, current tRBS behavior), this family is informational. If the thesis pursues the **stretch contribution of explicit Pareto fronts**, this family becomes central.

### 6.1 NSGA-II (Deb, Pratap, Agarwal & Meyarivan, 2002)

Already covered in [w1_optimization_methods.md §2.3](../w1_optimization_methods.md) for the single-objective case. The multi-objective machinery is its primary contribution: fast non-dominated sorting + crowding distance for Pareto front diversity. Standard for 2–3 objectives.

### 6.2 NSGA-III (Deb & Jain, 2014)

**What it is.** Extension of NSGA-II for many-objective problems (≥ 4 objectives). Replaces crowding distance with reference-point-based selection, maintaining Pareto front coverage in higher-dimensional objective spaces where crowding distance becomes degenerate. Recent ESG-portfolio application: [Larni-Fooeik et al. (2025)](https://papers.ssrn.com/sol3/Delivery.cfm/3c1a2997-b403-44c1-8c91-5bd66f348c50-MECA.pdf?abstractid=5210265&mirid=1&type=2).

**Fit for tRBS.** Worth considering if the number of KPIs exceeds 3 (typical for the IZZ, DSM, NEMO cases). The 2024 benchmark by [Sharma et al.](https://link.springer.com/article/10.1007/s00500-024-09872-z) gives NSGA-III a hypervolume of 835.6 vs. NSGA-II's 709.8 vs. SMS-EMOA's 743.1 on a comparable problem, which is a solid argument for NSGA-III when objectives ≥ 4.

### 6.3 MOEA/D (Decomposition-based)

**What it is.** Zhang & Li (2007). Decompose the multi-objective problem into a set of single-objective subproblems via weight vectors and solve them concurrently with neighborhood cooperation. Recent runtime analyses ([2024 arXiv](https://arxiv.org/pdf/2404.11433)) clarify its theoretical properties.

**Fit for tRBS.** Strong candidate; some empirical work shows it dominates NSGA-II on many-objective problems with smooth structure. Worth mentioning, not implementing.

### 6.4 SMS-EMOA (S-Metric Selection)

**What it is.** Beume, Naujoks & Emmerich (2007). Selection based on hypervolume contribution — solutions that contribute most to the hypervolume of the current Pareto approximation are retained. Strong theoretical convergence properties ([2024 runtime analysis](https://arxiv.org/pdf/2312.10290)).

**Fit for tRBS.** Computationally expensive (hypervolume computation is #P-hard for many objectives). Worth a mention; skip for the primary benchmark.

### 6.5 Verdict for the thesis

If pursuing the multi-objective extension: NSGA-II for 2–3 KPIs, NSGA-III for 4+. MOEA/D and SMS-EMOA can be cited as alternatives without implementation. The novel algorithm B in the companion document (scenario-robust NSGA-II with Bertsimas-Sim budgets) builds directly on this family.

### 6.6 References

- Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). *NSGA-II*. *IEEE TEC*, 6(2), 182–197.
- Deb, K. & Jain, H. (2014). *NSGA-III*. *IEEE TEC*, 18(4), 577–601.
- Zhang, Q. & Li, H. (2007). *MOEA/D*. *IEEE TEC*, 11(6), 712–731.
- Beume, N., Naujoks, B. & Emmerich, M. (2007). *SMS-EMOA*. *European Journal of Operational Research*, 181(3), 1653–1669.
- Verma, S., et al. (2021). [A Comprehensive Survey on NSGA-II](https://dl.acm.org/doi/10.1007/s10462-023-10526-z). *Artificial Intelligence Review*.

---

## 7. Family 6 — Surrogate / Bayesian methods

### 7.1 Bayesian Optimization (BO) with Gaussian Processes

**What it is.** Build a Gaussian process posterior over f after each evaluation; choose the next evaluation by maximizing an acquisition function (expected improvement, UCB, or knowledge gradient). Mockus (1989); modern ML treatment Snoek, Larochelle & Adams (2012); tutorial [Frazier (2018)](https://arxiv.org/pdf/1807.02811).

**Complexity.** GP fitting is **O(n^3)** in the number of accumulated samples; **acquisition optimization is itself a global optimization problem**. Each iteration is dominated by surrogate-model overhead.

**Fit for tRBS.**
- ❌ **tRBS evaluations are cheap (milliseconds).** The fundamental trade-off in BO is "spend more compute on the surrogate to save expensive function evaluations." For cheap evaluations, the GP overhead dominates.
- ❌ Native simplex handling requires kernel modifications or reparameterization.
- ⚠ A recent line of work (e.g., [Park et al., 2024](https://arxiv.org/pdf/2502.06178)) proposes kernel-regression alternatives that reduce O(n^3) → O(n^2). Still over-engineered for tRBS.

**Verdict.** **Defensibly rule out BO in the thesis methodology.** The argument: BO is most useful when function evaluation is expensive (deep-learning hyperparameter tuning, simulation runs of hours); tRBS evaluations are ms-scale, so the GP overhead is a net loss. Cite Frazier (2018) for the standard guidance, cite the GP-overhead caveats from the literature, mention as future work if PwC's real case turns out to be expensive.

### 7.2 TRIKE / Trust-Region Surrogate Methods

**What it is.** Combine trust-region globalization with surrogate models (Kriging or polynomial). Industrial use in engineering design where each evaluation involves a CFD or FEM simulation.

**Fit for tRBS.** Same verdict as BO: over-engineered for ms-scale evaluations.

### 7.3 References

- Mockus, J. (1989). *Bayesian Approach to Global Optimization*. Kluwer.
- Snoek, J., Larochelle, H. & Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. *NeurIPS 2012*.
- Frazier, P. I. (2018). [A Tutorial on Bayesian Optimization](https://arxiv.org/pdf/1807.02811). arXiv:1807.02811.

---

## 8. Family 7 — Robust / uncertainty-aware methods

This family is the methodological backbone for RQ3 (scenario-robust allocations).

### 8.1 Bertsimas-Sim uncertainty budgets

**What it is.** Bertsimas & Sim (2003, 2004) introduced **polyhedral uncertainty sets** parametrized by a "budget" Γ that controls the number of uncertain parameters allowed to deviate simultaneously. The robust counterpart of a linear program with budgeted uncertainty has the same complexity as the nominal problem — a major theoretical breakthrough. Original applications: robust shortest path, robust scheduling. Now standard in finance and operations research.

**Why it matters for tRBS.** In tRBS terms, the "uncertainty" is in scenario weights w_s. Currently the user specifies them; a robust formulation would say "allow at most Γ scenarios to deviate by up to δ from their nominal weights, find the allocation that maximizes worst-case appreciation under this uncertainty set." This is the formal language for RQ3.

**Recent work.** [Robust multi-objective portfolio optimization using Bertsimas method (2017)](https://arxiv.org/pdf/1711.03716); [Effective budget of uncertainty for classes of robust optimization (2019)](https://arxiv.org/pdf/1907.02917); [budgeted interdiction uncertainty (2024)](https://link.springer.com/article/10.1007/s00291-024-00772-0).

### 8.2 Scenario decomposition (L-shaped method)

**What it is.** Benders' decomposition specialized for two-stage stochastic programs. The first stage decides x; the second stage evaluates each scenario s and adds optimality cuts back. Originally Van Slyke & Wets (1969); modern treatment Birge & Louveaux (2011).

**Fit for tRBS.** Conceptually applicable if scenarios were stochastic — but tRBS scenarios are deterministic alternative futures, not realizations of a random variable. The cleaner connection is **multi-scenario robust optimization** (max over x, min over scenarios) rather than expected-value stochastic programming. Worth a paragraph; not the primary technical tool.

### 8.3 CVaR / Conditional Value-at-Risk

**What it is.** Rockafellar & Uryasev (2000). Replace the expected-value objective with the expected value conditional on being in the worst α% of outcomes. Widely used in financial portfolio optimization.

**Fit for tRBS.** Strong candidate for an alternative robustness formulation: "maximize the expected appreciation over the worst 30% of scenarios." Implementationally clean (linear program in many cases). Worth one paragraph in the methodology; perhaps a sensitivity analysis.

### 8.4 References

- Bertsimas, D. & Sim, M. (2003). *Robust discrete optimization and network flows*. *Mathematical Programming*, 98(1–3), 49–71.
- Bertsimas, D. & Sim, M. (2004). *The price of robustness*. *Operations Research*, 52(1), 35–53.
- Ben-Tal, A., El Ghaoui, L. & Nemirovski, A. (2009). *Robust Optimization*. Princeton University Press.
- Rockafellar, R. T. & Uryasev, S. (2000). *Optimization of conditional value-at-risk*. *Journal of Risk*, 2, 21–42.
- Birge, J. R. & Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd ed., Springer.

---

## 9. Gap analysis — where novel algorithms can live

Synthesizing the seven families against the eight tRBS-specific structural properties, three "gaps" emerge that are simultaneously (a) under-explored in the literature and (b) directly addressable by combining established components in new ways:

### Gap 1 — Simplex-native methods are absent from MCDA / decision-support literature.

Frank-Wolfe and mirror descent are the canonical algorithms for simplex-constrained smooth optimization, with strong convergence theory and O(k) per-iteration cost. They are extensively used in online learning (Beck & Teboulle, 2003), machine learning (Jaggi, 2013), and modern portfolio optimization ([2025 application](https://www.mdpi.com/2227-7390/13/18/3038)). **Not a single application to MCDA or simulation-based decision support appears in the literature surveyed.** This is the largest gap. → Novel algorithm A: Mirror-Descent + Basin-Hopping hybrid.

### Gap 2 — Multi-start strategies for nonconvex NLP are still using random restarts.

Multi-start SLSQP with Dirichlet sampling is the default; MLSL clustering improves slightly. **No published method uses bandit-style adaptive allocation of restart budget.** Yet bandit theory has matured substantially (Auer et al., 2002; Russo & Van Roy, 2014) and has been successfully applied to closely related domains: SAT solver restarts ([2024 RL-based reset](https://arxiv.org/pdf/2404.03753)), hyperparameter optimization (Hyperband, Li et al., 2018). The transfer to nonconvex NLP restart is mechanical but apparently unmade. → Novel algorithm C: Bandit-Based Adaptive Multi-Start.

### Gap 3 — Explicit Pareto-front + scenario-robust optimization for MCDA is rare.

NSGA-II for multi-objective is standard; robust NSGA-II under noise has been studied ([2023 arXiv](https://arxiv.org/pdf/2306.04525)); robust multi-objective with surrogates ([2022 arXiv](https://arxiv.org/pdf/2203.01996)). But the specific combination of (a) NSGA-II treating KPIs as objectives, (b) Bertsimas-Sim Γ-budget controlling scenario robustness, (c) applied to MCDA / decision-support, does not appear. → Novel algorithm B: Scenario-Robust NSGA-II with Bertsimas-Sim Budget.

### Gaps that exist but are *less promising* (and why)

- **Differentiable dependency-graph (JAX rewrite of `Evaluate`) → projected Newton.** Real opportunity, but requires major refactor of shared code (flagged for supervisor approval in [w1_codebase_audit.md §3.4](../w1_codebase_audit.md)). High reward, high risk. Held as a stretch goal.
- **Sinusoidal-basis trust-region.** Theoretically appealing because tRBS value functions are *known* to be sinusoidal; trust-region literature uses quadratic models. But proving convergence for non-quadratic trust-region models is hard (Davidon's conic models are the closest precedent, 1980), and the appeal evaporates when tRBS has *mixed* linear and sinusoidal value functions.
- **Graph-decomposition coordinate descent.** Exploits the DAG structure of `Evaluate` to decompose into independent subproblems on simplex faces. Beautiful idea, but tRBS dependency graphs in practice have multiplicative dependencies that prevent clean decomposition.

These three rejected ideas are revisited in the long-list section of [novel_algorithm_proposals.md](novel_algorithm_proposals.md).
