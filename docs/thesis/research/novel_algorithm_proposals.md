# Three Novel Algorithm Proposals for tRBS Capital Allocation

**Author:** Joep Weterman
**Date:** 2026-05-21
**Status:** Draft for kick-off meeting; selection of the 3 to develop further pending Louise's input
**Companion document:** [`existing_methods_survey.md`](existing_methods_survey.md)

---

## 1. Selection criteria

Six candidate ideas were generated from the gap analysis in [§9 of the existing-methods survey](existing_methods_survey.md#9-gap-analysis--where-novel-algorithms-can-live). Three were selected for detailed development against the following criteria, each on a 1–5 scale:

| Criterion | What it measures | Why it matters for an 8.0+ thesis |
|---|---|---|
| Defensible novelty | Likelihood that the contribution is not pre-empted by existing literature | A thesis stakes a claim; novelty must survive a defensive literature scan |
| tRBS fit | How directly the algorithm exploits tRBS-specific structure (simplex, sinusoidal value functions, dependency graph, multi-scenario) | Generic improvements are weaker contributions than structure-exploiting ones |
| Theoretical depth | Available rigorous analysis (convergence rates, complexity bounds, regret) | An 8.0+ thesis is expected to do more than empirical benchmarking |
| Implementation feasibility | Effort to implement and integrate with the existing `vlinder` codebase within the W4–W9 window | Scope discipline; an unimplemented algorithm cannot be benchmarked |
| Publishability | Could it stand alone as a workshop or journal paper? | A genuine differentiator for an 8.5+ grade |

### Scoring

| Idea | Novelty | tRBS fit | Theory | Impl. | Publishability | **Total** |
|---|---|---|---|---|---|---|
| A. Mirror-Descent + Basin-Hopping (MDBH) | 5 | 5 | 5 | 4 | 4 | **23** ★ |
| B. Scenario-Robust NSGA-II + Bertsimas Budget (SR-NSGA) | 4 | 5 | 4 | 3 | 4 | **20** ★ |
| C. Bandit-Based Adaptive Multi-Start (BAMS) | 4 | 4 | 5 | 5 | 4 | **22** ★ |
| D. Sinusoid-Aware Trust-Region (STR) | 4 | 4 | 2 | 3 | 3 | 16 |
| E. Differentiable Dependency-Graph + Projected Newton (DDG-PN) | 3 | 5 | 4 | 1 | 3 | 16 |
| F. Graph-Decomposition Coordinate Descent (GDCD) | 4 | 3 | 3 | 2 | 3 | 15 |

The three selected (A, B, C) span the proposal's existing three-method comparison cleanly:

- A. MDBH **improves on the proposal's basin-hopping** by replacing its SLSQP inner loop with mirror descent.
- B. SR-NSGA **extends the proposal's GA** with explicit Pareto-front + scenario robustness.
- C. BAMS **improves on the proposal's SLSQP multi-start** with bandit-based budget allocation.

This is methodologically clean — each novel algorithm has a clear *baseline* to beat, and the three together test the proposal's three claims under genuinely improved conditions.

§§ 2–4 develop each. § 5 briefly explains why D, E, F were not selected. § 6 lays out a unified implementation roadmap.

---

## 2. Algorithm A — Mirror-Descent + Basin-Hopping Hybrid (MDBH)

### 2.1 Core insight

The proposal's basin-hopping uses SLSQP as the local engine. SLSQP treats the simplex Σx_i = B as a generic linear-equality constraint and pays O(k^3) per QP subproblem to handle it. **Mirror descent with the negative-entropy mirror map handles the simplex natively in O(k) per iteration via a multiplicative-weights update**, and its iterates remain strictly in the interior (no boundary degeneracy at vertices). Wrapping mirror descent inside basin-hopping yields a method that combines (a) simplex-native local search with (b) global stochastic escape from local optima — the two properties the tRBS landscape simultaneously demands.

The combination has been used in **online learning and game theory** (e.g., [Bauschke et al. 2017 on Bregman proximal methods](https://arxiv.org/pdf/1610.00076), [Krichene et al. 2015 on accelerated mirror descent](https://bayen.berkeley.edu/sites/default/files/accelerated_mirror_descent.pdf)) but **not, to my surveyed knowledge, applied to MCDA or simulation-based decision support**. The defensive literature scan ([WebSearch on "mirror descent basin hopping hybrid global optimization simplex constraint"](https://danmackinlay.name/notebook/gd_mirror.html)) returned the expected components in isolation but no direct combination paper for this application.

### 2.2 Algorithm

```
Algorithm MDBH(f, B, k, T_local, T_hop, η, σ):
  Input:
    f      — black-box objective (tRBS appreciation, to minimize as −F)
    B      — total budget (simplex sum)
    k      — number of decision variables
    T_local — mirror-descent steps per local optimization
    T_hop  — number of basin hops
    η      — mirror-descent step size
    σ      — perturbation scale on the simplex tangent
  Initialize:
    x_0 ← (B/k, B/k, ..., B/k)        # simplex centroid
    x*, f* ← x_0, f(x_0)
  for t = 1, ..., T_hop:
    # Local optimization via entropic mirror descent
    x ← MirrorDescent(f, x_{t-1}, T_local, η)
    if f(x) < f*: x*, f* ← x, f(x)

    # Stochastic perturbation on the simplex
    ξ ← sample a tangent direction (project Gaussian onto {Σ = 0})
    x_perturb ← Π_Δ(x + σ · ξ)        # project back if needed
    Δ ← f(x_perturb) − f(x)
    if Δ < 0 or random() < exp(−Δ / T_metropolis):
      x_t ← x_perturb
    else:
      x_t ← x
  return x*, f*

Subroutine MirrorDescent(f, x_0, T, η):
  x ← x_0
  for s = 1, ..., T:
    g ← ∇f(x)                    # finite differences if no AD
    x_i ← x_i · exp(−η · g_i / B) for all i
    x ← x · B / Σx_i             # renormalize to simplex
  return x
```

Two design choices worth defending:

1. **Tangent-space perturbation.** Naive Gaussian perturbation of x leaves the simplex; we project back. A cleaner alternative is to sample directly in the tangent space TΔ = {ξ : Σξ_i = 0} and use the exponential map x ↦ x · exp(σξ) (geodesic on the simplex in the Fisher-Rao metric). This is the form used in information geometry (Amari, 2016, *Information Geometry and Its Applications*) and aligns with the entropic mirror map. The thesis methodology should document the choice.
2. **Step size η.** Adaptive step size via line search inside the mirror-descent loop is standard. For the thesis, decreasing schedule η_t = η_0 / √t is the textbook default (Beck & Teboulle, 2003).

### 2.3 Theoretical properties

**Local convergence (within a single basin).** Entropic mirror descent on the simplex achieves O(1/√t) convergence for non-smooth convex objectives and O(1/t) for smooth convex objectives (Beck & Teboulle, 2003, Theorems 4.1 and 4.2), with convergence constants depending on dimension only as O(√log k). For tRBS this gives a per-basin convergence guarantee that *is* dimension-mild — relevant if PwC's case scales to k ≈ 20.

**Global convergence (basin-hopping wrapper).** Inherits the standard basin-hopping convergence story (Wales & Doye 1997, Leary 2000): probabilistic convergence to the global optimum as T_hop → ∞ under the assumption that every basin of attraction has nonzero probability of being reached by the perturbation kernel. For the simplex with a Gaussian-in-tangent kernel, this assumption is straightforward to verify (the kernel has full support on the tangent space).

**Per-iteration complexity.** O(k) for the mirror-descent step, dominated by the gradient evaluation (k+1 function calls via finite differences). For T_local = 50 and T_hop = 100, total cost ≈ T_hop × T_local × (k+1) = 100 × 50 × 7 = 35,000 function evaluations — comparable to the proposal's multi-start SLSQP (N=100 starts × ~50 iters × ~7 gradient calls).

### 2.4 Expected empirical advantage over the proposal's basin-hopping

- **Comparable solution quality** on smooth unimodal problems (Frank-Wolfe and SLSQP both find the unique optimum).
- **Better solution quality on multimodal problems with vertex-near optima**, because mirror descent avoids the boundary stalling that SLSQP can exhibit at active-bound vertices.
- **2-5× faster per-iteration** because the QP subproblem is replaced by an exponential-weights update.
- **Negligible memory overhead** vs. SLSQP's stored Hessian approximation.

These predictions are testable in the W8–W11 experiment phase of the thesis. The benchmark structure is symmetric to the proposal's basin-hopping-with-SLSQP-inner-loop, so the comparison is methodologically clean.

### 2.5 Failure modes

1. **Multiplicative-weights iterates can collapse to zero** for variables with persistently positive gradient. Mitigation: lower-bound iterates at ε · B/k for small ε (standard regularization).
2. **Convergence rate is slower than SLSQP on smooth unimodal problems.** This is structural; the right framing is "MDBH wins on multimodal landscapes, loses on smooth ones." Empirical landscape characterization (RQ2) is the methodological bridge.
3. **Tangent-space perturbation has a free hyperparameter σ.** Default σ = 0.1 · B works in preliminary mental simulation; the thesis methodology should report sensitivity.

### 2.6 Publishability

**Yes** — at the workshop or short-paper level. The combination is novel, the theoretical analysis combines two well-established results, and the empirical advantage on simplex-constrained nonconvex problems would be of interest to the constrained-optimization community independent of the MCDA application. Suitable target venues: EvoCOP workshop, EURO conference, *Operations Research Letters*.

### 2.7 Implementation roadmap

| Phase | Work | Effort |
|---|---|---|
| W4 | Implement `MirrorDescent` subroutine + unit tests on convex test problems | 8 h |
| W5 | Wrap in basin-hopping outer loop + simplex projection | 6 h |
| W6 | Integrate as `ContinuousOptimize.optimize_mdbh(scenario, ...)` | 4 h |
| W8–W9 | Benchmark on Beerwiser, DSM, IZZ, Refugee, NEMO + PwC case | 12 h |
| W10–W11 | Sensitivity analysis (η, σ, T_local, T_hop) | 8 h |
| **Total** | | **38 h** |

External dependencies: none beyond scipy + numpy. No JAX rewrite needed (finite-difference gradients suffice for k ≤ 15).

---

## 3. Algorithm B — Scenario-Robust NSGA-II with Bertsimas-Sim Budget (SR-NSGA)

### 3.1 Core insight

The proposal's GA optimizes a scalar weighted-sum aggregation of (a) KPIs within scenarios and (b) scenarios across the problem. **The weighted sum hides two distinct decisions**: how to trade off KPIs against each other (a values question) and how to trade off optimality against scenario robustness (a risk question). SR-NSGA exposes both by:

1. Treating KPI appreciations as **explicit objectives** (NSGA-II machinery for Pareto front).
2. Replacing scenario weights with a **Bertsimas-Sim Γ-budget** on scenario weight perturbation. The decision-maker chooses Γ ∈ {0, 1, ..., S} — the maximum number of scenarios allowed to deviate from their nominal weight — and the optimizer returns the Pareto front of "(expected appreciation, worst-case appreciation under Γ-budget)" tradeoffs.

Output: a 2D Pareto front parametrized by Γ that gives the decision-maker an immediate visual of *"how much expected performance do I give up for one additional unit of robustness?"* — directly answering RQ3.

### 3.2 Algorithm

```
Algorithm SR-NSGA(f, B, k, scenarios, Γ_max, pop_size, n_gen):
  # f is now a vector-valued function: f(x, s) = (v_1, ..., v_K) — K KPI appreciations
  # for scenario s, NOT scalarized
  Initialize population P_0 of pop_size individuals sampled uniformly on Δ^k
  for g = 1, ..., n_gen:
    # Per-individual evaluation
    for x in P_g:
      kpi_vec[x] = Σ_s w_s · f(x, s)               # expected KPI vector
      worst_vec[x] = min over scenario sets S' with |S'|=Γ_max of
                      Σ_{s ∈ S'} (1/|S'|) · f(x, s)  # worst Γ-budget aggregation
      obj[x] = concat(kpi_vec[x], worst_vec[x])    # 2K-dim objective vector

    # Standard NSGA-II non-dominated sorting + crowding distance
    fronts = FastNonDominatedSort(P_g ∪ Offspring(P_g), obj)
    P_{g+1} = SelectByCrowdingDistance(fronts, pop_size)

    # Generate offspring via SBX crossover + polynomial mutation, with simplex
    # projection to maintain feasibility
    Offspring = []
    for j = 1, ..., pop_size / 2:
      p_a, p_b = TournamentSelect(P_{g+1}, k=2)
      c_a, c_b = SBX(p_a, p_b, η_c=15)
      c_a, c_b = PolynomialMutation(c_a, η_m=20), PolynomialMutation(c_b, ...)
      c_a, c_b = SimplexProject(c_a, B), SimplexProject(c_b, B)
      Offspring.append(c_a); Offspring.append(c_b)
  return Pareto-front(P_{n_gen})
```

### 3.3 Theoretical properties

**Convergence of NSGA-II.** Recent runtime analyses ([Wietheger & Doerr 2024, IEEE TEC](https://arxiv.org/pdf/2407.17687)) prove that NSGA-II finds the Pareto front of standard benchmark functions in expected time polynomial in the population size and the front size. The general convergence-to-Pareto-front result holds; only the rate is problem-dependent.

**Robust-objective interpretation.** The worst_vec computation is the **adversarial recourse** in the Bertsimas-Sim formulation: a Γ-budget uncertainty set forces the adversary to pick the worst Γ scenarios. Bertsimas & Sim (2004) prove that the robust counterpart of a linear program with budgeted uncertainty has the **same complexity** as the nominal LP — i.e., adding robustness is "free" computationally for LPs. For our nonlinear case the result does not transfer literally, but the worst_vec computation is still tractable: it requires evaluating f at each scenario (already done for kpi_vec) and a simple sort.

**Per-evaluation complexity.** Each individual requires |S| function evaluations (one per scenario). With pop_size = 100 and n_gen = 200 and |S| = 3, total = 100 × 200 × 3 = 60,000 evaluations. Tractable at ms scale.

### 3.4 Expected empirical advantage over the proposal's GA

- **Replaces a single number (scalarized appreciation) with a Pareto front** — decision-makers see the actual tradeoffs rather than implicitly making them via opaque weights.
- **Explicit robustness parameter Γ** — actionable for PwC stakeholders who can articulate "I want a recommendation that's good under any 1 of my 3 scenarios going badly."
- **Same hyperparameter cost as standard NSGA-II.**

### 3.5 Failure modes

1. **The 2K-dimensional objective vector** (K KPI appreciations × 2 for expected + worst-case) becomes hard to visualize for K > 3. Mitigation: dimensionality reduction in the visualization layer (e.g., parallel coordinates plot or principal Pareto front).
2. **Bertsimas-Sim budget Γ is a discrete hyperparameter.** Running the optimizer for each Γ in {0, ..., S} gives the full robustness frontier; for S = 3 this is 4 runs, manageable.
3. **Pareto front of size > 100** can be unwieldy. Standard NSGA-II crowding distance handles this; alternatively MOEA/D's decomposition gives a more controllable Pareto-front size.

### 3.6 Publishability

**Yes** — this is the most directly publishable of the three, because it answers a real applied question (how to do MCDA with explicit robustness) using established components in a clean combination. Target venues: *EJOR* (European Journal of Operational Research), *Decision Sciences*, *International Journal of Robust and Nonlinear Control*. Possibly a tier above MDBH because the applied motivation is sharper.

### 3.7 Implementation roadmap

| Phase | Work | Effort |
|---|---|---|
| W6 | Adapt pymoo NSGA-II to use vector objectives = KPI appreciations | 8 h |
| W7 | Implement Bertsimas-Sim worst-case scenario aggregation | 6 h |
| W8 | Wire into `ContinuousOptimize.optimize_sr_nsga(scenarios, Γ, ...)` | 4 h |
| W9 | Visualization: 2D and 3D Pareto front plots; Γ-frontier curves | 8 h |
| W10–W11 | Benchmark on cases with K ≥ 3 KPIs; sensitivity on Γ | 14 h |
| **Total** | | **40 h** |

External dependencies: pymoo (already widely used in optimization research). One Pipfile addition.

---

## 4. Algorithm C — Bandit-Based Adaptive Multi-Start (BAMS)

### 4.1 Core insight

Multi-start SLSQP is the proposal's canonical handling of multimodality: sample N starting points uniformly from the simplex (via Dirichlet(1, ..., 1)), run SLSQP from each, keep the best optimum. **The "uniform" allocation is wasteful.** If after 20 starts you have discovered three basins with appreciation values 92, 85, and 60, continuing to spend equal effort on the 60-basin is bad allocation. Multi-armed bandit theory gives the optimal way to trade off exploration (discovering new basins) and exploitation (refining good basins): allocate the next restart to the basin maximizing an Upper Confidence Bound (UCB) over its true global appreciation, or sample from a posterior over true appreciation (Thompson sampling). UCB regret bounds (Auer, Cesa-Bianchi & Fischer, 2002, *Machine Learning*) and Thompson sampling regret bounds ([Russo & Van Roy 2014, *Mathematics of Operations Research*](https://arxiv.org/pdf/1209.3353)) give principled budget allocation.

The transfer of bandit theory to nonconvex NLP restarts is mechanical but apparently unmade. Closest published work: bandit-based restart in SAT solvers ([2024 arXiv RL-based reset policy](https://arxiv.org/pdf/2404.03753)), Hyperband for hyperparameter optimization (Li et al., 2018, *JMLR*), but those address different reset semantics.

### 4.2 Algorithm

```
Algorithm BAMS(f, B, k, N_total, exploration_phase=N_init):
  # Phase 1: Exploration — N_init uniform Dirichlet starts to discover basins
  basins = []  # each basin: (centroid, best_f, count, sample_history)
  for i = 1, ..., N_init:
    x_0 = Dirichlet(1, ..., 1) · B
    x*, f* = SLSQP(f, x_0)
    # Cluster x* into an existing basin (Euclidean distance < ε) or create new
    b = NearestBasin(x*, basins, threshold=ε)
    if b: b.update(x*, f*)
    else: basins.append(NewBasin(x*, f*))

  # Phase 2: Exploitation — bandit-allocated remaining starts
  for j = N_init+1, ..., N_total:
    # UCB acquisition: select basin with highest f* + c · √(log j / count_b)
    b_select = argmax_b (b.best_f + c · sqrt(log(j) / b.count))
    # Sample a new starting point near b_select's centroid but with perturbation
    x_0 = SamplePerturbationNearCentroid(b_select.centroid, scale=σ_explore)
    x*, f* = SLSQP(f, x_0)
    b = NearestBasin(x*, basins, threshold=ε)
    if b: b.update(x*, f*)
    else: basins.append(NewBasin(x*, f*))

  return argmax basin's best_f, with full Pareto curve of (budget, best-so-far)
```

Hyperparameter c trades off exploration vs. exploitation; standard UCB1 sets c = √2, but the c that's optimal for a given problem is itself a learnable quantity (BayesUCB, Kaufmann et al., 2012).

### 4.3 Theoretical properties

**Regret bound.** For the K-armed bandit with sub-Gaussian rewards, UCB1 achieves regret O(√(K T log T)) over T pulls (Auer et al., 2002), matching the information-theoretic lower bound up to log factors (Lai & Robbins, 1985). In the BAMS setting, "regret" is the cumulative excess of best-found-appreciation over the global optimum. The bound suggests BAMS will converge to the global optimum at a rate that improves on naive multi-start by a factor of √(K log T) — concretely, for T = 100 starts and K = 5 discovered basins, this is ~5× improvement in cumulative regret.

**Caveat on assumptions.** The standard bandit setting assumes i.i.d. rewards from each arm. The "reward" here — best-found f* after a fresh SLSQP run from a perturbed centroid — is **not i.i.d.** within a basin (later runs explore the basin more thoroughly, so reward is monotone-increasing-in-count). This violates the standard analysis. Two principled fixes: (a) use a non-stationary bandit (discounted UCB, Garivier & Moulines 2008) where recent rewards weigh more, or (b) reframe as a best-arm-identification problem (Audibert et al., 2010). The thesis would defend the choice in the methodology.

**Per-iteration complexity.** O(K) for the UCB scan + one full SLSQP run. Same per-iteration cost as naive multi-start; the gain is in *which* starting points are chosen, not in the per-run cost.

### 4.4 Expected empirical advantage over the proposal's naive multi-start

- **2-5× compute-efficiency** to reach a given best-found appreciation, based on bandit-regret-bound extrapolation.
- **Adaptive to landscape difficulty.** Easy unimodal problems converge fast (few basins discovered, exploitation phase short). Hard multimodal problems get more thorough exploration.
- **Anytime property** — the "best so far" curve is monotone-improving and informative throughout the run, vs. naive multi-start which is best-after-all-runs.

### 4.5 Failure modes

1. **Basin clustering threshold ε is a free hyperparameter.** Too large → distinct basins merged (under-exploration); too small → spurious basins (over-exploration). The thesis methodology should report sensitivity to ε.
2. **The non-i.i.d. reward issue (see §4.3).** Defensive approach: report results both with standard UCB and with a non-stationary variant; show robustness.
3. **Performance gain depends on number of basins K.** For truly unimodal problems (K=1) BAMS reduces to naive multi-start. The gain is zero on the easy problems. This is the right behavior, but the thesis benchmark must include both unimodal and multimodal cases.

### 4.6 Publishability

**Yes, with the strongest standalone story of the three.** The technique is application-agnostic — it improves multi-start NLP for any nonconvex problem, not just MCDA — and the theoretical backing from bandit regret bounds is rigorous and well-established. Suitable venues: *Mathematical Programming Computation*, *JOTA*, *INFORMS Journal on Computing*. Could even be a conference paper at ICML/NeurIPS if framed under the "Bayesian optimization meets nonconvex NLP" angle.

### 4.7 Implementation roadmap

| Phase | Work | Effort |
|---|---|---|
| W4 | Basin clustering + UCB acquisition machinery | 6 h |
| W5 | Wrap around existing SLSQP runs | 4 h |
| W6 | Integrate as `ContinuousOptimize.optimize_slsqp_bandit(...)` | 4 h |
| W8–W9 | Benchmark on all 5 demo cases + PwC; vs naive multi-start of equal compute | 10 h |
| W10–W11 | Sensitivity: c (UCB constant), ε (basin threshold), N_init / N_total split | 8 h |
| **Total** | | **32 h** |

External dependencies: none (pure scipy + numpy).

---

## 5. Rejected ideas — long-list with rationale

These three were generated but did not make the top 3:

### D. Sinusoid-Aware Trust-Region (STR)

**Idea.** tRBS uses *known* sinusoidal value functions on some KPIs. Standard trust-region methods build a *quadratic* model of the objective; STR would build a model from a **sinusoidal basis** when the active KPI is sinusoidal, providing a better local fit and faster convergence.

**Why not selected.** Theoretical convergence analysis for non-quadratic trust-region models is hard (Davidon's conic models, 1980, are the closest precedent; they have only been partially analyzed in 45 years). The risk of investing W3–W4 in theory that doesn't close is high. Empirical gain is also uncertain because tRBS mixes linear and sinusoidal value functions — STR would only win on cases where sinusoidal dominates.

### E. Differentiable Dependency-Graph + Projected Newton (DDG-PN)

**Idea.** Rewrite the `Evaluate` dependency-graph traversal in JAX → automatic reverse-mode AD for analytical gradients → exact Hessian via forward-over-reverse (cheap for k ≤ 15) → projected Newton on the simplex. Bridges differentiable programming and constrained NLP.

**Why not selected.** Major refactor of shared code (flagged for supervisor approval in [w1_codebase_audit.md §3.4](../w1_codebase_audit.md)). If the supervisor or PwC team rejects the JAX rewrite, the entire algorithm is unimplementable. High reward, but the risk to the thesis timeline is unacceptable for the primary three. **Holding as a stretch contribution** for after the main three are benchmarked.

### F. Graph-Decomposition Coordinate Descent (GDCD)

**Idea.** Use the DAG structure of the dependency graph to identify independent subproblems and decompose the optimization into coordinate descent on simplex faces.

**Why not selected.** tRBS dependency graphs in practice contain multiplicative dependencies (operator `*` in the operator set, see `evaluate.py::operators_dict`) that couple subproblems and prevent clean decomposition. The idea is elegant but the empirical decomposability is unclear — would require an investigation of all 5 demo cases just to know if it's applicable.

---

## 6. Unified implementation roadmap

Total estimated effort: 38 + 40 + 32 = **110 h** over W4–W11, against a planning budget of approximately 130 h for those weeks (per [Thesis_Planning_tRBS.xlsx](file:///C:/Users/joepw/Desktop/Thesis_Planning_tRBS.xlsx)). Within budget with ~15% slack.

**Dependencies:**

1. **Pure-function evaluation wrapper** (flagged in [w1_codebase_audit.md §3.4](../w1_codebase_audit.md)) — needed by all three. Implement in W4 contingent on supervisor approval.
2. **pymoo** library — needed by SR-NSGA. Add to Pipfile.
3. No JAX rewrite needed for any of A, B, C. The stretch-goal DDG-PN remains the only candidate that requires it.

**Benchmark design.** All three novel algorithms benchmark against (a) the existing tRBS grid search, (b) the proposal's three methods (SLSQP + multi-start, basin-hopping with SLSQP, real-coded GA). Metrics:

- Best-found appreciation (RQ1: solution quality)
- Wall-clock time and function-evaluation count (RQ1: runtime)
- Multi-start basin-discovery curves (RQ2: landscape characterization)
- Per-scenario optima divergence + Pareto-front for SR-NSGA (RQ3: scenario robustness)

Each of the 5 demo cases (Beerwiser, DSM, IZZ, Refugee, NEMO) plus the PwC case is run 10 times with different seeds; results reported as mean ± std.

---

## 7. Open questions for the supervisor

1. **Of the three (MDBH, SR-NSGA, BAMS), which two are most defensible if I have to drop one?** My current recommendation: keep MDBH (strongest novelty) and BAMS (strongest theory + lowest implementation risk); drop SR-NSGA if time pressure (it's the largest implementation effort and depends on agreeing to extend to multi-objective).
2. **Are the publishability targets (workshop / EJOR / Math Prog Computation) realistic from the thesis viewpoint, or is "spinning out a paper" out of scope and I should stay in thesis-only deliverables?**
3. **Stretch goal DDG-PN — is a JAX rewrite of `Evaluate` something the tRBS / PwC team would entertain, or is that a non-starter that I should drop from the long-list permanently?**
4. **Bertsimas-Sim Γ-budget in SR-NSGA — is the "Γ scenarios deviate" semantics the right one for tRBS, or should the robustness be parametrized differently (e.g., uncertainty in KPI weights rather than scenario weights)?**
