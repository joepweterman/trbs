# W1 — Continuous Optimization Methods for tRBS: Academic Shortlist

**Author:** Joep Weterman
**Date:** 2026-05-21
**Status:** Draft for kick-off meeting

---

## 1. Problem characterization

Before defending the choice of methods, the structural properties of the tRBS optimization problem need to be made explicit, because each property maps to a specific algorithmic requirement:

| Property | Value in tRBS | Implication for method choice |
|---|---|---|
| Dimensionality (k) | 2–6 internal variables per DMO (up to ~10–15 in the PwC case) | **Low-dimensional.** Favors gradient-based methods; metaheuristics are typically over-powered for k < 10 (Rios & Sahinidis, 2013). |
| Feasible set | Simplex: Σx_i = B, x_i ≥ 0 | Linear equality + bound constraints. Native handling required. |
| Objective smoothness | Piecewise smooth: linear value functions are C^∞; sinusoidal value functions are C^∞ on (s, e) but capped at boundaries → kinks at value 0 and 100 | Methods must tolerate non-smoothness at clipping boundaries, or starting points must be feasible interior. |
| Convexity | **Non-convex in general.** Sinusoidal value functions are concave on [s, e] (sin is concave on [0, π/2]), so v(g(x)) is concave when g is affine in x. When g involves multiplicative dependencies (typical in tRBS graphs), v(g(x)) can become non-concave and multimodal. | Local methods alone insufficient — need either multi-start or globalization. |
| Gradient availability | None analytic. `Evaluate` traverses a dependency graph operator-by-operator; AD would require reimplementation. | Methods must work with finite differences (cost: k+1 evaluations per gradient) or be gradient-free. |
| Evaluation cost | Cheap: dependency-graph traversal in milliseconds for the demo cases. | Permits budgets of 10⁴–10⁵ function evaluations. Sample-efficient methods (Bayesian opt) not motivated. |
| Output | Scalar (weighted appreciation) under current scalarization, or vector of KPI appreciations if Pareto front desired | Method must support **either** scalar (weighted sum) **or** vector objectives. Choice connects to RQ3. |

Together these properties define a **low-dimensional, simplex-constrained, finite-budget, non-convex, derivative-free (or finite-difference) black-box optimization problem** — a class well-studied in operations research, with several canonical algorithm families.

## 2. The three shortlisted methods (per thesis proposal)

The thesis proposal commits to SLSQP, basin-hopping, and a genetic algorithm. The remainder of this document defends each choice against the problem characterization above and lays out the references that justify the comparison.

### 2.1 Method A — Sequential Least Squares Programming (SLSQP)

**Family:** local, gradient-based, active-set sequential quadratic programming.
**Implementation:** `scipy.optimize.minimize(method="SLSQP")`, originally Kraft (1988) Fortran code.

**Mathematical structure.** SLSQP solves the constrained NLP

$$ \min_x f(x) \;\; \text{s.t.} \;\; c_i(x) = 0, \; c_j(x) \geq 0 $$

by approximating the Lagrangian with a quadratic model and the constraints with linear models at each iterate, solving the resulting QP subproblem with a least-squares method, and updating via line search. Convergence is **locally quadratic** under standard regularity (LICQ, second-order sufficient conditions; Nocedal & Wright, 2006, Ch. 18).

**Why it fits tRBS:**

1. **Native handling of the simplex constraint.** SLSQP accepts equality (Σx_i = B) and inequality (x_i ≥ 0) constraints directly via the `constraints` argument and bounds, without reformulation. This contrasts with penalty/barrier methods that require tuning a penalty parameter.
2. **Low-dimensional advantage.** For k ≤ 15, SLSQP typically converges in 10–100 iterations and seconds of wall time on portfolio problems (Brenndoerfer, 2024). It is the default method in scipy-based portfolio allocation tutorials precisely because of this efficiency.
3. **KKT multipliers as shadow prices.** The dual variable on the budget equality is the marginal value of an additional unit of capital — a directly interpretable quantity for PwC stakeholders. This connects RQ1 directly to the practical decision-support value the proposal claims.
4. **Mature finite-difference support.** When analytical gradients are absent (the tRBS case), scipy computes forward differences with k+1 evaluations per gradient. For tRBS's millisecond evaluation cost, this is negligible.

**Risks for tRBS:**

- **Local convergence only.** Sinusoidal value functions combined with multiplicative dependencies in the graph produce multimodal landscapes (the proposal flags this; RQ2 is explicitly about characterizing it). SLSQP from a single starting point will return a local optimum.
- **Mitigation:** Multi-start with N = 100–1000 starts sampled uniformly from the simplex via the Dirichlet(1, 1, ..., 1) distribution. This is the standard technique in portfolio NLP (Boyd & Vandenberghe, 2004, Ch. 11). The cost is N × (per-start SLSQP cost), still tractable given tRBS evaluation speed.
- **Ill-conditioning at appreciation boundaries.** Sinusoidal `v(g(x))` has zero gradient at v = 0 and v = 100 (clipping plateaus). SLSQP can stall here. Detection: monitor gradient norm; restart from perturbed point if below tolerance.

**Key references for the thesis:**
- Kraft, D. (1988). *A software package for sequential quadratic programming*. DFVLR-FB 88-28. — Original SLSQP.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Springer, Ch. 18 (SQP) and Ch. 12 (KKT theory). — Convergence theory.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*, Cambridge University Press, Ch. 11. — Simplex-constrained NLP.
- Brenndoerfer, M. (2024). [*Quadratic Programming for Portfolio Optimization*](https://mbrenndoerfer.com/writing/quadratic-programming-portfolio-optimization). — Modern empirical justification for SLSQP as the default in portfolio NLP.
- Joshi, A. et al. (2024). [*PySLSQP: A transparent Python package for SLSQP*](https://arxiv.org/html/2408.13420v1). — Up-to-date implementation analysis; useful if convergence diagnostics needed.

---

### 2.2 Method B — Basin-Hopping

**Family:** hybrid global method combining local optimization with stochastic perturbation under Metropolis acceptance.
**Implementation:** `scipy.optimize.basinhopping`, originally Wales & Doye (1997).

**Mathematical structure.** At iterate x_t, basin-hopping:
1. Perturbs: x'_t = x_t + ε, ε drawn from a configurable proposal (default uniform).
2. Locally optimizes: x*'_t = LocalMin(x'_t) (default L-BFGS-B; configurable to SLSQP for tRBS's constraints).
3. Accepts/rejects: x_{t+1} = x*'_t with probability min(1, exp(−Δf / T)), else x_{t+1} = x_t.

This produces a Markov chain over local minima — exploring the **landscape of minima** rather than the original landscape. Convergence to the global minimum is guaranteed in the limit T → 0 with a logarithmic cooling schedule (Hajek, 1988 — classical simulated annealing convergence), though in practice fixed-T basin-hopping is the standard.

**Why it fits tRBS:**

1. **Designed for the exact landscape type tRBS produces.** Wales & Doye (1997) developed basin-hopping for energy surfaces with the structure "smooth basins separated by saddle barriers" — a description that fits sinusoidal value functions composed with multiplicative dependencies almost verbatim. The recent benchmark by [Sanvito & Cattaneo (2024)](https://arxiv.org/html/2403.05877v1) shows basin-hopping is competitive with state-of-the-art metaheuristics on standard multimodal benchmarks (Rastrigin, Schwefel, Ackley) while being algorithmically simpler.
2. **Inherits SLSQP's constraint handling.** By specifying `minimizer_kwargs={"method": "SLSQP", "constraints": ..., "bounds": ...}`, basin-hopping respects the simplex throughout. Each "hop" is a perturbation followed by SLSQP local optimization, so feasibility is restored at every accepted iterate.
3. **Bridges A and C.** Cleaner comparison story: SLSQP = pure local; basin-hopping = local + stochastic globalization; GA = pure population-based stochastic. This three-way comparison directly answers RQ1's "balance between solution quality and runtime" question by spanning the methodological spectrum.

**Risks for tRBS:**

- **Computational cost.** Each hop is a full SLSQP run. For N_hops = 100 and ~50 SLSQP iterations each, expect 5,000 + iterations vs. ~100 for a single SLSQP run. Tractable but materially slower.
- **Hyperparameter sensitivity.** Step size of the perturbation and temperature T require tuning. Default scipy values are reasonable but worth a brief sensitivity analysis (mention in methodology section).
- **Stochastic** — same run gives different results. Need to report mean ± std over multiple runs.

**Key references for the thesis:**
- Wales, D. J. & Doye, J. P. K. (1997). *Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms*. *Journal of Physical Chemistry A*, 101(28), 5111–5116. — Original.
- Olson, B., Hashmi, I., Molloy, K. & Shehu, A. (2012). *Basin Hopping as a General and Versatile Optimization Framework*. *Advances in Artificial Intelligence*. — Demonstrates applicability beyond physics; useful for citing in interdisciplinary positioning.
- Sanvito, F. & Cattaneo, M. (2024). [*A Performance Analysis of Basin Hopping Compared to Established Metaheuristics for Global Optimization*](https://arxiv.org/html/2403.05877v1). — Recent benchmark, directly relevant for the comparison.
- Locatelli, M. (2022). [*Hopping between distant basins*](https://link.springer.com/article/10.1007/s10898-022-01153-z). *Journal of Global Optimization*. — Theory of long-range hops; relevant if tRBS landscape proves to have widely-separated basins.

---

### 2.3 Method C — Genetic Algorithm (NSGA-II or real-coded GA)

**Family:** population-based metaheuristic; derivative-free, stochastic.
**Implementation choice:** two variants to discuss with supervisor:
- **C1 — Real-coded GA with weighted-sum scalarization.** Treats the same scalar appreciation function as A and B; population-based search. Implementation: `pymoo.algorithms.soo.nonconvex.GA`.
- **C2 — NSGA-II for explicit multi-objective.** Treats the per-KPI appreciation vector directly, returning the Pareto front rather than a single weighted-sum optimum. Implementation: `pymoo.algorithms.moo.nsga2`.

**Mathematical structure.** A real-coded GA evolves a population of candidate solutions through (i) tournament selection, (ii) simulated binary crossover (SBX; Deb & Agrawal, 1995), and (iii) polynomial mutation. NSGA-II adds fast non-dominated sorting and crowding distance to maintain Pareto front diversity (Deb et al., 2002 — IEEE TEC; cited ~50,000 times).

**Why it fits tRBS:**

1. **No gradient or smoothness assumption.** Robust to the appreciation boundary kinks and non-smoothness that can trouble SLSQP. Good "always works" baseline.
2. **NSGA-II offers a clean methodological extension.** The current tRBS scalarizes via weighted sum of KPI appreciations — an old MCDA assumption that requires committing to weights ex ante. NSGA-II would let decision-makers see the **entire Pareto frontier** of financial / social / environmental tradeoffs, then choose post hoc. This is the central insight of Pedersen, Fitzgibbons & Pomorski (2021)'s ESG-Efficient Frontier — exactly the literature the thesis proposal positions itself against.
3. **Active research area.** Larni-Fooeik et al. (2025) [show NSGA-III improvements over NSGA-II for ESG portfolio selection](https://papers.ssrn.com/sol3/Delivery.cfm/3c1a2997-b403-44c1-8c91-5bd66f348c50-MECA.pdf?abstractid=5210265&mirid=1&type=2) — useful for motivating the choice and for the "future work" section of the thesis.

**Risks for tRBS:**

- **Slower convergence on smooth low-dim problems.** Rios & Sahinidis (2013) — the canonical benchmark of derivative-free optimization — finds GAs typically dominated by gradient-based methods when smoothness is present and k < 20. This is a *feature, not a bug* for the thesis: SLSQP is expected to win on speed and likely on quality for unimodal cases; the interesting question is how the gap evolves as multimodality increases.
- **Hyperparameter burden.** Population size, generations, crossover probability, mutation probability, distribution indices (η_c, η_m). Standard NSGA-II defaults (pop=100, η_c=15, η_m=20) are well-established but worth one paragraph of sensitivity analysis.
- **No KKT multipliers / no natural sensitivity output.** Loses one of SLSQP's advantages. Mitigation: report Pareto front itself as the "sensitivity output" — arguably more informative than dual variables for decision-makers.

**Recommendation:** start with **C1 (real-coded GA + weighted-sum)** as the proposal-canonical comparison. Add **C2 (NSGA-II + Pareto front)** as an extension if time allows — it would substantially strengthen the contribution but expands scope and is best discussed with the supervisor before committing.

**Key references for the thesis:**
- Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197. — Canonical NSGA-II reference.
- Deb, K. & Agrawal, R. B. (1995). *Simulated Binary Crossover for Continuous Search Space*. *Complex Systems*, 9, 115–148. — SBX crossover operator used by pymoo.
- Rios, L. M. & Sahinidis, N. V. (2013). *Derivative-Free Optimization: A Review of Algorithms and Comparison of Software Implementations*. *Journal of Global Optimization*, 56(3), 1247–1293. — Sets the empirical expectations for the comparison; essential citation.
- Larni-Fooeik, A. et al. (2025). *Applying NSGA-III to Multi-Objective Portfolio Optimization*. *SSRN*. — Recent, ESG-specific.
- Pedersen, L. H., Fitzgibbons, S. & Pomorski, L. (2021). *Responsible Investing: The ESG-Efficient Frontier*. *Journal of Financial Economics*, 142(2), 572–597. — Theoretical motivation for the Pareto-frontier framing.

---

## 3. Honourable mentions (not in primary comparison)

Two methods worth referencing in the methodology section to demonstrate methodological awareness, but not included in the primary three-way comparison unless results suggest otherwise:

- **Trust-region constrained** (`scipy.optimize.minimize(method="trust-constr")`). Better theoretical guarantees than SLSQP on non-convex problems (Conn, Gould & Toint, 2000), but more conservative step-taking and similar local-only limitation. Use as a robustness check on SLSQP results.
- **Differential Evolution** (Storn & Price, 1997; `scipy.optimize.differential_evolution`). Often outperforms standard GA on continuous problems with simpler hyperparameter tuning. Could substitute for the GA if early experiments show GA struggling.

## 4. Mapping methods to research questions

| RQ | SLSQP | Basin-hopping | GA |
|---|---|---|---|
| RQ1: best balance quality/runtime | **Expected fastest, possibly highest quality on unimodal cases** | Mid-cost, highest robustness on multimodal | Slowest, most robust to non-smoothness |
| RQ2: characterize landscape (convexity, multimodality) | Multi-start SLSQP run count of distinct converged points = empirical multimodality measure | Acceptance rate and basin-distance metrics characterize landscape ruggedness | Population diversity over generations reveals landscape topology |
| RQ3: scenario-robust vs. scenario-specific allocation | Re-run per scenario, compare Lagrange multipliers | Re-run per scenario, compare basin structures | NSGA-II variant can treat scenarios as additional objectives → Pareto frontier across scenarios |

Each method contributes complementary evidence to each RQ, which makes the three-way comparison defensible and not redundant.

## 5. Recommendation

Proceed with the three methods as proposed: **SLSQP + multi-start (A)**, **Basin-hopping with SLSQP inner loop (B)**, **Real-coded GA with weighted-sum scalarization (C1)**. Hold **NSGA-II (C2)** as a stretch goal contingent on supervisor agreement and time budget.

This shortlist:
- Spans the methodological spectrum (local → hybrid → global) — defensible methodologically.
- Has clear academic provenance (each method has 25+ years of literature and 1,000+ citations).
- Maps cleanly to all three research questions.
- Is implementable in scipy + pymoo without exotic dependencies.
- Allows direct benchmark against the existing grid search on the 5 tRBS demo cases plus the PwC case.
