# W1 — Kick-Off Brief

**Author:** Joep Weterman
**Date:** 2026-05-21
**Status:** Draft for kick-off meeting with Louise
**Supporting docs:** [`w1_codebase_audit.md`](w1_codebase_audit.md) · [`w1_optimization_methods.md`](w1_optimization_methods.md)

---

## 1. Setup checklist

| Item | Status |
|---|---|
| VS Code 1.121 + Python/Pylance/Jupyter/GitLens/Ruff/Black extensions | ✅ |
| Repo forked → `github.com/joepweterman/trbs`; `origin` points to fork, `upstream` to `responsible-business-decision-making/trbs` | ✅ |
| Python 3.11.9 + pipenv venv at `~/.virtualenvs/tRBS-DclBJWVi-python.exe`; `pip install -e .` installs the local `vlinder` package | ✅ |
| Test suite: **171/172 pass**. One pre-existing Windows-only failure: `test_make_report.py::test_create_report[Optimistic]` (illegal `:` in PDF filename from `strftime("%H:%M:%S")`) | ⚠ noted |
| pre-commit hooks installed | ✅ |

## 2. Code placement decision (one-paragraph summary)

The new continuous optimization layer lives in a **new module `src/vlinder/optimize_continuous.py`** as a sibling to the existing `optimize.py`, not as a modification of it. This keeps the existing `case.optimize(scenario)` grid-search API intact (tests, demo notebook, downstream PwC use), gives a clean side-by-side benchmark structure (required by RQ1), and removes regression risk against the 171 existing tests. A new public method `case.optimize_continuous(scenario, method="slsqp"|"basin_hopping"|"genetic")` is added to `trbs.py`. Full audit in [`w1_codebase_audit.md`](w1_codebase_audit.md), including class skeleton and a flagged refactor (extracting a pure-function evaluation wrapper) that touches shared code and warrants supervisor sign-off before implementation.

## 3. Optimization methods — shortlist and academic justification (one-paragraph per method)

Full justification per method, with the references and how each maps to the three research questions, is in [`w1_optimization_methods.md`](w1_optimization_methods.md). Headline summary:

- **A — SLSQP (Sequential Least Squares Programming).** Local, gradient-based active-set SQP. Native simplex handling, locally quadratic convergence (Nocedal & Wright, 2006, Ch. 18), KKT multipliers as shadow prices on the budget — directly addresses RQ1's "decision-support value" angle. Risk: local-only; mitigated with N=100–1000 Dirichlet multi-start. Standard portfolio NLP method (Brenndoerfer 2024; Boyd & Vandenberghe 2004 Ch. 11).
- **B — Basin-hopping.** Wales & Doye (1997) hybrid: stochastic perturbation + SLSQP inner loop, Metropolis acceptance. **Designed for exactly the landscape type tRBS produces** ("smooth basins separated by saddle barriers" — sinusoidal value functions composed with multiplicative dependencies). Inherits SLSQP's constraint handling. Recent benchmark by [Sanvito & Cattaneo 2024](https://arxiv.org/html/2403.05877v1) shows competitiveness with state-of-the-art metaheuristics.
- **C — Genetic Algorithm.** Real-coded GA with weighted-sum scalarization (Deb 2001; Deb & Agrawal 1995 SBX). Derivative-free, robust to non-smoothness. Expected to be slowest on smooth low-dim problems per Rios & Sahinidis (2013), but provides the gradient-free comparison the proposal needs. **NSGA-II Pareto-front extension** is held as a stretch goal — would meaningfully strengthen the multi-objective angle (per Pedersen et al. 2021's ESG-Efficient Frontier framing) but expands scope.

These three span the methodological spectrum (local → hybrid → metaheuristic), which is what makes the comparison defensible rather than redundant.

## 4. Questions for the kick-off meeting

1. **Scope of decision variables.** Is the continuous optimization over a single DMO's `internal_variable_inputs` (matches my reading of the proposal), or should it also choose between DMOs (mixed combinatorial-continuous)? Affects problem dimensionality and method selection.
2. **Multi-scenario aggregation.** Optimize the scenario-weighted aggregate, run per-scenario optima and analyze divergence, or both? RQ3 reads as "both"; confirming the scope.
3. **NSGA-II Pareto front as a stretch contribution.** Worth the scope expansion, or stay scalar with weighted sums?
4. **PwC case.** Is there a concrete anonymized case in scope, or will the contribution rest on the five public demo cases?
5. **Repository placement.** Land `optimize_continuous.py` in the public `vlinder` package from the start, or stage it under `thesis/` until graded?
6. **Pre-existing Windows bug.** Worth a PR upstream now, or after the thesis ships?

## 5. Week 2 plan (subject to kick-off outcomes)

- Implement the pure-function evaluation wrapper (pending Q1 in §4).
- Stand up `ContinuousOptimize.optimize_slsqp` with multi-start + report on Beerwiser case.
- Begin formal mathematical specification of the optimization problem in LaTeX (methodology Ch. 3 starter).
- Begin annotated bibliography on portfolio optimization (planning W2–W3 lit review).
