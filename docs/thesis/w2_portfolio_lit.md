# W2 — Portfolio optimization literature review (v1)

> **Canonical location:** `C:\Users\joepw\thesis-knowledge\02-literature\portfolio_optimization_v1.md`.
> This file is a pointer; update the canonical version, not this one.

## 30-second summary

First-pass scaffold for §2.1 of the thesis Lit Review chapter. **11 verified papers** organized along de Bliek's *Writing a Thesis* framework (seminal → first studies → broad topic → specialized topic → relationship/crossover → synthesis):

- **Seminal:** Markowitz 1952 — origin of "optimal capital allocation on the simplex"
- **First studies:** Tobin 1958 (separation), Sharpe 1964 (CAPM)
- **Broad — parameter uncertainty handling:** Jorion 1986 (Bayes-Stein mean), Black & Litterman 1992 (equilibrium priors + views), Ledoit & Wolf 2004 (covariance shrinkage)
- **Specialized — explicit robust optimization:** Goldfarb & Iyengar 2003 (SOCP), Bertsimas & Sim 2004 (Γ-budget)
- **Crossover to multi-criteria:** Steuer & Na 2003 (MCDM-in-finance survey), Hirschberger, Steuer, Utz, Wimmer & Qi 2013 (tri-criterion exact algorithm)
- **Crossover to ESG:** Friede, Busch & Bassen 2015 (2,200-study meta-analysis), Pedersen, Fitzgibbons & Pomorski 2021 (ESG-efficient frontier)

**Verification:** every citation checked end-to-end (author list, year, journal, volume, issue, page range) against publisher / Scopus / RePEc on 2026-05-28. Verification log in §10 of canonical doc. Two hallucinations caught and corrected during the verification pass (Hirschberger 2013 has 5 authors not 4; it is in *Operations Research* not EJOR).

**Synthesis:** The research gap is the absence of a continuous-optimization algorithm that handles **non-convex** simplex-constrained multi-KPI appreciation aggregation — every prior method in the lit review requires convex (quadratic) plus linear functional form, which tRBS's sinusoidal value functions break.

**Items for v2 (W3):** modern (2018+) continuous-NLP portfolio applications, a simplex-native optimization paper (Frank-Wolfe / mirror descent), at least one sceptical-of-shrinkage paper (DeMiguel et al. 1/N), and direct quotations to replace `[from-abstract]` paraphrases.
