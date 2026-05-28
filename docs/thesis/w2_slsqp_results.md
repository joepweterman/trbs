# W2 — SLSQP scaffold empirical results

> **Canonical location:** This report lives in the thesis knowledge base at
> `C:\Users\joepw\thesis-knowledge\04-experiments\exp01-slsqp-vs-grid-beerwiser.md`.
> This file in the repo is a pointer + a quick summary for code reviewers.
> Update the canonical version, not this one, when re-running the experiment.

## 30-second summary

- **Experiment script:** [`experiments/w2_slsqp_vs_grid.py`](../../experiments/w2_slsqp_vs_grid.py) (re-runnable).
- **Raw results JSON:** `experiments/out/w2_slsqp_vs_grid.json`.
- **Headline result:** On Beerwiser (k=2, Base case), SLSQP with Dirichlet multi-start hits one of **two competing basins**:
  - Interior basin ~[122k, 178k], appreciation ≈ **65.700** (9 / 12 SLSQP runs)
  - Near-corner basin ~[25k, 275k], appreciation ≈ **65.711** (2 / 12 SLSQP runs; **grid finds this deterministically**)
- **Implication for the thesis:** Plain SLSQP + random multi-start is *insufficient* on multimodal landscapes — even on the smallest tRBS case. This is the cleanest possible motivation for W3 basin-hopping work.
- **Tests:** 5 new tests in `tests/test_optimize_continuous.py`, all passing as of commit `9bde5a7`.

For full numbers, per-run allocations, methodology discussion, caveats, and the W3+ next-steps list, see the canonical report.
