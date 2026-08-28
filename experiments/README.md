# Experiments

Research code around the `vlinder` optimizer. Nothing in this folder ships with the package;
the package itself lives under `src/vlinder/`. Every experiment reads the packaged cases (or
generates synthetic ones) through the unmodified pipeline, so the numbers below reproduce by
running the file they belong to.

The `expNN_` prefix marks the numbered exploratory experiments; the remaining files are the
study drivers and analysis scripts referenced by the accompanying MSc thesis (Weterman, 2026),
which keep their thesis-cited names.

## Exploratory experiments

| File | What it does | Key result |
| --- | --- | --- |
| `w2_slsqp_vs_grid.py` | The first SLSQP-vs-grid harness on Beerwiser (week 2). | Historical; superseded by exp04 after the solver-stall diagnosis. |
| `exp02_budget_constraint_monotonicity.py` | Tests whether appreciation is monotone in total spend, per packaged case. | IZZ improves along 8-11% of radial directions when under-spending, so the feasible set became the capped simplex with `spend_all` as the switch. |
| `exp04_basin_hopping_beerwiser.py` | Diagnoses the two silent solver failures and benchmarks SLSQP vs basin-hopping on Beerwiser. | int64 truncation and the budget-scale stall fixed; both methods then recover the global clip-kink optimum [25000, 275000] on 12/12 seeds at 1-3% of grid's evaluations. |
| `exp05_basin_hopping_tuning.py` | Varies one basin-hopping knob at a time: restarts vs chain length, temperature (fixed and case-derived), kick width, and Powell/COBYLA as inner solver. | On Beerwiser and IZZ nothing beats the default configuration. On Refugee the derivative-free inner solvers win, COBYLA at half the evaluations; a case-derived temperature loosens acceptance enough to hurt. |

## Comparisons on the packaged cases

| File | What it does | Key result |
| --- | --- | --- |
| `method_comparison.py` | Every method at its own defaults, per packaged case: appreciation, evaluations, seconds. | Four of the five cases agree within 0.05 points across the continuous methods; Refugee separates them, and basin-hopping wins there. This table motivated basin-hopping as the package default. |
| `time_limited_comparison.py` | Every method in both budget modes under the same wall-clock limits (15/30/60/120 s), 5 seeds, read at the median. | The continuous methods are done within 15 s on every case; the refining grid climbs with time but never catches them. |
| `figure_time_limited.py` / `table_time_limited.py` | The figure (median line, min-max band) and the LaTeX table behind the timed comparison. | See `out/time_limited/`. |
| `real_case_study.py` | The thesis's descriptive benchmark on the packaged cases (Chapter 5). | At k=9 (IZZ) the continuous methods beat the grid on quality and cost simultaneously. |

## Helpers

`bundled_cases.py`, `analyze_nonconvexity.py`, `gen_dependency_graphs.py` inspect the packaged
cases (non-convexity sources, dependency-graph structure).

## `synthetic/`

The pre-registered Monte-Carlo study of the thesis: the case generator (`case_factory.py`),
the certified oracles (`oracle.py`), the locked study harness and drivers
(`run_confirmatory.py`, `run_mdbh_benchmark.py`, `run_equal_footing.py`,
`scip_benchmark.py`), and the analysis scripts that regenerate every table and figure from the
frozen result stores. `PREREGISTRATION.md` holds the design and its dated amendments. The
stores are append-only and must not change: they carry the evidence chain of the thesis.
Headline results: multi-start SLSQP recovers the certified optimum in 1,260 of 1,260
concave-regime runs; SCIP confirms every certificate without ever branching; the grid baseline
recovers nothing at k=15, where its bounded evaluation budget buys a lattice too coarse to
contain the optimum.
