# W1 — tRBS Codebase Audit and Code Placement Decision

**Author:** Joep Weterman
**Date:** 2026-05-21
**Status:** Draft for kick-off meeting

---

## 1. tRBS Architecture (current state)

The `vlinder` package implements the Responsible Business Simulator as a four-stage pipeline. The user-facing entry point is `TheResponsibleBusinessSimulator` (in `src/vlinder/trbs.py`), instantiated with a case folder + format + case name. The pipeline:

| Stage | Class | File | Responsibility |
|-------|-------|------|----------------|
| 1. Build | `CaseImporter` | `case_importer.py` | Read Excel/JSON/CSV into `input_dict` (decision variables, KPIs, dependency graph, weights) |
| 2. Evaluate | `Evaluate` | `evaluate.py` | Walk the dependency graph for each (scenario, DMO) pair → key-output values |
| 3. Appreciate | `Appreciate` | `appreciate.py` | Transform KPI values into 0–100 appreciation scores (linear OR sinusoidal), weight by KPI × theme × scenario, aggregate |
| 4. Optimize (existing) | `Optimize` | `optimize.py` | Brute-force grid search over discrete allocations summing to total budget |

The decision variable in optimization terms is the vector
$$ \mathbf{x} = (x_1, \ldots, x_k) \in \mathbb{R}^k_{\geq 0}, \quad \sum_{i=1}^k x_i = B $$
where `x` corresponds to `input_dict["decision_makers_option_value"][dmo_index]` — the allocation of total budget B across the k `internal_variable_inputs`.

## 2. How the current `Optimize` class works (and why it must be replaced for continuous allocation)

`optimize.py::Optimize.optimize_single_scenario` executes the following pipeline (line numbers from current `main`):

1. **Find the current best DMO** (`find_dict_values`, L28–52) — identifies the highest-appreciation DMO and its allocation as a starting reference.
2. **Scale budget** (`scale_max_investment`, L54–69) — rescales `max_investment` to a manageable grid resolution (rounds to nearest 100 within the order of magnitude).
3. **Compute step size** (`calculate_step_size`, L71–96) — solves for the largest step size such that `C(units + k − 1, k − 1) ≤ max_combinations` (stars-and-bars combinatorial cap).
4. **Generate combinations** (`generate_combinations`, L98–113) — uses `itertools.combinations_with_replacement` + `permutations` to enumerate all discrete allocations summing to `max_investment`.
5. **Grid search** (`grid_search`, L115–176) — for each combination, mutates `input_dict["decision_makers_option_value"]`, calls `Evaluate(...).evaluate_selected_scenario(scenario)` + `Appreciate(...)`, records appreciation, tracks the best.

**Fundamental limitation:** The combinatorial cost is C(B/s + k − 1, k − 1). For B = 1000, k = 4, step size s = 10: C(103, 3) ≈ 178,000 evaluations. For k = 6 and finer resolution this explodes (curse of dimensionality, Bergstra & Bengio 2012). Coarse grids miss the true optimum; fine grids are computationally infeasible.

## 3. The new continuous optimization layer — where it lives

### 3.1 Decision: new module `src/vlinder/optimize_continuous.py`

A new sibling to `optimize.py`, **not a modification of it**. Rationale:

- **Backwards compatibility**: existing `case.optimize(scenario)` API used in `vlinder_demo.ipynb` and tests stays untouched. PwC users and the open-source community can continue to use the grid search.
- **Clean comparison**: thesis methodology requires benchmarking the new methods against the existing grid search. Keeping both modules in parallel makes this trivial — the benchmark experiments simply instantiate both classes against the same case.
- **Reduced risk of regression**: 172 tests currently exist and reference `Optimize`. Adding a new class avoids breaking them.

### 3.2 Class skeleton

```python
# src/vlinder/optimize_continuous.py
from typing import Callable, Literal
import numpy as np
from scipy.optimize import minimize, basinhopping
from vlinder.appreciate import Appreciate
from vlinder.evaluate import Evaluate

Method = Literal["slsqp", "basin_hopping", "genetic"]

class ContinuousOptimize:
    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self._k = len(input_dict["internal_variable_inputs"])
        self._budget = None  # set by optimize()

    def _build_objective(self, scenario: str, dmo_name: str) -> Callable[[np.ndarray], float]:
        """Returns f(x) = -appreciation(x) for minimization."""
        # Uses a deep copy of input_dict to avoid mutation across optimizer iterations
        ...

    def optimize(self, scenario: str, method: Method = "slsqp",
                 dmo_name: str = "Optimized (continuous)",
                 n_multistart: int = 100) -> dict:
        """Dispatch to the selected continuous optimization method."""
        ...
```

### 3.3 User-facing API addition (in `trbs.py`)

```python
def optimize_continuous(self, scenario: str, method: str = "slsqp", **kwargs):
    self._validate_evaluated_and_appreciated()
    ContinuousOptimize(self.input_dict, self.output_dict).optimize(
        scenario=scenario, method=method, **kwargs
    )
```

Demo notebook gains a section parallel to the existing grid-search demo, side-by-side comparison.

### 3.4 Refactor candidates flagged for discussion with supervisor

These are improvements I'd want to make but would touch shared code — discuss before doing:

1. **Pure-function evaluation wrapper.** Today `Optimize.grid_search` mutates `input_dict` in-place during the search loop. For a continuous optimizer this is dangerous (optimizer may call f(x) hundreds of times, and partial state across calls is fragile). I'd extract a pure helper `evaluate_allocation(input_dict, x, scenario) -> float` that takes a copy. Risk: minor, but it touches `optimize.py`.
2. **Gradient interface.** None of the methods I'm starting with require analytical gradients (SLSQP can use finite differences). But if I later test trust-region or interior-point methods, I'd need automatic differentiation. JAX-compatible reimplementation of `evaluate_all_dependencies` would be the path; large scope, only worth it if benchmarking demands it.

## 4. Known issue surfaced during setup

`tests/test_make_report.py::test_create_report[Optimistic]` fails on Windows because the report filename uses `strftime("%H:%M:%S")` — `:` is illegal in Windows paths. 171/172 tests pass; this is a pre-existing repo bug unrelated to the thesis. Trivial fix (`%H-%M-%S`), happy to PR upstream after the thesis ships.

## 5. Open questions for the supervisor

1. Is **continuous over a single DMO's internal variables** the right scope, or should the optimizer also choose **between DMOs** (mixed combinatorial-continuous)? The proposal reads as the former; confirming.
2. For multi-scenario problems, is the goal to optimize the **scenario-weighted aggregate** (current `_apply_scenario_weights` logic), or to compute **per-scenario optima** and then analyze robustness — or both? RQ3 in the proposal points toward both.
3. Should the new module land in the public `vlinder` package, or be kept in a `thesis/` subdirectory until the thesis is graded?
