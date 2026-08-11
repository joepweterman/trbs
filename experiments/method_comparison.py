"""
Compare every optimization method on the bundled cases.

Produces the table behind the review answer: per case and per method the appreciation reached,
the number of objective evaluations spent, and the wall-clock time. Also reports which method
``method="auto"`` picks for each case and why.

Run it with the repository's virtual environment:

    python experiments/method_comparison.py
    python experiments/method_comparison.py --cases beerwiser refugee --seed 7

Grid search is skipped above ``GRID_MAX_K`` levers. It enumerates every combination that spends
the budget exactly and then expands each one into all its distinct permutations, so its true
cost grows with k factorial: at nine levers a single run is already hours rather than seconds.
That is a property of the baseline, not of the harness, and it is the reason the continuous
methods exist.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from vlinder.optimize import Optimize
from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.utils import suppress_print

CASES = ["beerwiser", "refugee", "IZZ", "DSM", "NEMO"]
#: "auto" runs last so its row can be read against the methods it chooses between.
METHODS = ["grid", "slsqp", "basin_hopping", "genetic_algorithm", "mdbh", "auto"]
#: Above this many levers grid search is not run; see the module docstring.
GRID_MAX_K = 3
#: Every method runs at its own default settings, which is what a user gets when they name a
#: method without arguments. Note that basin-hopping's default is a single long chain, while
#: "auto" gives it ten shorter chains; on a case with a deep second optimum that difference
#: matters more than the number of hops does.
METHOD_KWARGS = {method: {} for method in METHODS}


@suppress_print
def build(name, root=None):
    """Build, evaluate and appreciate a case, ready to optimize."""
    case = (
        TheResponsibleBusinessSimulator(name)
        if root is None
        else TheResponsibleBusinessSimulator(name, file_path=Path(root), file_extension="csv")
    )
    case.build()
    case.evaluate()
    case.appreciate()
    return case


@suppress_print
def run_one(name, method, scenario, seed, root=None):
    """Run a single method on a freshly built case and return its result."""
    case = build(name, root)
    kwargs = dict(METHOD_KWARGS[method])
    if method != "grid":
        kwargs["seed"] = seed
    started = time.perf_counter()
    case.optimize(scenario, method=method, **kwargs)
    return case.optimization_result, time.perf_counter() - started


@suppress_print
def auto_choice(name, root=None):
    """Which method the decision tree picks for this case, and why."""
    case = build(name, root)
    optimizer = Optimize(case.input_dict, case.output_dict)
    return optimizer.select_method_for(case.input_dict["scenarios"][0])


def compare(cases, methods, seed, root=None):
    """Run every (case, method) cell and return the rows."""
    rows = []
    for name in cases:
        case = build(name, root)
        scenario = case.input_dict["scenarios"][0]
        k = len(case.input_dict["internal_variable_inputs"])
        choice = auto_choice(name, root)
        print(f"\n{name} (k={k}, scenario '{scenario}')")
        print(f"  auto picks: {choice.method} - {choice.reason}")
        print(f"  {'method':20s} {'appreciation':>13s} {'evaluations':>12s} {'seconds':>9s}")

        for method in methods:
            if method == "grid" and k > GRID_MAX_K:
                print(f"  {method:20s} {'not run':>13s} {'-':>12s} {'-':>9s}   (k > {GRID_MAX_K})")
                rows.append({"case": name, "k": k, "method": method, "appreciation": None, "note": "skipped"})
                continue
            result, elapsed = run_one(name, method, scenario, seed, root)
            evaluations = result.n_function_evals if result.n_function_evals is not None else -1
            # "auto" reports the method it landed on, so name both.
            label = f"auto -> {result.method}" if method == "auto" else method
            print(f"  {label:20s} {result.appreciation:13.6f} {evaluations:12d} {elapsed:9.2f}")
            rows.append(
                {
                    "case": name,
                    "k": k,
                    "method": method,
                    "appreciation": float(result.appreciation),
                    "n_function_evals": result.n_function_evals,
                    "seconds": elapsed,
                    "budget": result.budget,
                    "budget_spent": result.budget_spent,
                    "allocation": np.asarray(result.allocation).tolist(),
                }
            )
    return rows


def main():
    """Parse the arguments and print the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=CASES, help="cases to compare")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="methods to compare")
    parser.add_argument("--seed", type=int, default=42, help="seed for the stochastic methods")
    parser.add_argument("--root", default=None, help="directory of generated cases (csv), for synthetic cases")
    args = parser.parse_args()

    rows = compare(args.cases, args.methods, args.seed, args.root)

    print("\nBest appreciation per case")
    for name in args.cases:
        scored = [r for r in rows if r["case"] == name and r.get("appreciation") is not None]
        if scored:
            best = max(scored, key=lambda r: r["appreciation"])
            print(f"  {name:12s} {best['appreciation']:12.6f}  ({best['method']})")


if __name__ == "__main__":
    main()
