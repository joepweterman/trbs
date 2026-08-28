"""
This file kept the capped-simplex grid baseline of the A4 equal-footing study while it
lived outside the package. The Vlinder team has since decided it ships with the package,
so the class now lives in ``vlinder.optimize`` as ``CappedGridSearch``, registered as
``"grid_capped"``. This module stays as a thin shim so the A4 drivers and the timed
comparison keep importing from the path they were written against.

Run (self-check against the packaged grid on a bundled case):
  python experiments/synthetic/grid_capped.py
"""

from vlinder.optimize import CappedGridSearch, Optimize  # noqa: F401  (re-export for the drivers)


def bounded_compositions(units, parts):
    """
    This function yields every tuple of ``parts`` non-negative integers summing to
    at most ``units``: the lattice points of the capped simplex. It delegates to the
    packaged implementation and stays for the older experiment imports.
    :param units: the resolution, in steps, of the budget
    :param parts: the number of internal variable inputs
    :return: a generator of integer tuples of length ``parts``
    """
    return CappedGridSearch._bounded_compositions(units, parts)  # pylint: disable=protected-access


def register():
    """
    This function used to add ``grid_capped`` to the registry; the package now ships it.
    Kept as a no-op so the existing drivers keep working unchanged.
    :return: None
    """
    Optimize.METHOD_REGISTRY.setdefault("grid_capped", CappedGridSearch)


def main():
    """Self-check: both baselines on Beerwiser, same budget, different lattices."""
    import os  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    import vlinder as vl  # pylint: disable=import-outside-toplevel
    from vlinder.trbs import TheResponsibleBusinessSimulator  # pylint: disable=import-outside-toplevel
    from vlinder.utils import suppress_print  # pylint: disable=import-outside-toplevel

    from eval_tracer import trace_evaluations  # pylint: disable=import-outside-toplevel

    register()
    data = Path(os.path.dirname(vl.__file__)) / "data"

    @suppress_print
    def build():
        sim = TheResponsibleBusinessSimulator("Beerwiser", file_path=data, file_extension="csv")
        sim.build()
        sim.evaluate()
        sim.appreciate()
        return sim

    @suppress_print
    def solve(sim, method):
        sim.optimize("Base case", method=method, dmo_name=f"Check ({method})", seed=0)

    for method in ("grid", "grid_capped"):
        sim = build()
        with trace_evaluations() as trace:
            solve(sim, method)
        result = sim.optimization_result
        spend = sum(result.allocation) / result.budget if result.budget else float("nan")
        print(
            f"{method:>12}: {trace.n_evals:>7} evaluations, appreciation {result.appreciation:.6f}, "
            f"spend {spend:.4f}"
        )


if __name__ == "__main__":
    main()
