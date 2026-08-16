"""
This file contains the capped-simplex grid baseline of the A4 equal-footing study.

The packaged ``GridSearch`` enumerates the budget face, every combination summing
to exactly the budget, while all continuous solvers work on the capped simplex
where under-spending is feasible. The baseline therefore searches a strictly
smaller set, and on a case whose appreciation is not monotone in total spend that
is a handicap rather than a technicality. This module adds a second baseline that
enumerates the capped simplex and is identical to the packaged one in every other
respect, so that the difference between the two measures the constraint and
nothing else.

Two deliberate properties:

  * the same scaling and divisibility rule as the packaged grid decides the step,
    so the two baselines are read at comparable resolutions;
  * the lattice is generated directly as the compositions of the resolution over
    k+1 parts, the last part being unspent budget, rather than by filtering
    multisets and expanding them through ``set(permutations(...))``. That is the
    enumeration a competent implementation would use, which is the point: the
    cost comparison should be against the method, not against its bookkeeping.

The class is registered into ``Optimize.METHOD_REGISTRY`` by :func:`register`,
which the driver calls. It is not added to the package: whether a thesis-side
solver belongs in the shipped registry is an open question with the vlinder team.

Run (self-check against the packaged grid on a bundled case):
  python experiments/synthetic/grid_capped.py
"""

from math import comb

from vlinder.optimize import GridSearch, Optimize


def bounded_compositions(units, parts):
    """
    This function yields every tuple of ``parts`` non-negative integers summing to
    at most ``units``: the lattice points of the capped simplex.
    :param units: the resolution, in steps, of the budget
    :param parts: the number of internal variable inputs
    :return: a generator of integer tuples of length ``parts``
    """
    if parts == 0:
        yield ()
        return
    for head in range(units + 1):
        for tail in bounded_compositions(units - head, parts - 1):
            yield (head,) + tail


class CappedGridSearch(GridSearch):
    """Grid search over ``sum(x) <= B`` instead of over the budget face.

    Everything the packaged baseline does is inherited unchanged: the budget is
    still derived from the best-performing decision-maker option, the best point
    is still written back, and an existing option that beats the whole lattice
    still wins. Only the set of points enumerated differs.
    """

    default_dmo_name = "Optimized (capped grid)"
    method_name = "grid_capped"
    method_label = "grid (capped simplex)"

    @staticmethod
    def calculate_step_size(max_investment, scaled_max_investment, num_internal_inputs, max_combinations):
        """
        This function picks the finest step whose capped lattice stays under the
        ceiling. The capped simplex holds ``comb(units + k, k)`` points against the
        face's ``comb(units + k - 1, k - 1)``, so at an equal ceiling this yields a
        step one or two units coarser than the packaged grid's.
        :param max_investment: the budget
        :param scaled_max_investment: the budget scaled for combinatorial purposes
        :param num_internal_inputs: the number of internal variable inputs
        :param max_combinations: upper bound on the number of lattice points
        :return: the step size, in budget units
        """
        step_size_tmp = 1
        while True:
            units = scaled_max_investment // step_size_tmp
            n_points = comb(units + num_internal_inputs, num_internal_inputs)
            if n_points <= max_combinations and scaled_max_investment % step_size_tmp == 0:
                break
            step_size_tmp += 1

        return max_investment / (scaled_max_investment / step_size_tmp)

    @staticmethod
    def generate_combinations(max_investment, step_size, num_internal_inputs):
        """
        This function builds every lattice point of the capped simplex at the given
        step, generated directly rather than by filtering and permuting.
        :param max_investment: the budget
        :param step_size: the step size, in budget units
        :param num_internal_inputs: the number of internal variable inputs
        :return: a list of allocation tuples, each summing to at most the budget
        """
        units = int(round(max_investment / step_size))
        return [
            tuple(count * step_size for count in point) for point in bounded_compositions(units, num_internal_inputs)
        ]


def register():
    """
    This function makes ``grid_capped`` available to ``sim.optimize(method=...)``.
    :return: None
    """
    Optimize.METHOD_REGISTRY["grid_capped"] = CappedGridSearch


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
