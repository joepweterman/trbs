"""
This file contains the shared evaluation tracer of the A4 equal-footing study.

Every solver in ``vlinder.optimize`` reaches the objective through the single
module-level function ``evaluate_allocation``: grid search calls it in its
enumeration loop, the SLSQP family calls it through ``BaseSolver._objective``,
and the genetic algorithm calls it in its fitness closure. Wrapping that one
function therefore records every method's evaluations with the same instrument,
which is what amendment A4 requires: no method keeps its own accounting, and no
solver code is touched.

Only improvements are stored, not every evaluation. The best-so-far staircase is
all the cost-to-target metric needs, and a full trace of a 600,000-evaluation
grid run would be an expensive way to store a step function.

Run (self-check on a bundled case):
  python experiments/synthetic/eval_tracer.py
"""

import time
from contextlib import contextmanager

import vlinder.optimize as vopt


class EvalTrace:
    """The best-so-far staircase of one solver run, in evaluations and seconds."""

    def __init__(self):
        self.n_evals = 0
        self.best_f = float("-inf")
        self.improvements = []
        self._t0 = time.perf_counter()

    def record(self, value):
        """
        This function books one objective evaluation and extends the staircase
        when it improves on everything seen so far.
        :param value: the appreciation returned by the objective
        :return: None
        """
        self.n_evals += 1
        if value > self.best_f:
            self.best_f = float(value)
            self.improvements.append((self.n_evals, time.perf_counter() - self._t0, float(value)))

    def cost_to_target(self, f_reference, tau):
        """
        This function reports the cost at which the run first came within ``tau``
        appreciation points of the reference optimum.
        :param f_reference: the reference optimum (certified f* or best known)
        :param tau: the target tolerance in appreciation points
        :return: a dict with the evaluations and seconds at that moment, or None
                 if the run never reached the target
        """
        threshold = f_reference - tau
        for n_evals, elapsed, value in self.improvements:
            if value >= threshold:
                return {"evals": n_evals, "seconds": elapsed}
        return None


@contextmanager
def trace_evaluations():
    """
    This context manager routes every objective evaluation through one trace.

    The patch is process-local and is always undone, so worker processes and
    sequential runs in the same process do not leak into one another.
    :return: the EvalTrace collecting the run inside the block
    """
    trace = EvalTrace()
    original = vopt.evaluate_allocation

    def traced(*args, **kwargs):
        value = original(*args, **kwargs)
        trace.record(value)
        return value

    vopt.evaluate_allocation = traced
    try:
        yield trace
    finally:
        vopt.evaluate_allocation = original


def main():
    """Self-check: trace two methods on Beerwiser and report what was captured."""
    import os  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    import vlinder as vl  # pylint: disable=import-outside-toplevel
    from vlinder.trbs import TheResponsibleBusinessSimulator  # pylint: disable=import-outside-toplevel
    from vlinder.utils import suppress_print  # pylint: disable=import-outside-toplevel

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
        sim.optimize("Base case", method=method, dmo_name=f"Trace ({method})", seed=0)

    for method in ("grid", "slsqp"):
        sim = build()
        with trace_evaluations() as trace:
            solve(sim, method)
        reported = sim.optimization_result.n_function_evals
        print(
            f"{method:>6}: traced {trace.n_evals} evaluations, solver reported {reported}, "
            f"{len(trace.improvements)} improvements, best {trace.best_f:.6f}"
        )


if __name__ == "__main__":
    main()
