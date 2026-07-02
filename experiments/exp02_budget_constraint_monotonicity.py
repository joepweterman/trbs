"""
exp02 — Budget-constraint formulation research: does any real tRBS case ever
prefer to UNDER-spend? Decides between Sum x = B (equality) and Sum x <= B
(capped simplex) for the thesis.

Method, per case x scenario:
  - sample the budget FACE  {x>=0, Sum x = B}     via Dirichlet(1..1)*B
  - sample the CAPPED simplex {x>=0, Sum x <= B}  via Dirichlet over k+1, drop slack
  - radial monotonicity: for each capped point x with s=Sum x, compare
        f(x)   vs   f(x * B/s)        (same ray, pushed to the budget face)
    Non-decreasing along every ray from 0 to the face  =>  optimum on the face
    =>  equality is WLOG. Violations => under-spending can strictly help.

Run:
  & "C:\\Users\\joepw\\.virtualenvs\\tRBS-DclBJWVi-python.exe\\Scripts\\python.exe" `
    C:\\Users\\joepw\\tRBS\\experiments\\exp02_budget_constraint_monotonicity.py
"""

# pylint: disable=invalid-name,protected-access,too-many-locals
# (math notation B/X/f; Optimize internals are the documented experiment surface)

import io
import json
import contextlib
import numpy as np
from vlinder.trbs import TheResponsibleBusinessSimulator
from vlinder.optimize import Optimize, evaluate_allocation

RNG = np.random.default_rng(12345)
CASES = ["Beerwiser", "Refugee", "IZZ"]
N_FACE = 4000
N_CAP = 4000
TOL = 1e-6  # appreciation points


def load(name):
    """Build/evaluate/appreciate a bundled case, silencing pipeline prints."""
    c = TheResponsibleBusinessSimulator(name)
    with contextlib.redirect_stdout(io.StringIO()):
        c.build()
        c.evaluate()
        c.appreciate()
    return c


def setup_probe(c):
    """Register the Probe DMO and freeze boundaries; return (opt, budget, k)."""
    opt = Optimize(c.input_dict, c.output_dict)
    budget = opt._infer_budget()
    ref = c.input_dict["decision_makers_option_value"][0].copy()
    with contextlib.redirect_stdout(io.StringIO()):
        opt._prepare_input_dict("Probe", ref)
    return opt, budget, opt._k


def f_vals(opt, X, sc):
    """Objective (weighted appreciation) at each allocation row of X."""
    return np.array([evaluate_allocation(opt.input_dict, x, sc, "Probe") for x in X])


def main():
    """Run the face/capped/radial sampling study over all cases and print JSON."""
    results = {}
    for name in CASES:
        c = load(name)
        opt, B, k = setup_probe(c)
        scenarios = [str(s) for s in c.input_dict["scenarios"]]
        case_res = {"k": int(k), "B": float(B), "scenarios": {}}
        for sc in scenarios:
            face = RNG.dirichlet(np.ones(k), size=N_FACE) * B
            cap = RNG.dirichlet(np.ones(k + 1), size=N_CAP)[:, :k] * B
            cap_sum = cap.sum(axis=1)
            push = cap * (B / np.where(cap_sum > 0, cap_sum, 1.0))[:, None]

            face_vals = f_vals(opt, face, sc)
            cap_vals = f_vals(opt, cap, sc)
            push_vals = f_vals(opt, push, sc)

            gains = cap_vals - push_vals  # >0 => under-spending this ray helps
            bi = int(cap_vals.argmax())
            case_res["scenarios"][sc] = {
                "best_face": round(float(face_vals.max()), 6),
                "best_capped": round(float(cap_vals.max()), 6),
                "capped_minus_face": round(float(cap_vals.max() - face_vals.max()), 6),
                "best_capped_spend_fraction": round(float(cap_sum[bi] / B), 4),
                "n_radial_underspend_strictly_better": int(np.sum(gains > TOL)),
                "max_radial_gain": round(float(gains.max()), 6),
                "median_radial_gain": round(float(np.median(gains)), 6),
                "n_cap": N_CAP,
            }
        results[name] = case_res
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
