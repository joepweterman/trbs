"""
Honest non-convexity audit per case. For each case determine:
  - which nodes are decision-dependent (reachable forward from an internal variable)
  - genuine bilinear products: op '*' with BOTH operands decision-dependent (these
    are the terms that are actually non-convex in x; a product with a parameter is
    still linear in x)
  - count of sinusoidal appreciation KPIs (key_outputs.linear == 0)
  - count of min/max clamp operators (kinks -> non-smooth, also break convexity)
"""

# pylint: disable=too-many-locals

import os
from pathlib import Path
import pandas as pd
import vlinder as vl

DATA = Path(os.path.dirname(vl.__file__)) / "data"


def is_number(s):
    """True if s parses as a number (comma thousands separators allowed)."""
    try:
        float(str(s).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def audit(case):
    """Audit one bundled case for genuine nonconvexity sources."""
    csv = DATA / case / "csv"
    deps = pd.read_csv(csv / "dependencies.csv", sep=";")
    ko = pd.read_csv(csv / "key_outputs.csv", sep=";")
    internals = set(pd.read_csv(csv / "decision_makers_options.csv", sep=";")["internal_variable_input"])

    # forward-propagate decision dependence to a fixpoint
    xdep = set(internals)
    changed = True
    while changed:
        changed = False
        for _, r in deps.iterrows():
            args = [a for a in (r["argument_1"], r["argument_2"]) if not is_number(a) and not pd.isna(a)]
            if any(str(a) in xdep for a in args) and str(r["destination"]) not in xdep:
                xdep.add(str(r["destination"]))
                changed = True

    bilinear = []
    for _, r in deps.iterrows():
        if str(r["operator"]) != "*":
            continue
        a1, a2 = r["argument_1"], r["argument_2"]
        if is_number(a1) or is_number(a2) or pd.isna(a1) or pd.isna(a2):
            continue
        if str(a1) in xdep and str(a2) in xdep:
            bilinear.append((str(a1), str(a2), str(r["destination"])))

    n_sin = int((ko["linear"] == 0).sum())
    n_lin = int((ko["linear"] == 1).sum())
    clamps = deps[deps["operator"].isin(["min", "max"])]
    n_clamp_xdep = sum(
        1
        for _, r in clamps.iterrows()
        if any(str(a) in xdep for a in (r["argument_1"], r["argument_2"]) if not is_number(a) and not pd.isna(a))
    )

    print(f"=== {case} ===")
    print(f"  decision-dependent nodes: {len(xdep & set(deps['destination'])) }(+{len(internals)} internals)")
    print(f"  KPIs: {n_lin} linear, {n_sin} sinusoidal")
    print(f"  min/max clamps on decision-dependent terms: {n_clamp_xdep}")
    print(f"  GENUINE bilinear products (variable x variable, both depend on x): {len(bilinear)}")
    for a1, a2, d in bilinear[:6]:
        print(f"      {a1}  x  {a2}  ->  {d}")
    if len(bilinear) > 6:
        print(f"      ... and {len(bilinear) - 6} more")


if __name__ == "__main__":
    for c in ["Beerwiser", "Refugee", "IZZ"]:
        audit(c)
