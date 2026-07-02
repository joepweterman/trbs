"""
Generate clean layered dependency-graph figures for the thesis proposal.

For each case (Beerwiser, Refugee, IZZ) this reads the packaged vlinder
dependency data and renders a left-to-right layered DAG:

  internal variables (decisions)  ->  intermediate computations  ->  key outputs
  external / fixed inputs feed in from the left as well.

Multiplicative couplings (operator '*') are highlighted, because a product of
two terms that both depend on the allocation is bilinear and is what can make
the aggregated appreciation non-convex (see the proposal's Theory section).

Output: dep_<case>.pdf (vector) + dep_<case>.png in the --out directory.
No graphviz needed: layering is computed by longest path, layout via
networkx.multipartite_layout.
"""

# pylint: disable=too-many-locals,wrong-import-position
# (matplotlib.use('Agg') must run before pyplot is imported)

import argparse
import os
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import vlinder as vl  # noqa: E402

# vlinder brand palette
C_INTERNAL = "#299D8F"  # decision (internal) variables
C_EXTERNAL = "#BBBBBB"  # external / fixed inputs
C_INTER = "#6A91B6"  # intermediate computations
C_OUTPUT = "#D04A02"  # key outputs
C_MULT = "#E72B33"  # multiplicative edge highlight
C_OTHER = "#9aa0a6"  # other edges

DATA = Path(os.path.dirname(vl.__file__)) / "data"


def is_number(s):
    """True if s parses as a number (comma thousands separators allowed)."""
    try:
        float(str(s).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def load_case(case):
    """Read dependencies, key outputs and internal variables for a bundled case."""
    csv = DATA / case / "csv"
    deps = pd.read_csv(csv / "dependencies.csv", sep=";")
    kos = pd.read_csv(csv / "key_outputs.csv", sep=";")["key_output"].tolist()
    dmo = pd.read_csv(csv / "decision_makers_options.csv", sep=";")
    internals = dmo["internal_variable_input"].unique().tolist()
    return deps, kos, internals


def build_graph(deps):
    """Dependency DAG with a bilinear flag on genuine variable-times-variable edges."""
    g = nx.DiGraph()
    for _, row in deps.iterrows():
        dest = str(row["destination"])
        op = str(row["operator"])
        a1, a2 = row["argument_1"], row["argument_2"]
        # a genuine bilinear coupling = product of two terms that are both
        # variables/computations (not a multiply-by-constant scaling)
        both_vars = not is_number(a1) and not pd.isna(a1) and not is_number(a2) and not pd.isna(a2)
        coupling = (op == "*") and both_vars
        g.add_node(dest)
        for arg in (a1, a2):
            if is_number(arg) or pd.isna(arg):
                continue
            arg = str(arg)
            g.add_node(arg)
            if g.has_edge(arg, dest):
                if coupling:
                    g[arg][dest]["coupling"] = True
            else:
                g.add_edge(arg, dest, op=op, coupling=coupling)
    return g


def assign_layers(g):
    """Cycle-safe longest-path layering via SCC condensation."""
    cond = nx.condensation(g)  # DAG of strongly connected components
    clayer = {n: 0 for n in cond.nodes}
    for n in nx.topological_sort(cond):
        for succ in cond.successors(n):
            clayer[succ] = max(clayer[succ], clayer[n] + 1)
    layer = {}
    for cn, members in cond.nodes(data="members"):
        for m in members:
            layer[m] = clayer[cn]
    nx.set_node_attributes(g, layer, "layer")
    return layer


def categorise(node, kos, internals, g):
    """Node category: output / internal / external / intermediate."""
    if node in kos:
        return "output"
    if node in internals:
        return "internal"
    if g.in_degree(node) == 0:
        return "external"
    return "intermediate"


def draw(case, out_dir):
    """Render the layered dependency DAG for one case to PDF + PNG."""
    deps, kos, internals = load_case(case)
    g = build_graph(deps)
    layer = assign_layers(g)
    cats = {n: categorise(n, kos, internals, g) for n in g.nodes}
    color_map = {"internal": C_INTERNAL, "external": C_EXTERNAL, "intermediate": C_INTER, "output": C_OUTPUT}
    node_colors = [color_map[cats[n]] for n in g.nodes]

    pos = nx.multipartite_layout(g, subset_key="layer", align="vertical", scale=1.0)

    n_layers = max(layer.values()) + 1
    max_in_layer = max(sum(1 for v in layer.values() if v == L) for L in range(n_layers))
    width = max(9, 2.0 * n_layers)
    height = max(6, 0.55 * max_in_layer)

    fig, ax = plt.subplots(figsize=(width, height))

    mult_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("coupling")]
    other_edges = [(u, v) for u, v, d in g.edges(data=True) if not d.get("coupling")]
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=other_edges,
        ax=ax,
        edge_color=C_OTHER,
        width=0.8,
        arrows=True,
        arrowsize=8,
        alpha=0.6,
        connectionstyle="arc3,rad=0.03",
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=mult_edges,
        ax=ax,
        edge_color=C_MULT,
        width=1.6,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.03",
    )
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=260, edgecolors="white", linewidths=0.6)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=6.0, font_color="#111111")

    legend = [
        Patch(facecolor=C_INTERNAL, label="internal variable (decision)"),
        Patch(facecolor=C_EXTERNAL, label="external / fixed input"),
        Patch(facecolor=C_INTER, label="intermediate computation"),
        Patch(facecolor=C_OUTPUT, label="key output (KPI)"),
        Line2D([0], [0], color=C_MULT, lw=2, label="multiplicative coupling: variable $\\times$ variable"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"dep_{case.lower()}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(
        f"{case}: nodes={g.number_of_nodes()} edges={g.number_of_edges()} "
        f"layers={n_layers} mult_edges={len(mult_edges)} -> {stem}.pdf/.png"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.cwd() / "dep_figs"))
    ap.add_argument("--cases", nargs="+", default=["Beerwiser", "Refugee", "IZZ"])
    args = ap.parse_args()
    for c in args.cases:
        draw(c, args.out)
