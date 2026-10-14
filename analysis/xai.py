"""
analysis/xai.py
===============
XAI (Explainable AI) visualizations for the Q-Learning agent.

Main functions
--------------
load_qtables(qtables_dir, instance_filter)
    Aggregates Q-tables across repetitions for a given instance group.

policy_heatmap(Q_avg, f1_bin, f2_bin, ax, title)
    Draws a 3x3 heatmap of the greedy policy over (f3_bin x f4_bin)
    for a fixed slice (f1_bin, f2_bin).

plot_all_slices(Q_avg, instance_label, out_dir)
    Produces a 2x2 figure:
      - early search  (f1=4, f2=4): high gap, unbalanced
      - late search   (f1=0, f2=0): near-optimal, balanced
      - mid-high gap  (f1=2, f2=2)
      - diversify     (f1=3, f2=1): stagnated, moderate imbalance

plot_q_values_heatmap(Q_avg, instance_label, out_dir)
    4-panel figure: one panel per operator, showing max Q-value
    over (f3_bin x f4_bin), aggregated over all (f1, f2) slices.

plot_visit_heatmap(visits_avg, instance_label, out_dir)
    Heatmap of state visitation frequency (aggregated over f1, f2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STATES   = 81
N_OPERATORS = 4
OPERATOR_NAMES = ["Op1\n(inter-swap)", "Op2\n(reinsert)", "Op3\n(intra)", "Op4\n(block)"]
OPERATOR_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# Feature bin labels
F1_LABELS = ["low gap", "mid gap", "high gap"]                   # f1: relative gap
F2_LABELS = ["balanced", "moderate", "unbalanced"]               # f2: load CV
F3_LABELS = ["early", "mid", "late"]                             # f3: progress
F4_LABELS = ["low stag", "mid stag", "high stag"]               # f4: stagnation


# ---------------------------------------------------------------------------
# Q-table aggregation
# ---------------------------------------------------------------------------

def load_qtables(
    qtables_dir: Path | str,
    instance_filter: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and averages Q-tables from all .npz files in qtables_dir whose
    filename contains instance_filter.

    Returns
    -------
    Q_avg : ndarray (625, 4)
        Element-wise mean of Q across all matching files.
    visits_avg : ndarray (625, 4)
        Element-wise mean of visit counts.
    """
    qtables_dir = Path(qtables_dir)
    files = sorted(qtables_dir.glob("*.npz"))
    if instance_filter:
        files = [f for f in files if instance_filter in f.name]

    if not files:
        raise FileNotFoundError(
            f"No .npz files found in {qtables_dir} "
            f"matching '{instance_filter}'"
        )

    Qs      = []
    visits  = []
    for fp in files:
        try:
            data = np.load(fp)
            Q = data["Q"]
            V = data["visits"]
            if Q.shape == (N_STATES, N_OPERATORS):
                Qs.append(Q)
                visits.append(V)
        except Exception:
            continue  # skip corrupted or stale files

    if not Qs:
        raise FileNotFoundError(
            f"No valid (N_STATES={N_STATES}) .npz files in {qtables_dir} "
            f"matching '{instance_filter}'"
        )

    return np.mean(Qs, axis=0), np.mean(visits, axis=0)


# ---------------------------------------------------------------------------
# Greedy policy extraction
# ---------------------------------------------------------------------------

def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """
    Returns the greedy action (0-3) for each state.
    Ties broken by lowest index.

    Parameters
    ----------
    Q : ndarray (625, 4)

    Returns
    -------
    policy : ndarray (625,) int
    """
    return np.argmax(Q, axis=1)


def policy_slice(
    policy: np.ndarray,
    f1_bin: int,
    f2_bin: int,
) -> np.ndarray:
    """
    Extracts the 3x3 policy grid for fixed (f1_bin, f2_bin),
    varying f3_bin (rows) and f4_bin (cols).

    State encoding: s = f1*27 + f2*9 + f3*3 + f4  (base-3)

    Returns
    -------
    grid : ndarray (3, 3)  -- values in {0,1,2,3}
    """
    grid = np.zeros((3, 3), dtype=int)
    for f3 in range(3):
        for f4 in range(3):
            state_id = f1_bin * 27 + f2_bin * 9 + f3 * 3 + f4
            grid[f3, f4] = policy[state_id]
    return grid


def qmax_slice(
    Q: np.ndarray,
    f1_bin: int,
    f2_bin: int,
) -> np.ndarray:
    """
    Returns the max Q-value grid (3x3) for fixed (f1, f2),
    varying f3 (rows) x f4 (cols).  State encoding base-3.
    """
    grid = np.zeros((3, 3))
    for f3 in range(3):
        for f4 in range(3):
            state_id = f1_bin * 27 + f2_bin * 9 + f3 * 3 + f4
            grid[f3, f4] = Q[state_id].max()
    return grid


# ---------------------------------------------------------------------------
# Single-slice heatmap
# ---------------------------------------------------------------------------

def policy_heatmap(
    Q: np.ndarray,
    f1_bin: int,
    f2_bin: int,
    ax: plt.Axes,
    title: str = "",
    show_text: bool = True,
) -> None:
    """
    Draws the greedy-policy heatmap for a (f1, f2) slice onto ax.
    Rows = f3 (progress), Cols = f4 (stagnation).
    Cell color = preferred operator; text = operator index (1-based).
    """
    policy  = greedy_policy(Q)
    grid    = policy_slice(policy, f1_bin, f2_bin)
    cmap    = matplotlib.colors.ListedColormap(OPERATOR_COLORS)
    norm    = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")

    # Axis labels
    ax.set_xticks(range(3))
    ax.set_xticklabels(F4_LABELS, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(F3_LABELS, fontsize=7)
    ax.set_xlabel("Stagnation (f4)", fontsize=8)
    ax.set_ylabel("Progress (f3)", fontsize=8)

    if show_text:
        for r in range(3):
            for c in range(3):
                op = grid[r, c]
                ax.text(c, r, f"Op{op+1}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

    if title:
        ax.set_title(title, fontsize=9, pad=4)


# ---------------------------------------------------------------------------
# Multi-slice figure
# ---------------------------------------------------------------------------

_SLICES = [
    # (f1_bin, f2_bin, subtitle) — indices in {0,1,2} for 3-bin space
    (2, 2, "High gap\n(unbalanced load)"),
    (2, 0, "High gap\n(balanced load)"),
    (1, 1, "Mid search\n(moderate)"),
    (0, 0, "Near-optimal\n(balanced)"),
]


def plot_all_slices(
    Q: np.ndarray,
    instance_label: str,
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    Produces a 2x2 figure of policy heatmaps for the four canonical
    (f1, f2) slices defined in _SLICES.

    Returns the path to the saved PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Q-Learning Greedy Policy -- {instance_label}\n"
        "(rows=progress f3, cols=stagnation f4; color=preferred operator)",
        fontsize=11,
    )

    for ax, (f1, f2, subtitle) in zip(axes.flat, _SLICES):
        policy_heatmap(Q, f1, f2, ax, title=subtitle)

    # Shared legend
    patches = [
        mpatches.Patch(color=OPERATOR_COLORS[k], label=f"Op{k+1}")
        for k in range(N_OPERATORS)
    ]
    fig.legend(
        handles=patches,
        title="Operator",
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = out_dir / f"policy_heatmap_{instance_label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Q-value magnitude heatmap (one panel per operator)
# ---------------------------------------------------------------------------

def plot_q_values_heatmap(
    Q: np.ndarray,
    instance_label: str,
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    4-panel heatmap showing the MEAN Q-value of each operator across all
    (f1, f2) slices, projected onto (f3 x f4).

    Useful for seeing which operator accumulates the most value overall
    regardless of the current gap/balance.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate Q over f1 and f2 dimensions
    # Q has shape (81, 4); decode each state (base-3: s = b1*27 + b2*9 + b3*3 + b4)
    mean_q = np.zeros((3, 3, N_OPERATORS))   # [f3, f4, op]
    count  = np.zeros((3, 3), dtype=int)

    for s in range(N_STATES):
        b1  = s // 27
        rem = s % 27
        b2  = rem // 9
        rem = rem % 9
        b3  = rem // 3
        b4  = rem % 3
        mean_q[b3, b4, :] += Q[s, :]
        count[b3, b4] += 1

    # Normalise
    for op in range(N_OPERATORS):
        mean_q[:, :, op] /= np.maximum(count, 1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(
        f"Mean Q-value per Operator (aggregated over f1, f2) -- {instance_label}",
        fontsize=11,
    )

    vmin = mean_q.min()
    vmax = mean_q.max()

    for k, ax in enumerate(axes):
        im = ax.imshow(
            mean_q[:, :, k],
            cmap="viridis",
            vmin=vmin, vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"Op{k+1}", fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(F4_LABELS, fontsize=6, rotation=30, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels(F3_LABELS, fontsize=6)
        ax.set_xlabel("Stagnation (f4)", fontsize=7)
        if k == 0:
            ax.set_ylabel("Progress (f3)", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = out_dir / f"qvalue_heatmap_{instance_label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# State-visitation heatmap
# ---------------------------------------------------------------------------

def plot_visit_heatmap(
    visits: np.ndarray,
    instance_label: str,
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    Heatmap of total state visits projected onto (f3 x f4),
    aggregated over all (f1, f2) slices.

    Reveals which regions of the state space the agent actually explores.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # visits shape: (625, 4) -- sum over operators first
    total_visits = visits.sum(axis=1)   # (625,)

    # Project onto (f3, f4)
    proj = np.zeros((3, 3))
    for s in range(N_STATES):
        rem = s % 9
        b3  = rem // 3
        b4  = rem % 3
        proj[b3, b4] += total_visits[s]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(proj, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_xticklabels(F4_LABELS, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(F3_LABELS, fontsize=8)
    ax.set_xlabel("Stagnation (f4)", fontsize=9)
    ax.set_ylabel("Progress (f3)", fontsize=9)
    ax.set_title(
        f"State Visitation (aggregated over gap f1 & CV f2)\n{instance_label}",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="Total visits")

    # Annotate cell values
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{proj[r,c]:.0f}", ha="center", va="center",
                    fontsize=7, color="black")

    plt.tight_layout()
    out_path = out_dir / f"visit_heatmap_{instance_label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: generate all XAI figures for one instance group
# ---------------------------------------------------------------------------

def generate_xai_figures(
    qtables_dir: Path | str,
    instance_filter: str,
    instance_label: str,
    out_dir: Path | str,
) -> list[Path]:
    """
    Loads aggregated Q-tables for instance_filter and generates:
      1. policy_heatmap_<label>.png   (4-slice policy grid)
      2. qvalue_heatmap_<label>.png   (per-operator mean Q)
      3. visit_heatmap_<label>.png    (state visitation)

    Returns list of paths to saved figures.
    """
    Q, visits = load_qtables(qtables_dir, instance_filter)
    paths = []
    paths.append(plot_all_slices(Q, instance_label, out_dir))
    paths.append(plot_q_values_heatmap(Q, instance_label, out_dir))
    paths.append(plot_visit_heatmap(visits, instance_label, out_dir))
    return paths


# ---------------------------------------------------------------------------
# Policy textual interpretation (XAI narrative)
# ---------------------------------------------------------------------------

# Short names for operator labels (no newline)
_OP_SHORT = ["Op1 (inter-swap)", "Op2 (reinsert)", "Op3 (intra-swap)", "Op4 (block-move)"]

# Human-readable descriptions of each operator
_OP_DESC = {
    0: "inter-machine job swap (Op1)",
    1: "job reinsertion (Op2)",
    2: "intra-machine job swap (Op3)",
    3: "block move (Op4)",
}

# Canonical scenario labels for narrative
_SCENARIOS = [
    ((4, 4), "early search with high imbalance"),
    ((4, 0), "early search with balanced load"),
    ((2, 2), "mid search with moderate conditions"),
    ((0, 0), "near-optimal, balanced load"),
    ((0, 4), "near-optimal, unbalanced load"),
    ((4, 4), "high gap, unbalanced"),  # same as first, used for stagnation slices
]


def _dominant_op(policy: np.ndarray, f1_bin: int, f2_bin: int) -> dict:
    """Returns a dict with dominant operator and its frequency for a (f1, f2) slice."""
    from collections import Counter
    grid = policy_slice(policy, f1_bin, f2_bin)
    flat = grid.flatten()
    counts = Counter(flat.tolist())
    dom_op = max(counts, key=counts.get)
    dom_freq = counts[dom_op] / len(flat)
    return {"op": dom_op, "freq": dom_freq, "counts": dict(counts)}


def _stagnation_preference(policy: np.ndarray) -> dict:
    """
    Aggregates operator preference for low-stagnation (f4=0,1) vs
    high-stagnation (f4=3,4) states across ALL (f1,f2,f3) slices.
    Returns {low_stag: dominant_op, high_stag: dominant_op}.
    """
    from collections import Counter
    low_counts  = Counter()
    high_counts = Counter()
    for s in range(N_STATES):
        f4 = s % 3
        if f4 == 0:
            low_counts[int(policy[s])]  += 1
        elif f4 == 2:
            high_counts[int(policy[s])] += 1

    low_dom  = max(low_counts,  key=low_counts.get)
    high_dom = max(high_counts, key=high_counts.get)
    low_freq  = low_counts[low_dom]  / sum(low_counts.values())
    high_freq = high_counts[high_dom] / sum(high_counts.values())
    return {
        "low_stag":  {"op": low_dom,  "freq": low_freq},
        "high_stag": {"op": high_dom, "freq": high_freq},
    }


def _progress_preference(policy: np.ndarray) -> dict:
    """
    Aggregates operator preference for early (f3=0,1) vs late (f3=3,4) progress bins
    across ALL (f1,f2,f4) slices.
    """
    from collections import Counter
    early_counts = Counter()
    late_counts  = Counter()
    for s in range(N_STATES):
        f3 = (s // 3) % 3
        if f3 == 0:
            early_counts[int(policy[s])] += 1
        elif f3 == 2:
            late_counts[int(policy[s])]  += 1

    early_dom = max(early_counts, key=early_counts.get)
    late_dom  = max(late_counts,  key=late_counts.get)
    early_freq = early_counts[early_dom] / sum(early_counts.values())
    late_freq  = late_counts[late_dom]   / sum(late_counts.values())
    return {
        "early": {"op": early_dom, "freq": early_freq},
        "late":  {"op": late_dom,  "freq": late_freq},
    }


def _gap_preference(policy: np.ndarray) -> dict:
    """
    Aggregates operator preference for high gap (f1=3,4) vs low gap (f1=0,1)
    across ALL (f2,f3,f4) slices.
    """
    from collections import Counter
    high_gap_counts = Counter()
    low_gap_counts  = Counter()
    for s in range(N_STATES):
        f1 = s // 27
        if f1 == 2:
            high_gap_counts[int(policy[s])] += 1
        elif f1 == 0:
            low_gap_counts[int(policy[s])]  += 1

    hg_dom = max(high_gap_counts, key=high_gap_counts.get)
    lg_dom = max(low_gap_counts,  key=low_gap_counts.get)
    hg_freq = high_gap_counts[hg_dom] / sum(high_gap_counts.values())
    lg_freq = low_gap_counts[lg_dom]  / sum(low_gap_counts.values())
    return {
        "high_gap": {"op": hg_dom, "freq": hg_freq},
        "low_gap":  {"op": lg_dom, "freq": lg_freq},
    }


def policy_report(
    Q: np.ndarray,
    instance_label: str,
    out_dir: Path | str | None = None,
    verbose: bool = True,
) -> str:
    """
    Produces a human-readable (and LaTeX-embeddable) textual interpretation
    of the Q-Learning agent's learned policy.

    Parameters
    ----------
    Q : ndarray (625, 4)
        Averaged Q-table for a given instance group.
    instance_label : str
        Label used in headings (e.g. "n=50, m=4").
    out_dir : Path or str, optional
        If provided, saves the report to <out_dir>/policy_report_<label>.txt
        and a LaTeX .tex version.
    verbose : bool
        If True, prints the report to stdout.

    Returns
    -------
    report : str
        Plain-text policy interpretation.
    """
    out_dir = Path(out_dir) if out_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    policy = greedy_policy(Q)

    # --- aggregate analyses -------------------------------------------------
    prog  = _progress_preference(policy)
    stag  = _stagnation_preference(policy)
    gap   = _gap_preference(policy)

    # --- per-scenario dominant ops ------------------------------------------
    early_hi = _dominant_op(policy, 2, 2)   # high gap, unbalanced
    early_lo = _dominant_op(policy, 2, 0)   # high gap, balanced
    late_hi  = _dominant_op(policy, 0, 2)   # near-opt, unbalanced
    late_lo  = _dominant_op(policy, 0, 0)   # near-opt, balanced
    mid_mid  = _dominant_op(policy, 1, 1)   # mid search, moderate

    # --- unique operators in policy -----------------------------------------
    unique_ops = sorted(set(policy.tolist()))
    unique_pct = {op: float((policy == op).mean()) for op in range(4)}

    # --- build text ---------------------------------------------------------
    lines = []
    lines.append(f"=== Policy Report: {instance_label} ===")
    lines.append("")
    lines.append("Operator usage (fraction of states where it is greedy-optimal):")
    for op in range(4):
        lines.append(f"  {_OP_SHORT[op]:30s}  {unique_pct[op]*100:5.1f}%")
    lines.append("")

    # Phase analysis
    ep = prog["early"]["op"]; ef = prog["early"]["freq"]
    lp = prog["late"]["op"];  lf = prog["late"]["freq"]
    lines.append("Search-phase analysis (f3 = progress):")
    lines.append(f"  Early iterations (f3 in [0,40%]): dominant operator = {_OP_DESC[ep]} "
                 f"({ef*100:.0f}% of states in this phase)")
    lines.append(f"  Late  iterations (f3 in [60,100%]): dominant operator = {_OP_DESC[lp]} "
                 f"({lf*100:.0f}% of states in this phase)")
    if ep != lp:
        lines.append("  --> The agent transitions its preferred operator as the search matures,")
        lines.append("      suggesting an implicit diversification-to-intensification strategy.")
    else:
        lines.append("  --> The agent consistently prefers the same operator regardless of progress,")
        lines.append("      indicating that this operator dominates across all search phases.")
    lines.append("")

    # Stagnation analysis
    sp = stag["low_stag"]["op"];  sf = stag["low_stag"]["freq"]
    hp = stag["high_stag"]["op"]; hf = stag["high_stag"]["freq"]
    lines.append("Stagnation analysis (f4 = fraction of iterations without improvement):")
    lines.append(f"  Low  stagnation (f4 in [0,30%]): dominant operator = {_OP_DESC[sp]} "
                 f"({sf*100:.0f}%)")
    lines.append(f"  High stagnation (f4 > 60%):        dominant operator = {_OP_DESC[hp]} "
                 f"({hf*100:.0f}%)")
    if sp != hp:
        lines.append("  --> Under high stagnation the agent switches operator, acting as an")
        lines.append("      implicit escape-from-local-optima (diversification) mechanism.")
    else:
        lines.append("  --> Stagnation level does not change the preferred operator,")
        lines.append("      suggesting the agent relies on perturbation (ILS kick) for escape.")
    lines.append("")

    # Gap analysis
    hg = gap["high_gap"]["op"]; hgf = gap["high_gap"]["freq"]
    lg = gap["low_gap"]["op"];  lgf = gap["low_gap"]["freq"]
    lines.append("Solution quality analysis (f1 = relative gap to lower bound):")
    lines.append(f"  High gap (f1 > 60%): dominant operator = {_OP_DESC[hg]} ({hgf*100:.0f}%)")
    lines.append(f"  Low  gap (f1 < 20%): dominant operator = {_OP_DESC[lg]} ({lgf*100:.0f}%)")
    if hg != lg:
        lines.append("  --> The agent applies different operators depending on solution quality,")
        lines.append("      consistent with an adaptive intensification strategy near the optimum.")
    else:
        lines.append("  --> The preferred operator is invariant to solution quality.")
    lines.append("")

    # Scenario-level interpretation
    lines.append("Canonical scenario summary:")
    lines.append(f"  High gap, unbalanced     (f1=2,f2=2): {_OP_DESC[early_hi['op']]} "
                 f"({early_hi['freq']*100:.0f}% of slice states)")
    lines.append(f"  High gap, balanced       (f1=2,f2=0): {_OP_DESC[early_lo['op']]} "
                 f"({early_lo['freq']*100:.0f}%)")
    lines.append(f"  Mid search               (f1=1,f2=1): {_OP_DESC[mid_mid['op']]} "
                 f"({mid_mid['freq']*100:.0f}%)")
    lines.append(f"  Near-optimal, unbalanced (f1=0,f2=2): {_OP_DESC[late_hi['op']]} "
                 f"({late_hi['freq']*100:.0f}%)")
    lines.append(f"  Near-optimal, balanced   (f1=0,f2=0): {_OP_DESC[late_lo['op']]} "
                 f"({late_lo['freq']*100:.0f}%)")
    lines.append("")
    lines.append("Summary paragraph (for paper):")
    lines.append("-" * 60)

    # Build paragraph
    para = _build_paper_paragraph(instance_label, prog, stag, gap, early_hi, late_lo, mid_mid)
    lines.append(para)
    lines.append("-" * 60)

    report = "\n".join(lines)

    if verbose:
        print(report)

    if out_dir is not None:
        safe = instance_label.replace(",", "").replace(" ", "_").replace("=", "")
        txt_path = out_dir / f"policy_report_{safe}.txt"
        txt_path.write_text(report, encoding="utf-8", newline="\n")
        tex_path = out_dir / f"policy_report_{safe}.tex"
        tex_path.write_text(_to_latex_paragraph(para, instance_label), encoding="utf-8", newline="\n")

    return report


def _build_paper_paragraph(
    label: str,
    prog: dict,
    stag: dict,
    gap: dict,
    early_hi: dict,
    late_lo: dict,
    mid_mid: dict,
) -> str:
    """Constructs a concise paragraph summarising the learned policy."""
    ep = prog["early"]["op"];  ef = prog["early"]["freq"]
    lp = prog["late"]["op"];   lf = prog["late"]["freq"]
    sp = stag["low_stag"]["op"];  sf = stag["low_stag"]["freq"]
    hp = stag["high_stag"]["op"]; hf = stag["high_stag"]["freq"]
    hg = gap["high_gap"]["op"];   hgf = gap["high_gap"]["freq"]
    lg = gap["low_gap"]["op"];    lgf = gap["low_gap"]["freq"]

    op_name = ["inter-machine job swap (Op1)", "job reinsertion (Op2)",
               "intra-machine job swap (Op3)", "block move (Op4)"]
    op_tex  = [r"\textsc{Op1}", r"\textsc{Op2}", r"\textsc{Op3}", r"\textsc{Op4}"]

    parts = []
    parts.append(
        f"For {label}, the Q-Learning agent converged to a policy in which "
        f"{op_tex[ep]} dominates in the early search phase "
        f"({ef*100:.0f}\\% of early states), "
        f"while {op_tex[lp]} becomes preferred in the final iterations "
        f"({lf*100:.0f}\\% of late states)."
    )
    if ep != lp:
        parts.append(
            "This shift mirrors a classical diversification-to-intensification "
            "transition: broadly exploring the neighbourhood early, then "
            "refining with finer moves near termination."
        )
    if sp != hp:
        parts.append(
            f"Regarding stagnation, the agent selects {op_tex[sp]} under low stagnation "
            f"({sf*100:.0f}\\%) but switches to {op_tex[hp]} when stagnation is high "
            f"({hf*100:.0f}\\%), effectively acting as an on-line escape mechanism "
            "without requiring explicit restarts."
        )
    else:
        parts.append(
            "Stagnation level does not trigger an operator switch; "
            "the ILS perturbation alone provides escape from local optima."
        )
    if hg != lg:
        parts.append(
            f"Solution quality also modulates behaviour: {op_tex[hg]} is preferred "
            f"when the gap is large ({hgf*100:.0f}\\% of high-gap states), "
            f"whereas {op_tex[lg]} is chosen near the lower bound "
            f"({lgf*100:.0f}\\% of near-optimal states), "
            "suggesting targeted intensification as the agent approaches optimality."
        )
    return " ".join(parts)


def _to_latex_paragraph(para: str, label: str) -> str:
    """Wraps the paragraph in a minimal LaTeX block."""
    lines = [
        "% Auto-generated by analysis/xai.py -- policy_report()",
        f"% Instance group: {label}",
        "",
        para,
        "",
    ]
    return "\n".join(lines)


def generate_policy_reports(
    qtables_dir: Path | str,
    instance_groups: list[tuple[str, str]],
    out_dir: Path | str,
) -> list[Path]:
    """
    Generates policy reports for multiple instance groups.

    Parameters
    ----------
    qtables_dir : Path | str
    instance_groups : list of (instance_filter, instance_label) tuples
    out_dir : Path | str

    Returns
    -------
    paths : list of Path -- paths to .txt report files
    """
    out_dir = Path(out_dir)
    paths = []
    for filt, label in instance_groups:
        try:
            Q, _ = load_qtables(qtables_dir, filt)
            policy_report(Q, label, out_dir=out_dir, verbose=False)
            safe = label.replace(",", "").replace(" ", "_").replace("=", "")
            paths.append(out_dir / f"policy_report_{safe}.txt")
        except FileNotFoundError as exc:
            print(f"[WARN] Skipping {label}: {exc}")
    return paths
