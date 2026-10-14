"""
analysis/plots.py
=================
Convergence curves and GAP boxplots for the PMSP Learnheuristic experiment.

Main functions
--------------
load_results(csv_path)
    Loads results.csv into a pandas DataFrame with proper dtypes.

plot_gap_boxplot(df, out_dir, instance_configs)
    One figure per instance size: boxplot of GAP% by method.

plot_gap_boxplot_summary(df, out_dir)
    Single figure: mean GAP% per method aggregated across all instances.

plot_time_boxplot(df, out_dir)
    Boxplot of CPU time per method x instance size.

Notes
-----
plot_convergence() reads its data from a separate JSON file produced by
experiment/convergence_entrada75.py, not from results.csv -- the
production runner (experiment/runner.py) never passes track_history=True,
so results.csv never contains per-iteration history (see that script's
docstring for the full rationale, Entrada 75).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Colour palette  (consistent with xai.py operator colours)
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "lpt":        "#7F7F7F",
    "ils_random": "#4C72B0",
    "ils_ucb":    "#DD8452",
    "ils_rf":     "#55A868",
    "ils_ql":     "#C44E52",
    "ils_rand_hi": "#E67E22",
    "ils_ql_ws":  "#9B59B6",
    "ils_ql_hi":  "#E74C3C",
}

METHOD_LABELS = {
    "lpt":        "LPT",
    "ils_random": "ILS-Rand",
    "ils_ucb":    "ILS-UCB",
    "ils_rf":     "ILS-RF",
    "ils_ql":     "ILS-QL",
    "ils_rand_hi": "ILS-Rand-HI",
    "ils_ql_ws":  "ILS-QL-WS",
    "ils_ql_hi":  "ILS-QL-HI",
}

METHOD_ORDER = ["lpt", "ils_random", "ils_ucb", "ils_rf", "ils_ql", "ils_rand_hi", "ils_ql_ws", "ils_ql_hi"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(csv_path: Path | str) -> pd.DataFrame:
    """
    Loads results.csv and casts columns to appropriate types.

    Expected columns:
        instance, method, rep, n, m, lb, makespan,
        gap_pct, cpu_time, n_improvements
    """
    df = pd.read_csv(csv_path)
    df["n"]   = df["n"].astype(int)
    df["m"]   = df["m"].astype(int)
    df["rep"] = df["rep"].astype(int)
    df["gap_pct"]   = df["gap_pct"].astype(float)
    df["cpu_time"]  = df["cpu_time"].astype(float)
    df["makespan"]  = df["makespan"].astype(int)
    df["lb"]        = df["lb"].astype(int)
    # instance size label  e.g. "n=20, m=2"
    df["size_label"] = df.apply(lambda r: f"n={r.n}, m={r.m}", axis=1)
    return df


# ---------------------------------------------------------------------------
# GAP boxplot per instance size
# ---------------------------------------------------------------------------

def _size_label_order(df: pd.DataFrame) -> list[str]:
    """Returns size_label values sorted by (n, m)."""
    sizes = df[["n", "m", "size_label"]].drop_duplicates()
    sizes = sizes.sort_values(["n", "m"])
    return sizes["size_label"].tolist()


def plot_gap_boxplot(
    df: pd.DataFrame,
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    One figure with one panel per (n, m) configuration.
    Each panel: boxplot of GAP% across 30 reps, one box per method.
    Methods that are always 0 (LPT on easy instances) show a flat line.

    Returns
    -------
    Path to saved PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_labels = _size_label_order(df)
    n_sizes = len(size_labels)
    ncols = 3
    nrows = (n_sizes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]

    for idx, slabel in enumerate(size_labels):
        ax = axes[idx]
        sub = df[df["size_label"] == slabel]
        data_per_method = [
            sub[sub["method"] == m]["gap_pct"].values
            for m in methods_present
        ]
        colors = [METHOD_COLORS[m] for m in methods_present]
        labels = [METHOD_LABELS[m] for m in methods_present]

        bp = ax.boxplot(
            data_per_method,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(slabel, fontsize=10)
        ax.set_xticks(range(1, len(methods_present) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("GAP (%)", fontsize=8)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide unused panels
    for idx in range(n_sizes, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("GAP% Distribution by Method and Instance Size\n(30 repetitions)", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / "gap_boxplot_by_size.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary boxplot: all instance sizes aggregated
# ---------------------------------------------------------------------------

def plot_gap_boxplot_summary(
    df: pd.DataFrame,
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    Single boxplot with one box per method, aggregating all instances.
    Good for a quick visual overview in the paper.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    data_per_method = [df[df["method"] == m]["gap_pct"].values for m in methods_present]
    colors = [METHOD_COLORS[m] for m in methods_present]
    labels = [METHOD_LABELS[m] for m in methods_present]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        data_per_method,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(methods_present) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("GAP (%)", fontsize=11)
    ax.set_title("GAP% Distribution -- All Instances (30 reps each)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    plt.tight_layout()
    out_path = out_dir / "gap_boxplot_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Mean GAP bar chart with error bars
# ---------------------------------------------------------------------------

def plot_mean_gap_bar(
    df: pd.DataFrame,
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    Grouped bar chart: mean GAP% per method per instance size.
    Error bars = 1 standard deviation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    size_labels = _size_label_order(df)

    n_sizes   = len(size_labels)
    n_methods = len(methods_present)
    x         = np.arange(n_sizes)
    width     = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(8, n_sizes * 1.8), 5))

    for i, m in enumerate(methods_present):
        sub  = df[df["method"] == m]
        means = [sub[sub["size_label"] == sl]["gap_pct"].mean() for sl in size_labels]
        stds  = [sub[sub["size_label"] == sl]["gap_pct"].std()  for sl in size_labels]
        offsets = x + (i - n_methods / 2 + 0.5) * width
        ax.bar(
            offsets, means, width * 0.9,
            yerr=stds, capsize=3,
            color=METHOD_COLORS[m], alpha=0.85,
            label=METHOD_LABELS[m],
            error_kw=dict(elinewidth=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean GAP (%)", fontsize=10)
    ax.set_title("Mean GAP% by Method and Instance Size\n(error bars = 1 std dev, 30 reps)", fontsize=10)
    ax.legend(fontsize=9, ncol=n_methods)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = out_dir / "mean_gap_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CPU time boxplot
# ---------------------------------------------------------------------------

def plot_time_boxplot(
    df: pd.DataFrame,
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    Boxplot of CPU time (seconds) per method, one panel per instance size.
    LPT is extremely fast so it will appear near zero.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_labels = _size_label_order(df)
    n_sizes = len(size_labels)
    ncols = 3
    nrows = (n_sizes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]

    for idx, slabel in enumerate(size_labels):
        ax = axes[idx]
        sub = df[df["size_label"] == slabel]
        data_per_method = [
            sub[sub["method"] == m]["cpu_time"].values
            for m in methods_present
        ]
        colors = [METHOD_COLORS[m] for m in methods_present]
        labels = [METHOD_LABELS[m] for m in methods_present]

        bp = ax.boxplot(
            data_per_method,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(slabel, fontsize=9)
        ax.set_xticks(range(1, len(methods_present) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("CPU time (s)", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for idx in range(n_sizes, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("CPU Time by Method and Instance Size (30 repetitions)", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / "time_boxplot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Profile plot: rank of methods per instance
# ---------------------------------------------------------------------------

def plot_rank_profile(
    df: pd.DataFrame,
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    For each instance (identified by instance name + rep), rank the ILS
    methods by GAP%.  Plots the fraction of instances where each method
    achieves rank 1 (best), 2, 3, 4 -- excluding LPT baseline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ils_methods = [m for m in METHOD_ORDER[1:] if m in df["method"].unique()]
    if not ils_methods:
        return None

    # Pivot: rows = (instance, rep), cols = methods
    pivot = df[df["method"].isin(ils_methods)].pivot_table(
        index=["instance", "rep"],
        columns="method",
        values="gap_pct",
    )

    rank_counts = {m: [0] * len(ils_methods) for m in ils_methods}
    total = len(pivot)
    for _, row in pivot.iterrows():
        ranked = row.sort_values().index.tolist()
        for rank_0, m in enumerate(ranked):
            rank_counts[m][rank_0] += 1

    # Convert to fractions
    x = np.arange(1, len(ils_methods) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ils_methods:
        fracs = [c / total for c in rank_counts[m]]
        ax.plot(x, fracs, marker="o", label=METHOD_LABELS[m],
                color=METHOD_COLORS[m], linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {i}" for i in x], fontsize=10)
    ax.set_ylabel("Fraction of instances", fontsize=10)
    ax.set_title("Method Ranking Profile (ILS variants, all instances)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = out_dir / "rank_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Convergence curves (Entrada 75) -- separate data source (see module
# docstring above and experiment/convergence_entrada75.py), NOT
# results.csv. Isolated color/label dicts on purpose: ig_ql/sa_li_perkins/
# bco are NOT added to METHOD_COLORS/METHOD_LABELS/METHOD_ORDER above,
# so plot_gap_boxplot/plot_mean_gap_bar/plot_time_boxplot/plot_rank_profile
# keep their current (already-in-use) behaviour unchanged.
# ---------------------------------------------------------------------------

CONVERGENCE_COLORS = {
    "ig_ql":         "#1F77B4",
    "ils_ql":        "#C44E52",
    "sa_li_perkins": "#2CA02C",
    "bco":           "#8C564B",
}

CONVERGENCE_LABELS = {
    "ig_ql":         "IG-QL",
    "ils_ql":        "ILS-QL",
    "sa_li_perkins": "SA (Li & Perkins, 2024)",
    "bco":           "BCO (Davidovic et al., 2012)",
}


def plot_convergence(
    data: list[dict],
    out_dir: Path | str,
    show: bool = False,
) -> Path:
    """
    Convergence curves (best-so-far GAP% vs. iteration) for a small set
    of representative instances, one panel per instance size.

    Parameters
    ----------
    data : list of dict
        Each element: {"n", "m", "lb", "seed", "max_iter",
        "curves": {method: {"history_gap": [...], "cpu_time": ...,
        "n_iters_ran": int}}} -- the exact structure saved by
        experiment/convergence_entrada75.py to
        results/convergence/convergence_entrada75.json. Any subset of
        methods present in "curves" is plotted (falls back to the raw
        method name as label/default color cycling if a method is not
        in CONVERGENCE_LABELS/CONVERGENCE_COLORS).

        "n_iters_ran" (optional; treated as len(history_gap) if absent,
        i.e. the whole curve solid) marks where a method's search
        ACTUALLY stopped, for methods with an early-stopping rule --
        e.g. sa_li_perkins stops once makespan <= lb*p_target (see
        pmsp/sa.py docstring). Points at index >= n_iters_ran are the
        correct "best-so-far" value forward-filled from the stopping
        point (provably constant, since no further search happened)
        but are drawn with a DASHED line so the figure never implies
        more search happened than actually did.
    out_dir : Path or str
    show : bool

    Returns
    -------
    Path to saved PNG (out_dir / "fig_convergence.png").
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_panels = len(data)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    legend_handles, legend_labels = None, None
    for ax, entry in zip(axes, data):
        n, m = entry["n"], entry["m"]
        for method, curve in entry["curves"].items():
            hist = curve["history_gap"]
            if not hist:
                continue
            n_ran = curve.get("n_iters_ran", len(hist))
            color = CONVERGENCE_COLORS.get(method, None)
            label = CONVERGENCE_LABELS.get(method, method)

            x_solid = np.arange(1, n_ran + 1)
            ax.plot(x_solid, hist[:n_ran], label=label, color=color,
                    linewidth=1.6)

            if n_ran < len(hist):
                # Trecho forward-filled apos parada antecipada (ver
                # docstring acima) -- tracejado, para nao sugerir busca
                # real. Comeca em n_ran-1 (nao n_ran) para a linha ficar
                # visualmente continua com o trecho solido.
                start = max(n_ran - 1, 0)
                x_dashed = np.arange(start + 1, len(hist) + 1)
                ax.plot(x_dashed, hist[start:], color=color, linewidth=1.6,
                        linestyle="--", alpha=0.7)
        ax.set_title(f"n={n}, m={m}", fontsize=10)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("GAP (%)", fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Legenda UNICA compartilhada acima dos paineis (nao uma por eixo):
    # os 3 paineis usam sempre os mesmos metodos/cores, e uma legenda
    # por eixo (loc='best') podia sobrepor uma curva perfeitamente
    # horizontal (caso real observado em n=100,m=20 -- sa_li_perkins
    # nao melhora nenhuma vez em 1000 iteracoes ali).
    fig.suptitle("Convergence: best-so-far GAP% vs. iteration", fontsize=11, y=1.02)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center",
                   bbox_to_anchor=(0.5, 0.98), ncol=len(legend_labels),
                   fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    out_path = out_dir / "fig_convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def generate_all_plots(
    csv_path: Path | str,
    out_dir: Path | str,
) -> list[Path]:
    """
    Loads results.csv and generates all standard plots.
    Returns list of paths to saved PNG files.
    """
    df = load_results(csv_path)
    paths = []
    paths.append(plot_gap_boxplot(df, out_dir))
    paths.append(plot_gap_boxplot_summary(df, out_dir))
    paths.append(plot_mean_gap_bar(df, out_dir))
    paths.append(plot_time_boxplot(df, out_dir))
    p = plot_rank_profile(df, out_dir)
    if p:
        paths.append(p)
    return paths
