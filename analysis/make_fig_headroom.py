"""Figure: headroom of the actions versus policy gain, per cell (type x n), on the standard R||Cmax benchmark.

Panel A: actions = destruction strategies (Entrada 137; law confirmed, rho ~ 0.74).
Panel B: actions = destruction size (Entrada 145; prospective test, law refuted, rho ~ -0.37).
Data: results/entrada137_rcmax_benchmark_cells.csv (columns cell, hd, pol) and
      results/entrada145_dpolicy_cells.csv (columns cell, hd, gain); hd and the policy column are mean
      relative makespan differences (%) with respect to random selection (negative = better than random).
Output: paper/fig_headroom.pdf and results/fig_headroom_rho.txt (the correlations, checked by the audit).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

a = pd.read_csv(RES / "entrada137_rcmax_benchmark_cells.csv").rename(columns={"pol": "gain"})
b = pd.read_csv(RES / "entrada145_dpolicy_cells.csv")

types = sorted(set(c.split("|")[0] for c in a.cell))
markers = ["o", "s", "^", "v", "D", "P", "X"]
mk = {t: markers[i % len(markers)] for i, t in enumerate(types)}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
lines = []
for ax, df, title in [(axes[0], a, "A. Actions: destruction strategies"), (axes[1], b, "B. Actions: destruction size")]:
    x = -df.hd  # headroom: gain of the per-instance oracle over random selection (% of makespan)
    y = -df.gain  # policy gain over random selection (% of makespan); negative = policy worse than random
    rho, p_two = stats.spearmanr(x, y)
    for t in types:
        m = df.cell.str.startswith(t + "|")
        ax.scatter(x[m], y[m], marker=mk[t], s=22, facecolor="none", edgecolor="C0", linewidths=0.9, label=t)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel("headroom: oracle gain over random (%)", fontsize=8.5)
    ax.set_title(title, fontsize=8.5)
    ax.text(0.04, 0.95, f"Spearman $\\rho$ = {rho:.2f}".replace("-", "−") + f"\n({len(df)} cells)", transform=ax.transAxes, va="top", fontsize=8)
    ax.tick_params(labelsize=8)
    lines.append(f"{title}: n={len(df)} rho={rho:.4f} p_two={p_two:.3g}")
axes[0].set_ylabel("policy gain over random (%)", fontsize=8.5)
axes[1].legend(fontsize=6.5, loc="lower right", frameon=False, ncol=2, title="instance type", title_fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / "paper" / "fig_headroom.pdf")
(RES / "fig_headroom_rho.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
