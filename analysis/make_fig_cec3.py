"""Figura 3 do artigo do CEC (cec/fig_cec3.pdf): 'a aprendizagem poderia ter aparecido?' -- (a) poder do teste por efeito plantado; (b) controle positivo: fracao de escolhas da acao plantada; (c) horizonte longo: ganho dos aprendizes e vantagem da acao validada por T.
Le results/entrada184_power.csv, results/plant_entrada185.csv e os CSV do relatorio da frente C (results/entrada185_long_*.csv). Uso: python -m analysis.make_fig_cec3"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
COL = {"ql": "#c0392b", "ucb": "#1f6fb2", "pg": "#3a9a92", "pm": "#8e6bb0", "ap": "#e08e0b", "alns": "#6b6b6b"}
NAME = {"ql": "Q-Learning", "ucb": "UCB1", "pg": "Policy grad.", "pm": "Prob. matching", "ap": "Adaptive pursuit", "alns": "ALNS roulette"}

fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.15), gridspec_kw={"width_ratios": [1, 1, 1.05]})

# (a) poder empirico por efeito plantado
P = pd.read_csv(R / "entrada184_power.csv")
deltas = [float(c.split("_")[1]) for c in P.columns if c.startswith("power_")]
M = P[[f"power_{d}" for d in deltas]].to_numpy()
clean = P.mean_d.abs() < 0.005
for i in range(len(P)):
    ax[0].plot(deltas, M[i], color="#bbbbbb", lw=0.7, zorder=1)
ax[0].plot(deltas, np.median(M[clean.to_numpy()], axis=0), color="#1f6fb2", lw=1.8, marker="o", ms=3, zorder=3, label="median (10 comparisons)")
ax[0].axhline(0.8, color="k", lw=0.5, ls="--")
ax[0].axvline(0.016, color="#c0392b", lw=0.8, ls=":")
ax[0].text(0.0165, 0.04, "largest observed\nlearner gain", color="#c0392b", fontsize=6.3, va="bottom")
ax[0].set_xlabel("planted policy gain (pp)")
ax[0].set_ylabel("detection rate (Wilcoxon, Holm 12)")
ax[0].set_title("(a) Power of the test", fontsize=8.5)
ax[0].set_ylim(0, 1.03)
ax[0].legend(fontsize=6.3, frameon=False, loc="upper left")

# (b) controle positivo: acao plantada, modo 1, S5
D = pd.read_csv(R / "plant_entrada185.csv")
D = D[(D.space == "S5") & (D["mode"] == 1)]
g = D.groupby(["learner", "bonus"]).apply(lambda x: x.hit2.sum() / (len(x) * (x["T"].iloc[0] - x["T"].iloc[0] // 2)) * x.A.iloc[0], include_groups=False).reset_index(name="x")
for k in COL:
    s = g[g.learner == k].sort_values("bonus")
    ax[1].plot(s.bonus, s.x, color=COL[k], marker="o", ms=3, lw=1.2, label=NAME[k])
ax[1].axhline(1.0, color="k", lw=0.5, ls="--")
ax[1].set_xscale("log")
from matplotlib.ticker import NullFormatter
ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[1].set_ylim(-1.5, 5.4)
ax[1].set_yticks([0, 1, 2, 3, 4, 5])
ax[1].set_xticks([0.5, 1, 2, 5])
ax[1].set_xticklabels(["0.5", "1", "2", "5"])
ax[1].set_xlabel("planted reward bonus (makespan units)")
ax[1].set_ylabel("choices of the planted action\n(multiples of uniform; max 5)")
ax[1].set_title("(b) Positive control", fontsize=8.5)
ax[1].legend(fontsize=5.8, frameon=False, loc="lower center", ncol=3, columnspacing=0.6, handlelength=1.2)

# (c) horizonte longo
A = pd.read_csv(R / "entrada185_long_aiter.csv")
B = pd.read_csv(R / "entrada185_long_bval.csv")
A0 = pd.read_csv(R / "entrada180_agents_aiter.csv")      # T = 1000 (Entrada 180, mesmas instancias)
A0 = A0[A0.id.str.startswith("Aiter-S5-")].assign(id=lambda d: d.id.str.replace("Aiter-S5-", "Aiter-T1000-"))
B0 = pd.read_csv(R / "entrada180_agents_b.csv")
B0 = B0[B0.id.str.startswith("B-S5-")].assign(id=lambda d: d.id.str.replace("B-S5-", "Bval-T1000-"))
A, B = pd.concat([A0, A], ignore_index=True), pd.concat([B0, B], ignore_index=True)
A["T"] = A.id.str.extract(r"T(\d+)").astype(int)
B["T"] = B.id.str.extract(r"T(\d+)").astype(int)
A["k"] = A.id.str.split("-").str[-1]
B["k"] = B.id.str.split("-").str[-1]
Ts = sorted(A["T"].unique())
for k in ("ql", "ucb", "pg"):
    a = A[A.k == k].sort_values("T")
    ax[2].plot(a["T"], -a.mean_d_pct, color=COL[k], marker="o", ms=3, lw=1.2, label=f"{NAME[k]} vs random")
b = B.groupby("T").mean_d_pct.mean()
ax[2].plot(b.index, -b.values, color="k", marker="s", ms=3, lw=1.2, ls="--", label="validated vs learners (mean)")
ax[2].axhline(0, color="k", lw=0.5)
ax[2].axhspan(-0.05, 0.05, color="#dddddd", alpha=0.5, zorder=0)
ax[2].set_xscale("log")
ax[2].set_xlabel("iterations $T$")
ax[2].set_ylabel("gain (pp; positive: helps)")
ax[2].set_title("(c) Longer horizons", fontsize=8.5)
ax[2].set_ylim(-0.02, 0.135)
ax[2].legend(fontsize=5.8, frameon=False, loc="upper right")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "cec" / "fig_cec3.pdf")
print("ok -> cec/fig_cec3.pdf")
