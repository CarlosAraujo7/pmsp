"""Figura 1 do artigo do CEC (cec/fig_cec4.pdf): folga do oraculo x ganho do Q-Learning a iteracoes iguais, por celula: 48 sinteticas (P||Cmax), 36 publicadas (descoberta e held-out) e 18 do flowshop. Uso: python -m analysis.make_fig_cec4"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

allx, ally = [], []
for sp in ("S3", "S5", "S15", "S30"):
    d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
    orc = d[d.method.str.startswith("arm")].groupby(["cls", "rep"]).makespan.min().rename("oracle").reset_index()
    w = ms.merge(orc, on=["cls", "rep"])
    g = pd.DataFrame({"h": -100 * (w.oracle - w.rand) / w.rand, "gain": -100 * (w.ql - w.rand) / w.rand, "cls": w.cls}).groupby("cls").mean()
    allx += list(g.h)
    ally += list(g.gain)
rho_syn = st.spearmanr(allx, ally)

fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.55))
ax.scatter(allx, ally, s=9, color="#b0b0b0", marker="o", label=f"synthetic ($\\rho={rho_syn.statistic:.2f}$)", zorder=1)
PC = pd.read_csv(R / "entrada192_published_cells.csv")
FAMC = {"B": ("#1f6fb2", "s"), "C": ("#c0392b", "o"), "E": ("#3a9a92", "^")}
for fam, (col, mk) in FAMC.items():
    for st_, fill in (("discovery", False), ("replication", True)):
        x = PC[(PC.fam == fam) & (PC.set == st_)]
        ax.scatter(-x.folga, -x.ql, s=17, marker=mk, facecolors=col if fill else "none", edgecolors=col, linewidths=0.9, zorder=3)
FC = pd.read_csv(R / "entrada197_pfsp_cells.csv")
FC = FC[FC.set == "syn"]
ax.scatter(FC.folga, FC.gq, s=14, marker="x", color="#7d3c98", linewidths=0.9, zorder=3)
rho_f = [st.spearmanr(FC[FC.space == s_].folga, FC[FC.space == s_].gq).statistic for s_ in ("F5", "F15")]
rep = PC[PC.set == "replication"]
rho_rep = st.spearmanr(-rep.folga, -rep.ql)
ax.scatter([], [], s=14, marker="x", color="#7d3c98", label=f"flowshop ($\\rho={rho_f[0]:.2f}$, ${rho_f[1]:.2f}$)")
ax.scatter([], [], s=17, marker="o", facecolors="#c0392b", edgecolors="#c0392b", label=f"published, held-out ($\\rho={rho_rep.statistic:.2f}$)")
rho_dsc = st.spearmanr(-PC[PC.set == "discovery"].folga, -PC[PC.set == "discovery"].ql)
ax.scatter([], [], s=17, marker="o", facecolors="none", edgecolors="#c0392b", label=f"published, discovery ($\\rho={rho_dsc.statistic:.2f}$)")
for fam, (col, mk) in FAMC.items():
    ax.scatter([], [], s=17, marker=mk, color=col, label=f"family {fam}")
ax.plot([0, 0.43], [0, 0.43], color="k", lw=0.5, ls="--")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Headroom of the oracle over random choice (pp)")
ax.set_ylabel("Gain of Q-Learning, equal iterations (pp)")
ax.legend(fontsize=5.6, frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "cec" / "fig_cec4.pdf")
print("ok -> cec/fig_cec4.pdf; rho sintetico", round(rho_syn.statistic, 3), "held-out", round(rho_rep.statistic, 3), "flowshop", [round(v, 3) for v in rho_f])
