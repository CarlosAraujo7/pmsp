"""Figura 1 do artigo do CEC, nova versao (cec/fig_cec7.pdf, duas colunas): (a) ganho do Q-Learning padrao e (b) do Q-Learning com desempate aleatorio sobre o aleatorio a iteracoes iguais contra o ganho da acao validada, por celula (69 celulas,
eixos simlog com trecho linear em +-0,1; a diagonal e' 'ganho igual ao da validada', a pontilhada a razao de captura agrupada do desempate aleatorio, a faixa cinza a margem de +-0,05); (c) a parcela do ganho da validada que cada variante de ordem
captura na R||Cmax do benchmark padrao (21 celulas). Dados: results/entrada283_cells.csv, entrada283b_capture69.csv, entrada283_order.csv. Uso: python -m analysis.make_fig_cec7"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, FixedLocator  # noqa: E402

plt.rcParams.update({"font.size": 7.2, "axes.linewidth": 0.6, "pdf.fonttype": 42, "legend.frameon": False})
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
C = pd.read_csv(R / "entrada283_cells.csv")
C = C[C.coll != "tsp"]
CAP = pd.read_csv(R / "entrada283b_capture69.csv")
O = pd.read_csv(R / "entrada283_order.csv")
STY = [("padrao", "standard benchmark (R||Cmax, 21)", "#0072B2", "o"), ("familias", "published R||Cmax families (6)", "#E69F00", "s"),
       ("publicado", "published P||Cmax (18)", "#D55E00", "D"), ("sintetico", "synthetic P||Cmax (24)", "#8c8c8c", "^")]
LIN = 0.1
fig = plt.figure(figsize=(7.16, 2.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.34, left=0.065, right=0.995, bottom=0.18, top=0.93)


def style(a):
    a.set_xscale("symlog", linthresh=LIN, linscale=0.55)
    a.set_yscale("symlog", linthresh=LIN, linscale=0.55)
    ticks = [-1, -0.1, 0, 0.1, 1, 5]
    for ax in (a.xaxis, a.yaxis):
        ax.set_major_locator(FixedLocator(ticks))
        ax.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_minor_locator(FixedLocator([]))
    a.set_xlim(-0.12, 7)
    a.set_ylim(-1.6, 7)
    a.axhspan(-0.05, 0.05, color="#dddddd", zorder=0, lw=0)
    a.axhline(0, color="k", lw=0.4, zorder=1)
    a.axvline(0, color="k", lw=0.4, zorder=1)
    xs = np.concatenate([-np.logspace(-3, 0.1, 50)[::-1], [0], np.logspace(-3, 0.85, 80)])
    a.plot(xs, xs, "k--", lw=0.6, zorder=1)
    a.spines[["top", "right"]].set_visible(False)


def panel(a, col, title):
    style(a)
    for coll, lab, colr, mk in STY:
        d = C[C.coll == coll]
        a.scatter(d.g_val, d[col], s=11, marker=mk, color=colr, alpha=0.85, linewidths=0.3, edgecolors="white", zorder=3, label=lab)
    a.set_title(title, fontsize=7.4, loc="left", pad=2)
    a.set_xlabel("gain of the validated action (pp)")


a = fig.add_subplot(gs[0, 0])
panel(a, "g_ql", "(a) standard Q-Learning")
a.set_ylabel("gain of Q-Learning (pp)")
a.legend(loc="upper left", fontsize=5.6, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.15, bbox_to_anchor=(0.0, 1.0))
b = fig.add_subplot(gs[0, 1])
panel(b, "g_rt", "(b) ties broken at random")
cr = CAP[(CAP.grupo == "todas as 69") & (CAP.aprendiz == "desempate aleatorio")].iloc[0]
xs = np.logspace(-3, 0.85, 80)
b.plot(xs, cr.captura * xs, color="#1b7837", ls=":", lw=1.0, zorder=2)
b.text(-0.1, 6.0, f"dashed: equal gain" + chr(10) + f"dotted: pooled capture ({100 * cr.captura:.0f}%)", fontsize=5.8, color="k", ha="left", va="top")

c = fig.add_subplot(gs[0, 2])
V = [("padrao (lista original)", "standard\n(original list)", "#7a7a7a"), ("lista invertida", "reversed\nlist", "#b4b4b4"), ("validada em primeiro", "validated\naction first", "#0072B2"),
     ("permutada por execucao", "permuted\nper run", "#D55E00"), ("desempate aleatorio", "ties broken\nat random", "#1b7837")]
w = 0.16
for i, (key, lab, colr) in enumerate(V):
    d = O[O.variante == key].set_index("espaco").loc[["B5", "B15", "B30"]]
    c.bar(np.arange(3) + (i - 2) * w, d.captura.values, width=w * 0.92, color=colr, label=lab.replace("\n", " "), zorder=3)
c.axhline(0, color="k", lw=0.5)
c.set_xticks(range(3))
c.set_xticklabels(["5 actions", "15 actions", "30 actions"])
c.set_ylabel("share of validated gain captured")
c.set_title("(c) order of the actions", fontsize=7.4, loc="left", pad=2)
c.set_ylim(-0.5, 1.5)
c.spines[["top", "right"]].set_visible(False)
c.legend(loc="upper center", fontsize=5.4, ncol=2, handlelength=0.9, columnspacing=0.6, labelspacing=0.2, bbox_to_anchor=(0.5, 1.0))
fig.savefig(ROOT / "cec" / "fig_cec7.pdf")
print("ok -> cec/fig_cec7.pdf", "cells", len(C), "pooled capture rt", round(float(cr.captura), 3))
