"""Figura 1 do artigo do CEC (cec/fig_cec5.pdf): folga do oraculo COM VALIDACAO CRUZADA DE SEMENTES x ganho do Q-Learning a iteracoes iguais, por celula, nos tres hospedeiros disponiveis (P||Cmax sintetico e publicado; flowshop; hospedeiro populacional quando existir), com o limiar unico h* (Entrada 210) e a mesma figura com a folga enviesada ao lado. Uso: python -m analysis.make_fig_cec5"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
C = pd.read_csv(R / "entrada209_cforacle_cells.csv")
GA = R / "entrada219_ga_cells.csv"
hstar = None
txt = (R / "entrada209_cforacle.txt").read_text(encoding="utf-8")
import re  # noqa: E402

hstar = float(re.search(r"limiar h\* escolhido na descoberta \(acuracia balanceada [\d.]+\): ([\d.]+)", txt).group(1))

STYLE = [
    ("P||Cmax synthetic", (C.problem == "pcmax") & (C.set == "syn"), "#b0b0b0", "o", True),
    ("published, discovery", (C.set == "discovery"), "#c0392b", "o", False),
    ("published, held-out", (C.set == "heldout"), "#c0392b", "o", True),
    ("flowshop, synthetic", (C.problem == "pfsp") & (C.set == "syn"), "#7d3c98", "x", True),
    ("flowshop, Taillard", (C.problem == "pfsp") & (C.set == "ta"), "#7d3c98", "+", True),
]
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
for k, (hcol, title) in enumerate((("h_cf", "(a) Cross-fitted oracle (seeds that choose differ from seeds that evaluate)"), ("h_biased", "(b) Best of one seed per action (biased)"))):
    a = ax[k]
    for lab, msk, col, mk, fill in STYLE:
        x = C[msk]
        if mk in ("x", "+"):
            a.scatter(x[hcol], x.gq, s=14, marker=mk, color=col, linewidths=0.9, zorder=3, label=lab if k == 0 else None)
        else:
            a.scatter(x[hcol], x.gq, s=14, marker=mk, facecolors=col if fill else "none", edgecolors=col, linewidths=0.8, zorder=3 if lab != "P||Cmax synthetic" else 1, label=lab if k == 0 else None)
    if GA.exists():
        G = pd.read_csv(GA)
        gcol = "h_cf" if hcol == "h_cf" else "h_biased"
        a.scatter(G[gcol], G.ql, s=14, marker="^", facecolors="none", edgecolors="#1f6fb2", linewidths=0.8, zorder=3, label="genetic algorithm" if k == 0 else None)
    a.axhline(0, color="k", lw=0.5)
    if hcol == "h_cf":
        a.axvline(hstar, color="k", lw=0.5, ls=":")
        a.text(hstar, 0.46, f" $h^*={hstar:.3f}$", fontsize=6.2, va="top")
    a.set_xlabel("Headroom of the oracle over random choice (pp)")
    a.set_title(title, fontsize=6.8)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("Gain of Q-Learning, equal iterations (pp)")
ax[0].legend(fontsize=5.2, frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.88))
ax[0].set_ylim(-0.27, 0.48)
fig.tight_layout()
fig.savefig(ROOT / "cec" / "fig_cec5.pdf")
rows = []
for nm, msk in (("pcmax held-out", C.set == "heldout"), ("pcmax discovery", C.set == "discovery"), ("pcmax synthetic", (C.problem == "pcmax") & (C.set == "syn")), ("flowshop synthetic", (C.problem == "pfsp") & (C.set == "syn"))):
    x = C[msk]
    rows.append((nm, len(x), st.spearmanr(x.h_biased, x.gq).statistic, st.spearmanr(x.h_cf, x.gq).statistic))
print("ok -> cec/fig_cec5.pdf; h* =", hstar)
for r in rows:
    print(r[0], r[1], round(r[2], 2), round(r[3], 2))
