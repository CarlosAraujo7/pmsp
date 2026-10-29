"""Figura 1 do artigo do CEC (cec/fig_cec6.pdf, uma coluna): folga do oraculo COM VALIDACAO CRUZADA de sementes x ganho do Q-Learning a iteracoes iguais, por celula, nos tres hospedeiros (P||Cmax sintetico e publicado; flowshop; algoritmo genetico), com o limiar unico h* (Entrada 210). Inclui as 24 celulas de tempos finos (Entrada 228). Uso: python -m analysis.make_fig_cec6"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
C = pd.read_csv(R / "entrada209_cforacle_cells.csv")
G = pd.read_csv(R / "entrada219_ga_cells.csv")
F = pd.read_csv(R / "entrada228_cf_fine_cells.csv")
hstar = float(re.search(r"limiar h\* escolhido na descoberta \(acuracia balanceada [\d.]+\): ([\d.]+)", (R / "entrada209_cforacle.txt").read_text(encoding="utf-8")).group(1))
STYLE = [
    ("P||Cmax synthetic", (C.problem == "pcmax") & (C.set == "syn"), "#b0b0b0", "o", True),
    ("published, discovery", (C.set == "discovery"), "#c0392b", "o", False),
    ("published, held-out", (C.set == "heldout"), "#c0392b", "o", True),
    ("flowshop, synthetic", (C.problem == "pfsp") & (C.set == "syn"), "#7d3c98", "x", True),
    ("flowshop, Taillard", (C.problem == "pfsp") & (C.set == "ta"), "#7d3c98", "+", True),
]
fig, a = plt.subplots(1, 1, figsize=(3.5, 1.82))
for lab, msk, col, mk, fill in STYLE:
    x = C[msk]
    if mk in ("x", "+"):
        a.scatter(x.h_cf, x.gq, s=14, marker=mk, color=col, linewidths=0.9, zorder=3, label=lab)
    else:
        a.scatter(x.h_cf, x.gq, s=14, marker=mk, facecolors=col if fill else "none", edgecolors=col, linewidths=0.8, zorder=3 if lab != "P||Cmax synthetic" else 1, label=lab)
a.scatter(G.h_cf, G.ql, s=16, marker="^", facecolors="none", edgecolors="#1f6fb2", linewidths=0.8, zorder=3, label="genetic algorithm")
a.scatter(F.h_cf, F.gq, s=14, marker="s", facecolors="none", edgecolors="#e67e22", linewidths=0.8, zorder=3, label="finer times")
a.axhline(0, color="k", lw=0.5)
a.axvline(hstar, color="k", lw=0.5, ls=":")
a.text(hstar + 0.005, 0.455, f"$h^*={hstar:.3f}$", fontsize=6.2, va="top")
a.set_xlabel("Cross-fitted oracle headroom over random choice (pp)")
a.set_ylabel("Q-Learning gain (pp)")
a.legend(fontsize=5.4, frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.93))
a.spines[["top", "right"]].set_visible(False)
a.set_xlim(-0.32, 0.45)
a.set_ylim(-0.32, 0.45)
off = int(((G.h_cf < -0.32) | (G.ql < -0.32) | (G.h_cf > 0.45)).sum())
a.text(0.44, -0.30, f"{off} GA cells off scale", fontsize=5.6, ha="right")
fig.tight_layout()
fig.savefig(ROOT / "cec" / "fig_cec6.pdf")
print("ok -> cec/fig_cec6.pdf; off-scale GA cells", off, "; h* =", hstar, "; GA cells", len(G), "gain range", round(G.ql.min(), 3), round(G.ql.max(), 3))
