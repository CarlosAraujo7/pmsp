"""Figura do artigo do CEC (cec/fig_cec.pdf): (a) onde esta' o ganho em P||Cmax; (b) o ganho da busca local por convencao de orcamento e implementacao. Valores lidos de resultados ja' auditados. Uso: python -m analysis.make_fig_cec"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})

fv = pd.read_csv(R / "fig_thesis_values_entrada162.csv")
fv = fv[fv.problem.str.startswith("Identical")]
lab_a = ["Search\nmechanism\n(IG vs ILS)", "Local\nsearch", "Online\npolicy", "Validated\nsize"]
val_a = [float(v) for v in fv.gain_pp]
assert len(val_a) == 4

p124 = pd.read_csv(R / "entrada124_igls_p_primary.csv").set_index("id")
q129 = pd.read_csv(R / "entrada129_igls_eq_primary.csv").set_index("id")
p162 = pd.read_csv(R / "entrada162_igls_fast_primary.csv").set_index("id")
p170 = pd.read_csv(R / "entrada170_numba_primary.csv").set_index("id")
rows_b = [("Equal\niterations", p124.loc["P1"]), ("Equal CPU,\nloops", q129.loc["Q1"]), ("Equal CPU,\nvectorized", p162.loc["D1"]), ("Equal CPU,\ncompiled", p170.loc["C1"]), ("Equal\nevaluations", p170.loc["C2"])]
val_b = np.array([-r.mean_d_pct for _, r in rows_b])
lo = np.array([-r.ci90_hi for _, r in rows_b])
hi = np.array([-r.ci90_lo for _, r in rows_b])

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1, 1.25]})
cols_a = ["#1f6fb2", "#3a9a92", "#c0392b", "#7f8c8d"]
ax[0].bar(range(4), val_a, color=cols_a, width=0.62)
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_yscale("symlog", linthresh=0.01)
ax[0].set_xticks(range(4))
ax[0].set_xticklabels(lab_a)
ax[0].set_ylabel("Gain in makespan (percentage points)")
ax[0].set_title("(a) Where the gain comes from", fontsize=8.5)
for i, v in enumerate(val_a):
    ax[0].annotate(f"{v:+.3f}", (i, v), ha="center", va="bottom" if v > 0 else "top", fontsize=7, xytext=(0, 2 if v > 0 else -2), textcoords="offset points")
ax[0].set_ylim(-0.1, 1.2)
cols_b = ["#8e9aa5", "#c0392b", "#3a9a92", "#1f6fb2", "#e08e0b"]
ax[1].bar(range(5), val_b, color=cols_b, width=0.62, yerr=[val_b - lo, hi - val_b], error_kw=dict(lw=0.8, capsize=2))
ax[1].set_xticks(range(5))
ax[1].set_xticklabels([l for l, _ in rows_b])
ax[1].set_ylabel("Gain of the local search (pp)")
ax[1].set_title("(b) The same gain under five budgets", fontsize=8.5)
for i, v in enumerate(val_b):
    ax[1].annotate(f"{v:.3f}", (i, hi[i]), ha="center", va="bottom", fontsize=7, xytext=(0, 1.5), textcoords="offset points")
ax[1].set_ylim(0, 0.19)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(ROOT / "cec" / "fig_cec.pdf")
print("ok -> cec/fig_cec.pdf", [round(v, 3) for v in val_a], [round(v, 3) for v in val_b])

# ---------------------------------------------------------------- Figura 2: (a) vantagem do IG sobre o ILS por classe a CPU igual; (b) folga x ganho do Q-Learning em 48 celulas
import re as _re
txt = (R / "entrada162_ils_fast.txt").read_text(encoding="utf-8").split("por classe")[1]
cls = []
for ln in txt.splitlines():
    m = _re.match(r"\s*rand_(\d+)j_(\d+)m\s+(-?\d\.\d+)\s+(-?\d\.\d+)", ln)
    if m:
        cls.append((int(m.group(1)), int(m.group(2)), -float(m.group(3)), -float(m.group(4))))
cls.sort()
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw={"width_ratios": [1.15, 1]})
xs = np.arange(len(cls))
ax[0].bar(xs, [c[2] for c in cls], color="#1f6fb2", width=0.62, label="equal CPU")
ax[0].scatter(xs, [c[3] for c in cls], marker="D", s=14, color="#e08e0b", zorder=3, label="equal iterations")
ax[0].set_yscale("symlog", linthresh=0.01)
ax[0].set_xticks(xs)
ax[0].set_xticklabels([f"{c[0]},{c[1]}" for c in cls], rotation=60, fontsize=6.5)
ax[0].set_xlabel("class $(n,m)$")
ax[0].set_ylabel("Gain of IG over ILS (pp)")
ax[0].set_title("(a) Search mechanism, by class", fontsize=8.5)
ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
from scipy import stats as _st
allx, ally = [], []
for sp in ("S3", "S5", "S15", "S30"):
    d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
    orc = d[d.method.str.startswith("arm")].groupby(["cls", "rep"]).makespan.min().rename("oracle").reset_index()
    w = ms.merge(orc, on=["cls", "rep"])
    g = pd.DataFrame({"h": -100 * (w.oracle - w.rand) / w.rand, "gain": -100 * (w.ql - w.rand) / w.rand, "cls": w.cls}).groupby("cls").mean()
    allx += list(g.h)
    ally += list(g.gain)
rho_syn = _st.spearmanr(allx, ally)
ax[1].scatter(allx, ally, s=9, color="#b0b0b0", marker="o", label=f"synthetic ($\\rho={rho_syn.statistic:.2f}$)", zorder=1)
PC = pd.read_csv(R / "entrada192_published_cells.csv")
FAMC = {"B": ("#1f6fb2", "s"), "C": ("#c0392b", "o"), "E": ("#3a9a92", "^")}
for fam, (col, mk) in FAMC.items():
    for st_, fill in (("discovery", False), ("replication", True)):
        x = PC[(PC.fam == fam) & (PC.set == st_)]
        ax[1].scatter(-x.folga, -x.ql, s=17, marker=mk, facecolors=col if fill else "none", edgecolors=col, linewidths=0.9, zorder=3)
rep = PC[PC.set == "replication"]
rho_rep = _st.spearmanr(-rep.folga, -rep.ql)
ax[1].scatter([], [], s=17, marker="o", facecolors="#c0392b", edgecolors="#c0392b", label=f"published, held-out ($\\rho={rho_rep.statistic:.2f}$)")
ax[1].scatter([], [], s=17, marker="o", facecolors="none", edgecolors="#c0392b", label="published, discovery")
for fam, (col, mk) in FAMC.items():
    ax[1].scatter([], [], s=17, marker=mk, color=col, label=f"family {fam}")
ax[1].plot([0, 0.43], [0, 0.43], color="k", lw=0.5, ls="--")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel("Headroom of the oracle over random choice (pp)")
ax[1].set_ylabel("Gain of Q-Learning at equal iterations (pp)")
ax[1].set_title("(b) Headroom vs gain", fontsize=8.5)
ax[1].legend(fontsize=5.6, frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
print("rho sintetico", round(rho_syn.statistic, 3), "rho held-out", round(rho_rep.statistic, 3))
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "cec" / "fig_cec2.pdf")
print("ok -> cec/fig_cec2.pdf", len(cls), "classes;", len(allx), "celulas; rho", round(rho_syn.statistic, 3))
