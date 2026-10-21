"""Figura 1 do artigo (Entrada 162/164): onde esta o ganho? Gano de makespan (pontos percentuais; positivo = o componente ajuda) de quatro componentes em tres problemas.

Componentes: (1) mecanismo de busca (IG vs ILS, a CPU igual); (2) busca local (a CPU igual); (3) politica online sobre o tamanho de destruicao (contra a escolha ao acaso); (4) tamanho validado offline (contra a escolha ao acaso).
Valores (todos lidos de arquivos de resultados ja' auditados; a origem de cada um esta em SOURCES e no arquivo results/fig_thesis_values_entrada162.csv):
  P||Cmax : (1) E1 da Entrada 164 (ILS eficiente a CPU igual); (2) D1 da Entrada 164 (busca local rapida a CPU igual); (3) A2 da Entrada 114 (ig_ql sobre d vs d ao acaso); (4) A1 da Entrada 114 (d validado vs d ao acaso)
  R||Cmax : (1) B1 da Entrada 137 (IG com busca local vs ILS a CPU igual, 700 instancias do benchmark padrao); (2) nao medido a CPU igual; (3) D2 e (4) D1/D3 da Entrada 145 (validado por tipo e n)
  Flowshop: Taillard classico (Entrada 159): P1, P2, P3 e P5.
Uso: python -m analysis.make_fig_thesis
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
SOURCES = {}


def primary(path, id_, col="mean_d_pct"):
    d = pd.read_csv(R / path).set_index("id")
    return float(d.loc[id_, col])


rows = []


def add(problem, comp, gain, source):
    rows.append(dict(problem=problem, component=comp, gain_pp=gain, source=source))


# --- P||Cmax
add("Identical machines (P||Cmax)", "Search mechanism\n(IG vs ILS, equal CPU)", -primary("entrada162_ils_fast_primary.csv", "E1"), "entrada162_ils_fast_primary.csv:E1")
add("Identical machines (P||Cmax)", "Local search\n(equal CPU)", -primary("entrada162_igls_fast_primary.csv", "D1"), "entrada162_igls_fast_primary.csv:D1")
f15 = pd.read_csv(R / "entrada114_dsel_primary.csv").set_index("id") if (R / "entrada114_dsel_primary.csv").exists() else None
if f15 is not None:
    add("Identical machines (P||Cmax)", "Online policy\nvs random size", -float(f15.loc["A2", "mean_d_pct"]), "entrada114_dsel_primary.csv:A2")
    add("Identical machines (P||Cmax)", "Validated size\nvs random size", -float(f15.loc["A1", "mean_d_pct"]), "entrada114_dsel_primary.csv:A1")
# --- R||Cmax (benchmark padrao)
add("Unrelated machines (R||Cmax)", "Search mechanism\n(IG vs ILS, equal CPU)", -primary("entrada137_rcmax_benchmark_primary.csv", "B1"), "entrada137_rcmax_benchmark_primary.csv:B1")
d145 = pd.read_csv(R / "entrada145_dpolicy_primary.csv").set_index("id")
add("Unrelated machines (R||Cmax)", "Online policy\nvs random size", -float(d145.loc["D2", "mean_d_pct"]), "entrada145_dpolicy_primary.csv:D2")
val_line = [l for l in (R / "entrada145_dpolicy.txt").read_text(encoding="utf-8").splitlines() if l.startswith("Ganho sobre rand_d")]
val_tn = float([x for x in val_line[0].split(",") if "val_tn" in x][0].split()[-1])
add("Unrelated machines (R||Cmax)", "Validated size\nvs random size", -val_tn, "entrada145_dpolicy.txt:val_tn")
# --- flowshop (Taillard)
P = pd.read_csv(R / "entrada159_taillard_primary.csv").set_index("id")
add("Permutation flowshop (Taillard)", "Search mechanism\n(IG vs ILS, equal CPU)", -float(P.loc["P1", "mean_d_pct"]), "entrada159_taillard_primary.csv:P1")
add("Permutation flowshop (Taillard)", "Local search\n(equal CPU)", -float(P.loc["P2", "mean_d_pct"]), "entrada159_taillard_primary.csv:P2")
add("Permutation flowshop (Taillard)", "Online policy\nvs random size", -float(P.loc["P3", "mean_d_pct"]), "entrada159_taillard_primary.csv:P3")
add("Permutation flowshop (Taillard)", "Validated size\nvs random size", -float(P.loc["P5", "mean_d_pct"]), "entrada159_taillard_primary.csv:P5")

V = pd.DataFrame(rows)
V.to_csv(R / "fig_thesis_values_entrada162.csv", index=False)

comps = ["Search mechanism\n(IG vs ILS, equal CPU)", "Local search\n(equal CPU)", "Online policy\nvs random size", "Validated size\nvs random size"]
colors = {"Search mechanism\n(IG vs ILS, equal CPU)": "#1b6ca8", "Local search\n(equal CPU)": "#4aa3a2", "Online policy\nvs random size": "#c0392b", "Validated size\nvs random size": "#7f8c8d"}
probs = list(dict.fromkeys(V.problem))
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.4), sharey=True)
for ax, pr in zip(axes, probs):
    g = V[V.problem == pr].set_index("component")
    xs, vals, cols = [], [], []
    for i, c in enumerate(comps):
        if c in g.index:
            xs.append(i)
            vals.append(g.loc[c, "gain_pp"])
            cols.append(colors[c])
    ax.bar(xs, vals, color=cols, width=0.7)
    for i, c in enumerate(comps):
        if c not in g.index:
            ax.annotate("n/a", (i, 0), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.5, color="0.4")
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:+.3f}".replace("-", "−"), (x, v), xytext=(0, 3 if v >= 0 else -9), textcoords="offset points", ha="center", fontsize=6.5)
    ax.axhline(0, color="0.4", lw=0.7)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(["mech.", "local\nsearch", "online\npolicy", "validated\nsize"], fontsize=7)
    ax.set_title(pr, fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
axes[0].set_ylabel("gain in makespan (percentage points)\nsymmetric-log scale", fontsize=7.5)
axes[0].set_ylim(-0.4, 4)
fig.tight_layout()
fig.savefig(ROOT / "paper" / "fig_thesis.pdf")
print(V.to_string(index=False))
print("ok -> paper/fig_thesis.pdf")
