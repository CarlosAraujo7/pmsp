"""
analysis/make_table_ties.py
Gera journal/tab_ties.tex a partir de results/entrada280_tiesall_summary.csv (Entrada 280: o Q-Learning com desempate aleatorio nas 85 celulas).
Uso: python -m analysis.make_table_ties
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
J = ROOT / "journal"
NAMES = {"padrao": "Standard benchmark (B5, B15, B30)", "familias": "Published families of $R||\\Cmax$", "sintetico": "Synthetic $P||\\Cmax$",
         "publicado": "Held-out published $P||\\Cmax$", "tsp": "TSP (iterated local search)"}


def main():
    S = pd.read_csv(R / "entrada280_tiesall_summary.csv").set_index("colecao")
    L = ["\\begin{table}[t]", "\\caption{The Q-Learning with ties broken at random at every decision (ql-rt) in the $85$ cells of the order diagnostic: mean gain over random choice at equal iterations (points of the makespan or tour length) of the standard learner, "
         "of the learner with the action list permuted once per run (ql-perm), of the random-tie learner and of the validated action; cells in which the random-tie learner gains beyond the margin, and cells in which the validated action beats it (Holm within each collection).}",
         "\\label{tab:ties}", "\\centering", "{\\scriptsize\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
         "Collection & Cells & Standard & ql-perm & ql-rt & Validated & ql-rt gains & Validated beats \\\\", "\\midrule"]
    for k in ("padrao", "familias", "sintetico", "publicado", "tsp"):
        r = S.loc[k]
        L.append(f"{NAMES[k]} & ${int(r.celulas)}$ & ${r.ql_padrao:.3f}$ & ${r.ql_perm:.3f}$ & ${r.ql_rt:.3f}$ & ${r.validada:.3f}$ & ${int(r.rt_alem_da_margem)}$ & ${int(r.validada_bate_rt)}$ \\\\")
    L += ["\\midrule", f"Total & ${int(S.celulas.sum())}$ & & & & & ${int(S.rt_alem_da_margem.sum())}$ & ${int(S.validada_bate_rt.sum())}$ \\\\", "\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    (J / "tab_ties.tex").write_text("\n".join(L), encoding="utf-8")
    print("tab_ties.tex escrito")


if __name__ == "__main__":
    main()
