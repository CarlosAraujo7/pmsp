"""
analysis/make_table_281_282.py
Gera journal/tab_scarce.tex (Entrada 281: poucos dados de ajuste) e journal/tab_cvrp.tex (Entrada 282: CVRP com a biblioteca de terceiros) a partir dos CSV de resultados.
Uso: python -m analysis.make_table_281_282
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
J = ROOT / "journal"
LEARN = ["srw", "aucb", "rw", "mab_eps", "mab_ucb1"]
NAMES = {"srw": "SRW", "aucb": "AlphaUCB", "rw": "RW", "mab_eps": "MAB-$\\varepsilon$", "mab_ucb1": "MAB-UCB1"}


def fmt(x, d=2):
    return f"${x:.{d}f}$"


def scarce():
    T = pd.read_csv(R / "entrada281_scarce_by_k.csv").set_index("k")
    L = ["\\begin{table}[t]", "\\caption{Where online learning could pay: scarce tuning data (standard benchmark, B30, $T=1000$). With $k$ of the $10$ tuning instances of each class ($4$ random subsets per $k$; $k=10$ is the full tuning), the static action, the optimistic Q-Learning warm-started "
         "with the prior of the same $k$ instances, the same prior without learning (learning rate zero) and the cold optimistic Q-Learning: mean gain over random choice in points (seven types), the differences, and the standard deviation across the four subsets of the mean gain of the static action and of the warm-started learner.}",
         "\\label{tab:scarce}", "\\centering", "{\\scriptsize\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{@{}rrrrrrrrr@{}}", "\\toprule",
         "$k$ & Static & Warm & Frozen & Cold & Warm$-$static & Warm$-$frozen & SD static & SD warm \\\\", "\\midrule"]
    for k in (1, 2, 3, 5, 10):
        r = T.loc[k]
        sds = (fmt(r.sd_static, 3), fmt(r.sd_warm, 3)) if k < 10 else ("---", "---")
        L.append(f"${k}$ & {fmt(r.static)} & {fmt(r.warm)} & {fmt(r.frozen)} & {fmt(r.cold)} & {fmt(r.warm_minus_static)} & {fmt(r.warm_minus_frozen)} & {sds[0]} & {sds[1]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    (J / "tab_scarce.tex").write_text("\n".join(L), encoding="utf-8")


def cvrp():
    C = pd.read_csv(R / "entrada282_cvrp_cells.csv")
    L = ["\\begin{table}[t]", "\\caption{The protocol on a third-party implementation (ALNS for the CVRP with the \\texttt{alns} library; $16$ heterogeneous actions, $T=1000$, library selectors with the values of its CVRP example, no tuning): gain over random choice in percent of the cost, "
         "per class (customers $n$; random R or clustered C; $6$ or $16$ customers per route), of the validated action and of the five library selectors (SRW: SegmentedRouletteWheel; RW: RouletteWheel; MAB: MABSelector), and the best selector as a percentage of the validated gain where the latter exceeds $0.3$.}",
         "\\label{tab:cvrp}", "\\centering", "{\\scriptsize\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
         "Class & Validated & " + " & ".join(NAMES[m] for m in LEARN) + " & Best/val. \\\\", "\\midrule"]
    for _, r in C.iterrows():
        best = max(r[f"g_{m}"] for m in LEARN)
        pct = ("none" if best <= 0 else f"${100 * best / r.g_val:.0f}\\%$") if r.g_val > 0.3 else "---"
        cls = r.cls.replace("_n", ", $n=") + "$"
        L.append(f"{cls} & {fmt(r.g_val)} & " + " & ".join(fmt(r[f'g_{m}']) for m in LEARN) + f" & {pct} \\\\")
    L += ["\\midrule", "Mean & " + fmt(C.g_val.mean()) + " & " + " & ".join(fmt(C[f"g_{m}"].mean()) for m in LEARN) + " & \\\\", "\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    (J / "tab_cvrp.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    scarce()
    cvrp()
    print("tab_scarce.tex e tab_cvrp.tex escritos")
