"""
analysis/make_table_cost.py
Gera journal/tab_cost.tex a partir de results/entrada277_cost_curve.csv e results/entrada277_cost_breakeven.csv (Entrada 277: o preco do conhecimento offline).
Uso: python -m analysis.make_table_cost
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
J = ROOT / "journal"


def main():
    C = pd.read_csv(R / "entrada277_cost_curve.csv").set_index("k")
    B = pd.read_csv(R / "entrada277_cost_breakeven.csv").set_index("k")
    L = ["\\begin{table}[t]", "\\caption{The price of offline knowledge on the standard benchmark (B30, $T=1000$): the class-validated action chosen with $k$ of the $10$ tuning instances of each class ($1{,}000$ resamples; $k=10$ is the validated action), "
         "its gain over random choice (5th--95th percentile of the resamples), its mean gap to the best known solutions, the cost of the tuning in units of one run of the validated action over the $25$ test instances of the class, and the number of instances per class beyond which "
         "its total cost, tuning included, is below that of random choice at $T=10^4$ ($4.31\\%$ above the best known; measured) and below that of random choice with the iterations that give the same gap (interpolated).}",
         "\\label{tab:cost}", "\\centering", "{\\scriptsize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}rrrrrrr@{}}", "\\toprule",
         "$k$ & Tuning runs & Gain (p5--p95) & Gap (\\%) & Tuning cost & $N$ at $T=10^4$ & $N^*$ \\\\", "\\midrule"]
    for k in (1, 2, 3, 5, 10):
        c, b = C.loc[k], B.loc[k]
        gain = f"${c.Gtn:.2f}$" + (f" (${c.Gtn_p05:.2f}$--${c.Gtn_p95:.2f}$)" if k < 10 else "")
        L.append(f"${k}$ & ${int(c.runs_tuning_per_class)}$ & {gain} & ${b.gap_static:.2f}$ & ${b.tune_over_test_cost:.1f}$ & ${b.N_10k:.0f}$ & ${b.Nstar_interp:.0f}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    (J / "tab_cost.tex").write_text("\n".join(L), encoding="utf-8")
    print("tab_cost.tex escrito")


if __name__ == "__main__":
    main()
