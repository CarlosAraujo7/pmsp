"""
analysis/explore_entrada134_policy_by_dist.py (Entrada 134)
ANALISE EXPLORATORIA POS HOC (decidida depois de ver que Q1 e Q2 da Entrada 133 nao foram confirmadas e que o efeito e' heterogeneo entre distribuicoes; nao faz parte do
pre-registro): politica (Q-Learning) vs selecao aleatoria por distribuicao, a igual T e a CPU igual, com Holm sobre as 4 distribuicoes dentro de cada comparacao, e a folga dos
bracos fixos (cada estrategia de destruicao sozinha vs a mistura uniforme). Saidas: results/entrada134_policy_by_dist.csv, results/entrada134_policy_by_dist.txt.
Uso: python -m analysis.explore_entrada134_policy_by_dist
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni  # noqa: E402

R = ROOT / "results"


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def main() -> None:
    d = pd.read_csv(R / "unrelated_policy_entrada133.csv")
    w = d.pivot(index=["lo", "hi", "rep"], columns="method", values="makespan").reset_index()
    cpu = d.pivot(index=["lo", "hi", "rep"], columns="method", values="cpu_time").reset_index()
    rows = []
    for (lo, hi), g in w.groupby(["lo", "hi"]):
        c = cpu[(cpu.lo == lo) & (cpu.hi == hi)]
        row = dict(dist=f"U[{lo},{hi}]", n=len(g))
        for lab, b in [("random", "ur_igls_random"), ("eq", "ur_igls_random_eq")]:
            x = rel(g.ur_igls_ql, g[b])
            nz = x[x != 0]
            row[f"d_{lab}"] = float(x.mean())
            row[f"better_{lab}"] = int((x < 0).sum())
            row[f"worse_{lab}"] = int((x > 0).sum())
            row[f"p_{lab}"] = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
        for k in range(4):
            row[f"fix{k}"] = float(rel(g[f"ur_igls_fix{k}"], g.ur_igls_random).mean())
        row["cpu_ql_over_random"] = float(c.ur_igls_ql.mean() / c.ur_igls_random.mean())
        rows.append(row)
    D = pd.DataFrame(rows)
    for lab in ("random", "eq"):
        D[f"p_holm4_{lab}"] = holm_bonferroni(D[f"p_{lab}"].fillna(1).tolist(), alpha=0.05)["p_adjusted"]
    D["best_fixed"] = D[[f"fix{k}" for k in range(4)]].min(axis=1)
    pd.set_option("display.width", 250, "display.float_format", lambda v: f"{v:.4g}")
    text = ("ANALISE EXPLORATORIA POS HOC (Entrada 134) -- politica vs selecao aleatoria por distribuicao, R||Cmax 1000 x 50, 24 instancias por distribuicao\n"
            "(d em %, negativo: Q-Learning melhor; fixK = estrategia de destruicao K sozinha vs mistura uniforme; Holm sobre as 4 distribuicoes)\n\n" + D.to_string(index=False))
    print(text)
    (R / "entrada134_policy_by_dist.txt").write_text(text, encoding="utf-8")
    D.to_csv(R / "entrada134_policy_by_dist.csv", index=False)


if __name__ == "__main__":
    main()
