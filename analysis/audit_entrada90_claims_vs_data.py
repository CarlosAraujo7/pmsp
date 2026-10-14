"""
analysis/audit_entrada90_claims_vs_data.py
============================================
Auditoria de 04/10/2026 (diario_pesquisa.md, Entrada 90): reproduz, direto
de results/results.csv, os numeros que confrontam as alegacoes de
abertura do manuscrito (Abstract/Highlights) com os dados.

Secoes impressas:
  1. gap medio (%) por classe: lpt, ig_ql, ig_random, ils_ql, ils_random,
     multifit, sa_li_perkins, bco  (+ cpu_time medio)
  2. ig_ql vs multifit/sa_li_perkins/bco/ig_random: pares, vitorias,
     derrotas, empates, p bruto (Wilcoxon pareado, SEM Holm -- para os
     numeros corrigidos, ver report_entrada37_sota_and_holm.py)
  3. sa_li_perkins == lpt? (gap medio identico em quantas classes)
  4. esparsidade do sinal de recompensa: n_improvements de ig_ql por
     classe e o cronograma de epsilon do QLearningAgent
  5. folga do LB: melhor-de-todos-os-metodos vs LB, por classe

Uso: python -m analysis.audit_entrada90_claims_vs_data
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "results" / "results.csv")
cls = df.groupby("instance").agg(n=("n", "first"), m=("m", "first")).sort_values(["n", "m"])
order = cls.index
keys = ["lpt", "ig_ql", "ig_random", "ils_ql", "ils_random", "multifit", "sa_li_perkins", "bco"]
pd.set_option("display.width", 220)

print("== 1. gap medio (%) por classe")
piv = df[df.method.isin(keys)].pivot_table(index="instance", columns="method", values="gap_pct").loc[order]
print(piv[keys].round(3).to_string())
print("\n   cpu_time medio (s)")
print(df[df.method.isin(keys)].pivot_table(index="instance", columns="method", values="cpu_time").loc[order][keys].round(4).to_string())


def paired(a, b, inst):
    x = df[(df.instance == inst) & (df.method == a)].set_index("rep").gap_pct
    y = df[(df.instance == inst) & (df.method == b)].set_index("rep").gap_pct
    idx = x.index.intersection(y.index)
    d = x.loc[idx] - y.loc[idx]
    p = 1.0 if (d == 0).all() else float(wilcoxon(x.loc[idx], y.loc[idx]).pvalue)
    return len(idx), int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum()), p


print("\n== 2. ig_ql vs outro: (n_pares, ig_ql melhor, pior, empate, p bruto)")
for other in ["multifit", "sa_li_perkins", "bco", "ig_random"]:
    print(f"  -- vs {other}")
    for inst in order:
        n_, b_, w_, t_, p_ = paired("ig_ql", other, inst)
        print(f"     {inst:15s} {n_:4d} {b_:4d} {w_:4d} {t_:4d}   p={p_:.3g}")

print("\n== 3. sa_li_perkins com gap medio identico ao LPT (3 casas)")
same = [i for i in order if round(piv.loc[i, "sa_li_perkins"], 3) == round(piv.loc[i, "lpt"], 3)]
print(f"   {len(same)} de {len(order)} classes: {same}")

print("\n== 4. sinal de recompensa (ig.py: reward = max(delta, 0), aceitacao gulosa)")
g = df[df.method == "ig_ql"].groupby("instance").n_improvements
print(pd.DataFrame({"media": g.mean().round(2), "mediana": g.median(),
                    "share<=5": g.apply(lambda s: (s <= 5).mean()).round(2)}).loc[order].to_string())
eps0, dec, emin = 1.0, 0.995, 0.05   # experiment/config.py: QL_EPSILON_*
print(f"   epsilon chega ao piso em t={math.log(emin / eps0) / math.log(dec):.0f}/1000; "
      f"epsilon medio na rodada = {np.mean([max(emin, eps0 * dec ** t) for t in range(1000)]):.3f}")

print("\n== 5. folga do LB: melhor-de-todos (23 metodos) vs LB")
rows = []
for inst in order:
    s = df[df.instance == inst]
    s = s[s.rep.isin(s[s.method == "ig_ql"].rep.unique())]
    mk = s.pivot_table(index="rep", columns="method", values="makespan", aggfunc="first")
    lb = s.groupby("rep").lb.first()
    best, ql = mk.min(axis=1), mk["ig_ql"]
    rows.append(dict(cls=inst, n_over_m=round(cls.loc[inst, "n"] / cls.loc[inst, "m"], 1),
                     gapLB_igql=round(100 * ((ql - lb) / lb).mean(), 3),
                     gapLB_best=round(100 * ((best - lb) / lb).mean(), 3),
                     pct_best_hits_LB=round(100 * (best == lb).mean()),
                     pct_igql_hits_LB=round(100 * (ql == lb).mean()),
                     pct_igql_equals_best=round(100 * (ql == best).mean())))
print(pd.DataFrame(rows).to_string(index=False))
