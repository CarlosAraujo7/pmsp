"""
analysis/audit_entrada101_lb_closure.py (Entrada 101)
Quanto do gap vs LB classico e' folga do limite? Para cada classe sintetica e instancia:
LB1 = max(pmax, ceil(S/m)); LB3 = max(LB1, p_(m)+p_(m+1)); UB = melhor makespan entre os 23 metodos.
Imprime: % de instancias com UB==LB1, UB==LB3 (otimo provado), folga media (UB-LB3)/LB3 e
(UB-LB1)/LB1. Os UBs sao do results.csv; as instancias sao regeneradas por seed.
Uso: python -m analysis.audit_entrada101_lb_closure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pmsp.exact import improved_lower_bound  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

df = pd.read_csv(ROOT / "results" / "results.csv")
rows = []
for inst, g in df.groupby("instance"):
    n, m = int(g.n.iloc[0]), int(g.m.iloc[0])
    ub = g.groupby("rep").makespan.min()
    lb1, lb3 = [], []
    for rep in ub.index:
        p = PMSPInstance.random(n=n, m=m, seed=int(rep), name=inst).p
        lb1.append(max(int(max(p)), -(-int(sum(p)) // m)))
        lb3.append(improved_lower_bound([int(x) for x in p], m))
    lb1, lb3, ubv = np.array(lb1), np.array(lb3), ub.to_numpy()
    rows.append(dict(cls=inst, n=n, m=m, N=len(ubv), opt_proven_LB1=100 * (ubv == lb1).mean(),
                     opt_proven_LB3=100 * (ubv == lb3).mean(),
                     slack_LB1_pct=100 * ((ubv - lb1) / lb1).mean(), slack_LB3_pct=100 * ((ubv - lb3) / lb3).mean(),
                     unresolved=int((ubv > lb3).sum())))
t = pd.DataFrame(rows).sort_values(["n", "m"])
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(t.to_string(index=False))
print("total instancias:", int(t.N.sum()), "| sem prova de otimalidade (UB > LB3):", int(t.unresolved.sum()))
t.to_csv(ROOT / "results" / "entrada101_lb_closure.csv", index=False)
