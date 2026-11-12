"""
experiment/run_cycle35_qperm_entrada260.py (Entrada 260, Ciclo 35)
O Q-Learning padrao INVARIANTE A ORDEM DAS ACOES (ql_perm): a cada execucao as acoes sao embaralhadas por uma permutacao aleatoria (equivale a desempatar o argmax ao acaso), o resto do agente e' o da Entrada 234.
Motivo (Entrada 258): com a validada no indice 0 o Q-Learning padrao recupera 81-86% do ganho da validada; o Q-Learning padrao repete o primeiro indice (55-73% das escolhas) e o seu resultado depende da ordem em que as acoes foram listadas.
Este teste refaz, com o Q-Learning invariante a ordem, as celulas em que o Q-Learning padrao parecia ganhar ou perder: o benchmark de Fanjul-Peyro e Ruiz (700 instancias de teste, B5/B15/B30, 3 sementes) e as familias publicadas
(500 instancias por familia e espaco, 5 sementes). Mesmas instancias e mesmos hiperparametros dos testes originais; sementes proprias.
Uso: python -m experiment.run_cycle35_qperm_entrada260 --bench fanjul --space B30 --shard 0 --nshards 4   |   --bench families --space S5 --shard 0 --nshards 4
Saidas: results/qperm_fanjul_{espaco}_s{i}of{n}_entrada260.csv, results/qperm_families_{espaco}_s{i}of{n}_entrada260.csv
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

R = ROOT / "results"
T = 1000


def run_q_perm(UA, P, a0, m, od, lb, st, d, ls, seed):
    A = len(st)
    perm = np.random.default_rng(seed).permutation(A)
    best, _, ev, used, _ = UA.ur_agent_core(P, a0, m, st[perm].copy(), d[perm].copy(), ls[perm].copy(), T, UA.AGENT_Q, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
    return int(best), float(ev)


def fanjul(space, shard, nshards):
    from experiment import run_cycle32_fanjul_entrada256 as F
    UA = F.UA
    man = F.manifest()
    man = man[man.role == "test"]
    F.warm()
    out = R / f"qperm_fanjul_{space}_s{shard}of{nshards}_entrada260.csv"
    rows, t0 = [], time.perf_counter()
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard:
            continue
        P, a0, od, lb = F.load(r)
        st, d, ls = F.arm_table_b(space, int(r.n))
        for s in range(F.N_SEED):
            seed = F.seed_of(space, int(r.ri), "ql", s, False) + 31_000_000
            mk, ev = run_q_perm(UA, P, a0, int(r.m), od, lb, st, d, ls, seed)
            rows.append(dict(space=space, type=r["type"], instance=int(r.instance), m=int(r.m), n=int(r.n), idx=int(r.idx), method="ql_perm", seed_i=s, makespan=mk, bks=int(r.best_known), evals=ev))
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[qperm fanjul {space} s{shard}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


def families(space, shard, nshards):
    from experiment import run_cycle30_ur_external_entrada254 as E
    from experiment.run_agents_numba_entrada180 import arm_table
    UA = E.UA
    E.U.warm()
    out = R / f"qperm_families_{space}_s{shard}of{nshards}_entrada260.csv"
    rows, t0 = [], time.perf_counter()
    for ci, (f, n, m) in enumerate(E.CLASSES):
        cls = E.cls_name(f, n, m)
        st, d, ls = arm_table(space, n)
        for rep in range(E.N_TEST):
            if (ci * E.N_TEST + rep) % nshards != shard:
                continue
            P, a0, od, lb = E.make(ci, rep, False)
            for s in range(5):
                seed = E.seed_of(space, ci, rep, "ql", s) + 31_000_000
                mk, ev = run_q_perm(UA, P, a0, m, od, lb, st, d, ls, seed)
                rows.append(dict(space=space, cls=cls, rep=rep, method="ql_perm", seed_i=s, makespan=mk, lb=lb, T=T, evals=ev))
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[qperm families {space} s{shard}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    (fanjul if a.bench == "fanjul" else families)(a.space, a.shard, a.nshards)
