"""
experiment/run_cycle37_qperm_longT_entrada263.py (Entrada 263, Ciclo 37)
O Q-Learning padrao INVARIANTE A ORDEM DAS ACOES (ql_perm, Entrada 260) no horizonte longo T = 10.000 do benchmark de Fanjul-Peyro e Ruiz, espaco B30, 700 instancias de teste, 3 sementes (as mesmas instancias da Entrada 258).
Motivo: na Entrada 258 (horizonte longo) o Q-Learning padrao ganhou do aleatorio alem da margem em 2 dos 7 tipos (previsto no maximo 1); a pergunta e' se isso tambem e' o travamento no indice 0 (Entrada 258, etapas 1-2).
Uso: python -m experiment.run_cycle37_qperm_longT_entrada263 --shard i --nshards n
Saidas: results/qpermL_test_B30_T10000_s{i}of{n}_entrada263.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle32_fanjul_entrada256 as F  # noqa: E402
from experiment import run_cycle35_qperm_entrada260 as Q  # noqa: E402

R = F.R
SPACE = "B30"
T_LONG = 10_000
SEED_SHIFT = 1_000_000_000          # o mesmo da Entrada 258 (independente das sementes de T = 1.000)


def run(shard, nshards):
    F.T = T_LONG
    Q.T = T_LONG
    man = F.manifest()
    man = man[man.role == "test"]
    F.warm()
    out = R / f"qpermL_test_{SPACE}_T{T_LONG}_s{shard}of{nshards}_entrada263.csv"
    rows, t0 = [], time.perf_counter()
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard:
            continue
        P, a0, od, lb = F.load(r)
        st, d, ls = F.arm_table_b(SPACE, int(r.n))
        for s in range(F.N_SEED):
            seed = F.seed_of(SPACE, int(r.ri), "ql", s, False) + SEED_SHIFT + 31_000_000
            mk, ev = Q.run_q_perm(F.UA, P, a0, int(r.m), od, lb, st, d, ls, seed)
            rows.append(dict(space=SPACE, type=r["type"], instance=int(r.instance), m=int(r.m), n=int(r.n), idx=int(r.idx), method="ql_perm", seed_i=s,
                             makespan=mk, bks=int(r.best_known), T=T_LONG, evals=ev))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[qpermL s{shard}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    run(a.shard, a.nshards)
