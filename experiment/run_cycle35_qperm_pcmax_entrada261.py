"""
experiment/run_cycle35_qperm_pcmax_entrada261.py (Entrada 261, Ciclo 35)
O Q-Learning padrao INVARIANTE A ORDEM (permutacao aleatoria das acoes a cada execucao) na bateria principal de P||Cmax (Entrada 180: 12 classes (n, m) x 100 instancias de teste, T = 1.000, 5 sementes, espacos S5 e S15,
mesmo lr do Q-Learning da bateria). No P||Cmax a lista de acoes e' ordenada por tamanho de destruicao crescente (5%, 10%, 20%, 30%, 40% de n), de modo que a acao 0 e' a MAIS BARATA: o travamento do Q-Learning padrao no primeiro
indice (Entrada 258) tornaria 'o Q-Learning ganha a CPU igual escolhendo acoes mais baratas' um efeito da ordem. Esta bateria mede o Q-Learning invariante a ordem a iteracoes iguais; a CPU igual fica para depois.
Uso: python -m experiment.run_cycle35_qperm_pcmax_entrada261 --space S5 --shard 0 --nshards 4     Saida: results/qperm_pcmax_{espaco}_s{i}of{n}_entrada261.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_agents_numba_entrada180 as E  # noqa: E402

R = E.R
G = E.G


def main(space, shard, nshards):
    ch = json.loads(E.choice_path(space).read_text())
    E.warm()
    out = R / f"qperm_pcmax_{space}_s{shard}of{nshards}_entrada261.csv"
    rows, t0 = [], time.perf_counter()
    for ci, (n, m) in enumerate(E.CLASSES):
        cls = f"rand_{n}j_{m}m"
        st, d, ls = E.arm_table(space, n)
        A = len(st)
        for rep in range(E.N_TEST):
            if (ci * E.N_TEST + rep) % nshards != shard:
                continue
            inst, p, a0, od = E.make(n, m, rep, False)
            for s in range(E.N_SEED):
                seed = E.seed_of(space, ci, rep, "ql", s) + 31_000_000
                perm = np.random.default_rng(seed).permutation(A)
                best, _, ev, used = G.ig_agent_core(p, a0, m, st[perm].copy(), d[perm].copy(), ls[perm].copy(), 1000, G.AGENT_Q, 0, ch["lr"], seed, od, inst.lower_bound)
                rows.append(dict(space=space, cls=cls, rep=rep, method="ql_perm", seed_i=s, makespan=int(best), lb=inst.lower_bound, T=1000, evals=float(ev)))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[qperm pcmax {space} s{shard}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    main(a.space, a.shard, a.nshards)
