"""
experiment/run_cycle36_neural_test_entrada262.py (Entrada 262, Ciclo 36)
TESTE da politica neural treinada offline (Entrada 259; pesos fixos, nenhuma aprendizagem durante a execucao) no benchmark de Fanjul-Peyro e Ruiz: as mesmas 700 instancias de teste, T = 1.000, 3 sementes, espacos B15 e B30.
Variantes: nn (com caracteristicas do estado e da instancia) e nn_const (mistura constante treinada com o mesmo orcamento; controle).
Uso: python -m experiment.run_cycle36_neural_test_entrada262 --space B30 --shard 0 --nshards 4     Saida: results/nn_test_{espaco}_s{i}of{n}_entrada262.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle32_fanjul_entrada256 as F  # noqa: E402
from pmsp import ur_numba_neural as NN  # noqa: E402

R = F.R
H = 16
A_OF = {"B5": 5, "B15": 15, "B30": 30}


def main(space, shard, nshards):
    A = A_OF[space]
    th = {v: np.load(R / f"nn_theta_{space}_{v}_entrada259.npy") for v in ("full", "const")}
    man = F.manifest()
    man = man[man.role == "test"]
    F.warm()
    rows, t0 = [], time.perf_counter()
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard:
            continue
        P, a0, od, lb = F.load(r)
        st, d, ls = F.arm_table_b(space, int(r.n))
        for name, v, uf in (("nn", "full", 1), ("nn_const", "const", 0)):
            W1, b1, W2, b2 = NN.unpack(th[v], A, H)
            for s in range(F.N_SEED):
                seed = F.seed_of(space, int(r.ri), "ql", s, False) + 62_000_000 + (1_000_000 if name == "nn_const" else 0)
                best, ev, used = NN.ur_neural_core(P, a0, int(r.m), st, d, ls, F.T, W1, b1, W2, b2, uf, seed, od, lb)
                rows.append(dict(space=space, type=r["type"], instance=int(r.instance), m=int(r.m), n=int(r.n), idx=int(r.idx), method=name, seed_i=s, makespan=int(best), bks=int(r.best_known), evals=float(ev)))
    pd.DataFrame(rows).to_csv(R / f"nn_test_{space}_s{shard}of{nshards}_entrada262.csv", index=False)
    print(f"[nn test {space} s{shard}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    main(a.space, a.shard, a.nshards)
