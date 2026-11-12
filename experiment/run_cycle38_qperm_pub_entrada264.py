"""
experiment/run_cycle38_qperm_pub_entrada264.py (Entrada 264, Ciclo 38)
O Q-Learning padrao INVARIANTE A ORDEM (ql_perm, Entrada 260: permutacao aleatoria das acoes a cada execucao) nas instancias publicadas HELD-OUT de P||Cmax (Entrada 190: familias B, C, E do benchmark de Ostojic et al.; espacos S5 e S15;
T = 1.000; 5 sementes; mesmos hiperparametros do Q-Learning por familia e mesmas instancias, mesmas sementes de instancia). Motivo: as hipoteses confirmatorias H1-H3 da Entrada 190 ('o Q-Learning ganha na familia C, tercis 1-2, e perde na
familia E') e a correlacao de Spearman entre a folga e o ganho do Q-Learning (0,95/0,97 com a folga com validacao cruzada) foram medidas com o Q-Learning padrao, que repete a acao listada primeiro (Entrada 258); nesse benchmark a acao 0 e' a de
menor destruicao (a mais barata). Pergunta: o que sobra dessas leituras com o Q-Learning invariante a ordem?
Uso: python -m experiment.run_cycle38_qperm_pub_entrada264 --space S5 --shard 0 --nshards 12     Saida: results/qpermpub_{espaco}_s{i}of{n}_entrada264.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle17_replication_entrada190 as RP  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from experiment.run_cycle16_entrada185 import write  # noqa: E402

R = P.R


def run_space(space, shard, nshards):
    info = json.loads(RP.groups_path().read_text())
    ch = json.loads(P.choice_path(space).read_text())
    test = P.load_set(RP.M_HOLD)[shard::nshards]
    out = R / f"qpermpub_{space}_s{shard}of{nshards}_entrada264.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    t00, new = time.perf_counter(), not out.exists()
    for i, rec in enumerate(test):
        fam, g = info["test"][rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        st, d, ls = arm_table(space, rec["n"])
        x, y = ch[fam]["hp"].get("ql", [0.1, 0.1])
        rows = []
        for s in range(P.N_SEED):
            seed = P.seed_of(space, "test", fam, rec["inst_id"], "ql", s) + 41_000_000
            perm = np.random.default_rng(seed).permutation(len(st))
            best, _, ev, _, _ = P.H.ig_agent_core3(rec["p"], a0, rec["m"], st[perm].copy(), d[perm].copy(), ls[perm].copy(), 1000, P.AGENT["ql"], 0, x, y, seed % (2 ** 31 - 1), od, lb, 0, 0, 0.0)
            rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method="ql_perm", seed_i=s, makespan=int(best), lb=lb, T=1000, evals=float(ev)))
        write(out, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[qpermpub {space} s{shard}] {i}/{len(test)} ({time.perf_counter() - t00:.0f}s)", flush=True)
    print(f"[qpermpub {space} s{shard}] concluido ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    run_space(a.space, a.shard, a.nshards)
