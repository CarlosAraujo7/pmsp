"""
experiment/run_cycle19_warmq_entrada211.py
============================================
Entrada 211 (Ciclo 19, frente B): o Q-LEARNING AQUECIDO. Todos os aprendizes online dos ciclos anteriores comecam FRIOS em cada execucao (tabela zerada, ~1.000 atualizacoes de uma tabela 81 x A) -- a critica 4 da revisao cega #2. Aqui a tabela Q e' PRE-TREINADA offline nas 5.720 instancias de treino do seletor offline (Entrada 200; disjuntas de todo ajuste e teste),
com recompensa INVARIANTE A ESCALA (pmsp/ig_numba_agents5.py, rmode 2: 100 melhora / LB), e depois usada como inicio da execucao online na instancia de teste.
Pre-treino (sem ajuste de hiperparametros; fixos aqui): uma passada pelas instancias em ordem embaralhada (semente 0), cada uma com T = 1000, alfa = 0,1, gama = 0,9, epsilon inicial 0,3 decaindo por 0,995 ate o piso 0,05 (reinicia em cada instancia), a tabela passando de uma instancia para a seguinte; espacos S5 e S15.
Metodos no teste (as 2.550 da descoberta e as 3.950 held-out; S5 e S15; 5 sementes; T = 1000): wq = Q aquecido com adaptacao online (alfa = 0,05, gama = 0,9, epsilon constante 0,05); wq0 = a mesma politica pre-treinada CONGELADA (alfa = 0, epsilon 0,05): uma politica por estado aprendida offline, sem aprendizado online.
Fases (retomaveis): --pretrain ESPACO | --run CONJUNTO ESPACO --shard i --nshards n. Saidas: results/warmq_Q_ESPACO_entrada211.npy, results/warmq_CONJUNTO_ESPACO_s*of*_entrada211.csv. Pre-registro: analysis/report_entrada211_warmq.py (docstring; hashes em results/entrada211_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from pmsp import ig_numba_agents5 as H5  # noqa: E402

R = ROOT / "results"
HP_PRE = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
HP_WQ = np.array([0.05, 0.9, 1.0, 0.05, 0.0])
HP_WQ0 = np.array([0.0, 0.9, 1.0, 0.05, 0.0])
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]


def run5(space, rec, lb, a0, od, seed, hp, Q0, eps0, T=1000):
    st, d, ls = arm_table(space, rec["n"])
    out = H5.ig_agent_core5(rec["p"], a0, rec["m"], st, d, ls, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 2, Q0, eps0)
    return int(out[0]), float(out[2]), out[5]


def q_path(space):
    return R / f"warmq_Q_{space}_entrada211.npy"


def pretrain(space):
    P.warm()
    recs = S.train_set()
    order = np.random.default_rng(0).permutation(len(recs))
    A = len(arm_table(space, 100)[0])
    Q = np.zeros((81, A))
    t0 = time.perf_counter()
    for i, j in enumerate(order):
        rec = recs[j]
        lb, a0, od = P.prepare_inst(rec)
        _, _, Q = run5(space, rec, lb, a0, od, 1000 + i, HP_PRE, Q, 0.3)
        if i % 500 == 0:
            print(f"[warmq pretrain {space}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s) |Q|={np.abs(Q).max():.3f}", flush=True)
    np.save(q_path(space), Q)
    print(f"[warmq pretrain {space}] concluido ({time.perf_counter() - t0:.0f}s); acao preferida por estado (histograma):", np.bincount(Q.argmax(axis=1), minlength=A).tolist(), flush=True)


def _append(out, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def run_space(conj, space, shard, nshards):
    Q0 = np.load(q_path(space))
    recs, info = S.test_records(conj)
    recs = recs[shard::nshards]
    out = R / f"warmq_{conj}_{space}_s{shard}of{nshards}_entrada211.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        fam, g = info[rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        rows = []
        for meth, hp in (("wq", HP_WQ), ("wq0", HP_WQ0)):
            for s in range(S.N_SEED_TEST):
                mk, ev, _ = run5(space, rec, lb, a0, od, P.seed_of(space, "test", fam, rec["inst_id"], meth, s), hp, Q0, 0.05)
                rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=meth, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
        _append(out, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[warmq {conj} {space} s{shard}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[warmq {conj} {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", default=None)
    ap.add_argument("--run", nargs=2, default=None, metavar=("CONJUNTO", "ESPACO"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.pretrain:
        pretrain(a.pretrain)
    if a.run:
        run_space(a.run[0], a.run[1], a.shard, a.nshards)
