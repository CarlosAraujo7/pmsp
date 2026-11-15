"""
experiment/run_cycle44_frozenqo_entrada272.py (Entrada 272, Ciclo 44)
CONTROLE da Entrada 271: o ganho do Q-Learning otimista aquecido por uma priori offline vem da APRENDIZAGEM durante a execucao ou so' de partir de uma boa priori (e de misturar as melhores acoes com 5% de exploracao)? Mesmas prioris, mesmo qo
congelado, mesmas instancias de teste do benchmark de Fanjul-Peyro e Ruiz (B30, 700 instancias, T = 1.000, 3 sementes), mas com a TAXA DE APRENDIZAGEM ZERO (alfa = 0): a tabela Q fica igual a Q0 durante toda a execucao e a acao e' a de maior Q0 com
probabilidade 0,95 (epsilon constante 0,05). Variantes 'qo_frozen_loso' e 'qo_frozen_full'. Fases: --run --shard i --nshards n (results/frozenqo_test_s{i}of{n}_entrada272.csv). Sementes proprias.
Uso: python -m experiment.run_cycle44_frozenqo_entrada272 --run --shard 0 --nshards 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle32_fanjul_entrada256 as F  # noqa: E402

R = F.R
SPACE = "B30"
FIELDS = ["type", "instance", "m", "n", "idx", "method", "seed_i", "makespan", "bks", "evals"]


def run_one(P, a0, m, od, lb, n, seed, qo_cfg, sc):
    st, d, ls = F.arm_table_b(SPACE, n)
    c, _ = qo_cfg
    hp = np.array([0.0, 0.9, 1.0, 0.05, 0.0])          # alfa = 0: a tabela nao aprende
    Q0 = np.tile(c * np.asarray(sc, dtype=np.float64), (81, 1))
    out = F.UA5.ur_agent_core5(P, a0, m, st, d, ls, F.T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 2, Q0, 0.05)
    return int(out[0]), float(out[2])


def run(shard, nshards):
    man = F.manifest()
    man = man[man.role == "test"]
    pri = json.loads((R / "warmqo_prior_entrada271.json").read_text())
    qo_cfg, _ = F.frozen(SPACE)
    out = R / f"frozenqo_test_s{shard}of{nshards}_entrada272.csv"
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"]) for r in csv.DictReader(open(out, newline=""))}
    F.warm()
    new = not out.exists()
    t0, cnt = time.perf_counter(), 0
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard or (r["type"], str(int(r.instance))) in done:
            continue
        P, a0, od, lb = F.load(r)
        m_, n_ = int(r.m), int(r.n)
        cls = f"{r['type']}|{n_}"
        rows = []
        for k, (meth, sc) in enumerate((("qo_frozen_loso", pri["loso"][cls]), ("qo_frozen_full", pri["full"][cls]))):
            for s in range(F.N_SEED):
                seed = (F.seed_of(SPACE, int(r.ri), "qo", s, False) + 72_000_000 + k * 1_000_000) % (2 ** 31 - 1)
                mk, ev = run_one(P, a0, m_, od, lb, n_, seed, qo_cfg, sc)
                rows.append(dict(type=r["type"], instance=int(r.instance), m=m_, n=n_, idx=int(r.idx), method=meth, seed_i=s, makespan=mk, bks=int(r.best_known), evals=ev))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if cnt % 20 == 0:
            print(f"[frozenqo s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[frozenqo s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.run:
        run(a.shard, a.nshards)
