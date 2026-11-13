"""
experiment/run_cycle40_tsp_entrada267.py (Entrada 267, Ciclo 40)
O protocolo do artigo num SEGUNDO DOMINIO, fora do escalonamento: a busca local iterada (2-opt + Or-opt com listas de vizinhos) do TSP euclidiano (pmsp/tsp_numba.py), com a perturbacao escolhida a cada iteracao por um seletor.
Instancias: pontos uniformes e agrupados (pmsp/instance_tsp.py), n em {200, 500, 1000, 2000}: 8 classes (tipo, n), 12 instancias de AJUSTE e 50 de TESTE por classe. Espacos de acoes: T5 (cinco tamanhos de uma perturbacao) e T15
(tres perturbacoes x cinco tamanhos). T = 1.000 iteracoes, 3 sementes por instancia no teste (2 no ajuste). Acao validada = a de menor razao media (comprimento final / comprimento do tour inicial) nas instancias de ajuste da classe.
Metodos no teste: rand, ql, ql_perm (Q-Learning com as acoes embaralhadas a cada execucao), ucb, pg, alns, ts, qo (hiperparametros CONGELADOS dos outros hospedeiros: Q 0,1/0,9/epsilon 1,0 a 0,05; PG 0,03 em T5 e 0,3 em T15; roleta (0,1; 0); Thompson sem
desconto; Q otimista com o (c, alfa) de results/ur_choice_qo_S15_entrada234.json) e TODAS as 15 acoes fixas (as 5 primeiras sao as de T5), para a analise da granularidade do conhecimento offline. Fases (retomaveis):
  --tune --shard i --nshards n  (results/tsp_tune_s{i}of{n}_entrada267.csv) | --choose (results/tsp_choice_entrada267.json) | --run --shard i --nshards n (results/tsp_test_s{i}of{n}_entrada267.csv)
Uso: python -m experiment.run_cycle40_tsp_entrada267 --tune --shard 0 --nshards 12
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pmsp import tsp_numba as TS  # noqa: E402
from pmsp.instance_tsp import build  # noqa: E402

R = ROOT / "results"
KINDS = ["unif", "clust"]
NS = [200, 500, 1000, 2000]
N_TUNE, N_TEST, N_SEED, T = 12, 50, 3, 1000
SPACES = ("T5", "T15")
PG_LR = {"T5": 0.03, "T15": 0.3}
HP_Q = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
METHODS = ["rand", "ql", "ql_perm", "ucb", "pg", "alns", "ts", "qo"]
FIELDS = ["space", "kind", "n", "rep", "inst", "method", "seed_i", "length", "len0", "evals", "share0"]


def classes():
    return [(k, n) for k in KINDS for n in NS]


def instances(role):
    """Lista deterministica de (id, tipo, n, rep, semente da instancia)."""
    out, i = [], 0
    nrep = N_TUNE if role == "tune" else N_TEST
    for k, n in classes():
        for rep in range(nrep):
            out.append(dict(inst=i, kind=k, n=n, rep=rep, seed=(7_000_000 if role == "tune" else 8_000_000) + 1000 * KINDS.index(k) + 100 * NS.index(n) + rep))
            i += 1
    return out


def seed_of(space, inst, method, s, tune=False):
    k = (METHODS.index(method) if method in METHODS else 20 + int(method[3:]))
    return 1_900_000_000 + 100_000_000 * SPACES.index(space) + (50_000_000 if tune else 0) + 100_000 * k + 7 * inst + s


def frozen_qo():
    return tuple(json.loads((R / "ur_choice_qo_S15_entrada234.json").read_text())["cfg"])


def run_one(space, name, D, NL, tour0, seed, qo_cfg, kinds=None, sizes=None):
    k0, s0 = TS.arm_table(space)
    if kinds is None:
        kinds, sizes = k0, s0
    A = len(kinds)
    if name.startswith("arm"):
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, T, TS.AGENT_FIXED, int(name[3:]), 0.1, 0.1, seed, HP_Q, 0.0)
    elif name == "rand":
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, T, TS.AGENT_RANDOM, 0, 0.1, 0.1, seed, HP_Q, 0.0)
    elif name == "ql":
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, T, TS.AGENT_Q, 0, 0.1, 0.1, seed, HP_Q, 0.0)
    elif name == "ql_perm":
        perm = np.random.default_rng(seed).permutation(A)
        r = TS.tsp_agent_core(D, NL, tour0, kinds[perm].copy(), sizes[perm].copy(), T, TS.AGENT_Q, 0, 0.1, 0.1, seed, HP_Q, 0.0)
    elif name == "ucb":
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, T, TS.AGENT_UCB, 0, 0.1, 0.1, seed, HP_Q, 0.0)
    elif name == "pg":
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, T, TS.AGENT_PG, 0, PG_LR[space], 0.1, seed, HP_Q, 0.0)
    elif name == "alns":
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, T, TS.AGENT_ALNS, 0, 0.1, 0.0, seed, HP_Q, 0.0)
    elif name == "ts":
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, T, TS.AGENT_TS, 0, 1.0, 0.0, seed, HP_Q, 0.0)
    else:  # qo
        c, a = qo_cfg
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, T, TS.AGENT_QO, 0, 0.1, 0.0, seed, np.array([a, 0.9, 1.0, 0.05, 0.0]), float(c))
    return float(r[0]), float(r[2]), float(r[3][0]) / T


def warm():
    D, NL = build("unif", 40, 1)
    tour0, _ = TS.construct(D, NL)
    for sp in SPACES:
        for nm in METHODS + ["arm0"]:
            run_one(sp, nm, D, NL, tour0, 1, frozen_qo())


def tune(shard, nshards):
    warm()
    out = R / f"tsp_tune_s{shard}of{nshards}_entrada267.csv"
    rows, t0 = [], time.perf_counter()
    for it in instances("tune"):
        if it["inst"] % nshards != shard:
            continue
        D, NL = build(it["kind"], it["n"], it["seed"])
        tour0, _ = TS.construct(D, NL)
        len0 = float(TS.tour_len(tour0, D))
        for a in range(15):
            for s in range(2):
                ln, _, _ = run_one("T15", f"arm{a}", D, NL, tour0, seed_of("T15", it["inst"], f"arm{a}", s, True), None)
                rows.append(dict(cls=f"{it['kind']}|{it['n']}", arm=a, ratio=ln / len0))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[tsp tune s{shard}] {len(rows)} linhas ({time.perf_counter() - t0:.0f}s)", flush=True)


def choose():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "tsp_tune_s*of*_entrada267.csv")))], ignore_index=True)
    val15 = {c: int(g.groupby("arm").ratio.mean().idxmin()) for c, g in d.groupby("cls")}
    d5 = d[d.arm < 5]
    val5 = {c: int(g.groupby("arm").ratio.mean().idxmin()) for c, g in d5.groupby("cls")}
    (R / "tsp_choice_entrada267.json").write_text(json.dumps(dict(T15=val15, T5=val5), indent=1))
    print("classes validadas:", len(val15))


def run(shard, nshards):
    warm()
    qo = frozen_qo()
    out = R / f"tsp_test_s{shard}of{nshards}_entrada267.csv"
    done = set()
    if out.exists():
        done = {int(r["inst"]) for r in csv.DictReader(open(out, newline=""))}
    new = not out.exists()
    t0, cnt = time.perf_counter(), 0
    for it in instances("test"):
        if it["inst"] % nshards != shard or it["inst"] in done:
            continue
        D, NL = build(it["kind"], it["n"], it["seed"])
        tour0, _ = TS.construct(D, NL)
        len0 = float(TS.tour_len(tour0, D))
        rows = []
        for sp in SPACES:
            for me in METHODS:
                for s in range(N_SEED):
                    ln, ev, sh0 = run_one(sp, me, D, NL, tour0, seed_of(sp, it["inst"], me, s), qo)
                    rows.append(dict(space=sp, kind=it["kind"], n=it["n"], rep=it["rep"], inst=it["inst"], method=me, seed_i=s, length=ln, len0=len0, evals=ev, share0=sh0))
        for a in range(15):
            for s in range(N_SEED):
                ln, ev, _ = run_one("T15", f"arm{a}", D, NL, tour0, seed_of("T15", it["inst"], f"arm{a}", s), None)
                rows.append(dict(space="T15", kind=it["kind"], n=it["n"], rep=it["rep"], inst=it["inst"], method=f"arm{a}", seed_i=s, length=ln, len0=len0, evals=ev, share0=0.0))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if cnt % 5 == 0:
            print(f"[tsp s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[tsp s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        tune(a.shard, a.nshards)
    if a.choose:
        choose()
    if a.run:
        run(a.shard, a.nshards)
