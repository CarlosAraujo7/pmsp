"""
experiment/run_cycle50_cvrp_entrada282.py (Entrada 282, Ciclo 50)
O PROTOCOLO, SEM ALTERACAO, NUMA IMPLEMENTACAO DE TERCEIROS: o ALNS para o CVRP com a biblioteca `alns` (Wouda e Lan, JOSS 2023, MIT) e os seus seletores de operadores (roteiro de 16/11/2026, item P1). Operadores heterogeneos (8 de destruicao x 2 de reparo = 16 acoes; pmsp/cvrp_alns.py);
aprendizes = RouletteWheel, SegmentedRouletteWheel, AlphaUCB e dois MABSelector (UCB1, epsilon-guloso), com os valores do exemplo de CVRP da biblioteca e SEM ajuste; controle aleatorio = RandomSelect; estatica = um par fixo. T = 1.000 iteracoes, 3 sementes por instancia e metodo.
Classes sinteticas (estilo Uchoa et al. 2017): n em {100, 200} x clientes {aleatorios R, agrupados C} x clientes por rota {6, 16}: 8 classes, 10 instancias de AJUSTE e 20 de TESTE cada; mais as 22 instancias X publicadas com n <= 200 (CVRPLIB; melhores solucoes conhecidas), so' no teste.
Metodos no teste: random, rw, srw, aucb, mab_ucb1, mab_eps, as mesmas cinco com a LISTA DE OPERADORES PERMUTADA (sufixo _perm; diagnostico de ordem) e as 16 acoes fixas arm0..arm15. A validada de cada classe, a global (G0) e a por (n, tamanho de rota) (Gnr) saem so' do AJUSTE (--tune, --choose).
Fases: --tune --shard i --nshards n | --choose | --run --shard i --nshards n (sinteticas) | --runx --shard i --nshards n (X). --pilot usa 2 instancias de AJUSTE.
Saidas: results/cvrp_tune_s{i}of{n}_entrada282.csv, results/cvrp_choice_entrada282.json, results/cvrp_test_s{i}of{n}_entrada282.csv, results/cvrp_x_s{i}of{n}_entrada282.csv
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
from pmsp import cvrp_alns as C  # noqa: E402

R = ROOT / "results"
XDIR = ROOT / "external_benchmarks" / "cvrp_x" / "raw"
CLASSES = [(n, pos, r) for n in (100, 200) for pos in ("R", "C") for r in (6, 16)]
N_TUNE, N_TEST, N_SEED, T = 10, 20, 3, 1000
XNAMES = ["X-n101-k25", "X-n106-k14", "X-n110-k13", "X-n115-k10", "X-n120-k6", "X-n125-k30", "X-n129-k18", "X-n134-k13", "X-n139-k10", "X-n143-k7", "X-n148-k46", "X-n153-k22",
          "X-n157-k13", "X-n162-k11", "X-n167-k10", "X-n172-k51", "X-n176-k26", "X-n181-k23", "X-n186-k15", "X-n190-k8", "X-n195-k51", "X-n200-k36"]
METHODS = ["random"] + C.LEARNERS + [f"{k}_perm" for k in C.LEARNERS] + [f"arm{a}" for a in range(16)]
FIELDS = ["set", "cls", "rep", "method", "seed_i", "cost", "secs", "init", "share_d0", "share_r0", "bks"]


def cls_name(c):
    n, pos, r = CLASSES[c]
    return f"{pos}{r}_n{n}"


def instance(role, ci, rep):
    n, pos, r = CLASSES[ci]
    return C.make_class_instance(n, pos, r, (7_000_000 if role == "tune" else 8_000_000) + 1000 * ci + rep)


def run_seed(role, ci, rep, mi, s):
    return (50_000_000 if role == "tune" else 0) + 5_000_000 + ci * 100_000 + rep * 1000 + mi * 10 + s


def one(inst, method, seed, init):
    if method.startswith("arm"):
        cost, dt, c0, _, info = C.run_alns(inst, "fixed", seed, T, fixed=int(method[3:]), init=init)
    else:
        perm = method.endswith("_perm")
        name = method[:-5] if perm else method
        cost, dt, c0, _, info = C.run_alns(inst, name, seed, T, init=init, perm=perm)
    return cost, dt, c0, info


def write(out, rows, new):
    with open(out, "w" if new else "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def warm():
    inst = C.make_class_instance(30, "R", 6, 1)
    init = C.initial_state(inst)
    for m in METHODS[:6] + ["aucb_perm", "arm0"]:
        one(inst, m, 1, init)


def run_tune(shard, nshards, pilot):
    warm()
    out = R / (f"cvrp_tune_pilot_entrada282.csv" if pilot else f"cvrp_tune_s{shard}of{nshards}_entrada282.csv")
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    new, t0, cnt = not out.exists(), time.perf_counter(), 0
    k = 0
    for ci in range(len(CLASSES)):
        for rep in range(N_TUNE):
            k += 1
            if (k - 1) % nshards != shard or (cls_name(ci), str(rep)) in done:
                continue
            inst = instance("tune", ci, rep)
            init = C.initial_state(inst)
            rows = []
            for a in range(16):
                for s in range(2):
                    cost, dt, c0, info = one(inst, f"arm{a}", run_seed("tune", ci, rep, 10 + a, s), init)
                    rows.append(dict(set="tune", cls=cls_name(ci), rep=rep, method=f"arm{a}", seed_i=s, cost=cost, secs=dt, init=c0, share_d0=info["share_d0"], share_r0=info["share_r0"], bks=""))
            write(out, rows, new)
            new = False
            cnt += 1
            if pilot and cnt >= 2:
                print("pilot", time.perf_counter() - t0)
                return
    print(f"[cvrp tune s{shard}] concluido, {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)


def choose():
    import pandas as pd
    import glob
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "cvrp_tune_s*of*_entrada282.csv")))], ignore_index=True)
    d["a"] = d.method.str[3:].astype(int)
    d["ref"] = d.groupby(["cls", "rep"]).cost.transform("min")
    d["ratio"] = d.cost / d.ref
    d["n"] = d.cls.str.split("_n").str[1].astype(int)
    d["r"] = d.cls.str.split("_").str[0].str[1:].astype(int)
    pick = lambda g: int(g.groupby("a").ratio.mean().idxmin())
    out = dict(val={c: pick(g) for c, g in d.groupby("cls")}, g0=pick(d), gnr={f"{n}|{r}": pick(g) for (n, r), g in d.groupby(["n", "r"])})
    (R / "cvrp_choice_entrada282.json").write_text(json.dumps(out, indent=1))
    print(out)


def run_test(shard, nshards, pilot):
    warm()
    out = R / (f"cvrp_test_pilot_entrada282.csv" if pilot else f"cvrp_test_s{shard}of{nshards}_entrada282.csv")
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    new, t0, cnt, k = not out.exists(), time.perf_counter(), 0, 0
    role = "tune" if pilot else "test"
    for ci in range(len(CLASSES)):
        for rep in range(N_TUNE if pilot else N_TEST):
            k += 1
            if (k - 1) % nshards != shard or (cls_name(ci), str(rep)) in done:
                continue
            inst = instance(role, ci, rep)
            init = C.initial_state(inst)
            rows = []
            for mi, m in enumerate(METHODS):
                for s in range(N_SEED):
                    cost, dt, c0, info = one(inst, m, run_seed(role, ci, rep, mi, s), init)
                    rows.append(dict(set="syn", cls=cls_name(ci), rep=rep, method=m, seed_i=s, cost=cost, secs=dt, init=c0, share_d0=info["share_d0"], share_r0=info["share_r0"], bks=""))
            write(out, rows, new)
            new = False
            cnt += 1
            if pilot and cnt >= 1:
                print("pilot", time.perf_counter() - t0)
                return
            if cnt % 5 == 0:
                print(f"[cvrp test s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[cvrp test s{shard}] concluido, {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)


def run_x(shard, nshards, pilot):
    import vrplib
    warm()
    out = R / f"cvrp_x_s{shard}of{nshards}_entrada282.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    new, t0, cnt = not out.exists(), time.perf_counter(), 0
    for xi, name in enumerate(XNAMES):
        if xi % nshards != shard or (name, str(xi)) in done:
            continue
        inst = C.instance_from_vrplib(vrplib.read_instance(str(XDIR / "inst" / f"{name}.vrp")), vrplib.read_solution(str(XDIR / "bks" / f"{name}.sol")))
        init = C.initial_state(inst)
        rows = []
        for mi, m in enumerate(METHODS):
            for s in range(N_SEED):
                cost, dt, c0, info = one(inst, m, 900_000_000 + xi * 1000 + mi * 10 + s, init)
                rows.append(dict(set="X", cls=name, rep=xi, method=m, seed_i=s, cost=cost, secs=dt, init=c0, share_d0=info["share_d0"], share_r0=info["share_r0"], bks=inst.bks))
        write(out, rows, new)
        new = False
        cnt += 1
    print(f"[cvrp x s{shard}] concluido, {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--runx", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        run_tune(a.shard, a.nshards, a.pilot)
    if a.choose:
        choose()
    if a.run:
        run_test(a.shard, a.nshards, a.pilot)
    if a.runx:
        run_x(a.shard, a.nshards, a.pilot)
