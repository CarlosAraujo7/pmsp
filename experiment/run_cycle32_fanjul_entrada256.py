"""
experiment/run_cycle32_fanjul_entrada256.py
============================================
Entrada 256 (Ciclo 32): o protocolo COMPILADO do artigo (Entradas 180-255) no benchmark PADRAO de R||Cmax de Fanjul-Peyro e Ruiz (EJOR 2010; 1.400 instancias: 7 tipos x m em {10,20,30,40,50} x n em {100,200,500,1000} x 10; copia MIT do repositorio Fraunhofer-Chalmers-Centre/RCmax, baixada em 10/10/2026, ver external_benchmarks/rcmax_fanjul_ruiz/README.md), com os MELHORES VALORES CONHECIDOS de CPLEX 11.1 (2 h), Gurobi 9.1.1 (300 s) e do metodo LR (300 s) (results/rcmax_manifest_entrada137.csv: best_known = minimo dos tres; proven_opt). A divisao e' a do manifesto da Entrada 137: idx 0-1 AJUSTE (280), idx 2-6 TESTE (700), idx 7-9 RESERVA (nao usadas).
A grade de acoes do artigo (d/n de 0,05 a 0,40, pensada para n <= 300) e' grossa demais para n = 1000 (a menor, d = 50 tarefas por iteracao, destroi muito); aqui a grade e' ajustada a ESCALA: d/n em {0,01; 0,02; 0,05; 0,10; 0,20} (B5: destruicao aleatoria; B15: tres estrategias x cinco tamanhos; B30: B15 sem e com busca local), d = max(2, round(f n)). Aprendizes com os hiperparametros CONGELADOS (Entradas 234/255: lr do pg 0,03 em B5 e 0,3 em B15/B30; roleta ALNS (0,1; 0); Q otimista de results/ur_choice_qo_*_entrada234.json; iq e fr de results/pubaos_ext_choice_*_entrada255.json; B15 e B30 usam os do S15). A acao validada e' escolhida por CLASSE = (tipo, n) (28 classes; 10 instancias de ajuste por classe, duas por m) como a de menor razao media makespan / melhor conhecido, em 2 sementes.
Teste: 700 instancias x metodos rand, ql, ucb, pg, alns, qo, iq, fr, val x 3 sementes, T = 1.000, a iteracoes iguais. Ganho = -d sobre o aleatorio; lacuna = 100 (makespan - melhor conhecido) / melhor conhecido.
Fases (retomaveis): --tune ESPACO --shard i --nshards n | --choose ESPACO | --run ESPACO --shard i --nshards n [--pilot]. Saidas: results/fr_tune_*_entrada256.csv, results/fr_choice_ESPACO_entrada256.json, results/fr_test_ESPACO_s*of*_entrada256.csv. Pre-registro: analysis/report_entrada256_fanjul.py (docstring; hashes em results/entrada256_prereg_hashes.txt).
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

from pmsp import ur_numba_agents as UA  # noqa: E402
from pmsp import ur_numba_agents5 as UA5  # noqa: E402
from pmsp import ur_numba_agents6 as U6  # noqa: E402
from pmsp.instance_ur import constructive_ur, lower_bound_ur  # noqa: E402

R = ROOT / "results"
BENCH = ROOT / "external_benchmarks" / "rcmax_fanjul_ruiz" / "extracted" / "instances"
MAN = R / "rcmax_manifest_entrada137.csv"
SPACES = ("B5", "B15", "B30")
FRB = [0.01, 0.02, 0.05, 0.10, 0.20]
METHODS = ["rand", "ql", "ucb", "pg", "alns", "qo", "iq", "fr", "val"]
N_SEED = 3
T = 1000
FIELDS = ["space", "type", "instance", "m", "n", "idx", "method", "seed_i", "makespan", "lb", "bks", "T", "evals"]
PG_LR = {"B5": 0.03, "B15": 0.3, "B30": 0.3}
ALNS_HP = (0.1, 0.0)
BASE_SPACE = {"B5": "S5", "B15": "S15", "B30": "S15"}      # espaco de origem dos hiperparametros congelados
AGENT6 = {"iq": 11, "fr": 12}


def arm_table_b(space, n):
    d = lambda f: max(2, int(round(f * n)))
    if space == "B5":
        return np.zeros(5, dtype=np.int64), np.array([d(f) for f in FRB], dtype=np.int64), np.zeros(5, dtype=np.int64)
    s15 = np.array([s for s in range(3) for _ in FRB], dtype=np.int64)
    d15 = np.array([d(f) for _ in range(3) for f in FRB], dtype=np.int64)
    if space == "B15":
        return s15, d15, np.zeros(15, dtype=np.int64)
    return np.concatenate([s15, s15]), np.concatenate([d15, d15]), np.concatenate([np.zeros(15, dtype=np.int64), np.ones(15, dtype=np.int64)])


def manifest():
    m = pd.read_csv(MAN).reset_index(drop=True)
    m["ri"] = np.arange(len(m))
    return m


def load(row):
    A = np.loadtxt(BENCH / row["type"] / f"{int(row['instance'])}.csv", delimiter=",", dtype=np.int64, ndmin=2)
    P = np.ascontiguousarray(A.T)
    assert P.shape == (int(row["n"]), int(row["m"])), (P.shape, row["n"], row["m"])
    a0, od = constructive_ur(P)
    return P, a0, od, float(lower_bound_ur(P))


def seed_of(space, ri, method, s, tune=False):
    k = METHODS.index(method) if not method.startswith("arm") else 10 + int(method[3:])
    return 1_600_000_000 + 200_000_000 * SPACES.index(space) + (50_000_000 if tune else 0) + 10_000 * k + 5 * ri + s


def frozen(space):
    bs = BASE_SPACE[space]
    qo = tuple(json.loads((R / f"ur_choice_qo_{bs}_entrada234.json").read_text())["cfg"])
    pub = json.loads((R / f"pubaos_ext_choice_{bs}_entrada255.json").read_text())
    return qo, {k: tuple(v["cfg"]) for k, v in pub.items()}


def run_fixed_or_agent(space, name, P, a0, m, od, lb, n, seed, val_arm=0, qo_cfg=None, pub_cfg=None, arm=None):
    st, d, ls = arm_table_b(space, n)
    A = len(st)
    if name == "qo":
        c, a = qo_cfg
        hp = np.array([a, 0.9, 1.0, 0.05, 0.0])
        out = UA5.ur_agent_core5(P, a0, m, st, d, ls, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 2, np.full((81, A), c), 0.05)
        return int(out[0]), float(out[2])
    if name in AGENT6:
        cfg = pub_cfg[name]
        if name == "iq":
            lr, lr2, hp = 0.1, float(cfg[0]), np.array([0.1, 0.9, cfg[1], 0.05, 0.5])
        else:
            lr, lr2, hp = cfg[0], cfg[1], U6.HP_DEFAULT
        best, _, ev, used, _ = U6.ur_agent_core6(P, a0, m, st, d, ls, T, AGENT6[name], 0, lr, lr2, seed, od, lb, 0, 0, 0.0, hp, 0)
        return int(best), float(ev)
    if name == "val" or name.startswith("arm"):
        fx = val_arm if name == "val" else int(name[3:])
        best, _, ev, used, _ = UA.ur_agent_core(P, a0, m, st, d, ls, T, UA.AGENT_FIXED, fx, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
        return int(best), float(ev)
    ag = {"rand": UA.AGENT_RANDOM, "ql": UA.AGENT_Q, "ucb": UA.AGENT_UCB, "pg": UA.AGENT_PG, "alns": UA.AGENT_ALNS}[name]
    lr, lr2 = (PG_LR[space], 0.1) if name == "pg" else (ALNS_HP if name == "alns" else (0.1, 0.1))
    best, _, ev, used, _ = UA.ur_agent_core(P, a0, m, st, d, ls, T, ag, 0, lr, lr2, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
    return int(best), float(ev)


def warm():
    P = np.random.default_rng(1).integers(1, 50, size=(30, 4)).astype(np.int64)
    a0, od = constructive_ur(P)
    lb = float(lower_bound_ur(P))
    for sp in SPACES:
        qo, pub = frozen(sp)
        for name in METHODS[:-1]:
            run_fixed_or_agent(sp, name, P, a0, 4, od, lb, 30, 1, 0, qo, pub)


def tune(space, shard, nshards):
    man = manifest()
    man = man[man.role == "tune"]
    warm()
    A = len(arm_table_b(space, 100)[0])
    out = R / f"fr_tune_{space}_s{shard}of{nshards}_entrada256.csv"
    rows = []
    t0 = time.perf_counter()
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard:
            continue
        P, a0, od, lb = load(r)
        for a in range(A):
            for s in range(2):
                mk, ev = run_fixed_or_agent(space, f"arm{a}", P, a0, int(r.m), od, lb, int(r.n), seed_of(space, int(r.ri), f"arm{a}", s, True))
                rows.append(dict(cls=f"{r['type']}|{int(r.n)}", arm=a, ratio=mk / float(r.best_known)))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[fr tune {space} s{shard}] {len(rows)} linhas ({time.perf_counter() - t0:.0f}s)", flush=True)


def choose(space):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"fr_tune_{space}_s*of*_entrada256.csv")))], ignore_index=True)
    val = {c: int(g.groupby("arm").ratio.mean().idxmin()) for c, g in d.groupby("cls")}
    (R / f"fr_choice_{space}_entrada256.json").write_text(json.dumps(dict(val=val), indent=1))
    print(space, "validadas:", len(val), "classes")


def run_space(space, shard, nshards, pilot):
    man = manifest()
    man = man[man.role == ("tune" if pilot else "test")]
    val = json.loads((R / f"fr_choice_{space}_entrada256.json").read_text())["val"]
    qo, pub = frozen(space)
    out = R / (f"fr_pilot_{space}_entrada256.csv" if pilot else f"fr_test_{space}_s{shard}of{nshards}_entrada256.csv")
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    new = not out.exists()
    t0 = time.perf_counter()
    cnt = 0
    for _, r in man.iterrows():
        if (int(r.ri) % nshards != shard) or ((r["type"], str(int(r.instance))) in done):
            continue
        if pilot and int(r.idx) != 0:
            continue
        P, a0, od, lb = load(r)
        m_, n_ = int(r.m), int(r.n)
        rows = []
        for me in METHODS:
            for s in range(N_SEED):
                mk, ev = run_fixed_or_agent(space, me, P, a0, m_, od, lb, n_, seed_of(space, int(r.ri), me, s, pilot), val[f"{r['type']}|{n_}"], qo, pub)
                rows.append(dict(space=space, type=r["type"], instance=int(r.instance), m=m_, n=n_, idx=int(r.idx), method=me, seed_i=s, makespan=mk, lb=lb, bks=int(r.best_known), T=T, evals=ev))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if cnt % 25 == 0:
            print(f"[fr {space} s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[fr {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default=None)
    ap.add_argument("--choose", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.tune:
        tune(a.tune, a.shard, a.nshards)
    if a.choose:
        choose(a.choose)
    if a.run:
        run_space(a.run, a.shard, a.nshards, a.pilot)
