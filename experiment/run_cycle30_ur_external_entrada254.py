"""
experiment/run_cycle30_ur_external_entrada254.py
=================================================
Entrada 254 (Ciclo 30): o protocolo no R||Cmax com as FAMILIAS PUBLICADAS de instancias (pmsp/instance_ur_ext.py: nao correlacionada U{10,100}, correlacionada por tarefa e por maquina; 25 tamanhos por familia, m em {3, 5, 10, 15, 20} e n em {30, 50, 80, 100, 200}), isto e', fora do gerador que ajustou tudo o mais (tempos independentes U[1, 100], 12 classes). As revisoes cegas das Entradas 246-252 apontam 'um so' gerador, 12 classes' e 'implementacoes proprias'.
Desenho (o da Entrada 234): classe = (familia, n, m) (75 por espaco); 20 instancias de AJUSTE por classe (a acao validada por classe: a melhor acao fixa em 2 sementes por instancia) e 20 de TESTE por classe; espacos S5 e S15; T = 1.000; metodos rand, ql, ucb, pg, alns, qo (5 sementes cada), val (5 sementes) e cada acao fixa (4 sementes: o oraculo estatico com validacao cruzada de sementes e' o da Entrada 234). Hiperparametros dos aprendizes CONGELADOS das Entradas 180/234 (lr do pg 0,03 em S5 e 0,3 em S15; roleta ALNS (0,1; 0); Q otimista de results/ur_choice_qo_*_entrada234.json): nada e' ajustado ao gerador novo, so' a acao validada por classe.
Fases (retomaveis): --tune ESPACO | --run ESPACO --shard i --nshards n [--pilot: 2 instancias de AJUSTE por classe, saida results/ur_ext_pilot_*]. Saidas: results/ur_ext_choice_ESPACO_entrada254.json, results/ur_ext_test_ESPACO_s*of*_entrada254.csv. Pre-registro: analysis/report_entrada254_ur_external.py (docstring; hashes em results/entrada254_prereg_hashes.txt).
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

from experiment import run_cycle22_ur_entrada234 as U  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from pmsp import ur_numba_agents as UA  # noqa: E402
from pmsp.instance_ur import constructive_ur, lower_bound_ur  # noqa: E402
from pmsp.instance_ur_ext import FAMILIES, SIZES, gen  # noqa: E402

R = ROOT / "results"
SPACES = U.SPACES
N_TUNE, N_TEST, N_SEED_ARM = 20, 20, 4
METHODS = ["rand", "ql", "ucb", "pg", "alns", "val", "qo"]
CLASSES = [(f, n, m) for f in FAMILIES for (n, m) in SIZES]
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]


def cls_name(f, n, m):
    return f"{f}|{n}x{m}"


def choice_path(space):
    return R / f"ur_ext_choice_{space}_entrada254.json"


def make(ci, rep, tune):
    f, n, m = CLASSES[ci]
    seed = 1_000_000_000 + 100_000 * ci + rep + (50_000 if tune else 0)
    P = gen(f, n, m, seed)
    a0, od = constructive_ur(P)
    return P, a0, od, lower_bound_ur(P)


def seed_of(space, ci, rep, method, s, tune=False):
    k = METHODS.index(method) if not method.startswith("arm") else 7 + int(method[3:])
    return (1_300_000_000 if tune else 800_000_000) + 100_000_000 * SPACES.index(space) + 4_000_000 * k + 20_000 * ci + 5 * rep + s


def run_method(space, method, P, a0, m, od, lb, n, seed, val_arm, qo_cfg):
    if method == "qo":
        return U.run_qo(space, P, a0, m, od, lb, n, qo_cfg, seed)
    if method == "val":
        return U.run(space, P, a0, m, od, lb, n, UA.AGENT_FIXED, val_arm, 0.1, 0.1, seed)
    if method.startswith("arm"):
        return U.run(space, P, a0, m, od, lb, n, UA.AGENT_FIXED, int(method[3:]), 0.1, 0.1, seed)
    ag, key = U.AGENTS[method]
    lr, lr2 = (U.PG_LR[space], 0.1) if key == "pg" else (U.ALNS_HP if key == "alns" else (0.1, 0.1))
    return U.run(space, P, a0, m, od, lb, n, ag, 0, lr, lr2, seed)


def tune(space):
    U.warm()
    A = len(arm_table(space, 100)[0])
    rows = []
    for ci, (f, n, m) in enumerate(CLASSES):
        for rep in range(N_TUNE):
            P, a0, od, lb = make(ci, rep, True)
            for a in range(A):
                for s in range(2):
                    mk, ev = run_method(space, f"arm{a}", P, a0, m, od, lb, n, seed_of(space, ci, rep, f"arm{a}", s, True), 0, None)
                    rows.append((cls_name(f, n, m), a, mk))
    import pandas as pd
    d = pd.DataFrame(rows, columns=["cls", "arm", "makespan"])
    val = {c: int(g.groupby("arm").makespan.mean().idxmin()) for c, g in d.groupby("cls")}
    choice_path(space).write_text(json.dumps(dict(val=val), indent=1))
    print(space, "ajuste concluido;", len(val), "classes")


def run_space(space, shard, nshards, pilot):
    val = json.loads(choice_path(space).read_text())["val"]
    qo_cfg = tuple(json.loads(U.choice_qo_path(space).read_text())["cfg"])
    out = R / (f"ur_ext_pilot_{space}_entrada254.csv" if pilot else f"ur_ext_test_{space}_s{shard}of{nshards}_entrada254.csv")
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    U.warm()
    A = len(arm_table(space, 100)[0])
    new = not out.exists()
    t0 = time.perf_counter()
    cnt = 0
    for ci, (f, n, m) in enumerate(CLASSES):
        cls = cls_name(f, n, m)
        for rep in range(2 if pilot else N_TEST):
            if (not pilot) and (ci * N_TEST + rep) % nshards != shard:
                continue
            if (cls, str(rep)) in done:
                continue
            P, a0, od, lb = make(ci, rep, bool(pilot))
            rows = []
            for me in METHODS:
                for s in range(5):
                    mk, ev = run_method(space, me, P, a0, m, od, lb, n, seed_of(space, ci, rep, me, s), val[cls], qo_cfg)
                    rows.append(dict(space=space, cls=cls, rep=rep, method=me, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
            for a in range(A):
                for s in range(N_SEED_ARM):
                    mk, ev = run_method(space, f"arm{a}", P, a0, m, od, lb, n, seed_of(space, ci, rep, f"arm{a}", s), 0, None)
                    rows.append(dict(space=space, cls=cls, rep=rep, method=f"arm{a}", seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
            with open(out, "w" if new else "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
                fh.flush()
                os.fsync(fh.fileno())
            new = False
            cnt += 1
            if cnt % 100 == 0:
                print(f"[ext {space} s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[ext {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.tune:
        tune(a.tune)
    if a.run:
        run_space(a.run, a.shard, a.nshards, a.pilot)
