"""
experiment/run_cycle27_closedloop_entrada249.py
================================================
Entrada 249 (Ciclo 27): ORACULO DE MALHA FECHADA (dependente do estado). As tres revisoes cegas da Entrada 246 observam que a folga do artigo e' a de um oraculo ESTATICO (a melhor acao fixa por instancia) e que o que um controle online pretende explorar e' a troca de acao dentro da execucao. O cronograma de tres fases (Entrada 238) ja' testou a troca no tempo por classe; aqui o limite e' por estado e por instancia: em cada iteracao, `m_roll` = 8 execucoes independentes de cada acao a partir da solucao atual (pmsp/closedloop_numba.py) escolhem a acao de maior melhora esperada, que e' executada com uma execucao nova. NAO e' justo no orcamento; e' um LIMITE para politicas que decidem pelo estado.
Desenho: P||Cmax sintetico e R||Cmax, 12 classes x 100 instancias de TESTE (as mesmas das baterias das Entradas 180 e 234, para parear com `rand`, `val` e as acoes fixas), espacos S5 e S15, T = 1000, 2 sementes por instancia; `la8` (oraculo, m_roll = 8) e `la0` (controle: mesmo codigo, acao sorteada, sem avaliacoes; deve reproduzir o aleatorio). Nada e' ajustado.
Fases (retomaveis): --run HOST ESPACO --shard i --nshards n [--pilot: 3 instancias de AJUSTE por classe, saida results/cl_pilot_*]. HOST = pc | ur. Saidas: results/cl_HOST_ESPACO_s*of*_entrada249.csv. Pre-registro: analysis/report_entrada249_closedloop.py (docstring; hashes em results/entrada249_prereg_hashes.txt).
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

from experiment import run_agents_numba_entrada180 as PC  # noqa: E402
from experiment import run_cycle22_ur_entrada234 as UR  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from pmsp import closedloop_numba as CL  # noqa: E402

R = ROOT / "results"
HOSTS = ("pc", "ur")
SPACES = ("S5", "S15")
N_TEST, N_SEED, M_ROLL = 100, 2, 8
M_ROLL_RO, K_DEC, H_TAIL, N_RO = 12, 50, -1, 20
FIELDS = ["host", "space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals", "evals_roll"]


def seed_of(host, space, ci, rep, oracle, s):
    return 500_000_000 + 100_000_000 * HOSTS.index(host) + 50_000_000 * SPACES.index(space) + 5_000_000 * oracle + 20_000 * ci + 5 * rep + s


def instance(host, n, m, rep, ci, tune):
    if host == "pc":
        inst, p, a0, od = PC.make(n, m, rep, tune)
        return p, a0, od, float(inst.lower_bound)
    P, a0, od, lb = UR.make(n, m, rep, ci, tune)
    return P, a0, od, float(lb)


def run_la(host, space, X, a0, m, od, n, oracle, seed, T=1000, m_roll=M_ROLL):
    st, d, ls = arm_table(space, n)
    fn = CL.pc_lookahead if host == "pc" else CL.ur_lookahead
    best, ev, evr, used = fn(X, a0, m, st, d, ls, T, m_roll, seed, od, oracle)
    return int(best), float(ev), float(evr)


def run_ro(host, space, X, a0, m, od, n, base, seed, T=1000):
    st, d, ls = arm_table(space, n)
    fn = CL.pc_rollout if host == "pc" else CL.ur_rollout
    best, ev, evr, used = fn(X, a0, m, st, d, ls, T, M_ROLL_RO, K_DEC, H_TAIL, base, seed, od)
    return int(best), float(ev), float(evr)


def base_arms(host, space):
    f = R / (f"agents_choice_{space}_entrada180.json" if host == "pc" else f"ur_choice_{space}_entrada234.json")
    return {k: int(v) for k, v in json.loads(f.read_text())["val"].items()}


def warm():
    for host in HOSTS:
        X, a0, od, lb = instance(host, 20, 4, 0, 0, True)
        for sp in SPACES:
            for oracle in (0, 1):
                run_la(host, sp, X, a0, 4, od, 20, oracle, 1, T=5, m_roll=2)
            run_ro(host, sp, X, a0, 4, od, 20, 0, 1, T=10)


def run_space(host, space, shard, nshards, pilot):
    out = R / (f"cl_pilot_{host}_{space}_entrada249.csv" if pilot else f"cl_{host}_{space}_s{shard}of{nshards}_entrada249.csv")
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    val = base_arms(host, space)
    new = not out.exists()
    t0 = time.perf_counter()
    cnt = 0
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(3 if pilot else N_TEST):
            if (not pilot) and (ci * N_TEST + rep) % nshards != shard:
                continue
            if (cls, str(rep)) in done:
                continue
            X, a0, od, lb = instance(host, n, m, rep, ci, bool(pilot))
            rows = []
            for oracle, name in ((0, "la0"), (1, "la8")):
                for s in range(N_SEED):
                    mk, ev, evr = run_la(host, space, X, a0, m, od, n, oracle, seed_of(host, space, ci, rep, oracle, s))
                    rows.append(dict(host=host, space=space, cls=cls, rep=rep, method=name, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev, evals_roll=evr))
            if rep < (3 if pilot else N_RO):
                mk, ev, evr = run_ro(host, space, X, a0, m, od, n, val[cls], seed_of(host, space, ci, rep, 2, 0))
                rows.append(dict(host=host, space=space, cls=cls, rep=rep, method="ro", seed_i=0, makespan=mk, lb=lb, T=1000, evals=ev, evals_roll=evr))
            with open(out, "w" if new else "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            new = False
            cnt += 1
            if cnt % 25 == 0:
                print(f"[cl {host} {space} s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[cl {host} {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, metavar=("HOST", "SPACE"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.run:
        run_space(a.run[0], a.run[1], a.shard, a.nshards, a.pilot)
