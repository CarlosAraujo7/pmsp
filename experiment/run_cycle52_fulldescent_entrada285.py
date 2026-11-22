"""
experiment/run_cycle52_fulldescent_entrada285.py (Entrada 285, Ciclo 52)
========================================================================
A ablacao da Entrada 284 tinha uma ressalva: a descida da variante com reinsercao aleatoria (R1) era a curta do IG (no maximo DUAS melhorias), que nao repara uma perturbacao pesada, de modo que R1 nao e' um ILS forte. Aqui a mesma
estrutura recebe a DESCIDA COMPLETA (pmsp/ig_recon2.py: troca entre maquinas e reinsercao a partir da maquina gargalo ate' o otimo local; tampo de 10^6 movimentos so' como salvaguarda): G2 = reconstrucao gulosa + descida completa, R2 = reinsercao
aleatoria + descida completa = o ILS classico para P||Cmax (perturbacao de d tarefas ao acaso, descida ate' o otimo local, aceita se nao piora). Mesmas 12 classes x 100 instancias de teste, partida LPT, T = 1.000, espaco S5 com a acao sorteada, 5 sementes, sem aprendiz.
CPU igual como no protocolo (K = CPU(primeiro)/CPU(segundo) de 1.000 iteracoes, medido em sequencia em 2 instancias por classe, maquina em repouso; o segundo braco recebe round(1000 K) iteracoes):
  Ea = R2 com K(G2/R2) (R2 com a CPU de G2), Eb = G0 com K(R2/G0) (o IG simples com a CPU do ILS classico), Ec = G1 com K(R2/G1) (o IG hibrido com a CPU do ILS classico).
G0, G1, R1 a T = 1.000 (iteracoes iguais) vem da Entrada 284 (mesmas instancias, outras sementes; pareamento por instancia). Fases: --calibrate ; --pilot (2 instancias de AJUSTE de 4 classes, sem resultado de teste) ; --run --shard i --nshards n.
Uso: python -m experiment.run_cycle52_fulldescent_entrada285 --calibrate | --pilot | --run --shard 0 --nshards 12
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

from experiment.run_agents_numba_entrada180 import CLASSES, arm_table, make  # noqa: E402
from pmsp import ig_recon2 as R2  # noqa: E402

R = ROOT / "results"
N_TEST, N_SEED = 100, 5
MAX_MOVES = 10 ** 6
KFILE = R / "fulldescent_cpu_ratio_entrada285.json"
FIELDS = ["cls", "rep", "arm", "seed_i", "makespan", "lb", "T", "evals", "moves", "capped", "descents"]
# braco -> (reconstrucao, descida aplicada?, modo da descida, chave de K ou None)
ARMS = {"G2": (0, 1, 2, None), "R2": (1, 1, 2, None), "Ea": (1, 1, 2, "G2/R2"), "Eb": (0, 0, 1, "R2/G0"), "Ec": (0, 1, 1, "R2/G1")}
BASE = {"G0": (0, 0, 1), "G1": (0, 1, 1), "G2": (0, 1, 2), "R2": (1, 1, 2)}
AID = {a: i for i, a in enumerate(ARMS)}
OFFSET = 600_000_000


def seed_of(ci, rep, arm, s):
    return rep + 1000 * ci + 50000 * s + 1_000_000 * AID[arm] + OFFSET


def run(p, a0, m, od, n, recon, ls_on, ls_mode, T, seed):
    st, d, ls = arm_table("S5", n)
    ls = np.full(len(st), ls_on, dtype=np.int64)
    out = R2.ig_agent_core_recon2(p, a0, m, st, d, ls, T, 1, 0, seed, od, recon, ls_mode, MAX_MOVES)
    return int(out[0]), float(out[2]), int(out[5]), int(out[6]), int(out[7])


def warm():
    inst, p, a0, od = make(20, 4, 0, False)
    for rc in (0, 1):
        for lo, lm in ((0, 1), (1, 1), (1, 2)):
            run(p, a0, 4, od, 20, rc, lo, lm, 20, 1)


def calibrate():
    warm()
    res = {}
    for ci, (n, m) in enumerate(CLASSES):
        ratios = {k: [] for k in ("G2/R2", "R2/G0", "R2/G1")}
        for rep in range(2):
            inst, p, a0, od = make(n, m, rep, False)

            def cpu(name):
                rc, lo, lm = BASE[name]
                reps, t0 = 0, time.process_time()
                while time.process_time() - t0 < 1.0:
                    run(p, a0, m, od, n, rc, lo, lm, 1000, reps + 7)
                    reps += 1
                return (time.process_time() - t0) / reps
            c = {k: cpu(k) for k in BASE}
            for key in ratios:
                x, y = key.split("/")
                ratios[key].append(c[x] / c[y])
        res[f"{n}x{m}"] = {k: float(np.median(v)) for k, v in ratios.items()}
        print(f"[calib] {n}x{m}: {res[f'{n}x{m}']}", flush=True)
    KFILE.write_text(json.dumps(res, indent=1))


def run_instance(ci, n, m, rep, tune, K):
    inst, p, a0, od = make(n, m, rep, tune)
    lb = inst.lower_bound
    cls = f"rand_{n}j_{m}m"
    rows = []
    for arm, (rc, lo, lm, kk) in ARMS.items():
        T = 1000 if kk is None else max(10, int(round(1000 * K[f"{n}x{m}"][kk])))
        for s in range(N_SEED):
            mk, ev, mv, cp, nd = run(p, a0, m, od, n, rc, lo, lm, T, seed_of(ci, rep, arm, s))
            rows.append(dict(cls=cls, rep=rep, arm=arm, seed_i=s, makespan=mk, lb=lb, T=T, evals=ev, moves=mv, capped=cp, descents=nd))
    return rows


def pilot():
    K = json.loads(KFILE.read_text())
    warm()
    for ci in (0, 5, 9, 11):
        n, m = CLASSES[ci]
        t0 = time.perf_counter()
        rows = []
        for rep in range(2):
            rows += run_instance(ci, n, m, rep, True, K)
        mk = {a: float(np.mean([r["makespan"] for r in rows if r["arm"] == a])) for a in ARMS}
        Ts = {a: next(r["T"] for r in rows if r["arm"] == a) for a in ARMS}
        cap = sum(r["capped"] for r in rows)
        print(f"[pilot] {n}x{m}: {time.perf_counter() - t0:.1f}s para 2 instancias de ajuste; makespan medio {mk}; T {Ts}; descidas truncadas {cap}", flush=True)


def run_shard(shard, nshards):
    K = json.loads(KFILE.read_text())
    out = R / f"fulldescent_test_s{shard}of{nshards}_entrada285.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    new = not out.exists()
    t00 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for ci, (n, m) in enumerate(CLASSES):
            if ci % nshards != shard:
                continue
            cls = f"rand_{n}j_{m}m"
            for rep in range(N_TEST):
                if (cls, str(rep)) in done:
                    continue
                w.writerows(run_instance(ci, n, m, rep, False, K))
                if rep % 10 == 9:
                    f.flush()
                    os.fsync(f.fileno())
            print(f"[{shard}] {n}x{m} ok ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=12)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.pilot:
        pilot()
    if a.run:
        run_shard(a.shard, a.nshards)
