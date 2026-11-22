"""
experiment/run_cycle52b_fixedsize_entrada285b.py (Entrada 285b, Ciclo 52)
=========================================================================
Na Entrada 285 o ILS classico (R2: reinsercao aleatoria + descida ate' o otimo local) sorteava o tamanho da perturbacao entre os cinco tamanhos (d/n em {0,05; 0,10; 0,20; 0,30; 0,40}); um ILS bem ajustado usaria um tamanho FIXO, em geral pequeno. Aqui o
R2 roda com CADA tamanho fixo (bracos F0 a F4 = tamanho k, agente fixo, T = 1.000, 5 sementes; as mesmas 12 classes x 100 instancias de teste) e o IG simples (G0, escolha aleatoria entre os cinco tamanhos) recebe a CPU de cada F_k (bracos H0 a H4: T = round(1000 K_k),
K_k = CPU(F_k)/CPU(G0) de 1.000 iteracoes, medido em sequencia em 2 instancias por classe, maquina em repouso; --calibrate). O relatorio escolhe, por classe, o MELHOR tamanho do R2 nas proprias instancias de teste (uma escolha 'oraculo', que FAVORECE o R2 e portanto
e' conservadora para a conclusao) e o compara com o IG simples a iteracoes iguais (G0 da Entrada 284) e a CPU igual (H_k*). Fases: --calibrate ; --pilot (2 instancias de AJUSTE de 4 classes, sem resultado de teste) ; --run --shard i --nshards n.
Uso: python -m experiment.run_cycle52b_fixedsize_entrada285b --calibrate | --pilot | --run --shard 0 --nshards 12
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
from experiment.run_cycle52_fulldescent_entrada285 import MAX_MOVES  # noqa: E402
from pmsp import ig_recon2 as R2  # noqa: E402

R = ROOT / "results"
N_TEST, N_SEED = 100, 5
NK = 5
KFILE = R / "fixedsize_cpu_ratio_entrada285b.json"
FIELDS = ["cls", "rep", "arm", "seed_i", "makespan", "lb", "T", "evals", "moves", "capped", "descents"]
AID = {f"{p}{k}": i for i, (p, k) in enumerate([(p, k) for p in ("F", "H") for k in range(NK)])}
OFFSET = 700_000_000


def seed_of(ci, rep, arm, s):
    return rep + 1000 * ci + 50000 * s + 1_000_000 * AID[arm] + OFFSET


def run(p, a0, m, od, n, recon, ls_on, ls_mode, T, seed, agent, fixed):
    st, d, ls = arm_table("S5", n)
    ls = np.full(len(st), ls_on, dtype=np.int64)
    out = R2.ig_agent_core_recon2(p, a0, m, st, d, ls, T, agent, fixed, seed, od, recon, ls_mode, MAX_MOVES)
    return int(out[0]), float(out[2]), int(out[5]), int(out[6]), int(out[7])


def run_arm(arm, p, a0, m, od, n, T, seed):
    k = int(arm[1])
    if arm[0] == "F":
        return run(p, a0, m, od, n, 1, 1, 2, T, seed, 0, k)       # ILS classico com o tamanho k fixo
    return run(p, a0, m, od, n, 0, 0, 1, T, seed, 1, 0)           # IG simples, escolha aleatoria entre os cinco tamanhos


def warm():
    inst, p, a0, od = make(20, 4, 0, False)
    for arm in ("F0", "H0"):
        run_arm(arm, p, a0, 4, od, 20, 20, 1)


def calibrate():
    warm()
    res = {}
    for ci, (n, m) in enumerate(CLASSES):
        ratios = {f"F{k}": [] for k in range(NK)}
        for rep in range(2):
            inst, p, a0, od = make(n, m, rep, False)

            def cpu(arm):
                reps, t0 = 0, time.process_time()
                while time.process_time() - t0 < 1.0:
                    run_arm(arm, p, a0, m, od, n, 1000, reps + 7)
                    reps += 1
                return (time.process_time() - t0) / reps
            base = cpu("H0")
            for k in range(NK):
                ratios[f"F{k}"].append(cpu(f"F{k}") / base)
        res[f"{n}x{m}"] = {k: float(np.median(v)) for k, v in ratios.items()}
        print(f"[calib] {n}x{m}: {res[f'{n}x{m}']}", flush=True)
    KFILE.write_text(json.dumps(res, indent=1))


def run_instance(ci, n, m, rep, tune, K):
    inst, p, a0, od = make(n, m, rep, tune)
    lb = inst.lower_bound
    cls = f"rand_{n}j_{m}m"
    rows = []
    for arm in AID:
        T = 1000 if arm[0] == "F" else max(10, int(round(1000 * K[f"{n}x{m}"][f"F{arm[1]}"])))
        for s in range(N_SEED):
            mk, ev, mv, cp, nd = run_arm(arm, p, a0, m, od, n, T, seed_of(ci, rep, arm, s))
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
        mk = {a: round(float(np.mean([r["makespan"] for r in rows if r["arm"] == a])), 1) for a in AID}
        Ts = {a: next(r["T"] for r in rows if r["arm"] == a) for a in AID}
        print(f"[pilot] {n}x{m}: {time.perf_counter() - t0:.1f}s para 2 instancias de ajuste; makespan medio {mk}; T {Ts}", flush=True)


def run_shard(shard, nshards):
    K = json.loads(KFILE.read_text())
    out = R / f"fixedsize_test_s{shard}of{nshards}_entrada285b.csv"
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
