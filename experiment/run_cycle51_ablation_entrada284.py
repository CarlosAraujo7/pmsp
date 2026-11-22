"""
experiment/run_cycle51_ablation_entrada284.py (Entrada 284, Ciclo 51)
=====================================================================
ABLACAO DOS COMPONENTES DA ESTRUTURA DO IG (resposta a revisao duplo-cega do CEC de 18/11/2026, ponto P4: o ILS do artigo e' um comparador fraco e nao ha' ablacao do IG). No IG compilado de P||Cmax (12 classes sinteticas x 100 instancias
de teste, as mesmas da Entrada 180; partida LPT; T = 1.000; aceita se nao piora; espaco S5: cinco tamanhos de destruicao d/n em {0,05; 0,10; 0,20; 0,30; 0,40}, destruicao aleatoria; a acao e' sorteada uniformemente a cada iteracao: SEM aprendiz)
cruza-se a RECONSTRUCAO (gulosa: nao crescente em tempo, para a maquina de menor carga; ou ALEATORIA: cada tarefa removida vai para uma maquina sorteada) com a BUSCA LOCAL apos a reconstrucao (sem; com a descida curta do IG):
  G0 = gulosa sem descida (o IG simples do artigo), R0 = aleatoria sem descida, G1 = gulosa com descida (o IG hibrido), R1 = aleatoria com descida (= um ILS com perturbacao FORTE: d tarefas reinseridas ao acaso, descida curta, aceitacao se nao piora).
5 sementes por instancia (como o aleatorio da Entrada 180). CPU igual como no protocolo: K = CPU(primeiro braco)/CPU(segundo) de 1.000 iteracoes, medido em sequencia em 2 instancias por classe (maquina em repouso; --calibrate); o segundo braco recebe round(1000 K) iteracoes:
  E1 = R0 com K(G0/R0), E2 = R1 com K(G1/R1), E3 = G0 com K(G1/G0), E4 = R0 com K(R1/R0), E5 = G0 com K(R1/G0).
Fases: --calibrate ; --pilot (2 instancias de AJUSTE da menor e da maior classe, sem resultado de teste) ; --run --shard i --nshards n (resultados em results/ablation_test_s{i}of{n}_entrada284.csv). Retomavel.
Uso: python -m experiment.run_cycle51_ablation_entrada284 --calibrate | --pilot | --run --shard 0 --nshards 12
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
from pmsp import ig_recon as RC  # noqa: E402

R = ROOT / "results"
N_TEST, N_SEED = 100, 5
KFILE = R / "ablation_cpu_ratio_entrada284.json"
FIELDS = ["cls", "rep", "arm", "seed_i", "makespan", "lb", "T", "evals"]
# braco -> (reconstrucao, busca local, K de calibracao ou None)
ARMS = {
    "G0": (0, 0, None), "R0": (1, 0, None), "G1": (0, 1, None), "R1": (1, 1, None),
    "E1": (1, 0, "G0/R0"), "E2": (1, 1, "G1/R1"), "E3": (0, 0, "G1/G0"), "E4": (1, 0, "R1/R0"), "E5": (0, 0, "R1/G0"),
}
AID = {a: i for i, a in enumerate(ARMS)}
BASE = {"G0": (0, 0), "R0": (1, 0), "G1": (0, 1), "R1": (1, 1)}
OFFSET = 500_000_000


def seed_of(ci, rep, arm, s):
    return rep + 1000 * ci + 50000 * s + 1_000_000 * AID[arm] + OFFSET


def run(p, a0, m, od, n, recon, ls_on, T, seed):
    st, d, ls = arm_table("S5", n)
    ls = np.full(len(st), ls_on, dtype=np.int64)
    best, _, ev, _, _ = RC.ig_agent_core_recon(p, a0, m, st, d, ls, T, 1, 0, seed, od, recon)
    return int(best), float(ev)


def warm():
    inst, p, a0, od = make(20, 4, 0, False)
    for rc in (0, 1):
        for ls_on in (0, 1):
            run(p, a0, 4, od, 20, rc, ls_on, 20, 1)


def calibrate():
    warm()
    res = {}
    for ci, (n, m) in enumerate(CLASSES):
        ratios = {k: [] for k in ("G0/R0", "G1/R1", "G1/G0", "R1/R0", "R1/G0")}
        for rep in range(2):
            inst, p, a0, od = make(n, m, rep, False)

            def cpu(name):
                rc, ls_on = BASE[name]
                reps, t0 = 0, time.process_time()
                while time.process_time() - t0 < 1.0:
                    run(p, a0, m, od, n, rc, ls_on, 1000, reps + 7)
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
    for arm, (rc, ls_on, kk) in ARMS.items():
        T = 1000 if kk is None else max(50, int(round(1000 * K[f"{n}x{m}"][kk])))
        for s in range(N_SEED):
            mk, ev = run(p, a0, m, od, n, rc, ls_on, T, seed_of(ci, rep, arm, s))
            rows.append(dict(cls=cls, rep=rep, arm=arm, seed_i=s, makespan=mk, lb=lb, T=T, evals=ev))
    return rows


def pilot():
    K = json.loads(KFILE.read_text())
    warm()
    for ci in (0, 11):
        n, m = CLASSES[ci]
        t0 = time.perf_counter()
        rows = []
        for rep in range(2):
            rows += run_instance(ci, n, m, rep, True, K)
        mk = {a: float(np.mean([r["makespan"] for r in rows if r["arm"] == a])) for a in ARMS}
        Ts = {a: next(r["T"] for r in rows if r["arm"] == a) for a in ARMS}
        print(f"[pilot] {n}x{m}: {time.perf_counter() - t0:.1f}s para 2 instancias de ajuste; makespan medio {mk}; T {Ts}", flush=True)


def run_shard(shard, nshards):
    K = json.loads(KFILE.read_text())
    out = R / f"ablation_test_s{shard}of{nshards}_entrada284.csv"
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
