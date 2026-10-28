"""
experiment/run_cycle19_ga_mix_entrada220.py
=============================================
Entrada 220 (Ciclo 19, frente F, ESTAGIO 2; registrado depois de ver o estagio 1 e antes de rodar): no hospedeiro populacional as acoes sao COMPLEMENTARES -- uma acao unica, mesmo validada, pode perder para a mistura uniforme (a validada ficou -0,057 pontos contra o aleatorio em G9). O competidor estatico adequado e' uma MISTURA validada:
vmix = escolha UNIFORME ao acaso entre as TRES melhores acoes unicas de cada classe (menor lacuna media ao limite inferior nas 30 instancias de AJUSTE; k = 3 fixado aqui, sem ajuste), mesmo orcamento. Metodos (5 sementes, T = 10.000, as mesmas 1.200 instancias de teste): vmix a iteracoes iguais e vmix_eq_<aprendiz> a CPU igual do aprendiz (T = round(10000 K_aprendiz / K_vmix);
K_vmix calibrado em sequencia como os demais). Fases (retomaveis; a calibracao SOZINHA): --calibrate | --run ESPACO --shard i --nshards n. Saidas: results/gamix_k_entrada220.json, results/gamix_test_ESPACO_s*of*_entrada220.csv. Pre-registro: analysis/report_entrada220_ga_mix.py (docstring; hashes em results/entrada220_prereg_hashes.txt).
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

from experiment import run_cycle19_ga_entrada219 as GA  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES  # noqa: E402
from pmsp import ga_numba as G  # noqa: E402

R = ROOT / "results"
K_TOP = 3


def top_arms():
    out = {}
    for sp in ("G9", "G18"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ga_tune_{sp}_s*of*_entrada219.csv")))], ignore_index=True)
        d["gap"] = 100 * (d.makespan - d.lb) / d.lb
        out[sp] = {}
        for cls, x in d.groupby("cls"):
            g = x[x.method.str.startswith("arm")].groupby("method").gap.mean().sort_values()
            out[sp][cls] = [int(m[3:]) for m in g.index[:K_TOP]]
    return out


def run_mix(space, subset, p, a0, m, od, lb, seed, T):
    xo, mu, ls = GA.arms(space)
    sub = np.array(subset)
    best, _, ev, used, _, _ = G.ga_agent_core(p, a0, m, xo[sub], mu[sub], ls[sub], T, G.AGENT_RANDOM, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, G.HP_DEFAULT, 0)
    return int(best), float(ev)


def calibrate():
    top = top_arms()
    GA.warm()
    res = {}
    for sp in ("G9", "G18"):
        res[sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            cls = f"{n}x{m}"
            ratios = []
            for rep in range(2):
                p, a0, od, lb = GA.inst(ci, n, m, rep, False)

                def cpu(fn):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        fn(reps + 7)
                        reps += 1
                    return (time.process_time() - t0) / reps
                base = cpu(lambda sd: GA.run(sp, p, a0, m, od, lb, G.AGENT_RANDOM, 0, 0.1, 0.1, sd))
                ratios.append(cpu(lambda sd: run_mix(sp, top[sp][cls], p, a0, m, od, lb, sd, GA.T0)) / base)
            res[sp][cls] = float(np.median(ratios))
            print(f"[gamix-calib {sp}] {cls}: vmix {res[sp][cls]:.2f}", flush=True)
    (R / "gamix_k_entrada220.json").write_text(json.dumps(res, indent=1))
    (R / "gamix_top_entrada220.json").write_text(json.dumps(top, indent=1))


def _append(out, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GA.FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def run_space(space, shard, nshards):
    top = json.loads((R / "gamix_top_entrada220.json").read_text())[space]
    Kv = json.loads((R / "gamix_k_entrada220.json").read_text())[space]
    Kl = json.loads(GA.k_path().read_text())[space]
    out = R / f"gamix_test_{space}_s{shard}of{nshards}_entrada220.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    GA.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    items = [(ci, n, m, rep) for ci, (n, m) in enumerate(CLASSES) for rep in range(GA.N_TEST)][shard::nshards]
    for i, (ci, n, m, rep) in enumerate(items):
        cls = f"{n}x{m}"
        if (cls, str(rep)) in done:
            continue
        p, a0, od, lb = GA.inst(ci, n, m, rep, False)
        rows = []
        for s in range(GA.N_SEED_TEST):
            mk, ev = run_mix(space, top[cls], p, a0, m, od, lb, GA.seed_of(space, "test", ci, rep, "vmix", s), GA.T0)
            rows.append(dict(space=space, cls=cls, rep=rep, method="vmix", seed_i=s, makespan=mk, lb=lb, T=GA.T0, evals=ev))
            for k in GA.LEARNERS:
                Teq = max(200, int(round(GA.T0 * Kl[cls][k] / Kv[cls])))
                mk, ev = run_mix(space, top[cls], p, a0, m, od, lb, GA.seed_of(space, "test", ci, rep, "vmixeq" + k, s), Teq)
                rows.append(dict(space=space, cls=cls, rep=rep, method=f"vmix_eq_{k}", seed_i=s, makespan=mk, lb=lb, T=Teq, evals=ev))
        _append(out, rows, new)
        new = False
        if i % 100 == 0:
            print(f"[gamix {space} s{shard}] {i}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[gamix {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.run:
        run_space(a.run, a.shard, a.nshards)
