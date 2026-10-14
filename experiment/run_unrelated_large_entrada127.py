"""
experiment/run_unrelated_large_entrada127.py
===============================================
Entrada 127 (Ciclo 5, frente 2): R||Cmax onde o MILP NAO alcanca. Na Entrada 118 o MILP-60s venceu as heuristicas ate n = 500; no piloto de desenho, em n = 1000, m = 50 o MILP de
60 s devolveu incumbentes ruins (ex.: 84 vs 62 do IG+LS, LB 52). Aqui: tamanho (1000, 50) x 4 distribuicoes (U[1,100], U[10,100], U[100,200], U[100,120]) x 8 reps = 32 instancias
(seed 300000 + 1000*dist + rep), T=1000 e T=5000. d/n escolhido em 24 instancias de AJUSTE separadas (seed 97000+; 6 por distribuicao; IG+LS aleatorio, T=1000) entre {0,02; 0,05; 0,10}
pela menor razao media makespan/LB classico. Estado: limiares pooled da Entrada 118. Exato: limite LP + MILP-60s.
Metodos: mct, ur_igls_random, ur_igls_ql (T=1000), ur_igls_ql_T5000. Saidas: results/unrelated_large_entrada127.csv, results/unrelated_large_exact_entrada127.csv,
results/dsel_unrelated_large_entrada127.json. Uso: python -m experiment.run_unrelated_large_entrada127 --tune --run --workers 6 ; --exact --workers 6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment.run_unrelated_entrada118 import DISTS, LS_STEPS, THR  # noqa: E402

N, M = 1000, 50
FRACS_L = [0.005, 0.01, 0.02, 0.05, 0.10]  # EMENDA 1 (Entrada 129): grade original {0,02; 0,05; 0,10} deu o otimo na borda (0,02); estendida so' no ajuste
N_REPS, N_TUNE = 8, 6
OUT = ROOT / "results" / "unrelated_large_entrada127.csv"
OUT_X = ROOT / "results" / "unrelated_large_exact_entrada127.csv"
DSEL = ROOT / "results" / "dsel_unrelated_large_entrada127.json"
FIELDS = ["lo", "hi", "n", "m", "rep", "method", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "d"]
METHODS = [("mct", 0), ("ur_igls_random", 1000), ("ur_igls_ql", 1000), ("ur_igls_ql_T5000", 5000)]


def inst_of(di, rep, base=300000):
    from pmsp.unrelated import UnrelatedInstance
    lo, hi = DISTS[di]
    return UnrelatedInstance.random(N, M, lo, hi, seed=base + 1000 * di + rep, name=f"urL_U{lo}_{hi}_r{rep}")


def _tune_one(args):
    di, rep, f = args
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, ur_ig_run
    inst = inst_of(di, rep, base=97000)
    r = ur_ig_run(inst, RandomSelector(seed=rep, n_actions=N_UR_DESTROY), max_iter=1000, d=max(2, round(f * N)), seed=rep, ls_steps=LS_STEPS)
    return f, r["makespan"] / inst.lower_bound


def tune(workers):
    jobs = [(di, r, f) for di in range(len(DISTS)) for r in range(N_TUNE) for f in FRACS_L]
    acc = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f, ratio in ex.map(_tune_one, jobs):
            acc.setdefault(f, []).append(ratio)
    means = {f: float(np.mean(v)) for f, v in acc.items()}
    best = min(means, key=means.get)
    print("[tune]", ", ".join(f"{f:.2f}:{v:.4f}" for f, v in means.items()), "->", best, flush=True)
    DSEL.write_text(json.dumps({"frac": best, "means": means}, indent=1))


def _run(t):
    from experiment import config as CFG
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, UrStateFn, mct_greedy, ur_ig_run
    di, rep, mt, T = t["di"], t["rep"], t["method"], t["T"]
    inst = inst_of(di, rep)
    thr = json.loads(THR.read_text())
    sf = UrStateFn({"f1": thr["f1"], "f2": thr["f2"]})
    d = max(2, round(json.loads(DSEL.read_text())["frac"] * N))
    seed = rep
    if mt == "mct":
        t0 = time.perf_counter()
        s = mct_greedy(inst)
        r = dict(makespan=s.makespan, gap_pct=100.0 * (s.makespan - inst.lower_bound) / inst.lower_bound, cpu_time=time.perf_counter() - t0, n_improvements=0)
    elif mt == "ur_igls_random":
        r = ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), max_iter=T, d=d, seed=seed, ls_steps=LS_STEPS)
    else:
        ql = QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN, epsilon_decay=CFG.QL_EPSILON_DECAY,
                            seed=seed, n_actions=N_UR_DESTROY)
        r = ur_ig_run(inst, ql, max_iter=T, d=d, seed=seed, state_fn=sf, ls_steps=LS_STEPS)
    lo, hi = DISTS[di]
    return dict(lo=lo, hi=hi, n=N, m=M, rep=rep, method=mt, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], d=d)


def _exact(t):
    from pmsp.unrelated_exact import lp_bound, milp_solve
    di, rep = t["di"], t["rep"]
    inst = inst_of(di, rep)
    lp = lp_bound(inst)
    i60, d60, p60, t60 = milp_solve(inst, 60.0)
    lo, hi = DISTS[di]
    return dict(lo=lo, hi=hi, n=N, m=M, rep=rep, lb_classic=inst.lower_bound, lb_lp=lp, milp60_inc=i60, milp60_dual=d60, milp60_proven=int(p60), milp60_t=t60)


def _drive(fn, tasks, out, fields, keyf, workers):
    done = set()
    if out.exists():
        done = {keyf(r) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if keyf(t) not in done]
    print(f"[tasks] {len(pend)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in pend}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 10 == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("tune", "run", "exact"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.tune:
        tune(a.workers)
    if a.run:
        tasks = [dict(di=di, rep=r, method=mt, T=T, lo=DISTS[di][0], hi=DISTS[di][1], n=N, m=M) for di in range(len(DISTS)) for r in range(N_REPS) for mt, T in METHODS]
        tasks.sort(key=lambda t: -t["T"])
        _drive(_run, tasks, OUT, FIELDS, lambda x: (str(x["lo"]), str(x["hi"]), str(x["rep"]), x["method"]), a.workers)
    if a.exact:
        tasks = [dict(di=di, rep=r, lo=DISTS[di][0], hi=DISTS[di][1]) for di in range(len(DISTS)) for r in range(N_REPS)]
        fields = ["lo", "hi", "n", "m", "rep", "lb_classic", "lb_lp", "milp60_inc", "milp60_dual", "milp60_proven", "milp60_t"]
        _drive(_exact, tasks, OUT_X, fields, lambda x: (str(x["lo"]), str(x["hi"]), str(x["rep"])), a.workers)


if __name__ == "__main__":
    main()
