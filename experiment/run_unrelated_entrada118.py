"""
experiment/run_unrelated_entrada118.py
=======================================
Entrada 118: o protocolo "mecanismo vs politica" em R||Cmax (maquinas nao relacionadas), variante em que os metodos exatos nao acompanham o
tamanho. 4 distribuicoes de p_ij (U[1,100], U[10,100], U[100,200], U[100,120]) x 5 tamanhos (n,m) in {(50,10), (100,10), (100,20), (200,20), (500,30)}
x 20 repeticoes = 400 instancias de TESTE (seed 100000 + 1000*celula + rep). T=1000 para todos os metodos iterativos.

Desenho (licao do P||Cmax: o tamanho de destruicao importa e nao deve ser fixado ao acaso):
  --tune       escolhe, por TAMANHO, a fracao de destruicao d/n em {0,02; 0,05; 0,10; 0,20; 0,30} para o IG com busca local e selecao aleatoria, em instancias
               de AJUSTE separadas (seed 96000+; 6 por celula = 24 por tamanho); criterio: menor razao media makespan/LB classico -> results/dsel_unrelated_entrada118.json
  --calibrate  limiares POOLED do estado relativo (`rel`, Entrada 98) com ig+LS aleatorio em instancias de CALIBRACAO separadas (seed 95000+, 3 por celula)
  --run        TESTE: mct; ur_ils_random, ur_ils_ql; ur_igls_random, ur_igls_ql, ur_igls_ucb, ur_igls_fix0..3 (estrategia de destruicao fixa; folga);
               ur_ig_random, ur_ig_ql (IG puro, sem busca local: ablacao da busca local). IG com busca local = descida mover/trocar, ate 20 melhorias por iteracao.
  --exact      LP (limite certificado) e MILP (HiGHS) com 5 s e 60 s nas repeticoes 0-7 de cada celula (160 instancias)
Saidas: results/unrelated_entrada118.csv, results/unrelated_exact_entrada118.csv, results/state_calibration_unrelated_entrada118.json.
Uso: python -m experiment.run_unrelated_entrada118 --tune --calibrate --run --workers 12 ; --exact --workers 8 ; --smoke
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

DISTS = [(1, 100), (10, 100), (100, 200), (100, 120)]
SIZES = [(50, 10), (100, 10), (100, 20), (200, 20), (500, 30)]
CELLS = [(lo, hi, n, m) for (lo, hi) in DISTS for (n, m) in SIZES]
FRACS = [0.02, 0.05, 0.10, 0.20, 0.30]
N_REPS, N_EXACT, N_CAL, N_TUNE, LS_STEPS = 20, 8, 3, 6, 20
METHODS = ["mct", "ur_ils_random", "ur_ils_ql", "ur_igls_random", "ur_igls_ql", "ur_igls_ucb", "ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3",
           "ur_ig_random", "ur_ig_ql"]
OUT = ROOT / "results" / "unrelated_entrada118.csv"
OUT_X = ROOT / "results" / "unrelated_exact_entrada118.csv"
THR = ROOT / "results" / "state_calibration_unrelated_entrada118.json"
DSEL = ROOT / "results" / "dsel_unrelated_entrada118.json"
FIELDS = ["lo", "hi", "n", "m", "rep", "method", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "d"]


def inst_of(cell, rep, base=100000):
    from pmsp.unrelated import UnrelatedInstance
    lo, hi, n, m = cell
    return UnrelatedInstance.random(n, m, lo, hi, seed=base + 1000 * CELLS.index(cell) + rep, name=f"ur_{n}j_{m}m_U{lo}_{hi}_r{rep}")


def d_of(n, m):
    return max(2, round(json.loads(DSEL.read_text())[f"{n}x{m}"] * n))


def _tune_one(args):
    cell, rep, f = args
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, ur_ig_run
    inst = inst_of(cell, rep, base=96000)
    r = ur_ig_run(inst, RandomSelector(seed=rep, n_actions=N_UR_DESTROY), max_iter=1000, d=max(2, round(f * inst.n)), seed=rep, ls_steps=LS_STEPS)
    return cell[2], cell[3], f, r["makespan"] / inst.lower_bound


def tune(workers):
    jobs = [(c, r, f) for c in CELLS for r in range(N_TUNE) for f in FRACS]
    jobs.sort(key=lambda j: -j[0][2] * j[0][3])
    acc = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, m, f, ratio in ex.map(_tune_one, jobs, chunksize=4):
            acc.setdefault((n, m, f), []).append(ratio)
    out = {}
    for (n, m) in SIZES:
        means = {f: float(np.mean(acc[(n, m, f)])) for f in FRACS}
        out[f"{n}x{m}"] = min(means, key=means.get)
        print(f"[tune] {n}x{m}: " + ", ".join(f"{f:.2f}:{v:.4f}" for f, v in means.items()) + f" -> {out[f'{n}x{m}']}", flush=True)
    DSEL.write_text(json.dumps(out, indent=1))


def _calib(cell):
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, UrStateFn, ur_ig_run
    out = []
    for rep in range(N_CAL):
        inst = inst_of(cell, rep, base=95000)
        log = []

        class Rec(UrStateFn):
            def __call__(self, sol, t, max_iter, no_improve):
                log.append(UrStateFn.features(sol))
                return 0

        ur_ig_run(inst, RandomSelector(seed=rep, n_actions=N_UR_DESTROY), max_iter=1000, d=d_of(inst.n, inst.m), seed=rep, state_fn=Rec({"f1": [0, 0], "f2": [0, 0]}), ls_steps=LS_STEPS)
        a = np.array(log)
        f10, f20 = a[0]
        out.append(np.column_stack([a[:, 0] / f10 if f10 > 0 else np.zeros(len(a)), a[:, 1] / f20 if f20 > 0 else np.zeros(len(a))]))
    return np.concatenate(out)


def calibrate(workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        arrs = list(ex.map(_calib, sorted(CELLS, key=lambda c: -c[2] * c[3])))
    rel = np.concatenate(arrs)
    q = [1 / 3, 2 / 3]
    thr = {"f1": [float(v) for v in np.quantile(rel[:, 0], q)], "f2": [float(v) for v in np.quantile(rel[:, 1], q)], "n_steps": int(len(rel))}
    THR.write_text(json.dumps(thr, indent=1))
    print("limiares pooled:", thr)


def _run(t):
    from experiment import config as CFG
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import NormalisedUCBSelector, RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, N_UR_OPS, UrStateFn, mct_greedy, ur_ig_run, ur_ils_run
    cell, rep, mt = tuple(t["cell"]), t["rep"], t["method"]
    inst = inst_of(cell, rep)
    thr = json.loads(THR.read_text())
    sf = UrStateFn({"f1": thr["f1"], "f2": thr["f2"]})
    seed, d = rep, d_of(inst.n, inst.m)

    def ql(k):
        return QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                              epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=k)

    if mt == "mct":
        t0 = time.perf_counter()
        s = mct_greedy(inst)
        r = dict(makespan=s.makespan, gap_pct=100.0 * (s.makespan - inst.lower_bound) / inst.lower_bound, cpu_time=time.perf_counter() - t0, n_improvements=0)
    elif mt == "ur_ils_ql":
        r = ur_ils_run(inst, ql(N_UR_OPS), max_iter=1000, seed=seed, state_fn=sf)
    elif mt == "ur_ils_random":
        r = ur_ils_run(inst, RandomSelector(seed=seed, n_actions=N_UR_OPS), max_iter=1000, seed=seed)
    else:
        ls = LS_STEPS if mt.startswith("ur_igls") else 0
        kw = dict(max_iter=1000, d=d, seed=seed, ls_steps=ls)
        if mt.endswith("_ql"):
            r = ur_ig_run(inst, ql(N_UR_DESTROY), state_fn=sf, **kw)
        elif mt.endswith("_random"):
            r = ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), **kw)
        elif mt.endswith("_ucb"):
            r = ur_ig_run(inst, NormalisedUCBSelector(C=CFG.UCB_C, seed=seed, n_actions=N_UR_DESTROY), **kw)
        else:
            r = ur_ig_run(inst, FixedActionSelector(int(mt[-1])), **kw)
    lo, hi, n, m = cell
    return dict(lo=lo, hi=hi, n=n, m=m, rep=rep, method=mt, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], d=d)


def _exact(t):
    from pmsp.unrelated_exact import lp_bound, milp_solve
    cell, rep = tuple(t["cell"]), t["rep"]
    inst = inst_of(cell, rep)
    lp = lp_bound(inst)
    i5, d5, p5, t5 = milp_solve(inst, 5.0)
    i60, d60, p60, t60 = milp_solve(inst, 60.0)
    lo, hi, n, m = cell
    return dict(lo=lo, hi=hi, n=n, m=m, rep=rep, lb_classic=inst.lower_bound, lb_lp=lp, milp5_inc=i5, milp5_dual=d5, milp5_proven=int(p5), milp5_t=t5,
                milp60_inc=i60, milp60_dual=d60, milp60_proven=int(p60), milp60_t=t60)


def _drive(fn, tasks, out, fields, keyf, workers, every=100):
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
                if i % every == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("tune", "calibrate", "run", "exact", "smoke"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.tune:
        tune(a.workers)
    if a.calibrate:
        calibrate(a.workers)
    if a.run or a.smoke:
        out = OUT.with_name("unrelated_entrada118_SMOKE.csv") if a.smoke else OUT
        key = lambda x: (str(x["lo"]), str(x["hi"]), str(x["n"]), str(x["m"]), str(x["rep"]), x["method"])
        cells = CELLS[:2] if a.smoke else CELLS
        tasks = [dict(cell=list(c), rep=r, method=mt, lo=c[0], hi=c[1], n=c[2], m=c[3]) for c in cells for r in range(2 if a.smoke else N_REPS) for mt in METHODS]
        tasks.sort(key=lambda t: -t["n"] * t["m"])
        _drive(_run, tasks, out, FIELDS, key, a.workers, every=200)
    if a.exact:
        key = lambda x: (str(x["lo"]), str(x["hi"]), str(x["n"]), str(x["m"]), str(x["rep"]))
        tasks = [dict(cell=list(c), rep=r, lo=c[0], hi=c[1], n=c[2], m=c[3]) for c in CELLS for r in range(N_EXACT)]
        tasks.sort(key=lambda t: -t["n"] * t["m"])
        fields = ["lo", "hi", "n", "m", "rep", "lb_classic", "lb_lp", "milp5_inc", "milp5_dual", "milp5_proven", "milp5_t", "milp60_inc", "milp60_dual", "milp60_proven", "milp60_t"]
        _drive(_exact, tasks, OUT_X, fields, key, a.workers, every=10)


if __name__ == "__main__":
    main()
