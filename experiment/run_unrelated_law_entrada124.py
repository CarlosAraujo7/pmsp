"""
experiment/run_unrelated_law_entrada124.py
============================================
Entrada 124 (Ciclo 4, frente 1): R||Cmax em instancias NOVAS (seed 200000+; as mesmas 20 celulas x 20 reps = 400 instancias, T=1000, d/n por tamanho e limiares do estado
da Entrada 118, nada reajustado) para (i) CONFIRMAR em dados novos a observacao exploratoria "o ganho da politica acompanha a folga das estrategias" (Entrada 118: Spearman 0,68
entre 20 celulas) e (ii) DECOMPOR o ganho da politica (estado x TD) com o desenho 2x2 do P||Cmax:
  ur_igls_random   selecao uniforme                          ur_igls_fix0..3   estrategia de destruicao fixa (folga)
  ur_igls_ql       Q-Learning com estado (rel)               ur_igls_ql_flat   Q-Learning SEM estado (TD, sem condicionamento)
  ur_igls_ucb      UCB1 normalizado (sem estado, sem TD)     ur_igls_ucb_ctx   UCB1 por estado (estado, sem TD)
Saida: results/unrelated_law_entrada124.csv (retomavel). Uso: python -m experiment.run_unrelated_law_entrada124 --workers 12 [--smoke]
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment.run_unrelated_entrada118 import CELLS, LS_STEPS, THR, d_of, inst_of  # noqa: E402

BASE = 200000
N_REPS = 20
METHODS = ["ur_igls_random", "ur_igls_ql", "ur_igls_ql_flat", "ur_igls_ucb", "ur_igls_ucb_ctx", "ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3"]
OUT = ROOT / "results" / "unrelated_law_entrada124.csv"
FIELDS = ["lo", "hi", "n", "m", "rep", "method", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "d"]


def _run(t):
    from experiment import config as CFG
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.qlearning import ContextualUCBSelector, FlatQLearningSelector, QLearningAgent
    from pmsp.selectors import NormalisedUCBSelector, RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, UrStateFn, ur_ig_run
    cell, rep, mt = tuple(t["cell"]), t["rep"], t["method"]
    inst = inst_of(cell, rep, base=BASE)
    thr = json.loads(THR.read_text())
    sf = UrStateFn({"f1": thr["f1"], "f2": thr["f2"]})
    seed, d = rep, d_of(inst.n, inst.m)
    kw = dict(max_iter=1000, d=d, seed=seed, ls_steps=LS_STEPS)
    if mt == "ur_igls_random":
        r = ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), **kw)
    elif mt == "ur_igls_ql":
        r = ur_ig_run(inst, QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                                           epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=N_UR_DESTROY), state_fn=sf, **kw)
    elif mt == "ur_igls_ql_flat":
        r = ur_ig_run(inst, FlatQLearningSelector(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                                                  epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=N_UR_DESTROY), state_fn=sf, **kw)
    elif mt == "ur_igls_ucb":
        r = ur_ig_run(inst, NormalisedUCBSelector(C=CFG.UCB_C, seed=seed, n_actions=N_UR_DESTROY), **kw)
    elif mt == "ur_igls_ucb_ctx":
        r = ur_ig_run(inst, ContextualUCBSelector(C=CFG.UCB_C, seed=seed, n_actions=N_UR_DESTROY), state_fn=sf, **kw)
    else:
        r = ur_ig_run(inst, FixedActionSelector(int(mt[-1])), **kw)
    lo, hi, n, m = cell
    return dict(lo=lo, hi=hi, n=n, m=m, rep=rep, method=mt, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], d=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT.with_name("unrelated_law_entrada124_SMOKE.csv") if a.smoke else OUT
    done = set()
    if out.exists():
        done = {(r["lo"], r["hi"], r["n"], r["m"], r["rep"], r["method"]) for r in csv.DictReader(open(out, newline=""))}
    cells = CELLS[:2] if a.smoke else CELLS
    tasks = [dict(cell=list(c), rep=r, method=mt, lo=c[0], hi=c[1], n=c[2], m=c[3]) for c in cells for r in range(2 if a.smoke else N_REPS) for mt in METHODS
             if (str(c[0]), str(c[1]), str(c[2]), str(c[3]), str(r), mt) not in done]
    tasks.sort(key=lambda t: -t["n"] * t["m"])
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_run, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 200 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
