"""
experiment/run_unrelated_budget_entrada121.py
===============================================
Entrada 121 (Ciclo 3, frente 3): R||Cmax com ORCAMENTO diferente nas 160 instancias das repeticoes 0-7 das 20 celulas (as mesmas do MILP):
  ur_igls_random_T5000, ur_igls_ql_T5000     IG com busca local, T = 5000 (o T=1000 esta' em results/unrelated_entrada118.csv)
  ur_ils_random_T8000, ur_ils_ql_T8000       ILS com T = 8000: CPU APROXIMADAMENTE igual a do IG+LS com T = 1000 (uma iteracao do ILS custa ~1/8)
Mesmos d/n por tamanho (results/dsel_unrelated_entrada118.json), limiares do estado e instancias da Entrada 118; seed = rep.
Saida: results/unrelated_budget_entrada121.csv (retomavel). Uso: python -m experiment.run_unrelated_budget_entrada121 --workers 12 [--smoke]
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

from experiment.run_unrelated_entrada118 import CELLS, LS_STEPS, N_EXACT, THR, d_of, inst_of  # noqa: E402

METHODS = [("ur_igls_random_T5000", 5000), ("ur_igls_ql_T5000", 5000), ("ur_ils_random_T8000", 8000), ("ur_ils_ql_T8000", 8000)]
OUT = ROOT / "results" / "unrelated_budget_entrada121.csv"
FIELDS = ["lo", "hi", "n", "m", "rep", "method", "T", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "d"]


def _run(t):
    from experiment import config as CFG
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, N_UR_OPS, UrStateFn, ur_ig_run, ur_ils_run
    cell, rep, mt, T = tuple(t["cell"]), t["rep"], t["method"], t["T"]
    inst = inst_of(cell, rep)
    thr = json.loads(THR.read_text())
    sf = UrStateFn({"f1": thr["f1"], "f2": thr["f2"]})
    seed, d = rep, d_of(inst.n, inst.m)

    def ql(k):
        return QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                              epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=k)

    if mt.startswith("ur_ils"):
        r = ur_ils_run(inst, ql(N_UR_OPS), max_iter=T, seed=seed, state_fn=sf) if "_ql_" in mt else ur_ils_run(inst, RandomSelector(seed=seed, n_actions=N_UR_OPS), max_iter=T, seed=seed)
    else:
        kw = dict(max_iter=T, d=d, seed=seed, ls_steps=LS_STEPS)
        r = ur_ig_run(inst, ql(N_UR_DESTROY), state_fn=sf, **kw) if "_ql_" in mt else ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), **kw)
    lo, hi, n, m = cell
    return dict(lo=lo, hi=hi, n=n, m=m, rep=rep, method=mt, T=T, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], d=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT.with_name("unrelated_budget_entrada121_SMOKE.csv") if a.smoke else OUT
    done = set()
    if out.exists():
        done = {(r["lo"], r["hi"], r["n"], r["m"], r["rep"], r["method"]) for r in csv.DictReader(open(out, newline=""))}
    cells = CELLS[:2] if a.smoke else CELLS
    tasks = [dict(cell=list(c), rep=r, method=mt, T=T, lo=c[0], hi=c[1], n=c[2], m=c[3]) for c in cells for r in range(2 if a.smoke else N_EXACT) for mt, T in METHODS
             if (str(c[0]), str(c[1]), str(c[2]), str(c[3]), str(r), mt) not in done]
    tasks.sort(key=lambda t: -t["n"] * t["m"] * t["T"])
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
                if i % 40 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
