"""
experiment/run_granularity_factorial_entrada98.py
===================================================
Item 3 da prioridade 0 (Entradas 97-98). O fatorial anterior (Secao 5.7)
comparava U[1,100] vs U[1,1e8] e o chamava de "variancia"; a Entrada 97 mostrou
que o CV e' identico (0,559 vs 0,565), que os metodos sao invariantes a escala e
que o que muda e' a duplicacao dos tempos (dup_ratio 0,49 -> 0,00). Este
experimento varia a FAIXA H (granularidade) e n/m de forma ortogonal, com N=30:

  n = 150 fixo; m em {50, 15, 5} (n/m = 3, 10, 30);
  p_j ~ U[1, H] com H em {1e2, 1e3, 1e4, 1e6, 1e8};
  30 instancias por celula (seed 20000+k, o MESMO k em todas as celulas);
  metodos: ig_ql, ils_ql, ig_random, ils_random (T=1000; ILS com
  perturbation_threshold=15, como nas classes com n>=150).
Registra dup_ratio e CV de cada instancia.

Uso: python -m experiment.run_granularity_factorial_entrada98 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N = 150
MS = [50, 15, 5]
HS = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 6, 10 ** 8]
K = 30
METHODS = ["ig_ql", "ils_ql", "ig_random", "ils_random"]
OUT = ROOT / "results" / "granularity_factorial_entrada98.csv"
FIELDS = ["n", "m", "H", "k", "dup_ratio", "cv", "method", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements"]


def _task(args):
    m, H, k = args
    import numpy as np
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.apriori_select import dup_ratio
    from pmsp.ig import ig_run
    from pmsp.ils import ils_run
    from pmsp.instance import PMSPInstance

    seed = 20000 + k
    inst = PMSPInstance.random(n=N, m=m, seed=seed, high=H, name=f"fac_{m}_{H}")
    p = np.asarray(inst.p, dtype=float)
    base = dict(n=N, m=m, H=H, k=k, dup_ratio=dup_ratio(inst), cv=float(p.std() / p.mean()), lb=inst.lower_bound)
    rows = []
    for method in METHODS:
        sel = make_ils_selector(method, CFG.BASE_SEED + seed)
        if method.startswith("ig_"):
            r = ig_run(instance=inst, selector=sel, max_iter=1000, seed=CFG.BASE_SEED + seed,
                       stagnation_window=CFG.STAGNATION_WINDOW)
        else:
            r = ils_run(instance=inst, selector=sel, max_iter=1000, seed=CFG.BASE_SEED + seed,
                        stagnation_window=CFG.STAGNATION_WINDOW, perturbation_threshold=15, n_perturb=CFG.N_PERTURB)
        rows.append(dict(base, method=method, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                         n_improvements=r["n_improvements"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        with open(OUT, newline="") as f:
            done = {(int(r["m"]), int(r["H"]), int(r["k"])) for r in csv.DictReader(f)}
    tasks = [(m, H, k) for m in MS for H in HS for k in range(K) if (m, H, k) not in done]
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} ja feitas)", flush=True)
    new = not OUT.exists()
    t0 = time.perf_counter()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(_task, t) for t in tasks]), 1):
                try:
                    w.writerows(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 15 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", OUT)


if __name__ == "__main__":
    main()
