"""
experiment/run_seed_variance_entrada109.py
===========================================
WP7 (diario, Entrada 109): variancia ENTRE SEEDS do algoritmo. Todas as baterias do artigo usam UMA execucao por instancia
(seed = BASE_SEED + rep). Aqui, 3 classes ((30,8), (75,15), (150,30)) x 20 instancias (rep 0-19, as mesmas de results.csv)
x 10 seeds de algoritmo (1000..1009, distintos do seed original) x 4 metodos (ig_ql, ig_random, ils_ql, ils_random), T=1000
(ILS: limiar de perturbacao 15 se n>=150, senao 50). Saida: results/seed_variance_entrada109.csv (retomavel).
Uso: python -m experiment.run_seed_variance_entrada109 --workers 12 [--smoke]
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

CLASSES = [(30, 8), (75, 15), (150, 30)]
N_INST = 20
SEEDS = list(range(1000, 1010))
METHODS = ["ig_ql", "ig_random", "ils_ql", "ils_random"]
OUT = ROOT / "results" / "seed_variance_entrada109.csv"
FIELDS = ["instance", "method", "rep", "seed", "n", "m", "lb", "makespan", "gap_pct", "cpu_time"]


def _run(task: dict) -> dict:
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.ig import ig_run
    from pmsp.ils import ils_run
    from pmsp.instance import PMSPInstance
    n, m, rep, seed, mt = task["n"], task["m"], task["rep"], task["seed"], task["method"]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    sel = make_ils_selector(mt, seed)
    if mt.startswith("ig_"):
        r = ig_run(instance=inst, selector=sel, max_iter=1000, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW)
    else:
        r = ils_run(instance=inst, selector=sel, max_iter=1000, perturbation_threshold=15 if n >= 150 else 50,
                    n_perturb=CFG.N_PERTURB, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW)
    return dict(instance=f"rand_{n}j_{m}m", method=mt, rep=rep, seed=seed, n=n, m=m, lb=inst.lower_bound,
                makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = OUT.with_name("seed_variance_entrada109_SMOKE.csv") if args.smoke else OUT
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["instance"], r["method"], r["rep"], r["seed"]) for r in csv.DictReader(f)}
    cl = CLASSES[:1] if args.smoke else CLASSES
    tasks = [dict(n=n, m=m, rep=rep, seed=s, method=mt) for n, m in cl for rep in range(2 if args.smoke else N_INST)
             for s in (SEEDS[:2] if args.smoke else SEEDS) for mt in METHODS
             if (f"rand_{n}j_{m}m", mt, str(rep), str(s)) not in done]
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)")
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut], e)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 200 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
