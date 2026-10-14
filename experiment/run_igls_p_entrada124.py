"""
experiment/run_igls_p_entrada124.py
=====================================
Entrada 124 (Ciclo 4, frente 2): no P||Cmax o IG (sem busca local) supera o ILS (so' busca local); no R||Cmax o IG PURO perde para o ILS e so' a COMBINACAO IG+LS ganha.
Para isolar destruir-reconstruir vs busca local no P||Cmax: IG hibrido = destruir/reconstruir + descida com os operadores do ILS (inter-troca e realocacao,
best-improvement, ate 2 passos por iteracao) -- metodos igls_random e igls_ql (estado padrao), T=1000, d = 0,3 n, 3 estrategias, partida LPT; 12 classes x 100 reps
(mesmas instancias/seeds de results.csv). Saida: results/igls_p_entrada124.csv (retomavel).
Uso: python -m experiment.run_igls_p_entrada124 --workers 12 [--smoke]
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

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
METHODS = ["igls_random", "igls_ql"]
OUT = ROOT / "results" / "igls_p_entrada124.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements"]
LS_PASSES = 2


def make_ls():
    from pmsp.operators import apply_best

    def ls(sol, rng):
        for _ in range(LS_PASSES):
            if apply_best(0, sol, rng) <= 0 and apply_best(1, sol, rng) <= 0:
                break
    return ls


def _run(t):
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.ig import ig_run
    from pmsp.instance import PMSPInstance
    n, m, rep, mt = t["n"], t["m"], t["rep"], t["method"]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    r = ig_run(instance=inst, selector=make_ils_selector("ig_ql" if mt == "igls_ql" else "ig_random", CFG.BASE_SEED + rep), max_iter=1000, seed=CFG.BASE_SEED + rep,
               stagnation_window=CFG.STAGNATION_WINDOW, ls_fn=make_ls())
    return dict(instance=inst.name, method=mt, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT.with_name("igls_p_entrada124_SMOKE.csv") if a.smoke else OUT
    done = set()
    if out.exists():
        done = {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    cl = CLASSES[:2] if a.smoke else CLASSES
    tasks = [dict(n=n, m=m, rep=r, method=mt) for n, m in cl for r in range(2 if a.smoke else 100) for mt in METHODS if (f"rand_{n}j_{m}m", mt, str(r)) not in done]
    tasks.sort(key=lambda t: -t["n"])
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
                if i % 100 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
