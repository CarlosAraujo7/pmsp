"""
experiment/run_seed_variance_all_entrada117.py
================================================
Entrada 117: estende a variancia entre seeds (Entrada 109: 3 classes) as 9 classes restantes, completando as 12:
(20,4), (20,6), (40,5), (50,10), (60,20), (100,20), (150,15), (250,25), (300,20); 20 instancias (rep 0-19) x 10 seeds (1000-1009) x
{ig_ql, ig_random, ils_ql, ils_random}, T=1000 (ILS: limiar 15 se n>=150, senao 50). Reusa `_run` da Entrada 109 sem altera-lo.
Saida: results/seed_variance_all_entrada117.csv (retomavel). Uso: python -m experiment.run_seed_variance_all_entrada117 --workers 12
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

from experiment.run_seed_variance_entrada109 import FIELDS, METHODS, N_INST, SEEDS, _run  # noqa: E402

CLASSES = [(20, 4), (20, 6), (40, 5), (50, 10), (60, 20), (100, 20), (150, 15), (250, 25), (300, 20)]
OUT = ROOT / "results" / "seed_variance_all_entrada117.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        with open(OUT, newline="") as f:
            done = {(r["instance"], r["method"], r["rep"], r["seed"]) for r in csv.DictReader(f)}
    tasks = [dict(n=n, m=m, rep=rep, seed=s, method=mt) for n, m in CLASSES for rep in range(N_INST) for s in SEEDS for mt in METHODS
             if (f"rand_{n}j_{m}m", mt, str(rep), str(s)) not in done]
    tasks.sort(key=lambda t: -(t["n"] * (3 if t["method"].startswith("ils") else 1)))
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)", flush=True)
    new = not OUT.exists()
    t0 = time.perf_counter()
    with open(OUT, "a", newline="") as f:
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
                if i % 300 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", OUT)


if __name__ == "__main__":
    main()
