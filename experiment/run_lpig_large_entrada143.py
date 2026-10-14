"""
experiment/run_lpig_large_entrada143.py
========================================
Entrada 143 (Ciclo 9, frente 2): o LP-IG (Entrada 141) bate o MILP de 60 s onde o IG+LS-QL perdia (R||Cmax, n = 1000, m = 50; Entrada 127/130: IG+LS-QL T = 1000 +2,24%, T = 5000 +0,89% PIOR que o MILP)?
As MESMAS 32 instancias da Entrada 127 (seed 300000 + 1000*dist + rep; 4 distribuicoes x 8 reps) e as referencias ja' calculadas (LP e MILP-60s: results/unrelated_large_exact_entrada127.csv).
Metodos: lpig (T = 1000), lpig_T5000, ambos com d/n = 0,02 (escolha do LP-IG para n = 1000, Entrada 141; nada reajustado). Saida: results/lpig_large_entrada143.csv.
Uso: python -m experiment.run_lpig_large_entrada143 --workers 10
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

from experiment.run_lpig_benchmark_entrada141 import CHOICE, _solve  # noqa: E402
from experiment.run_unrelated_entrada118 import DISTS  # noqa: E402
from experiment.run_unrelated_large_entrada127 import M, N, inst_of  # noqa: E402

OUT = ROOT / "results" / "lpig_large_entrada143.csv"
FIELDS = ["lo", "hi", "n", "m", "rep", "method", "lb", "makespan", "cpu_time", "T", "d"]
METHODS = [("lpig", 1000), ("lpig_T5000", 5000)]


def _run(t):
    di, rep, mt, T = t["di"], t["rep"], t["method"], t["T"]
    inst = inst_of(di, rep)
    frac = json.loads(CHOICE.read_text())["1000"]["frac"]
    ms, cpu, d = _solve(inst, mt, T, frac, rep)
    lo, hi = DISTS[di]
    return dict(lo=lo, hi=hi, n=N, m=M, rep=rep, method=mt, lb=inst.lower_bound, makespan=int(ms), cpu_time=cpu, T=T, d=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        done = {(r["lo"], r["hi"], r["rep"], r["method"]) for r in csv.DictReader(open(OUT, newline=""))}
    tasks = [dict(di=di, rep=r, method=mt, T=T) for di in range(len(DISTS)) for r in range(8) for mt, T in METHODS if (str(DISTS[di][0]), str(DISTS[di][1]), str(r), mt) not in done]
    tasks.sort(key=lambda t: -t["T"])
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
                    print("[ERRO]", e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 10 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", OUT)


if __name__ == "__main__":
    main()
