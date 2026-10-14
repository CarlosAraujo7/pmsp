"""
experiment/run_igls_eq_entrada129.py
======================================
Entrada 129 (Ciclo 5, frente 3): o ganho da busca local no IG hibrido do P||Cmax (P1 da Entrada 124: igls_random vs ig_random, -0,150%) sobrevive a CPU igual?
A Entrada 127 (F2) mostrou que, na receita (d ajustado + partida forte), a CPU igual a busca local PERDE para iterar mais sem ela. Aqui o mesmo teste no cenario da Entrada 124
(d = 0,3 n, 3 estrategias aleatorias, partida LPT): `ig_random_eq` = ig_random com T_eq = max(1000, round(1000 K)) iteracoes, K = razao mediana de CPU igls_random/ig_random
medida por classe em 3 instancias (reps 0-2) numa fase de CALIBRACAO sequencial com a maquina em repouso (results/igls_eq_cpu_ratio_entrada129.json). 12 classes x 100 reps
(mesmas instancias/seeds de results.csv e da Entrada 124). Saida: results/igls_eq_entrada129.csv (retomavel).
Uso: python -m experiment.run_igls_eq_entrada129 --calibrate ; --run --workers 11
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

from experiment.run_igls_p_entrada124 import CLASSES, make_ls  # noqa: E402

RATIO = ROOT / "results" / "igls_eq_cpu_ratio_entrada129.json"
OUT = ROOT / "results" / "igls_eq_entrada129.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "T"]


def _solve(n, m, rep, T, ls):
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.ig import ig_run
    from pmsp.instance import PMSPInstance
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    r = ig_run(instance=inst, selector=make_ils_selector("ig_random", CFG.BASE_SEED + rep), max_iter=T, seed=CFG.BASE_SEED + rep,
               stagnation_window=CFG.STAGNATION_WINDOW, ls_fn=make_ls() if ls else None)
    return inst, r


def calibrate():
    out = {}
    for n, m in CLASSES:
        rs = []
        for rep in range(3):
            a = _solve(n, m, rep, 1000, False)[1]["cpu_time"]
            b = _solve(n, m, rep, 1000, True)[1]["cpu_time"]
            rs.append(b / a)
        k = sorted(rs)[1]
        out[f"{n}x{m}"] = dict(K=k, T_eq=max(1000, int(round(1000 * k))))
        print(f"[calib] {n}x{m}: K = {k:.2f} -> T_eq = {out[f'{n}x{m}']['T_eq']}", flush=True)
    RATIO.write_text(json.dumps(out, indent=1))


def _run(t):
    n, m, rep, T = t["n"], t["m"], t["rep"], t["T"]
    inst, r = _solve(n, m, rep, T, False)
    return dict(instance=inst.name, method="ig_random_eq", rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], T=T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.run:
        ratio = json.loads(RATIO.read_text())
        done = set()
        if OUT.exists():
            done = {(r["instance"], r["rep"]) for r in csv.DictReader(open(OUT, newline=""))}
        tasks = [dict(n=n, m=m, rep=r, T=ratio[f"{n}x{m}"]["T_eq"]) for n, m in CLASSES for r in range(100) if (f"rand_{n}j_{m}m", str(r)) not in done]
        tasks.sort(key=lambda t: -t["n"] * t["T"])
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
                    if i % 100 == 0 or i == len(tasks):
                        el = time.perf_counter() - t0
                        print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
        print("concluido ->", OUT)


if __name__ == "__main__":
    main()
