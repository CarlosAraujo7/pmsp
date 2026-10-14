"""
experiment/run_recipe_budget_entrada139.py
============================================
Entrada 139 (H3 da auditoria do Ciclo 4): curva qualidade x orcamento da RECEITA sem aprendizado (`igcv_seed`: IG com destruicao aleatoria, d/n validado por classe na Entrada 114/121,
partida = melhor de LPT/MULTIFIT/LDM), sem busca local. 6 classes ((30,8), (50,10), (75,15), (100,20), (150,15), (250,25)) x 100 reps (as mesmas instancias/seeds da Entrada 121) x
T in {250, 4000, 16000, 64000}; o ponto T = 1000 e' o ja' existente (`results/igcv_seed_entrada121.csv`). Como o seletor e' fixo e nada depende de T, a trajetoria de T menor e' prefixo
da de T maior (mesmo seed): o melhor-ate-agora e' monotono em T por construcao (verificado no relatorio). Saida: results/recipe_budget_entrada139.csv (retomavel).
Uso: python -m experiment.run_recipe_budget_entrada139 --workers 10
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

from experiment.run_igcvls_entrada127 import _solve, d_choice  # noqa: E402

CLASSES = [(30, 8), (50, 10), (75, 15), (100, 20), (150, 15), (250, 25)]
TS = [250, 4000, 16000, 64000]
OUT = ROOT / "results" / "recipe_budget_entrada139.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "cpu_time", "T", "frac"]


def _run(t):
    from experiment import config as CFG
    from pmsp.instance import PMSPInstance
    n, m, rep, T = t["n"], t["m"], t["rep"], t["T"]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    r = _solve(inst, t["frac"], CFG.BASE_SEED + rep, T, False)
    return dict(instance=inst.name, method=f"igcv_seed_T{T}", rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], cpu_time=r["cpu_time"], T=T, frac=t["frac"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    ch = d_choice()
    done = set()
    if OUT.exists():
        done = {(r["instance"], r["rep"], r["T"]) for r in csv.DictReader(open(OUT, newline=""))}
    tasks = [dict(n=n, m=m, rep=r, T=T, frac=ch[f"rand_{n}j_{m}m"]) for n, m in CLASSES for r in range(100) for T in TS if (f"rand_{n}j_{m}m", str(r), str(T)) not in done]
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
                    print("[ERRO]", e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 100 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", OUT)


if __name__ == "__main__":
    main()
