"""
experiment/run_unrelated_valid_entrada135.py
==============================================
Entrada 135 (Ciclo 7): em R||Cmax 1000 x 50, a VALIDACAO OFFLINE (escolher, por distribuicao, a melhor entre 5 candidatos -- as 4 estrategias de destruicao fixas e a mistura uniforme -- em
instancias de ajuste separadas) captura o que o Q-Learning captura? (analogo, para a escolha de estrategia, ao que a Entrada 121 mostrou para o tamanho de destruicao no P||Cmax.)
Ajuste: as 24 instancias de ajuste da Entrada 127 (seed 97000 + 1000*dist + rep, rep 0-5), T = 1000, d/n = 0,02, IG com busca local; 5 candidatos; escolha por distribuicao = menor razao
media makespan/LB classico. Teste: 96 instancias NOVAS (seed 600000 + 1000*dist + rep, 24 por distribuicao), metodos ur_igls_random, ur_igls_ql, ur_igls_fix0..3 (a 'validada' e' a
candidata escolhida; avaliada com os resultados do proprio braco). Saidas: results/valid_tune_entrada135.csv, results/valid_choice_entrada135.json, results/unrelated_valid_entrada135.csv. Retomavel.
Uso: python -m experiment.run_unrelated_valid_entrada135 --tune --workers 10 ; --run --workers 10
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

from experiment.run_unrelated_entrada118 import DISTS  # noqa: E402
from experiment.run_unrelated_large_entrada127 import M, N, inst_of  # noqa: E402
from experiment.run_unrelated_policy_entrada133 import _solve  # noqa: E402

BASE = 600000
N_REPS = 24
CAND = ["ur_igls_random", "ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3"]
METHODS = ["ur_igls_random", "ur_igls_ql", "ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3"]
TUNE = ROOT / "results" / "valid_tune_entrada135.csv"
CHOICE = ROOT / "results" / "valid_choice_entrada135.json"
OUT = ROOT / "results" / "unrelated_valid_entrada135.csv"
FIELDS = ["lo", "hi", "n", "m", "rep", "method", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "d", "T"]


def _run(t):
    di, rep, mt, base = t["di"], t["rep"], t["method"], t["base"]
    inst = inst_of(di, rep, base=base)
    r, d = _solve(inst, mt, rep, 1000)
    lo, hi = DISTS[di]
    return dict(lo=lo, hi=hi, n=N, m=M, rep=rep, method=mt, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], d=d, T=1000)


def _drive(tasks, out, workers, every):
    done = set()
    if out.exists():
        done = {(r["lo"], r["hi"], r["rep"], r["method"]) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if (str(DISTS[t["di"]][0]), str(DISTS[t["di"]][1]), str(t["rep"]), t["method"]) not in done]
    print(f"[tasks] {len(pend)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run, t): t for t in pend}
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
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.tune:
        tasks = [dict(di=di, rep=r, method=mt, base=97000) for di in range(len(DISTS)) for r in range(6) for mt in CAND]
        _drive(tasks, TUNE, a.workers, 20)
        import pandas as pd
        t = pd.read_csv(TUNE)
        t["ratio"] = t.makespan / t.lb
        choice = {}
        for (lo, hi), g in t.groupby(["lo", "hi"]):
            mean = g.groupby("method").ratio.mean()
            choice[f"{lo}_{hi}"] = dict(best=mean.idxmin(), means={k: float(v) for k, v in mean.items()})
            print(f"[tune] U[{lo},{hi}] -> {mean.idxmin()} ({', '.join(f'{k[-5:]}:{v:.4f}' for k, v in mean.items())})", flush=True)
        CHOICE.write_text(json.dumps(choice, indent=1))
    if a.run:
        tasks = [dict(di=di, rep=r, method=mt, base=BASE) for di in range(len(DISTS)) for r in range(N_REPS) for mt in METHODS]
        _drive(tasks, OUT, a.workers, 50)
        print("concluido ->", OUT)


if __name__ == "__main__":
    main()
