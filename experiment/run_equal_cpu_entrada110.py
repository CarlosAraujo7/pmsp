"""
experiment/run_equal_cpu_entrada110.py
========================================
WP3b (diario, Entrada 110): IG vs ILS em CPU IGUAL. As baterias do artigo comparam ILS e IG com o mesmo numero de
iteracoes (T=1000), mas uma iteracao do ILS custa mais: na mesma sessao/maquina (results/timing_entrada107.csv),
CPU(ILS-QL)/CPU(IG-QL) = 1,28x (n=20) ... 24,8x (n=300) e CPU(ILS-Random)/CPU(IG-Random) = 0,97x ... 8,8x. Aqui o IG recebe
T_IG = round(1000*K(n)) iteracoes, com K interpolado (log n) nessas razoes e CONGELADO abaixo; o ILS usa T=1000 (resultados ja
em results/new_distributions_entrada108.csv, mesmas instancias). Celulas: distribuicoes u1_10000 (fina: ILS vence em
iteracoes iguais) e u1_100 (grossa), 5 tamanhos x 30 repeticoes = 300 instancias.
Metodos: ig_ql_eqt (K de ILS-QL/IG-QL), ig_random_eqt (K de ILS-Random/IG-Random).
Saida: results/equal_cpu_entrada110.csv. Uso: python -m experiment.run_equal_cpu_entrada110 --workers 12 [--smoke]
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

DISTS = ["u1_10000", "u1_100"]
OUT = ROOT / "results" / "equal_cpu_entrada110.csv"
# n -> K (multiplicador de iteracoes do IG). Interpolacao log-linear em n sobre as razoes medidas (timing_entrada107):
#   ILS-QL/IG-QL:  20:1.285 30:1.74 40:2.24 50:2.60 60:3.38 75:4.77 100:6.20 150:10.79 250:19.46 300:24.77
#   ILS-Rand/IG-Rand: 20:0.975 30:1.16 40:1.29 50:1.62 60:1.78 75:2.26 100:2.54 150:4.58 250:8.08 300:8.84
K_QL = {25: 1.5, 50: 2.6, 60: 3.4, 100: 6.2, 200: 15.0}
K_RAND = {25: 1.1, 50: 1.6, 60: 1.8, 100: 2.5, 200: 6.3}
FIELDS = ["dist", "n", "m", "rep", "method", "T", "lb", "makespan", "gap_pct", "cpu_time"]


def _run(task: dict) -> dict:
    from experiment import config as CFG
    from experiment.run_new_distributions_entrada108 import _instance
    from experiment.runner import make_ils_selector
    from pmsp.ig import ig_run
    dist, n, m, rep, mt = task["dist"], task["n"], task["m"], task["rep"], task["method"]
    inst = _instance(dist, n, m, rep)
    base = "ig_ql" if mt == "ig_ql_eqt" else "ig_random"
    T = int(round(1000 * (K_QL if mt == "ig_ql_eqt" else K_RAND)[n]))
    sel = make_ils_selector(base, CFG.BASE_SEED + rep)
    r = ig_run(instance=inst, selector=sel, max_iter=T, seed=CFG.BASE_SEED + rep, stagnation_window=CFG.STAGNATION_WINDOW)
    return dict(dist=dist, n=n, m=m, rep=rep, method=mt, T=T, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"],
                cpu_time=r["cpu_time"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    from experiment.run_new_distributions_entrada108 import SIZES
    out = OUT.with_name("equal_cpu_entrada110_SMOKE.csv") if args.smoke else OUT
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["dist"], r["n"], r["m"], r["rep"], r["method"]) for r in csv.DictReader(f)}
    sizes = SIZES[:1] if args.smoke else SIZES
    tasks = [dict(dist=d, n=n, m=m, rep=rep, method=mt) for d in DISTS for n, m in sizes for rep in range(2 if args.smoke else 30)
             for mt in ("ig_ql_eqt", "ig_random_eqt") if (d, str(n), str(m), str(rep), mt) not in done]
    print(f"[tasks] {len(tasks)} pendentes")
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
                if i % 50 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
