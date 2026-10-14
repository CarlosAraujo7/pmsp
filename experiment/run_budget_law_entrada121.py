"""
experiment/run_budget_law_entrada121.py
=========================================
Entrada 121 (Ciclo 3, frente 1): a politica aprendida so' vale quando ha folga E o ORCAMENTO e' vinculante? No P||Cmax a busca satura em T=1000
(folga 0,04%, Entrada 104); no R||Cmax (Entrada 118) o orcamento vincula e o Q-Learning captura 55-90% da folga. Aqui o P||Cmax e' rodado com orcamento
APERTADO: T in {100, 250} (T=1000 ja' existe: results.csv e Entradas 98/104), 12 classes x 100 reps (mesmas instancias/seeds de results.csv),
metodos: ig_random, ig_ql (estado padrao), ig_ql_cal (estado `cal`), ig_fix0, ig_fix1, ig_fix2 (3 estrategias de destruicao, d = 0,3 n, partida LPT).
Saida: results/budget_law_entrada121.csv (retomavel). Uso: python -m experiment.run_budget_law_entrada121 --workers 12 [--smoke]
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
TS = [100, 250]
METHODS = ["ig_random", "ig_ql", "ig_ql_cal", "ig_fix0", "ig_fix1", "ig_fix2"]
OUT = ROOT / "results" / "budget_law_entrada121.csv"
FIELDS = ["instance", "method", "T", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements"]


def _run(t):
    from experiment import config as CFG
    from experiment.run_signal_headroom_entrada104 import _agent, _state_fn
    from experiment.runner import make_ils_selector
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ig import ig_run
    from pmsp.instance import PMSPInstance
    n, m, rep, T, mt = t["n"], t["m"], t["rep"], t["T"], t["method"]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    seed = CFG.BASE_SEED + rep
    kw = dict(instance=inst, max_iter=T, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW)
    if mt in ("ig_random", "ig_ql"):
        r = ig_run(selector=make_ils_selector(mt, seed), **kw)
    elif mt == "ig_ql_cal":
        r = ig_run(selector=_agent(seed), state_fn=_state_fn(n, m), **kw)
    else:
        r = ig_run(selector=FixedActionSelector(int(mt[-1])), **kw)
    return dict(instance=inst.name, method=mt, T=T, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"],
                cpu_time=r["cpu_time"], n_improvements=r["n_improvements"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT.with_name("budget_law_entrada121_SMOKE.csv") if a.smoke else OUT
    done = set()
    if out.exists():
        done = {(r["instance"], r["method"], r["T"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    cl = CLASSES[:2] if a.smoke else CLASSES
    tasks = [dict(n=n, m=m, rep=r, T=T, method=mt) for n, m in cl for r in range(2 if a.smoke else 100) for T in TS for mt in METHODS
             if (f"rand_{n}j_{m}m", mt, str(T), str(r)) not in done]
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
                if i % 500 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
