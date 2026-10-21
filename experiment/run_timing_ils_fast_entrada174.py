"""
experiment/run_timing_ils_fast_entrada174.py
==============================================
Entrada 174 (Ciclo 14, frente J): a figura qualidade x tempo (Figura 'fig:pareto', analysis/make_figure_pareto.py) e as razoes de CPU do ILS vieram de timing_entrada107.csv, medido com os operadores em lacos Python (o ILS e' 6,2 vezes mais lento
que o ILS vetorizado, com resultados identicos; Entrada 162). Aqui o ILS (ils_random, ils_ql) e' REMEDIDO com `best_fn=apply_best_fast`, na mesma sessao que duas pontes, ig_random e ig_ql_seed (reexecutadas), 12 classes x 5 reps (as mesmas instancias),
tempo de CPU do processo (`time.process_time`), 3 repeticoes de cada medida (minimo), para expressar os tempos novos do ILS na escala da sessao antiga: ils_fast_ajustado = ils_fast_novo x (CPU antigo / CPU novo das pontes, media geometrica por classe).
Saida: results/timing_ils_fast_entrada174.csv. Uso: python -m experiment.run_timing_ils_fast_entrada174
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
OUT = ROOT / "results" / "timing_ils_fast_entrada174.csv"


def main() -> None:
    from experiment import config as CFG
    from experiment.runner import make_ils_selector, run_single
    from pmsp.ils import ils_run
    from pmsp.instance import PMSPInstance
    from pmsp.operators_fast import apply_best_fast

    def timed(fn):
        best = float("inf")
        r = None
        for _ in range(3):
            t = time.process_time()
            r = fn()
            best = min(best, time.process_time() - t)
        return r, best

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time"])
        t0 = time.perf_counter()
        for n, m in CLASSES:
            for rep in range(5):
                inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
                jobs = {mt: (lambda mt=mt: ils_run(instance=inst, selector=make_ils_selector(mt, CFG.BASE_SEED + rep), max_iter=1000, seed=CFG.BASE_SEED + rep,
                                                  stagnation_window=CFG.STAGNATION_WINDOW, best_fn=apply_best_fast)) for mt in ("ils_random", "ils_ql")}
                jobs.update({mt: (lambda mt=mt: run_single(inst, mt, rep, max_iter=1000)) for mt in ("ig_random", "ig_ql_seed")})
                for mt, fn in jobs.items():
                    r, cpu = timed(fn)
                    name = "ils_random_fast" if mt == "ils_random" else "ils_ql_fast" if mt == "ils_ql" else mt + "_s2"
                    w.writerow([inst.name, name, rep, n, m, inst.lower_bound, r["makespan"], r["gap_pct"], cpu])
                    f.flush()
            print(f"  {n}x{m} ok {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
