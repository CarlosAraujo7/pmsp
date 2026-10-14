"""
experiment/run_timing_entrada107.py
====================================
WP3 (diario, Entrada 107): tempos de CPU na MESMA sessao/maquina, SEQUENCIAL (um processo, maquina em repouso), para as
razoes de CPU e a figura Pareto qualidade x tempo. Os cpu_time de results.csv vem de maquinas/condicoes diferentes e nao
servem para razoes entre metodos de baterias distintas. 12 classes x 5 repeticoes (rep 0-4), mesmas instancias de results.csv.
Metodos: lpt, multifit, ldm, sa_li_perkins, bco (orcamento de movimentos igual), bco_I100 (publicado), ils_random, ils_ql,
ig_random, ig_ql, ig_ql_seed.   Saida: results/timing_entrada107.csv   Uso: python -m experiment.run_timing_entrada107
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
METHODS = ["lpt", "multifit", "ldm", "sa_li_perkins", "bco", "bco_I100", "ils_random", "ils_ql", "ig_random", "ig_ql", "ig_ql_seed"]
OUT = ROOT / "results" / "timing_entrada107.csv"


def main() -> None:
    from experiment import config as CFG
    from experiment.runner import run_single
    from pmsp.bco import bco_run
    from pmsp.instance import PMSPInstance
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time"])
        t0 = time.perf_counter()
        for n, m in CLASSES:
            for rep in range(5):
                inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
                for mt in METHODS:
                    t = time.perf_counter()
                    r = bco_run(inst, max_iter=100 * 5 * n, seed=CFG.BASE_SEED + rep) if mt == "bco_I100" else run_single(inst, mt, rep, max_iter=1000)
                    wall = time.perf_counter() - t
                    w.writerow([inst.name, mt, rep, n, m, inst.lower_bound, r["makespan"], r["gap_pct"], wall])
                    f.flush()
            print(f"  {n}x{m} ok {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
