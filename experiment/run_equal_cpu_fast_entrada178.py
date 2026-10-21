"""
experiment/run_equal_cpu_fast_entrada178.py
=============================================
Entrada 178 (Ciclo 14, frente J2): IG versus ILS a CPU igual com o ILS VETORIZADO nas distribuicoes fina U[1,10^4] e grossa U[1,100] da Entrada 110 (suplemento, 'IG versus ILS at equal CPU time'). Na Entrada 110 o IG recebia T_IG = 1000 K(n) iteracoes com K = CPU(ILS)/CPU(IG)
medido com os operadores do ILS em laços (K ate' 24,8); com o ILS vetorizado a razao e' menor ou igual a 1,1 (Entrada 174), e a comparacao a CPU igual passa a dar mais iteracoes ao ILS: aqui o IG usa T = 1000 (resultados armazenados em results/new_distributions_entrada108.csv, mesmas
instancias) e o ILS vetorizado recebe T_ILS = round(1000 / K_fast(n)), K_fast interpolado (log n) nas razoes medidas na Entrada 174 (results/entrada107_cpu_ratios.txt, ILS vetorizado na escala da sessao de 107) e CONGELADO abaixo.
Metodos: ils_ql_eqfast (K de ILS-QL/IG-QL), ils_random_eqfast (K de ILS-Random/IG-Random); configuracao do ILS igual a da Entrada 108 (perturbation_threshold 15 se n >= 150, senao 50; N_PERTURB do config; best_fn = apply_best_fast, resultados identicos).
Saida: results/equal_cpu_fast_entrada178.csv. Uso: python -m experiment.run_equal_cpu_fast_entrada178  (um processo, retomavel)
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DISTS = ["u1_10000", "u1_100"]
OUT = ROOT / "results" / "equal_cpu_fast_entrada178.csv"
# n -> K_fast = CPU(ILS vetorizado) / CPU(IG) a T igual, interpolado log-linear em n nas razoes por classe da Entrada 174:
#   ILS-QL/IG-QL:     20:1,01 30:1,04 40:0,86 50:1,02 60:1,08 75:0,94 100:0,89 150:0,74 250:0,57 300:0,45
#   ILS-Rand/IG-Rand: 20:0,99 30:0,87 40:0,79 50:0,76 60:0,85 75:0,68 100:0,55 150:0,51 250:0,40 300:0,34
K_QL = {25: 1.02, 50: 1.02, 60: 1.08, 100: 0.89, 200: 0.64}
K_RAND = {25: 0.93, 50: 0.76, 60: 0.85, 100: 0.55, 200: 0.45}
FIELDS = ["dist", "n", "m", "rep", "method", "T", "lb", "makespan", "gap_pct", "cpu_time"]


def main() -> None:
    from experiment import config as CFG
    from experiment.run_new_distributions_entrada108 import SIZES, _instance
    from experiment.runner import make_ils_selector
    from pmsp.ils import ils_run
    from pmsp.operators_fast import apply_best_fast
    done = set()
    if OUT.exists():
        done = {(r["dist"], r["n"], r["m"], r["rep"], r["method"]) for r in csv.DictReader(open(OUT, newline=""))}
    tasks = [(d, n, m, rep, mt) for d in DISTS for n, m in SIZES for rep in range(30) for mt in ("ils_ql_eqfast", "ils_random_eqfast") if (d, str(n), str(m), str(rep), mt) not in done]
    print(f"[tasks] {len(tasks)} pendentes", flush=True)
    new = not OUT.exists()
    t0 = time.perf_counter()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, (d, n, m, rep, mt) in enumerate(tasks, 1):
            base = "ils_ql" if mt == "ils_ql_eqfast" else "ils_random"
            T = int(round(1000 / (K_QL if mt == "ils_ql_eqfast" else K_RAND)[n]))
            inst = _instance(d, n, m, rep)
            seed = CFG.BASE_SEED + rep
            r = ils_run(instance=inst, selector=make_ils_selector(base, seed), max_iter=T, perturbation_threshold=15 if n >= 150 else 50, n_perturb=CFG.N_PERTURB, seed=seed,
                        stagnation_window=CFG.STAGNATION_WINDOW, best_fn=apply_best_fast)
            w.writerow(dict(dist=d, n=n, m=m, rep=rep, method=mt, T=T, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"]))
            f.flush()
            os.fsync(f.fileno())
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido ->", OUT)


if __name__ == "__main__":
    main()
