"""
experiment/run_lpigsr_benchmark_entrada149.py
================================================
Entrada 149 (Ciclo 9, frente 4): LP-IG + MILP RESTRITO (`pmsp/unrelated_srmilp.py`) em R||Cmax no benchmark padrao de Fanjul-Peyro e Ruiz. Projeto (SO' em instancias de AJUSTE; 168 instancias:
m in {10, 30, 50} x n in {100, 200, 500, 1000}; `results/lpigsr_design_exploration_entrada149.csv`): GAP medio ao melhor conhecido do LP-IG (T = 1000) 0,878%; LP-IG + MILP restrito a k = 3 maquinas por tarefa
(limite 30 s) 0,270%; k = 6 0,224% (CPU medio ~22 s contra 2,9 s). Metodo escolhido para o teste: k = 6, limite de tempo 30 s (o melhor no ajuste; nada reajustado).
Teste: as 420 instancias RESERVADAS (idx 7-9; as mesmas da Entrada 141, onde o LP-IG e o LP-IG T = 5000 ja' foram rodados). Metodos novos: `lpig_sr6` (LP-IG T = 1000 com d/n da Entrada 141 -> MILP restrito k = 6, 30 s) e
`lpig_T8000` (controle a CPU aproximadamente igual: LP-IG com T = 8000). O cpu_time inclui LP, descida, IG e MILP. Saida: results/lpigsr_test_entrada149.csv. Retomavel.
Uso: python -m experiment.run_lpigsr_benchmark_entrada149 --run --workers 10
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

import experiment.run_lpig_benchmark_entrada141 as L  # noqa: E402
import experiment.run_rcmax_benchmark_entrada137 as B  # noqa: E402

K_SR, TL_SR = 6, 30.0
OUT = ROOT / "results" / "lpigsr_test_entrada149.csv"
FIELDS = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "T", "best_known", "lb", "milp_makespan"]
METHODS = ["lpig_sr6", "lpig_T8000"]


def _task(t):
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, _descent, ur_ig_run
    from pmsp.unrelated_lp import lp_start
    from pmsp.unrelated_srmilp import restricted_milp
    inst = B.load_instance(t["type"], t["instance"])
    seed = int(t["instance"])
    frac = json.loads(L.CHOICE.read_text())[t["n"]]["frac"]
    d = max(2, round(frac * inst.n))
    t0 = time.perf_counter()
    start, _, _ = lp_start(inst)
    _descent(start, 10 ** 6)
    T = 8000 if t["method"] == "lpig_T8000" else 1000
    r = ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), max_iter=T, d=d, seed=seed, initial_solution=start, ls_steps=B.LS_STEPS)
    sol, mk_milp = r["solution"], None
    if t["method"] == "lpig_sr6":
        sol, mk_milp, _ = restricted_milp(inst, sol, k=K_SR, time_limit=TL_SR)
    cpu = time.perf_counter() - t0
    return dict(type=t["type"], instance=t["instance"], m=inst.m, n=inst.n, idx=t["idx"], method=t["method"], makespan=int(sol.makespan), cpu_time=cpu, T=T, best_known=t["best_known"],
                lb=inst.lower_bound, milp_makespan=mk_milp if mk_milp is not None else "")


def run(workers):
    done = set()
    if OUT.exists():
        done = {(r["type"], r["instance"], r["method"]) for r in csv.DictReader(open(OUT, newline=""))}
    tasks = [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], method=mt, best_known=r["best_known"]) for r in B.manifest("reserve") for mt in METHODS
             if (r["type"], r["instance"], mt) not in done]
    tasks.sort(key=lambda t: -(int(t["n"]) * int(t["m"])))
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)", flush=True)
    new = not OUT.exists()
    t0 = time.perf_counter()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_task, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut]["type"], futs[fut]["instance"], futs[fut]["method"], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 40 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.run:
        run(a.workers)


if __name__ == "__main__":
    main()
