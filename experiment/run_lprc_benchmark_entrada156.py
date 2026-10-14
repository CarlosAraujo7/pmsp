"""
experiment/run_lprc_benchmark_entrada156.py
=============================================
Entrada 156 (Ciclo 11): MILP restrito com candidatos escolhidos pelo CUSTO REDUZIDO DO LP (LP-RC) em R||Cmax no benchmark padrao de Fanjul-Peyro e Ruiz. Projeto (SO' em instancias de AJUSTE; 72 instancias dos tipos
mais fracos: jobs correlacionados, maquinas correlacionadas, U[10,100], U[1,100]; n in {100, 200, 500}; m in {10, 30, 50}; `results/lprc_design_probe_entrada156.csv`): o MILP restrito a 6 maquinas por tarefa tem GAP medio
0,489% com as 6 de menor p_ij e 0,341% com as 6 de menor custo reduzido y_i p_ij - u_j do LP de atribuicao (maquinas correlacionadas 0,298 -> 0,017%; U[10,100] 0,708 -> 0,404%; jobs correlacionados 0,753 -> 0,759%).
Teste: as 700 instancias idx 2-6 do benchmark (nunca rodadas com os metodos LP-IG + MILP). Em cada instancia: LP-IG (T = 1000, d/n da Entrada 141) -> UMA solucao, e dela dois MILPs restritos (k = 6, 30 s): `lpig_sr6` (candidatas por p_ij)
e `lpig_rc6` (candidatas por custo reduzido; inclui a CPU de resolver o LP com duais); `lpig` (a solucao do IG antes do MILP) e `lpig_T8000` (controle a CPU aproximadamente igual: LP-IG com T = 8000, tarefa separada).
O cpu_time inclui LP, descida, IG e MILP. Saidas: results/lprc_test_entrada156.csv. Retomavel. Uso: python -m experiment.run_lprc_benchmark_entrada156 --run --workers 10
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
OUT = ROOT / "results" / "lprc_test_entrada156.csv"
FIELDS = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "T", "best_known", "lb"]
TASKS = ["main", "lpig_T8000"]


def _task(t):
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, _descent, ur_ig_run
    from pmsp.unrelated_lp import lp_duals, lp_start
    from pmsp.unrelated_srmilp import restricted_milp
    inst = B.load_instance(t["type"], t["instance"])
    seed = int(t["instance"])
    frac = json.loads(L.CHOICE.read_text())[t["n"]]["frac"]
    d = max(2, round(frac * inst.n))
    base = dict(type=t["type"], instance=t["instance"], m=inst.m, n=inst.n, idx=t["idx"], best_known=t["best_known"], lb=inst.lower_bound)
    t0 = time.perf_counter()
    start, _, _ = lp_start(inst)
    _descent(start, 10 ** 6)
    T = 8000 if t["task"] == "lpig_T8000" else 1000
    r = ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), max_iter=T, d=d, seed=seed, initial_solution=start, ls_steps=B.LS_STEPS)
    sol = r["solution"]
    cpu_ig = time.perf_counter() - t0
    if t["task"] == "lpig_T8000":
        return [dict(base, method="lpig_T8000", makespan=int(sol.makespan), cpu_time=cpu_ig, T=T)]
    rows = [dict(base, method="lpig", makespan=int(sol.makespan), cpu_time=cpu_ig, T=T)]
    new, _, c1 = restricted_milp(inst, sol, k=K_SR, time_limit=TL_SR)
    rows.append(dict(base, method="lpig_sr6", makespan=int(new.makespan), cpu_time=cpu_ig + c1, T=T))
    t1 = time.perf_counter()
    _, _, rc = lp_duals(inst)
    c_lp = time.perf_counter() - t1
    new2, _, c2 = restricted_milp(inst, sol, k=K_SR, time_limit=TL_SR, rank=rc)
    rows.append(dict(base, method="lpig_rc6", makespan=int(new2.makespan), cpu_time=cpu_ig + c_lp + c2, T=T))
    return rows


def run(workers):
    done = set()
    if OUT.exists():
        done = {(r["type"], r["instance"], r["method"]) for r in csv.DictReader(open(OUT, newline=""))}
    tasks = []
    for r in B.manifest("test"):
        for tk in TASKS:
            need = ["lpig", "lpig_sr6", "lpig_rc6"] if tk == "main" else ["lpig_T8000"]
            if all((r["type"], r["instance"], mname) in done for mname in need):
                continue
            tasks.append(dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], best_known=r["best_known"], task=tk))
    tasks.sort(key=lambda t: -(int(t["n"]) * int(t["m"])))
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} linhas feitas)", flush=True)
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
                    for row in fut.result():
                        w.writerow(row)
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut]["type"], futs[fut]["instance"], futs[fut]["task"], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 50 == 0 or i == len(tasks):
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
