"""
experiment/run_cycle56_credit_entrada291.py (Entrada 291, Ciclo 56)
=====================================================================
O DILEMA INFORMACAO-ALINHAMENTO: os sete aprendizes da Entrada 289 com dois creditos ALINHADOS com o objetivo (pmsp/de_credit.py): log_incumbent (a reducao de log10 do erro do incumbente; usa f_opt: limite superior, nao
praticavel) e rel_incumbent (a melhora do incumbente relativa a' dispersao da populacao; praticavel). Mesmas 24 funcoes BBOB, instancias de teste 6-15, 3 sementes, T = 1000, as MESMAS sementes da Entrada 289 (comparacao pareada com o
credito por sucesso). So' gera dados; a analise e as previsoes estao em analysis/report_entrada291_credit.py.
  base   controle aleatorio com registro por decisao do credito (as duas versoes)  -> results/de_credit_base_entrada291.csv, results/logs291/f<fun>_<credito>.npz
  learn  os sete aprendizes com cada credito                                        -> results/de_credit_learn_entrada291.csv
Uso: python -m experiment.run_cycle56_credit_entrada291 --stage base|learn [--workers 10]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

R = ROOT / "results"
LOGS = R / "logs291"
CREDITS = ["log_incumbent", "rel_incumbent"]
FIELDS = ["fun", "inst", "credit", "method", "seed_i", "err", "lg", "crew", "u0", "u1", "u2", "u3", "u4"]


def do_task(args):
    fun, inst, methods = args
    from experiment.run_cycle55_de_entrada289 import N_SEED, T, seed_of
    from pmsp import de_credit as DC
    rows, logs = [], []
    for credit in CREDITS:
        for method in methods:
            for s in range(N_SEED):
                out = DC.run_de_credit(fun, inst, method, seed_of(fun, inst, method, s), credit, T=T, log=True)
                u = np.bincount(out["A"].astype(np.int64), minlength=5) / T
                rows.append(dict(fun=fun, inst=inst, credit=credit, method=method, seed_i=s, err=out["err"], lg=out["lg"], crew=out["crew"], u0=u[0], u1=u[1], u2=u[2], u3=u[3], u4=u[4]))
                if method == "rand":
                    logs.append((credit, inst, s, out["A"], out["Rc"]))
    return fun, inst, rows, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["base", "learn"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    from experiment.run_cycle55_de_entrada289 import FUNS, LEARNERS, METHOD_ID, TEST_INST
    methods = ["rand"] if a.stage == "base" else LEARNERS
    out = R / ("de_credit_base_entrada291.csv" if a.stage == "base" else "de_credit_learn_entrada291.csv")
    tasks = sorted(((f, i, methods) for f in FUNS for i in TEST_INST), key=lambda t: -(t[0] in (21, 22)))
    t0 = time.perf_counter()
    rows, rand_logs = [], {(f, c): [] for f in FUNS for c in CREDITS}
    with Pool(a.workers) as pool:
        for k, (fun, inst, r, logs) in enumerate(pool.imap_unordered(do_task, tasks)):
            rows += r
            for (credit, i, s, A, Rc) in logs:
                rand_logs[(fun, credit)].append((i, s, A, Rc))
            if (k + 1) % 24 == 0:
                print(f"[credit291 {a.stage}] {k + 1}/{len(tasks)} tarefas ({time.perf_counter() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["fun"], r["inst"], r["credit"], METHOD_ID[r["method"]], r["seed_i"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if a.stage == "base":
        LOGS.mkdir(parents=True, exist_ok=True)
        for (fun, credit), lg in rand_logs.items():
            lg.sort(key=lambda x: (x[0], x[1]))
            np.savez_compressed(LOGS / f"f{fun:02d}_{credit}.npz", A=np.array([x[2] for x in lg]), Rc=np.array([x[3] for x in lg]))
    print(f"[credit291 {a.stage}] {len(rows)} execucoes em {time.perf_counter() - t0:.0f}s -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
