"""
experiment/run_cycle55_de_entrada289.py (Entrada 289, Ciclo 55)
=================================================================
O protocolo do artigo aplicado a um hospedeiro PADRAO de computacao evolutiva: evolucao diferencial com um pool de 5 estrategias (pmsp/de_host.py) nas 24 funcoes BBOB sem ruido, D = 10, NP = 100, T = 1000 geracoes (uma decisao por
geracao), orcamento de 100.100 avaliacoes IGUAL para todos os metodos (nao ha' diferenca de custo entre estrategias: a divisao decisao/custo nao se aplica). Recompensa = taxa de sucesso da geracao (sinal denso).
Classes = as 24 funcoes. Instancias BBOB 1-5 para o AJUSTE (a estrategia validada de cada funcao = a de menor log10 do erro medio) e 6-15 para o TESTE; 3 sementes por instancia.
Estagios (cada um so' gera dados; as analises e as previsoes estao em analysis/report_entrada289_de.py e analysis/predict_entrada289_twin.py):
  tune   acoes fixas 0-4 nas instancias de ajuste            -> results/de_tune_entrada289.csv
  base   controle aleatorio (com registro por decisao) e acoes fixas 0-4 nas instancias de teste -> results/de_test_base_entrada289.csv, results/logs289/f<fun>_rand.npz
  learn  os sete aprendizes (ql_std, ql_rt, ucb, pg, pm, ap, etc) nas instancias de teste          -> results/de_test_learn_entrada289.csv
Cada linha: funcao, instancia, metodo, semente, erro final (f_best - f_opt), log10 do erro (piso 1e-8), soma das recompensas (taxa de sucesso por geracao) e a fracao das decisoes em cada acao (u0..u4).
Uso: python -m experiment.run_cycle55_de_entrada289 --stage tune|base|learn [--workers 10]
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
LOGS = R / "logs289"
FUNS = list(range(1, 25))
TUNE_INST, TEST_INST = [1, 2, 3, 4, 5], list(range(6, 16))
N_SEED, T = 3, 1000
LEARNERS = ["ql_std", "ql_rt", "ucb", "pg", "pm", "ap", "etc"]
METHOD_ID = {m: i for i, m in enumerate(["rand", "arm0", "arm1", "arm2", "arm3", "arm4"] + LEARNERS)}
SEED0 = 91_000_000
FIELDS = ["fun", "inst", "method", "seed_i", "err", "lg", "rew", "u0", "u1", "u2", "u3", "u4"]


def seed_of(fun, inst, method, s):
    return SEED0 + fun * 1_000_000 + inst * 10_000 + METHOD_ID[method] * 100 + s


def do_task(args):
    fun, inst, methods = args
    from pmsp import de_host as DH
    rows, logs = [], []
    for method in methods:
        for s in range(N_SEED):
            out = DH.run_de(fun, inst, method, seed_of(fun, inst, method, s), T=T, log=True)
            u = np.bincount(out["A"].astype(np.int64), minlength=5) / T
            rows.append(dict(fun=fun, inst=inst, method=method, seed_i=s, err=out["err"], lg=out["lg"], rew=float(out["Rw"].sum()), u0=u[0], u1=u[1], u2=u[2], u3=u[3], u4=u[4]))
            if method == "rand":
                logs.append((inst, s, out["A"], out["Rw"], out["S"]))
    return fun, inst, rows, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "base", "learn"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.stage == "tune":
        insts, methods, out = TUNE_INST, [f"arm{k}" for k in range(5)], R / "de_tune_entrada289.csv"
    elif a.stage == "base":
        insts, methods, out = TEST_INST, ["rand"] + [f"arm{k}" for k in range(5)], R / "de_test_base_entrada289.csv"
    else:
        insts, methods, out = TEST_INST, LEARNERS, R / "de_test_learn_entrada289.csv"
    tasks = sorted(((f, i, methods) for f in FUNS for i in insts), key=lambda t: -(t[0] in (21, 22)))    # as funcoes lentas (21, 22) primeiro
    t0 = time.perf_counter()
    rows, rand_logs = [], {f: [] for f in FUNS}
    with Pool(a.workers) as pool:
        for k, (fun, inst, r, logs) in enumerate(pool.imap_unordered(do_task, tasks)):
            rows += r
            rand_logs[fun] += logs
            if (k + 1) % 24 == 0:
                print(f"[de289 {a.stage}] {k + 1}/{len(tasks)} tarefas ({time.perf_counter() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["fun"], r["inst"], METHOD_ID[r["method"]], r["seed_i"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    if a.stage == "base":
        LOGS.mkdir(parents=True, exist_ok=True)
        for fun, lg in rand_logs.items():
            lg.sort(key=lambda x: (x[0], x[1]))
            np.savez_compressed(LOGS / f"f{fun:02d}_rand.npz", A=np.array([x[2] for x in lg]), Rw=np.array([x[3] for x in lg]), S=np.array([x[4] for x in lg]),
                                inst=np.array([x[0] for x in lg]), seed=np.array([x[1] for x in lg]))
    print(f"[de289 {a.stage}] {len(rows)} execucoes em {time.perf_counter() - t0:.0f}s -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
