"""
experiment/run_dpolicy_entrada145.py
======================================
Entrada 145 (Ciclo 9, frente 3): TESTE PROSPECTIVO DA LEI DA FOLGA em um espaco de acoes COM folga real: o TAMANHO DE DESTRUICAO d em R||Cmax (benchmark padrao de Fanjul-Peyro e Ruiz), sobre o LP-IG
(Entrada 141; destruicao aleatoria; T = 1000). Acoes: d/n em F = {0,01; 0,02; 0,05; 0,10; 0,20}. No ajuste (280 instancias) a razao makespan/melhor conhecido varia 1-4% entre as acoes
(ex.: n = 500: 1,0056 em 0,01 contra 1,0168 em 0,20), isto e', a folga da selecao aleatoria de d sobre a melhor escolha e' grande, ao contrario do P||Cmax (0,09%).
Teste: as 700 instancias idx 2-6 de cada celula (as mesmas da Entrada 137; metodos NOVOS, pergunta nova). Metodos (todos LP-IG, T = 1000, seed = id da instancia, estrategia de destruicao aleatoria):
  fix_0..fix_4 : d/n fixo em F[k] (folga e braco validado)         rand_d : d/n sorteado uniformemente a cada iteracao (sem aprendizado)
  ql_d         : Q-Learning online sobre as 5 acoes (estado `rel` da Entrada 118)   ucb_d : UCB1 normalizado sobre as 5 acoes
  ql_d_pre     : Q-Learning com tabela PRE-TREINADA offline (instancias de ajuste do mesmo (tipo, n); 10 instancias; epsilon inicial 1,0 na 1a instancia do treino e 0,5 nas seguintes; 0,1 no teste)
Fases: --pretrain (tabelas por (tipo, n) nas instancias de ajuste; results/dpolicy_qtables_entrada145.npz) ; --run (teste). Retomavel.
Uso: python -m experiment.run_dpolicy_entrada145 --pretrain --workers 11 ; --run --workers 10
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

import experiment.run_rcmax_benchmark_entrada137 as B  # noqa: E402

F = [0.01, 0.02, 0.05, 0.10, 0.20]
QT = ROOT / "results" / "dpolicy_qtables_entrada145.npz"
OUT = ROOT / "results" / "dpolicy_test_entrada145.csv"
FIELDS = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "T", "best_known", "lb"]
METHODS = [f"fix_{k}" for k in range(5)] + ["rand_d", "ql_d", "ucb_d", "ql_d_pre"]


def _agent(kind, seed, q_init=None, eps0=1.0):
    from experiment import config as CFG
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import NormalisedUCBSelector, RandomSelector
    if kind == "ucb_d":
        return NormalisedUCBSelector(C=CFG.UCB_C, seed=seed, n_actions=len(F))
    if kind == "rand_d":
        return RandomSelector(seed=seed, n_actions=len(F))
    from pmsp.fixed_selector import FixedActionSelector
    if kind.startswith("fix_"):
        return FixedActionSelector(int(kind[-1]))
    return QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=eps0, epsilon_min=CFG.QL_EPSILON_MIN, epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=len(F), q_init=q_init)


def _solve(inst, kind, seed, q_init=None, eps0=1.0, start=None):
    from pmsp.unrelated import UrStateFn, _descent, ur_ig_run
    from pmsp.unrelated_lp import lp_start
    t0 = time.perf_counter()
    if start is None:
        start, _, _ = lp_start(inst)
        _descent(start, 10 ** 6)
    t_start = time.perf_counter() - t0
    thr = json.loads(B.THR.read_text())
    sf = UrStateFn({"f1": thr["f1"], "f2": thr["f2"]}) if kind in ("ql_d", "ql_d_pre") else None
    sel = _agent(kind, seed, q_init, eps0)
    r = ur_ig_run(inst, sel, max_iter=1000, d=None, seed=seed, initial_solution=start, state_fn=sf, ls_steps=B.LS_STEPS, action_map=lambda a, n: (0, max(2, round(F[a] * n))))
    return r["makespan"], r["cpu_time"] + t_start, sel


def _pretrain_group(args):
    typ, n = args
    rows = [r for r in B.manifest("tune") if r["type"] == typ and int(r["n"]) == n]
    rows.sort(key=lambda r: (int(r["m"]), int(r["idx"])))
    q = None
    for r in rows:
        inst = B.load_instance(r["type"], r["instance"])
        _, _, sel = _solve(inst, "ql_d", int(r["instance"]), q_init=q, eps0=0.5 if q is not None else 1.0)
        q = sel.Q.copy()
    return f"{typ}|{n}", q, len(rows)


def pretrain(workers):
    groups = [(t, n) for t in B.TYPES for n in B.NS]
    out = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, q, k in ex.map(_pretrain_group, groups):
            out[key] = q
            print(f"  pretrained {key} ({k} instancias) {time.perf_counter() - t0:.0f}s", flush=True)
    np.savez(QT, **out)
    print("salvo ->", QT)


def _task(t):
    inst = B.load_instance(t["type"], t["instance"])
    seed = int(t["instance"])
    kind = t["method"]
    q_init, eps0 = None, 1.0
    if kind == "ql_d_pre":
        q_init, eps0 = np.load(QT)[f"{t['type']}|{t['n']}"], 0.1
    ms, cpu, _ = _solve(inst, kind, seed, q_init=q_init, eps0=eps0)
    return dict(type=t["type"], instance=t["instance"], m=inst.m, n=inst.n, idx=t["idx"], method=kind, makespan=int(ms), cpu_time=cpu, T=1000, best_known=t["best_known"], lb=inst.lower_bound)


def run(workers):
    done = set()
    if OUT.exists():
        done = {(r["type"], r["instance"], r["method"]) for r in csv.DictReader(open(OUT, newline=""))}
    tasks = [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], method=mt, best_known=r["best_known"]) for r in B.manifest("test") for mt in METHODS
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
                if i % 200 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.pretrain:
        pretrain(a.workers)
    if a.run:
        run(a.workers)


if __name__ == "__main__":
    main()
