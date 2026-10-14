"""
experiment/run_lpig_benchmark_entrada141.py
=============================================
Entrada 141 (Ciclo 9, frente 1): um IG para R||Cmax que PARTE DA RELAXACAO LINEAR (LP-IG; `pmsp/unrelated_lp.py`: LP de atribuicao do HiGHS, arredondamento da solucao basica, descida completa)
no benchmark padrao de Fanjul-Peyro e Ruiz. Projeto (instancias de AJUSTE, idx 0-1, sem tocar o teste; `results/lpig_design_exploration_entrada141.csv`, 126 instancias): LP-IG com T = 1000 teve GAP medio ao
melhor conhecido de 0,83% contra 2,02% do IG a partir da construtiva (mesma CPU); em n = 1000: 0,31% contra 2,25%; descida mais forte (100 passos) sem efeito; d/n pela metade melhor (0,80%).
Teste: as 420 instancias RESERVADAS (idx 7-9 de cada celula; nunca usadas nas Entradas 137-138). Fases (cada uma congelada em hash antes da seguinte):
  --tune-d     : d/n do LP-IG por n em {0,01; 0,02; 0,05; 0,10; 0,20} (ajuste, T = 1000, 280 instancias; menor razao media makespan/melhor conhecido)
  --calibrate  : razao de CPU sequencial por n (instancias de ajuste, m = 30, 3 tipos): K = CPU(LP-IG, T = 1000)/CPU(IG-LS aleatorio, T = 1000, d/n da Entrada 137) -> T_eq do baseline = max(1000, round(1000 K))
  --run        : teste: ur_igls_random (T = 1000; d/n da Entrada 137), ur_igls_random_eq (T_eq), lpig (T = 1000), lpig_ql (T = 1000, Q-Learning sobre as 4 estrategias), lpig_T5000
O cpu_time dos metodos LP-IG inclui o LP, o arredondamento e a descida. Saidas: results/lpig_*_entrada141.{csv,json}. Retomavel.
Uso: python -m experiment.run_lpig_benchmark_entrada141 --tune-d --workers 11 ; --calibrate ; --run --workers 10
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

FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]
TUNE = ROOT / "results" / "lpig_tune_d_entrada141.csv"
CHOICE = ROOT / "results" / "lpig_choice_d_entrada141.json"
CALIB = ROOT / "results" / "lpig_cpu_ratio_entrada141.json"
OUT = ROOT / "results" / "lpig_test_entrada141.csv"
FIELDS = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "T", "d", "best_known", "lb"]
METHODS = ["ur_igls_random", "ur_igls_random_eq", "lpig", "lpig_ql", "lpig_T5000"]


def _solve(inst, mt, T, frac, seed):
    """Devolve (makespan, cpu_total, d). Os metodos 'lpig*' incluem a CPU do LP, do arredondamento e da descida."""
    from experiment import config as CFG
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, UrStateFn, _descent, ur_ig_run
    from pmsp.unrelated_lp import lp_start
    d = max(2, round(frac * inst.n))
    if mt in ("ur_igls_random", "ur_igls_random_eq"):
        r = ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), max_iter=T, d=d, seed=seed, ls_steps=B.LS_STEPS)
        return r["makespan"], r["cpu_time"], d
    t0 = time.perf_counter()
    start, _, _ = lp_start(inst)
    _descent(start, 10 ** 6)
    t_start = time.perf_counter() - t0
    if mt == "lpig_ql":
        thr = json.loads(B.THR.read_text())
        sf = UrStateFn({"f1": thr["f1"], "f2": thr["f2"]})
        sel = QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN, epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed,
                             n_actions=N_UR_DESTROY)
        r = ur_ig_run(inst, sel, max_iter=T, d=d, seed=seed, initial_solution=start, state_fn=sf, ls_steps=B.LS_STEPS)
    else:
        r = ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), max_iter=T, d=d, seed=seed, initial_solution=start, ls_steps=B.LS_STEPS)
    return r["makespan"], r["cpu_time"] + t_start, d


def _task(t):
    inst = B.load_instance(t["type"], t["instance"])
    base = t["method"].split("@")[0]
    ms, cpu, d = _solve(inst, base, t["T"], t["frac"], int(t["instance"]))
    return dict(type=t["type"], instance=t["instance"], m=inst.m, n=inst.n, idx=t["idx"], method=t["method"], makespan=int(ms), cpu_time=cpu, T=t["T"], d=d, best_known=t["best_known"], lb=inst.lower_bound)


def _drive(tasks, out, workers, every=50):
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"], r["method"]) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if (t["type"], t["instance"], t["method"]) not in done]
    pend.sort(key=lambda t: -(int(t["n"]) * int(t["m"]) * int(t["T"])))
    print(f"[tasks] {len(pend)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_task, t): t for t in pend}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut]["type"], futs[fut]["instance"], futs[fut]["method"], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % every == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def tune_d(workers):
    tasks = [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], method=f"lpig@{fr}", T=1000, frac=fr, best_known=r["best_known"]) for r in B.manifest("tune") for fr in FRACS]
    _drive(tasks, TUNE, workers, 100)
    import pandas as pd
    d = pd.read_csv(TUNE)
    d["frac"] = d.method.str.split("@").str[1].astype(float)
    d["ratio"] = d.makespan / d.best_known
    choice = {}
    for n, g in d.groupby("n"):
        mean = g.groupby("frac").ratio.mean()
        choice[str(int(n))] = dict(frac=float(mean.idxmin()), means={str(k): float(v) for k, v in mean.items()})
        print(f"[tune-d] n = {int(n)} -> d/n = {mean.idxmin()} ({', '.join(f'{k}:{v:.4f}' for k, v in mean.items())})", flush=True)
    CHOICE.write_text(json.dumps(choice, indent=1))


def calibrate():
    cd0, cd1 = json.loads(B.CHOICE_D.read_text()), json.loads(CHOICE.read_text())
    out = {}
    for n in B.NS:
        rs = []
        for r in B.manifest("tune"):
            if int(r["n"]) != n or int(r["m"]) != 30 or r["type"] not in ("U1a100", "U100a200", "JobsCorre"):
                continue
            inst = B.load_instance(r["type"], r["instance"])
            seed = int(r["instance"])
            c0 = _solve(inst, "ur_igls_random", 1000, cd0[str(n)]["frac"], seed)[1]
            c1 = _solve(inst, "lpig", 1000, cd1[str(n)]["frac"], seed)[1]
            rs.append(c1 / c0)
        rs.sort()
        k = float(np.median(rs))
        out[str(n)] = dict(K=k, T_eq=max(1000, int(round(1000 * k))), n_inst=len(rs), ratios=rs)
        print(f"[calib] n = {n}: K = {k:.2f} -> T_eq (baseline) = {out[str(n)]['T_eq']} ({len(rs)} instancias)", flush=True)
    CALIB.write_text(json.dumps(out, indent=1))


def run(workers):
    cd0, cd1, cal = json.loads(B.CHOICE_D.read_text()), json.loads(CHOICE.read_text()), json.loads(CALIB.read_text())
    tasks = []
    for r in B.manifest("reserve"):
        n = r["n"]
        for mt in METHODS:
            frac = cd0[n]["frac"] if mt.startswith("ur_") else cd1[n]["frac"]
            T = 5000 if mt == "lpig_T5000" else (cal[n]["T_eq"] if mt == "ur_igls_random_eq" else 1000)
            tasks.append(dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=n, m=r["m"], method=mt, T=T, frac=frac, best_known=r["best_known"]))
    _drive(tasks, OUT, workers, 50)
    print("concluido ->", OUT)


def main():
    ap = argparse.ArgumentParser()
    for k in ("tune-d", "calibrate", "run"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.tune_d:
        tune_d(a.workers)
    if a.calibrate:
        calibrate()
    if a.run:
        run(a.workers)


if __name__ == "__main__":
    main()
