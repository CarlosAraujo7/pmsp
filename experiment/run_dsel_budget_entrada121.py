"""
experiment/run_dsel_budget_entrada121.py
==========================================
Entrada 121 (Ciclo 3, frente 2): na Entrada 114 o Q-Learning NAO capturou a folga do espaco de acoes d (5 fracoes de destruicao) com T=1000.
Duas objecoes: (i) falta orcamento para aprender online; (ii) o aprendizado deve ser CRUZADO entre instancias (pre-treino).
  (i)  T = 5000, 6 classes ((30,8),(50,10),(75,15),(100,20),(150,30),(300,20)) x 50 reps (rep 0-49 das instancias de results.csv):
       igd_fix05..40, igd_random, igd_ql_cal, igd_ucb; AJUSTE do melhor d fixo ao mesmo T (seeds 90000+, 30 por classe; so' d fixos).
  (ii) T = 1000: igd_ql_ws_cal = Q-Learning (estado `cal`) PRE-TREINADO por classe em 50 instancias independentes (seeds 30000+; Q carregada entre
       episodios; epsilon inicial de avaliacao 0,5 como o ils_ql_ws) nas mesmas 6 classes x 50 reps (o T=1000 dos demais braços esta' em dsel_test_entrada114.csv).
Saidas: results/dsel_budget_entrada121.csv, results/dsel_budget_tuning_entrada121.csv, results/qtables_entrada121/*.npy. Retomavel.
Uso: python -m experiment.run_dsel_budget_entrada121 --tune --pretrain --run --workers 12
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment.run_dsel_entrada114 import FIX, FRACS, TUNE_SEED0, action_map  # noqa: E402

CLASSES = [(30, 8), (50, 10), (75, 15), (100, 20), (150, 30), (300, 20)]
N_TEST, N_TUNE, N_TRAIN, TRAIN_SEED0, T_BIG = 50, 30, 50, 30000, 5000
OUT = ROOT / "results" / "dsel_budget_entrada121.csv"
OUT_TUNE = ROOT / "results" / "dsel_budget_tuning_entrada121.csv"
QDIR = ROOT / "results" / "qtables_entrada121"
FIELDS = ["instance", "method", "T", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements"]
BIG_METHODS = FIX + ["igd_random", "igd_ql_cal", "igd_ucb"]


def _agent(seed, q_init=None, eps0=None):
    from experiment import config as CFG
    from pmsp.qlearning import QLearningAgent
    return QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START if eps0 is None else eps0, epsilon_min=CFG.QL_EPSILON_MIN,
                          epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=5, q_init=q_init)


def _pretrain(t):
    from experiment import config as CFG
    from experiment.run_signal_headroom_entrada104 import _state_fn
    from pmsp.ig import ig_run
    from pmsp.instance import PMSPInstance
    n, m = t["n"], t["m"]
    ag = _agent(TRAIN_SEED0)
    fn = _state_fn(n, m)
    for k in range(N_TRAIN):
        inst = PMSPInstance.random(n=n, m=m, seed=TRAIN_SEED0 + k, name=f"train_{n}j_{m}m")
        ig_run(instance=inst, selector=ag, max_iter=1000, seed=TRAIN_SEED0 + k, stagnation_window=CFG.STAGNATION_WINDOW, state_fn=fn, action_map=action_map)
        ag.carry_forward()
    QDIR.mkdir(parents=True, exist_ok=True)
    np.save(QDIR / f"{n}x{m}_d.npy", ag.Q)
    return f"{n}x{m}"


def _run(t):
    from experiment import config as CFG
    from experiment.run_signal_headroom_entrada104 import _state_fn
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ig import ig_run
    from pmsp.instance import PMSPInstance
    from pmsp.selectors import NormalisedUCBSelector, RandomSelector
    n, m, rep, T, mt, tune = t["n"], t["m"], t["rep"], t["T"], t["method"], t["tune"]
    inst = PMSPInstance.random(n=n, m=m, seed=(TUNE_SEED0 + rep) if tune else (CFG.BASE_SEED + rep), name=f"rand_{n}j_{m}m")
    seed = CFG.BASE_SEED + rep
    kw = dict(instance=inst, max_iter=T, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW, action_map=action_map)
    if mt in FIX:
        r = ig_run(selector=FixedActionSelector(FIX.index(mt)), **kw)
    elif mt == "igd_random":
        r = ig_run(selector=RandomSelector(seed=seed, n_actions=5), **kw)
    elif mt == "igd_ql_cal":
        r = ig_run(selector=_agent(seed), state_fn=_state_fn(n, m), **kw)
    elif mt == "igd_ql_ws_cal":
        q = np.load(QDIR / f"{n}x{m}_d.npy")
        r = ig_run(selector=_agent(seed, q_init=q, eps0=CFG.QL_EPSILON_START_WS), state_fn=_state_fn(n, m), **kw)
    elif mt == "igd_ucb":
        r = ig_run(selector=NormalisedUCBSelector(C=CFG.UCB_C, seed=seed, n_actions=5), **kw)
    else:
        raise ValueError(mt)
    return dict(instance=inst.name, method=mt, T=T, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"])


def _drive(tasks, out, workers):
    done = set()
    if out.exists():
        done = {(r["instance"], r["method"], r["T"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if (f"rand_{t['n']}j_{t['m']}m", t["method"], str(t["T"]), str(t["rep"])) not in done]
    pend.sort(key=lambda t: -t["n"] * t["T"])
    print(f"[tasks] {len(pend)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run, t): t for t in pend}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 100 == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("tune", "pretrain", "run"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.tune:
        _drive([dict(n=n, m=m, rep=r, T=T_BIG, method=mt, tune=True) for n, m in CLASSES for r in range(N_TUNE) for mt in FIX], OUT_TUNE, a.workers)
    if a.pretrain:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(CLASSES))) as ex:
            for r in ex.map(_pretrain, [dict(n=n, m=m) for n, m in CLASSES]):
                print("pretrained", r, flush=True)
    if a.run:
        tasks = [dict(n=n, m=m, rep=r, T=T_BIG, method=mt, tune=False) for n, m in CLASSES for r in range(N_TEST) for mt in BIG_METHODS]
        tasks += [dict(n=n, m=m, rep=r, T=1000, method="igd_ql_ws_cal", tune=False) for n, m in CLASSES for r in range(N_TEST)]
        _drive(tasks, OUT, a.workers)


if __name__ == "__main__":
    main()
