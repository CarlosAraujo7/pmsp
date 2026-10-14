"""
experiment/run_signal_headroom_entrada104.py
=============================================
WP4 do plano de revisao (diario, Entrada 104): POR QUE a politica Q-Learning agrega ~0,01% sobre a selecao
aleatoria no IG? Duas explicacoes rivais, testadas com pre-especificacao:

(A) FOLGA (headroom): as 3 estrategias de destruicao sao quase intercambiaveis -- nenhum seletor teria o que
    aprender. Mede-se rodando o IG com CADA estrategia fixa (ig_fix0/1/2) e comparando com a selecao uniforme
    (ig_random, em results.csv).
(B) SINAL: a recompensa original max(delta,0) e' esparsa (~3 valores nao nulos por execucao) e/ou o agente aprende
    do zero a cada execucao. Testam-se: recompensa DENSA (reducao relativa da soma dos quadrados das cargas),
    PRE-TREINO por classe (50 instancias independentes, seeds 30000-30049, Q carregada entre episodios) e ambos.

Todas as variantes de Q-Learning usam o estado recalibrado `cal` (Entrada 98; limiares em
results/state_calibration_entrada98.json), pois o estado padrao tem f1, f2 degenerados. Mesmas instancias/seeds de
results.csv (instancia = PMSPInstance.random(n, m, seed=rep); seed da execucao = BASE_SEED+rep), T=1000, N=100/classe.
Pre-treinados avaliam com epsilon inicial 0,5 (CFG.QL_EPSILON_START_WS, como o ils_ql_ws).

Saida: results/signal_headroom_entrada104.csv (retomavel; NAO toca results.csv).
Uso: python -m experiment.run_signal_headroom_entrada104 --pretrain --workers 12
     python -m experiment.run_signal_headroom_entrada104 --check
     python -m experiment.run_signal_headroom_entrada104 --run --workers 12
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

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20),
           (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
METHODS = ["ig_fix0", "ig_fix1", "ig_fix2", "ig_ql_dense_cal", "ig_ql_ws_cal", "ig_ql_dense_ws_cal"]
N_REPS = 100
N_TRAIN = 50
TRAIN_SEED0 = 30000
OUT = ROOT / "results" / "signal_headroom_entrada104.csv"
QDIR = ROOT / "results" / "qtables_entrada104"
CAL = ROOT / "results" / "state_calibration_entrada98.json"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements"]


def _thr(n: int, m: int) -> dict:
    return json.loads(CAL.read_text())["ig"][f"{n}x{m}"]["cal"]


def _agent(seed: int, q_init=None, eps0=None):
    from experiment import config as CFG
    from pmsp.ig import N_DESTROY_STRATEGIES
    from pmsp.qlearning import QLearningAgent
    return QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA,
                          epsilon_start=CFG.QL_EPSILON_START if eps0 is None else eps0,
                          epsilon_min=CFG.QL_EPSILON_MIN, epsilon_decay=CFG.QL_EPSILON_DECAY,
                          seed=seed, n_actions=N_DESTROY_STRATEGIES, q_init=q_init)


def _state_fn(n: int, m: int):
    from experiment.run_state_recalibration_entrada98 import make_state_fn
    return make_state_fn("cal", _thr(n, m), {"f1": [0, 0, 0], "f2": [0, 0, 0], "states": set()})


def _pretrain(task: dict) -> str:
    from experiment import config as CFG
    from pmsp.ig import ig_run
    from pmsp.instance import PMSPInstance
    from pmsp.rewards import dense_balance_reward
    n, m, kind = task["n"], task["m"], task["kind"]
    agent = _agent(seed=TRAIN_SEED0)
    fn = _state_fn(n, m)
    for k in range(N_TRAIN):
        inst = PMSPInstance.random(n=n, m=m, seed=TRAIN_SEED0 + k, name=f"train_{n}j_{m}m")
        ig_run(instance=inst, selector=agent, max_iter=1000, seed=TRAIN_SEED0 + k,
               stagnation_window=CFG.STAGNATION_WINDOW, state_fn=fn,
               reward_fn=dense_balance_reward if kind == "dense" else None)
        agent.carry_forward()
    QDIR.mkdir(parents=True, exist_ok=True)
    np.save(QDIR / f"{n}x{m}_{kind}.npy", agent.Q)
    return f"{n}x{m}_{kind}"


def run_one(method: str, n: int, m: int, rep: int) -> dict:
    from experiment import config as CFG
    from experiment.runner import run_single
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ig import ig_run
    from pmsp.instance import PMSPInstance
    from pmsp.rewards import dense_balance_reward
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    seed = CFG.BASE_SEED + rep
    kw = dict(instance=inst, max_iter=1000, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW)
    if method.startswith("ig_fix"):
        r = ig_run(selector=FixedActionSelector(int(method[-1])), **kw)
    elif method == "ig_ql_cal":            # reproducao de uma variante ja armazenada (checagem de determinismo)
        r = ig_run(selector=_agent(seed), state_fn=_state_fn(n, m), **kw)
    elif method == "ig_ql_dense_cal":
        r = ig_run(selector=_agent(seed), state_fn=_state_fn(n, m), reward_fn=dense_balance_reward, **kw)
    elif method == "ig_ql_ws_cal":
        q = np.load(QDIR / f"{n}x{m}_sparse.npy")
        r = ig_run(selector=_agent(seed, q_init=q, eps0=CFG.QL_EPSILON_START_WS), state_fn=_state_fn(n, m), **kw)
    elif method == "ig_ql_dense_ws_cal":
        q = np.load(QDIR / f"{n}x{m}_dense.npy")
        r = ig_run(selector=_agent(seed, q_init=q, eps0=CFG.QL_EPSILON_START_WS), state_fn=_state_fn(n, m),
                   reward_fn=dense_balance_reward, **kw)
    else:
        raise ValueError(method)
    return dict(instance=f"rand_{n}j_{m}m", method=method, rep=rep, n=n, m=m, lb=inst.lower_bound,
                makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"])


def _run(task: dict) -> dict:
    return run_one(task["method"], task["n"], task["m"], task["rep"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.check:     # ig_ql_cal reproduzido pelo gancho state_fn == armazenado (Entrada 98)?
        import pandas as pd
        ref = pd.read_csv(ROOT / "results" / "state_recalibration_entrada98.csv").query("method == 'ig_ql_cal'")
        bad = tot = 0
        for (n, m) in [(20, 6), (50, 10), (100, 20)]:
            for rep in range(5):
                row = ref[(ref.instance == f"rand_{n}j_{m}m") & (ref.rep == rep)]
                if row.empty:
                    continue
                tot += 1
                mk = run_one("ig_ql_cal", n, m, rep)["makespan"]
                bad += int(mk != int(row.makespan.iloc[0]))
                print(f"  {n}x{m} rep{rep}: novo {mk} vs armazenado {int(row.makespan.iloc[0])}")
        print(f"[check] {tot - bad}/{tot} identicos")
        return

    if args.pretrain:
        t0 = time.perf_counter()
        tasks = [dict(n=n, m=m, kind=k) for n, m in (CLASSES[:2] if args.smoke else CLASSES) for k in ("sparse", "dense")]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(_pretrain, t) for t in tasks]):
                print("  pretrained", fut.result(), f"{time.perf_counter() - t0:.0f}s", flush=True)
        return

    if args.run:
        classes = CLASSES[:1] if args.smoke else CLASSES
        out = OUT.with_name("signal_headroom_entrada104_SMOKE.csv") if args.smoke else OUT
        done = set()
        if out.exists():
            with open(out, newline="") as f:
                done = {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(f)}
        tasks = [dict(n=n, m=m, rep=rep, method=mt) for n, m in classes for rep in range(3 if args.smoke else N_REPS)
                 for mt in METHODS if (f"rand_{n}j_{m}m", mt, str(rep)) not in done]
        print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)")
        new = not out.exists()
        t0 = time.perf_counter()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_run, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futs), 1):
                    try:
                        w.writerow(fut.result())
                    except Exception as e:  # noqa: BLE001
                        print("[ERRO]", futs[fut], e)
                        continue
                    f.flush()
                    os.fsync(f.fileno())
                    if i % 200 == 0 or i == len(tasks):
                        el = time.perf_counter() - t0
                        print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
        print("concluido ->", out)


if __name__ == "__main__":
    main()
