"""
experiment/run_state_recalibration_entrada98.py
=================================================
Itens 1 e 2 da prioridade 0 (diario_pesquisa.md, Entradas 97-98).

ITEM 1 -- espaco de estados recalibrado. Em execucoes reais f1 (gap) e f2 (CV
das cargas) ficam no bin 0 em 100% dos passos (limiares 0,10/0,35 e 0,15/0,40
altos demais). Dois bracos, mesmo agente Q-Learning, mesmos hiperparametros,
mesmas seeds; so' o estado muda:
  *_ql_cal : f1 e f2 discretizados com limiares = quantis 1/3 e 2/3 da propria
             feature (valor absoluto) em execucoes de CALIBRACAO (20 instancias
             com seeds 10000-10019, selecao aleatoria, mesma classe e framework);
  *_ql_rel : f1 e f2 RELATIVOS ao valor no inicio da execucao (f/f(0); 0 se
             f(0)=0), com limiares = quantis 1/3 e 2/3 da razao nas mesmas
             execucoes de calibracao -- remove a componente "dificuldade da
             instancia", constante dentro da corrida e inutil para um agente que
             aprende do zero em cada corrida.
  f3 (progresso) e f4 (estagnacao) inalterados.
ITEM 2 -- UCB1 com recompensa normalizada (pmsp.selectors.NormalisedUCBSelector):
  ils_ucb_norm, ig_ucb_norm; e, nas 6 classes originais, ils_random / ils_ucb /
  ils_ql (estado padrao) em N=100 (os resultados armazenados la' sao N=30).

Todas as execucoes: T=1000, mesmas instancias/seeds de results.csv
(PMSPInstance.random(seed=BASE_SEED+rep)), N=100 por classe, 12 classes. Para o
ILS usa-se o mesmo perturbation_threshold das baterias existentes: 15 para
n>=150, 50 caso contrario (Entradas 48/68).

Uso (da raiz do repositorio):
    python -m experiment.run_state_recalibration_entrada98 --calibrate
    python -m experiment.run_state_recalibration_entrada98 --check
    python -m experiment.run_state_recalibration_entrada98 --run --workers 8
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
ORIG6 = {(20, 4), (20, 6), (30, 8), (50, 10), (75, 15), (100, 20)}
N_REPS = 100
CAL_SEEDS = list(range(10000, 10020))
OUT = ROOT / "results" / "state_recalibration_entrada98.csv"
CAL = ROOT / "results" / "state_calibration_entrada98.json"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements",
          "distinct_states", "f1_bins", "f2_bins"]
BINS34 = ([0.33, 0.67], [0.25, 0.60])     # f3, f4 (inalterados)


def ils_threshold(n: int) -> int:
    return 15 if n >= 150 else 50


def tasks_for(n: int, m: int):
    ms = ["ig_ql_cal", "ig_ql_rel", "ig_ucb_norm", "ils_ql_cal", "ils_ql_rel", "ils_ucb_norm"]
    if (n, m) in ORIG6:
        ms += ["ils_random", "ils_ucb", "ils_ql"]
    return ms


# ---------------------------------------------------------------------------
# funcao de estado alternativa (monkeypatch de compute_state_cfg dentro do worker)
# ---------------------------------------------------------------------------
def make_state_fn(kind: str, thr: dict, counters: dict):
    from pmsp.bounds import f1_relative_gap, f2_load_cv, f3_progress, f4_stagnation, _digitize
    init = {"f1": 0.0, "f2": 0.0}

    def fn(sol, t, max_iter, iters_no_improve, window, cfg):
        f1, f2 = f1_relative_gap(sol), f2_load_cv(sol)
        if kind == "rel":
            if t == 0:
                init["f1"], init["f2"] = f1, f2
            f1 = f1 / init["f1"] if init["f1"] > 0 else 0.0
            f2 = f2 / init["f2"] if init["f2"] > 0 else 0.0
        b1, b2 = _digitize(f1, thr["f1"]), _digitize(f2, thr["f2"])
        b3 = _digitize(f3_progress(t, max_iter), BINS34[0])
        b4 = _digitize(f4_stagnation(iters_no_improve, window), BINS34[1])
        counters["f1"][b1] += 1
        counters["f2"][b2] += 1
        s = b1 * 27 + b2 * 9 + b3 * 3 + b4
        counters["states"].add(s)
        return s
    return fn


def _instance(n, m, rep):
    from pmsp.instance import PMSPInstance
    return PMSPInstance.random(n=n, m=m, seed=rep, name=f"rand_{n}j_{m}m")


def _run_method(method: str, n: int, m: int, rep: int, thr_all: dict | None):
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.bounds import DEFAULT_STATE_CONFIG
    from pmsp.qlearning import QLearningAgent
    import pmsp.ig as ig_mod
    import pmsp.ils as ils_mod

    inst = _instance(n, m, rep)
    seed = CFG.BASE_SEED + rep
    extra = dict(distinct_states="", f1_bins="", f2_bins="")
    fw = "ig" if method.startswith("ig_") else "ils"
    ils_kw = dict(perturbation_threshold=ils_threshold(n), n_perturb=CFG.N_PERTURB)

    if method in ("ig_ql_cal", "ig_ql_rel", "ils_ql_cal", "ils_ql_rel"):
        kind = "cal" if method.endswith("_cal") else "rel"
        thr = thr_all[fw][f"{n}x{m}"][kind]
        counters = {"f1": [0, 0, 0], "f2": [0, 0, 0], "states": set()}
        fn = make_state_fn(kind, thr, counters)
        if fw == "ig":
            agent = QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START,
                                   epsilon_min=CFG.QL_EPSILON_MIN, epsilon_decay=CFG.QL_EPSILON_DECAY,
                                   seed=seed, n_actions=ig_mod.N_DESTROY_STRATEGIES)
            orig = ig_mod.compute_state_cfg
            ig_mod.compute_state_cfg = fn
            try:
                r = ig_mod.ig_run(instance=inst, selector=agent, max_iter=1000, seed=seed,
                                  stagnation_window=CFG.STAGNATION_WINDOW, state_cfg=DEFAULT_STATE_CONFIG)
            finally:
                ig_mod.compute_state_cfg = orig
        else:
            agent = QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START,
                                   epsilon_min=CFG.QL_EPSILON_MIN, epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed)
            orig = ils_mod.compute_state_cfg
            ils_mod.compute_state_cfg = fn
            try:
                r = ils_mod.ils_run(instance=inst, selector=agent, max_iter=1000, seed=seed,
                                    stagnation_window=CFG.STAGNATION_WINDOW, state_cfg=DEFAULT_STATE_CONFIG, **ils_kw)
            finally:
                ils_mod.compute_state_cfg = orig
        tot1, tot2 = sum(counters["f1"]), sum(counters["f2"])
        extra = dict(distinct_states=len(counters["states"]),
                     f1_bins="|".join(f"{c / tot1:.3f}" for c in counters["f1"]),
                     f2_bins="|".join(f"{c / tot2:.3f}" for c in counters["f2"]))
    else:
        sel = make_ils_selector(method, seed)
        if fw == "ig":
            r = ig_mod.ig_run(instance=inst, selector=sel, max_iter=1000, seed=seed,
                              stagnation_window=CFG.STAGNATION_WINDOW)
        else:
            r = ils_mod.ils_run(instance=inst, selector=sel, max_iter=1000, seed=seed,
                                stagnation_window=CFG.STAGNATION_WINDOW, **ils_kw)
    row = dict(instance=inst.name, method=method, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"],
               gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"])
    row.update(extra)
    return row


def _task(args):
    method, n, m, rep, thr_all = args
    return _run_method(method, n, m, rep, thr_all)


# ---------------------------------------------------------------------------
# calibracao
# ---------------------------------------------------------------------------
def _calib_one(args):
    fw, n, m = args
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    import pmsp.ig as ig_mod
    import pmsp.ils as ils_mod
    mod = ig_mod if fw == "ig" else ils_mod
    orig = mod.compute_features
    logs = []

    def logger(solution, t, max_iter, iters_no_improve, stagnation_window=50):
        v = orig(solution, t, max_iter, iters_no_improve, stagnation_window)
        logs[-1].append((float(v[0]), float(v[1])))
        return v

    mod.compute_features = logger
    try:
        for sd in CAL_SEEDS:
            inst = _instance(n, m, sd)
            logs.append([])
            seed = CFG.BASE_SEED + sd
            if fw == "ig":
                ig_mod.ig_run(instance=inst, selector=make_ils_selector("ig_random", seed), max_iter=1000, seed=seed,
                              stagnation_window=CFG.STAGNATION_WINDOW)
            else:
                ils_mod.ils_run(instance=inst, selector=make_ils_selector("ils_random", seed), max_iter=1000, seed=seed,
                                stagnation_window=CFG.STAGNATION_WINDOW, perturbation_threshold=ils_threshold(n),
                                n_perturb=CFG.N_PERTURB)
    finally:
        mod.compute_features = orig
    absf = np.concatenate([np.array(x) for x in logs])
    rel = []
    for x in logs:
        a = np.array(x)
        f10, f20 = a[0, 0], a[0, 1]
        rel.append(np.column_stack([a[:, 0] / f10 if f10 > 0 else np.zeros(len(a)),
                                    a[:, 1] / f20 if f20 > 0 else np.zeros(len(a))]))
    relf = np.concatenate(rel)
    q = [1 / 3, 2 / 3]
    return fw, f"{n}x{m}", {
        "cal": {"f1": [float(v) for v in np.quantile(absf[:, 0], q)], "f2": [float(v) for v in np.quantile(absf[:, 1], q)]},
        "rel": {"f1": [float(v) for v in np.quantile(relf[:, 0], q)], "f2": [float(v) for v in np.quantile(relf[:, 1], q)]},
    }


def calibrate(workers: int):
    jobs = [(fw, n, m) for fw in ("ig", "ils") for (n, m) in CLASSES]
    jobs.sort(key=lambda j: -(j[1] * (3 if j[0] == "ils" else 1)))
    out = {"ig": {}, "ils": {}, "calibration_seeds": [CAL_SEEDS[0], CAL_SEEDS[-1]], "quantiles": [1 / 3, 2 / 3]}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_calib_one, j) for j in jobs]):
            fw, cls, thr = fut.result()
            out[fw][cls] = thr
            print(f"[calib] {fw} {cls}: cal f1={np.round(thr['cal']['f1'], 4).tolist()} f2={np.round(thr['cal']['f2'], 4).tolist()} | "
                  f"rel f1={np.round(thr['rel']['f1'], 3).tolist()} f2={np.round(thr['rel']['f2'], 3).tolist()}", flush=True)
    CAL.write_text(json.dumps(out, indent=1, sort_keys=True))
    print("written", CAL)


def check():
    import pandas as pd
    res = pd.read_csv(ROOT / "results" / "results.csv")
    bad = 0
    for method, n, m in [("ils_random", 150, 15), ("ils_random", 40, 5), ("ils_random", 50, 10), ("ig_ucb", 50, 10), ("ils_ucb", 50, 10)]:
        for rep in range(3):
            r = _run_method(method, n, m, rep, None)
            s = res[(res.instance == f"rand_{n}j_{m}m") & (res.method == method) & (res.rep == rep)]
            if len(s):
                ok = int(s.makespan.iloc[0]) == int(r["makespan"])
                bad += (not ok)
                print(f"[check] {method} {n}x{m} rep{rep}: new={r['makespan']} stored={int(s.makespan.iloc[0])} {'OK' if ok else 'DIFF'}")
    print("determinism check:", "PASS" if bad == 0 else f"FAIL ({bad})")


def run(workers: int):
    thr_all = json.loads(CAL.read_text())
    done = set()
    if OUT.exists():
        with open(OUT, newline="") as f:
            done = {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(f)}
    tasks = []
    for (n, m) in CLASSES:
        for method in tasks_for(n, m):
            for rep in range(N_REPS):
                if (f"rand_{n}j_{m}m", method, str(rep)) not in done:
                    tasks.append((method, n, m, rep, thr_all))
    tasks.sort(key=lambda t: -(t[1] * (3 if t[0].startswith("ils") else 1)))
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} ja feitas)", flush=True)
    new = not OUT.exists()
    t0 = time.perf_counter()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_task, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 100 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.workers)
    if a.check:
        check()
    if a.run:
        run(a.workers)
