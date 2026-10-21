"""
experiment/run_dqn_entrada173.py
==================================
Entrada 173 (Ciclo 14, frente I): APRENDIZ NEURAL MAIOR E ESPACO DE ACAO MAIOR. Limite (i) do artigo: 'o aprendiz neural era pequeno (uma camada de 16 unidades, sem replay) e os espacos de acao, pequenos'.
Aprendiz: pmsp/dqn.py (DQN 2 x 64, replay, rede-alvo, Adam, contexto com a taxa de sucesso de cada acao e a ultima acao); o unico hiperparametro ajustado e' a taxa de aprendizado, lr in {0,001; 0,003; 0,01}, em instancias de AJUSTE.
Ambientes:
  R  : tamanho de destruicao d/n in {0,01; 0,02; 0,05; 0,10; 0,20} do LP-IG em R||Cmax no benchmark padrao (as 700 instancias de teste e as 280 de ajuste da Entrada 145/141; bracos fixos e sorteio ja' rodaram la'; resultados do nq_d pequeno: Entrada 162).
  P5 : tamanho de destruicao d/n in {0,05; 0,10; 0,20; 0,30; 0,40} do IG em P||Cmax (Entrada 114: 12 classes x 100 instancias de teste, 30 de ajuste por classe com seeds 90000+; bracos fixos, sorteio e Q-learning ja' rodaram la').
  P15: espaco CONJUNTO estrategia x tamanho (3 x 5 = 15 acoes) do IG em P||Cmax, as mesmas instancias; NOVO: 15 bracos fixos (compilados, pmsp/ig_numba.py: ajuste e teste) para a validacao por classe e o oraculo (folga);
       sorteio (rand15), Q-learning tabular (ql15), LinUCB (linucb15), DQN (dqn15) e o braco validado (val15) em Python no teste, com a mesma semente por instancia.
Fases: --arms15-tune ; --arms15-test (compiladas, rapidas) ; --tune-p5 / --tune-p15 / --tune-r --shard i --nshards N (ajuste do lr) ; --choose ; --run-p5 / --run-p15 / --run-r --shard i --nshards N ; --merge NOME. Retomavel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import config as CFG  # noqa: E402

R = ROOT / "results"
CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
FR = [0.05, 0.10, 0.20, 0.30, 0.40]
LRS = [0.001, 0.003, 0.01]
N_TUNE, N_TEST, TUNE_SEED0 = 30, 100, 90000
CHOICE = R / "dqn_choice_entrada173.json"
FP = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "lr", "arm"]
FA = ["instance", "rep", "arm", "n", "m", "lb", "makespan"]
FRC = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "T", "lr", "best_known", "lb"]


def out_path(name):
    return R / f"dqn_{name}_entrada173.csv"


def amap5(a, n):
    return 0, max(2, round(FR[a] * n))


def amap15(a, n):
    return a // 5, max(2, round(FR[a % 5] * n))


def inst_of(n, m, rep, tune):
    from pmsp.instance import PMSPInstance
    return PMSPInstance.random(n=n, m=m, seed=(TUNE_SEED0 + rep) if tune else (CFG.BASE_SEED + rep), name=f"rand_{n}j_{m}m")


# ---------------------------------------------------------------------------------------------- bracos fixos compilados (P15)
def arms15(tune):
    from pmsp import ig_numba as N
    from pmsp.constructive import lpt
    out = OUT = out_path("arms15_" + ("tune" if tune else "test"))
    done = set()
    if out.exists():
        done = {(r["instance"], r["rep"], r["arm"]) for r in csv.DictReader(open(out, newline=""))}
    new = not out.exists()
    first = inst_of(20, 4, 0, tune)
    N.ig_core(np.asarray(first.p, dtype=np.int64), lpt(first).assignment.astype(np.int64), 4, 2, 0, 10, 0.0, False, 0, np.argsort(first.p)[::-1].astype(np.int64), 1.0)
    t0 = time.perf_counter()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FA)
        if new:
            w.writeheader()
        for n, m in CLASSES:
            for rep in range(N_TUNE if tune else N_TEST):
                if (f"rand_{n}j_{m}m", str(rep), "0") in done:
                    continue
                inst = inst_of(n, m, rep, tune)
                p = np.asarray(inst.p, dtype=np.int64)
                od = np.argsort(inst.p)[::-1].astype(np.int64)
                a0 = lpt(inst).assignment.astype(np.int64)
                for arm in range(15):
                    s, d = amap15(arm, n)
                    best, _, _, _, _ = N.ig_core(p, a0, m, d, s, 1000, 0.0, False, CFG.BASE_SEED + rep + 100000 * arm, od, 1.0)
                    w.writerow(dict(instance=inst.name, rep=rep, arm=arm, n=n, m=m, lb=inst.lower_bound, makespan=int(best)))
            f.flush()
            print(f"[arms15 {'tune' if tune else 'test'}] {n}x{m} ok ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------- P||Cmax em Python
def solve_p(env, method, n, m, rep, tune, lr=None, arm=None):
    from pmsp.dqn import DQNSelector
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ig import ig_run
    from pmsp.linucb import LinUCBSelector
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import RandomSelector
    inst = inst_of(n, m, rep, tune)
    rseed = CFG.BASE_SEED + rep
    A, amap = (5, amap5) if env == "p5" else (15, amap15)
    if method.startswith("dqn"):
        sel = DQNSelector(A, lr=lr, seed=rseed)
    elif method == "rand15":
        sel = RandomSelector(seed=rseed, n_actions=15)
    elif method == "ql15":
        sel = QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN, epsilon_decay=CFG.QL_EPSILON_DECAY, seed=rseed, n_actions=15)
    elif method == "linucb15":
        sel = LinUCBSelector(15, seed=rseed)
    elif method == "val15":
        sel = FixedActionSelector(int(arm))
    else:
        raise ValueError(method)
    r = ig_run(instance=inst, selector=sel, max_iter=1000, seed=rseed, stagnation_window=CFG.STAGNATION_WINDOW, action_map=amap)
    return dict(instance=inst.name, method=method, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"], lr=lr, arm=arm)


def p_tasks(env, kind):
    ch = json.loads(CHOICE.read_text()) if CHOICE.exists() else {}
    tasks = []
    for n, m in CLASSES:
        c = f"{n}x{m}"
        if kind == "tune":
            tasks += [dict(env=env, method=f"dqn@{lr}", n=n, m=m, rep=r, tune=True, lr=lr) for r in range(N_TUNE) for lr in LRS]
        elif env == "p5":
            tasks += [dict(env=env, method="dqn_d", n=n, m=m, rep=r, tune=False, lr=ch["lr_p5"]) for r in range(N_TEST)]
        else:
            for r in range(N_TEST):
                tasks += [dict(env=env, method="rand15", n=n, m=m, rep=r, tune=False), dict(env=env, method="ql15", n=n, m=m, rep=r, tune=False), dict(env=env, method="linucb15", n=n, m=m, rep=r, tune=False),
                          dict(env=env, method="dqn15", n=n, m=m, rep=r, tune=False, lr=ch["lr_p15"]), dict(env=env, method="val15", n=n, m=m, rep=r, tune=False, arm=ch["val15"][c])]
    return tasks


def run_p(env, kind, shard, nshards):
    out = out_path(f"{kind}_{env}")
    part = out.with_name(out.stem + f"_s{shard}.csv")
    done = set()
    for src in (out, part):
        if src.exists() and src.stat().st_size > 0:
            done |= {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(open(src, newline=""))}
    pend = [t for t in p_tasks(env, kind) if (f"rand_{t['n']}j_{t['m']}m", t["method"], str(t["rep"])) not in done]
    pend.sort(key=lambda t: -t["n"] * t["m"])
    pend = [t for k, t in enumerate(pend) if k % nshards == shard]
    print(f"[fatia {shard}/{nshards} {kind} {env}] {len(pend)} pendentes -> {part.name}", flush=True)
    new = not part.exists()
    t0 = time.perf_counter()
    with open(part, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FP)
        if new:
            w.writeheader()
        for i, t in enumerate(pend, 1):
            w.writerow(solve_p(t["env"], t["method"], t["n"], t["m"], t["rep"], t["tune"], t.get("lr"), t.get("arm")))
            f.flush()
            os.fsync(f.fileno())
            if i % 100 == 0 or i == len(pend):
                print(f"  [fatia {shard} {kind} {env} {i}/{len(pend)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido fatia", shard, kind, env)


# ---------------------------------------------------------------------------------------------- R||Cmax (LP-IG)
def solve_r(inst, lr, seed):
    from experiment.run_ctx_entrada159 import F
    from pmsp.dqn import DQNSelector
    from pmsp.unrelated import _descent, ur_ig_run
    from pmsp.unrelated_lp import lp_start
    import experiment.run_rcmax_benchmark_entrada137 as B
    t0 = time.perf_counter()
    start, _, _ = lp_start(inst)
    _descent(start, 10 ** 6)
    t_start = time.perf_counter() - t0
    sel = DQNSelector(n_actions=len(F), lr=lr, seed=seed)
    r = ur_ig_run(inst, sel, max_iter=1000, d=None, seed=seed, initial_solution=start, state_fn=None, ls_steps=B.LS_STEPS, action_map=lambda a, n: (0, max(2, round(F[a] * n))))
    return r["makespan"], r["cpu_time"] + t_start


def r_tasks(kind):
    import experiment.run_rcmax_benchmark_entrada137 as B
    if kind == "tune":
        return [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], lr=lr, method=f"dqn@{lr}", best_known=r["best_known"]) for r in B.manifest("tune") for lr in LRS]
    lr = json.loads(CHOICE.read_text())["lr_r"]
    return [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], lr=lr, method="dqn_big", best_known=r["best_known"]) for r in B.manifest("test")]


def run_r(kind, shard, nshards):
    import experiment.run_rcmax_benchmark_entrada137 as B
    out = out_path(f"{kind}_r")
    part = out.with_name(out.stem + f"_s{shard}.csv")
    done = set()
    for src in (out, part):
        if src.exists() and src.stat().st_size > 0:
            done |= {(r["type"], r["instance"], r["method"]) for r in csv.DictReader(open(src, newline=""))}
    pend = [t for t in r_tasks(kind) if (t["type"], t["instance"], t["method"]) not in done]
    pend.sort(key=lambda t: -(int(t["n"]) * int(t["m"])))
    pend = [t for k, t in enumerate(pend) if k % nshards == shard]
    print(f"[fatia {shard}/{nshards} {kind} r] {len(pend)} pendentes -> {part.name}", flush=True)
    new = not part.exists()
    t0 = time.perf_counter()
    with open(part, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRC)
        if new:
            w.writeheader()
        for i, t in enumerate(pend, 1):
            inst = B.load_instance(t["type"], t["instance"])
            ms, cpu = solve_r(inst, t["lr"], int(t["instance"]))
            w.writerow(dict(type=t["type"], instance=t["instance"], m=inst.m, n=inst.n, idx=t["idx"], method=t["method"], makespan=int(ms), cpu_time=cpu, T=1000, lr=t["lr"], best_known=t["best_known"], lb=inst.lower_bound))
            f.flush()
            os.fsync(f.fileno())
            if i % 20 == 0 or i == len(pend):
                print(f"  [fatia {shard} {kind} r {i}/{len(pend)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido fatia", shard, kind, "r")


# ---------------------------------------------------------------------------------------------- escolhas (somente dados de ajuste) e fusao
def choose():
    import pandas as pd
    ch = json.loads(CHOICE.read_text()) if CHOICE.exists() else {}
    for env, key in (("p5", "lr_p5"), ("p15", "lr_p15")):
        f = out_path(f"tune_{env}")
        if f.exists():
            d = pd.read_csv(f)
            ch[key] = float(d.groupby("lr").gap_pct.mean().idxmin())
            ch[key + "_mean_gap"] = {str(k): float(v) for k, v in d.groupby("lr").gap_pct.mean().items()}
    f = out_path("tune_r")
    if f.exists():
        d = pd.read_csv(f)
        d["ratio"] = d.makespan / d.best_known
        ch["lr_r"] = float(d.groupby("lr").ratio.mean().idxmin())
        ch["lr_r_mean_ratio"] = {str(k): float(v) for k, v in d.groupby("lr").ratio.mean().items()}
    f = out_path("arms15_tune")
    if f.exists():
        d = pd.read_csv(f)
        d["cls"] = d.n.astype(str) + "x" + d.m.astype(str)
        ch["val15"] = {c: int(g.groupby("arm").makespan.mean().idxmin()) for c, g in d.groupby("cls")}
    CHOICE.write_text(json.dumps(ch, indent=1))
    print(json.dumps({k: v for k, v in ch.items() if not k.endswith("mean_gap") and not k.endswith("mean_ratio")}, indent=1))


def merge(name):
    import pandas as pd
    out = out_path(name)
    parts = sorted(out.parent.glob(out.stem + "_s*.csv"))
    frames = [pd.read_csv(out, dtype={"instance": str})] if out.exists() and out.stat().st_size > 0 else []
    frames += [pd.read_csv(q, dtype={"instance": str}) for q in parts]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True)
    key = ["type", "instance", "method"] if "type" in d.columns else ["instance", "method", "rep"]
    d = d.drop_duplicates(key)
    d.to_csv(out, index=False)
    print(f"merge {name} -> {out.name}: {len(d)} linhas de {len(parts)} fatias")


def main():
    ap = argparse.ArgumentParser()
    for k in ("arms15-tune", "arms15-test", "tune-p5", "tune-p15", "tune-r", "run-p5", "run-p15", "run-r", "choose"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--merge", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if getattr(a, "arms15_tune"):
        arms15(True)
    if getattr(a, "arms15_test"):
        arms15(False)
    if getattr(a, "tune_p5"):
        run_p("p5", "tune", a.shard, a.nshards)
    if getattr(a, "tune_p15"):
        run_p("p15", "tune", a.shard, a.nshards)
    if getattr(a, "tune_r"):
        run_r("tune", a.shard, a.nshards)
    if a.choose:
        choose()
    if getattr(a, "run_p5"):
        run_p("p5", "test", a.shard, a.nshards)
    if getattr(a, "run_p15"):
        run_p("p15", "test", a.shard, a.nshards)
    if getattr(a, "run_r"):
        run_r("test", a.shard, a.nshards)
    if a.merge:
        merge(a.merge)


if __name__ == "__main__":
    main()
