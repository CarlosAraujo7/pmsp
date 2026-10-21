"""
experiment/run_online_replic_entrada167.py
============================================
Entrada 167 (Ciclo 13, frente F): a limitacao 'os aprendizes de outras classes foram testados em UM ambiente' (Entradas 159 e 163). Aqui o bandit contextual linear (LinUCB, pmsp/linucb.py) e o Q-learning neural
online (pmsp/neuralq.py) escolhem o TAMANHO DE DESTRUICAO nos outros dois problemas, com as mesmas instancias, sementes, acoes e baselines ja' rodados:
  ambiente P (P||Cmax, Entrada 114): acoes d/n in {0,05; 0,10; 0,20; 0,30; 0,40}, destruicao aleatoria, partida LPT, T = 1000; teste = 12 classes x 100 reps (seeds BASE_SEED + rep); ajuste = seeds 90000 + rep, rep 0-29 (por classe).
  ambiente F (flowshop, Entradas 152/159): acoes d in {2, 3, 4, 6, 8}, IG de Ruiz e Stutzle com busca local, T = 300; teste = 120 instancias geradas (4 tamanhos x 30; seed da execucao 32000 + 1000 si + rep) + 40 instancias classicas de Taillard
  (seed 5000 + numero); ajuste = 10 instancias geradas por tamanho (seed 31000 + 1000 si + rep).
Hiperparametros ajustados SO nas instancias de ajuste de cada ambiente (menor media do makespan relativo ao limite inferior): LinUCB alpha in {0,25; 0,5; 1,0}; Q neural lr in {0,005; 0,02; 0,05} (hidden = 16).
Fases (por ambiente, em N processos independentes): --tune --env E --shard i --nshards N ; --merge-tune --env E ; --choose --env E ; --run --env E --shard i --nshards N ; --merge-test --env E.
Saidas: results/online_tune_{E}_entrada167.csv, online_choice_{E}_entrada167.json, online_test_{E}_entrada167.csv. Retomavel.
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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_benchmarks" / "pfsp_taillard"))

R = ROOT / "results"
FRACS = [0.05, 0.10, 0.20, 0.30, 0.40]
CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
GRID = {"linucb": [0.25, 0.5, 1.0], "nq": [0.005, 0.02, 0.05]}
FIELDS = ["env", "src", "n", "m", "rep", "learner", "hp", "makespan", "lb", "T", "cpu_time"]


def _selector(learner, hp, seed, na):
    from pmsp.linucb import LinUCBSelector
    from pmsp.neuralq import NeuralQSelector
    if learner == "linucb":
        return LinUCBSelector(n_actions=na, alpha=hp, seed=seed)
    return NeuralQSelector(n_actions=na, hidden=16, lr=hp, seed=seed)


def tasks(env, kind):
    out = []
    if env == "P":
        for n, m in CLASSES:
            for rep in range(30 if kind == "tune" else 100):
                out.append(dict(env="P", src="gen", n=n, m=m, rep=rep, si=-1))
    else:
        import experiment.run_pfsp_entrada152 as PF
        for si, (n, m) in enumerate(PF.SIZES):
            for rep in range(10 if kind == "tune" else 30):
                out.append(dict(env="F", src="gen", n=n, m=m, rep=rep, si=si))
            if kind == "test":
                for rep in range(10):
                    out.append(dict(env="F", src="tai", n=n, m=m, rep=rep, si=si))
    return out


def solve(t, learner, hp, kind):
    if t["env"] == "P":
        from experiment import config as CFG
        from experiment.run_dsel_entrada114 import TUNE_SEED0, action_map
        from pmsp.ig import ig_run
        from pmsp.instance import PMSPInstance
        iseed = (TUNE_SEED0 + t["rep"]) if kind == "tune" else (CFG.BASE_SEED + t["rep"])
        rseed = CFG.BASE_SEED + t["rep"]
        inst = PMSPInstance.random(n=t["n"], m=t["m"], seed=iseed, name=f"rand_{t['n']}j_{t['m']}m")
        r = ig_run(instance=inst, selector=_selector(learner, hp, rseed, 5), max_iter=1000, seed=rseed, stagnation_window=CFG.STAGNATION_WINDOW, action_map=action_map)
        return int(r["makespan"]), int(inst.lower_bound), 1000, r["cpu_time"]
    import experiment.run_pfsp_entrada152 as PF
    import experiment.run_pfsp_taillard_entrada159 as TA
    from pmsp.pfsp import pfsp_ig_run
    if t["src"] == "tai":
        inst, num = TA.taillard_instance(t["si"], t["rep"])
        seed = 5000 + num
    elif kind == "tune":
        inst = PF.inst_of(t["si"], t["rep"], base=31000)
        seed = 31000 + 1000 * t["si"] + t["rep"]
    else:
        inst = PF.inst_of(t["si"], t["rep"], base=32000)
        seed = 32000 + 1000 * t["si"] + t["rep"]
    r = pfsp_ig_run(inst, _selector(learner, hp, seed, len(PF.D_CHOICES)), PF.D_CHOICES, max_iter=PF.T_IG, seed=seed, state_fn=None, use_ls=True)
    return int(r["makespan"]), int(inst.lower_bound), PF.T_IG, r["cpu_time"]


def _paths(env, kind):
    return R / f"online_{kind}_{env}_entrada167.csv"


def _key(r):
    return (r["env"], r["src"], str(r["n"]), str(r["m"]), str(r["rep"]), r["learner"], str(r["hp"]))


def run_shard(env, kind, shard, nshards):
    out = _paths(env, kind)
    part = out.with_name(out.stem + f"_s{shard}.csv")
    chosen = json.loads((R / f"online_choice_{env}_entrada167.json").read_text()) if kind == "test" else None
    jobs = []
    for t in tasks(env, kind):
        for learner in GRID:
            for hp in ([chosen[learner]["hp"]] if kind == "test" else GRID[learner]):
                jobs.append((t, learner, hp))
    done = set()
    for src in (out, part):
        if src.exists() and src.stat().st_size > 0:
            done |= {_key(r) for r in csv.DictReader(open(src, newline=""))}
    pend = [(t, l, h) for t, l, h in jobs if (t["env"], t["src"], str(t["n"]), str(t["m"]), str(t["rep"]), l, str(h)) not in done]
    pend.sort(key=lambda j: -(j[0]["n"] * j[0]["m"]))
    pend = [j for k, j in enumerate(pend) if k % nshards == shard]
    print(f"[fatia {shard}/{nshards} {env} {kind}] {len(pend)} pendentes -> {part.name}", flush=True)
    new = not part.exists()
    t0 = time.perf_counter()
    with open(part, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, (t, learner, hp) in enumerate(pend, 1):
            ms, lb, T, cpu = solve(t, learner, hp, kind)
            w.writerow(dict(env=t["env"], src=t["src"], n=t["n"], m=t["m"], rep=t["rep"], learner=learner, hp=hp, makespan=ms, lb=lb, T=T, cpu_time=cpu))
            f.flush()
            os.fsync(f.fileno())
            if i % 50 == 0 or i == len(pend):
                print(f"  [fatia {shard} {env} {kind} {i}/{len(pend)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido fatia", shard, env, kind)


def merge(env, kind):
    out = _paths(env, kind)
    parts = sorted(out.parent.glob(out.stem + "_s*.csv"))
    frames = [pd.read_csv(out)] if out.exists() and out.stat().st_size > 0 else []
    frames += [pd.read_csv(p) for p in parts]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True).drop_duplicates(["env", "src", "n", "m", "rep", "learner", "hp"])
    d.to_csv(out, index=False)
    print(f"merge {env} {kind} -> {out.name}: {len(d)} linhas de {len(parts)} fatias")


def choose(env):
    d = pd.read_csv(_paths(env, "tune"))
    d["ratio"] = d.makespan / d.lb
    res = {}
    for learner, g in d.groupby("learner"):
        mean = g.groupby("hp").ratio.mean()
        res[learner] = dict(hp=float(mean.idxmin()), mean_ratio={str(k): float(v) for k, v in mean.items()})
    (R / f"online_choice_{env}_entrada167.json").write_text(json.dumps(res, indent=1))
    print(env, {k: v["hp"] for k, v in res.items()}, {k: v["mean_ratio"] for k, v in res.items()})


def main():
    ap = argparse.ArgumentParser()
    for k in ("tune", "run", "merge-tune", "merge-test", "choose"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--env", choices=["P", "F"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        run_shard(a.env, "tune", a.shard, a.nshards)
    if getattr(a, "merge_tune"):
        merge(a.env, "tune")
    if a.choose:
        choose(a.env)
    if a.run:
        run_shard(a.env, "test", a.shard, a.nshards)
    if getattr(a, "merge_test"):
        merge(a.env, "test")


if __name__ == "__main__":
    main()
