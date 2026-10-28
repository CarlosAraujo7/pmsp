"""
experiment/run_cycle19_ga_entrada219.py
=========================================
Entrada 219 (Ciclo 19, frente F): o TERCEIRO HOSPEDEIRO, POPULACIONAL: um algoritmo genetico de estado estacionario para P||Cmax (pmsp/ga_numba.py; cruzamento x mutacao x intensidade escolhidos por um seletor) com o MESMO protocolo (controle aleatorio, CPU igual com K calibrado em sequencia, acao validada por classe, folga do oraculo, pre-registro).
Instancias: sinteticas com tempos U[1, 10^4] (o regime de folga maior do artigo), 12 classes (n, m) como no estudo; ajuste 30 por classe (sementes 90000+) e teste 100 por classe (sementes BASE_SEED+rep, as mesmas instancias do estudo do P||Cmax com tempos finos); T = 10.000 iteracoes (avaliacoes de filho); 5 sementes por instancia no teste (3 no ajuste).
Espacos: G9 = 3 cruzamentos (uniforme, por maquinas, clone) x 3 mutacoes (realoca, gulosa, troca) com intensidade baixa; G18 = esses x 2 intensidades (indice = intensidade * 9 + 3 * cruzamento + mutacao). Aprendizes: ql, ucb, pg, alns, ts (hiperparametros e acao validada por classe ajustados nas 30 instancias de ajuste).
Fases (retomaveis): --tune ESPACO --shard i --nshards n | --choose | --calibrate | --run ESPACO --shard i --nshards n. Saidas: results/ga_tune_ESPACO_s*of*_entrada219.csv, ga_choice_entrada219.json, ga_cpu_ratio_entrada219.json, ga_test_ESPACO_s*of*_entrada219.csv. Pre-registro: analysis/report_entrada219_ga.py (docstring; hashes em results/entrada219_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment.run_agents_numba_entrada180 import CLASSES  # noqa: E402
from experiment.run_cycle16_entrada185 import make  # noqa: E402
from pmsp import ga_numba as G  # noqa: E402

R = ROOT / "results"
N_TUNE, N_TEST, N_SEED_TUNE, N_SEED_TEST, T0 = 30, 100, 3, 5, 10000
HIGH = 10000
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]
LEARNERS = {"ql": G.AGENT_Q, "ucb": G.AGENT_UCB, "pg": G.AGENT_PG, "alns": G.AGENT_ALNS, "ts": G.AGENT_TS}
GRID = {"pg": [(r, 0.1) for r in (0.03, 0.1, 0.3)], "alns": [(r, 0.0) for r in (0.1, 0.3, 0.5)], "ts": [(g, 0.0) for g in (1.0, 0.99, 0.95)]}


def arms(space):
    xo, mu = np.repeat([0, 1, 2], 3), np.tile([0, 1, 2], 3)
    if space == "G9":
        return xo.astype(np.int64), mu.astype(np.int64), np.zeros(9, dtype=np.int64)
    return np.concatenate([xo, xo]).astype(np.int64), np.concatenate([mu, mu]).astype(np.int64), np.concatenate([np.zeros(9), np.ones(9)]).astype(np.int64)


def inst(ci, n, m, rep, tune):
    i, p, a0, od = make(n, m, rep, tune, high=HIGH)
    return p, a0, od, i.lower_bound


def run(space, p, a0, m, od, lb, agent, fixed, lr, lr2, seed, T=T0):
    xo, mu, ls = arms(space)
    best, _, ev, used, _, _ = G.ga_agent_core(p, a0, m, xo, mu, ls, T, agent, fixed, lr, lr2, seed, od, lb, 0, 0, 0.0, G.HP_DEFAULT, 0)
    return int(best), float(ev)


def seed_of(space, kind, ci, rep, method, s, extra=0):
    return (rep * 131 + ci * 15485863 + s * 7919 + zlib.crc32(method.encode()) % 100003 * 104729 + ["G9", "G18"].index(space) * 32452843 + {"tune": 1, "test": 2, "calib": 4}[kind] * 49979687 + extra) % (2 ** 31 - 1)


def warm():
    p, a0, od, lb = inst(0, 20, 4, 0, True)
    for sp in ("G9", "G18"):
        for ag in (0, 1, 2, 3, 4, 7, 8):
            run(sp, p, a0, 4, od, lb, ag, 0, 0.5, 0.1, 1, T=30)


def _append(out, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def choice_path():
    return R / "ga_choice_entrada219.json"


def k_path():
    return R / "ga_cpu_ratio_entrada219.json"


def tune(space, shard, nshards):
    warm()
    A = len(arms(space)[0])
    out = R / f"ga_tune_{space}_s{shard}of{nshards}_entrada219.csv"
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"{n}x{m}"
        for rep in range(N_TUNE):
            if (ci * N_TUNE + rep) % nshards != shard:
                continue
            p, a0, od, lb = inst(ci, n, m, rep, True)
            base = dict(space=space, cls=cls, rep=rep, lb=lb, T=T0)
            for s in range(N_SEED_TUNE):
                mk, ev = run(space, p, a0, m, od, lb, G.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, "tune", ci, rep, "rand", s))
                rows.append(dict(base, method="rand", seed_i=s, makespan=mk, evals=ev))
                for a in range(A):
                    mk, ev = run(space, p, a0, m, od, lb, G.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, "tune", ci, rep, f"arm{a}", s))
                    rows.append(dict(base, method=f"arm{a}", seed_i=s, makespan=mk, evals=ev))
            for k, grid in GRID.items():
                for (x, y) in grid:
                    mk, ev = run(space, p, a0, m, od, lb, LEARNERS[k], 0, x, y, seed_of(space, "tune", ci, rep, f"{k}@{x},{y}", 0))
                    rows.append(dict(base, method=f"{k}@{x},{y}", seed_i=0, makespan=mk, evals=ev))
        print(f"[ga tune {space} s{shard}] classe {cls} ok", flush=True)
    _append(out, rows, True)


def choose():
    import glob

    import pandas as pd
    ch = {}
    for sp in ("G9", "G18"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ga_tune_{sp}_s*of*_entrada219.csv")))], ignore_index=True)
        d["gap"] = 100 * (d.makespan - d.lb) / d.lb
        ch[sp] = {}
        for cls, x in d.groupby("cls"):
            arm = x[x.method.str.startswith("arm")].groupby("method").gap.mean()
            hp = {}
            for k in GRID:
                g = x[x.method.str.startswith(k + "@")].groupby("method").gap.mean()
                hp[k] = [float(v) for v in g.idxmin().split("@")[1].split(",")]
            ch[sp][cls] = dict(val=int(arm.idxmin()[3:]), hp=hp)
    choice_path().write_text(json.dumps(ch, indent=1))
    print("escolhas:", {sp: {c: v["val"] for c, v in ch[sp].items()} for sp in ch})


def calibrate():
    ch = json.loads(choice_path().read_text())
    warm()
    res = {}
    for sp in ("G9", "G18"):
        res[sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            cls = f"{n}x{m}"
            ratios = {k: [] for k in list(LEARNERS) + ["val"]}
            for rep in range(2):
                p, a0, od, lb = inst(ci, n, m, rep, False)

                def cpu(agent, fixed=0, x=0.1, y=0.1):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        run(sp, p, a0, m, od, lb, agent, fixed, x, y, reps + 7)
                        reps += 1
                    return (time.process_time() - t0) / reps
                base = cpu(G.AGENT_RANDOM)
                for k, ag in LEARNERS.items():
                    x, y = ch[sp][cls]["hp"].get(k, [0.1, 0.1])
                    ratios[k].append(cpu(ag, 0, x, y) / base)
                ratios["val"].append(cpu(G.AGENT_FIXED, ch[sp][cls]["val"]) / base)
            res[sp][cls] = {k: float(np.median(v)) for k, v in ratios.items()}
            print(f"[ga-calib {sp}] {cls}: " + ", ".join(f"{k} {res[sp][cls][k]:.2f}" for k in res[sp][cls]), flush=True)
    k_path().write_text(json.dumps(res, indent=1))


def run_space(space, shard, nshards):
    ch = json.loads(choice_path().read_text())[space]
    K = json.loads(k_path().read_text())[space]
    out = R / f"ga_test_{space}_s{shard}of{nshards}_entrada219.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    A = len(arms(space)[0])
    t00 = time.perf_counter()
    new = not out.exists()
    items = [(ci, n, m, rep) for ci, (n, m) in enumerate(CLASSES) for rep in range(N_TEST)][shard::nshards]
    for i, (ci, n, m, rep) in enumerate(items):
        cls = f"{n}x{m}"
        if (cls, str(rep)) in done:
            continue
        p, a0, od, lb = inst(ci, n, m, rep, False)
        c = ch[cls]
        rows = []

        def add(method, s, mk, ev, TT):
            rows.append(dict(space=space, cls=cls, rep=rep, method=method, seed_i=s, makespan=mk, lb=lb, T=TT, evals=ev))
        for s in range(N_SEED_TEST):
            mk, ev = run(space, p, a0, m, od, lb, G.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, "test", ci, rep, "rand", s))
            add("rand", s, mk, ev, T0)
            for k, ag in LEARNERS.items():
                x, y = c["hp"].get(k, [0.1, 0.1])
                mk, ev = run(space, p, a0, m, od, lb, ag, 0, x, y, seed_of(space, "test", ci, rep, k, s))
                add(k, s, mk, ev, T0)
                if k in ("ql", "alns", "ts"):
                    Teq = max(200, int(round(T0 * K[cls][k])))
                    mk, ev = run(space, p, a0, m, od, lb, G.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, "test", ci, rep, "rand_eq" + k, s), T=Teq)
                    add(f"rand_eq_{k}", s, mk, ev, Teq)
                Tv = max(200, int(round(T0 * K[cls][k] / K[cls]["val"])))
                mk, ev = run(space, p, a0, m, od, lb, G.AGENT_FIXED, c["val"], 0.1, 0.1, seed_of(space, "test", ci, rep, "valeq" + k, s), T=Tv)
                add(f"val_eq_{k}", s, mk, ev, Tv)
            mk, ev = run(space, p, a0, m, od, lb, G.AGENT_FIXED, c["val"], 0.1, 0.1, seed_of(space, "test", ci, rep, "val", s))
            add("val", s, mk, ev, T0)
        for a in range(A):
            for s in range(4):                                                  # 4 sementes por acao: o oraculo com validacao cruzada (sementes {0, 1} escolhem, {2, 3} avaliam)
                mk, ev = run(space, p, a0, m, od, lb, G.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, "test", ci, rep, f"arm{a}", s))
                add(f"arm{a}", s, mk, ev, T0)
        _append(out, rows, new)
        new = False
        if i % 50 == 0:
            print(f"[ga {space} shard {shard}] {i}/{len(items)} ({time.perf_counter() - t00:.0f}s)", flush=True)
    print(f"[ga {space} shard {shard}] concluido ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("choose", "calibrate"):
        ap.add_argument("--" + f, action="store_true")
    for f in ("tune", "run"):
        ap.add_argument("--" + f, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        tune(a.tune, a.shard, a.nshards)
    if a.choose:
        choose()
    if a.calibrate:
        calibrate()
    if a.run:
        run_space(a.run, a.shard, a.nshards)
