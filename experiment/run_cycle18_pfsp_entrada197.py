"""
experiment/run_cycle18_pfsp_entrada197.py
===========================================
Entrada 197 (Ciclo 18, frente D): o SEGUNDO PROBLEMA, nativo e compilado: o flowshop de permutacao (Fm|prmu|Cmax) com o IG de Ruiz e Stutzle (pmsp/pfsp_numba.py), os mesmos seletores de acao e o mesmo protocolo (controle aleatorio, CPU igual com K calibrado em sequencia,
acao validada por classe tambem a CPU igual com seu proprio K_val, folga do oraculo, pre-registro). Acoes: F5 = tamanho de destruicao d em {2, 3, 4, 6, 8} com temperatura 0,4; F15 = esses cinco tamanhos x fator de temperatura {0,1; 0,4; 0,8} (indice = temp * 5 + tamanho). T = 1000, 3 sementes por instancia.
Aprendizes: ql, ucb, pg (lr em {0,03; 0,1; 0,3}), alns (rho em {0,1; 0,3; 0,5}), ts (gama em {1; 0,99; 0,95}); hiperparametros e acao validada por classe ajustados em 20 instancias por classe (menor lacuna media ao limite inferior); o ajuste tambem registra, por instancia, 3 sementes do aleatorio e 3 sementes de cada acao
(insumo do pre-teste de folga/ruido). Classes: (n, m) com n em {20, 50, 100} e m em {5, 10, 20} (Taillard 1993, p ~ U{1..99}); teste: 40 instancias sinteticas por classe (PfsInstance.random) e as 90 instancias ta001-ta090 de Taillard (10 por classe), todas sem relacao com as de ajuste.
Fases (retomaveis): --tune ESPACO --shard i --nshards n (sobre os pares (classe, instancia)) | --choose | --calibrate | --run ESPACO --shard i --nshards n. Pre-registro: analysis/report_entrada197_pfsp.py (docstring, congelado antes de rodar; hashes em results/entrada197_prereg_hashes.txt).
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

from pmsp import pfsp_numba as N  # noqa: E402
from pmsp.pfsp import PfsInstance  # noqa: E402

R = ROOT / "results"
TAI = ROOT / "external_benchmarks" / "pfsp_taillard" / "instances"
CLASSES = [(n, m) for n in (20, 50, 100) for m in (5, 10, 20)]
D5 = [2, 3, 4, 6, 8]
TPS = [0.1, 0.4, 0.8]
N_TUNE, N_TEST, N_SEED, TUNE_SEED0, TEST_SEED0 = 20, 40, 3, 90000, 0
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]
LEARNERS = {"ql": N.AGENT_Q, "ucb": N.AGENT_UCB, "pg": N.AGENT_PG, "alns": N.AGENT_ALNS, "ts": N.AGENT_TS}
GRID = {"pg": [(r, 0.1) for r in (0.03, 0.1, 0.3)], "alns": [(r, 0.0) for r in (0.1, 0.3, 0.5)], "ts": [(g, 0.0) for g in (1.0, 0.99, 0.95)]}


def arms(space):
    if space == "F5":
        return np.array(D5, dtype=np.int64), np.full(5, 0.4)
    return np.array(D5 * 3, dtype=np.int64), np.array([tp for tp in TPS for _ in D5], dtype=np.float64)


def synth(n, m, ci, rep, tune):
    seed = (TUNE_SEED0 if tune else TEST_SEED0) + 1000 * ci + rep
    inst = PfsInstance.random(n, m, seed)
    return np.ascontiguousarray(inst.p), inst.lower_bound


def taillard(idx):
    lines = open(TAI / f"ta{idx:03d}.txt").read().split("\n")
    n, m = (int(x) for x in lines[0].split())
    p = np.array([[int(x) for x in lines[1 + i].split()] for i in range(m)], dtype=np.int64).T
    inst = PfsInstance(p, f"ta{idx:03d}")
    return np.ascontiguousarray(inst.p), inst.lower_bound, n, m


def run(space, p, lb, agent, fixed, lr, lr2, seed, T=1000):
    d, tp = arms(space)
    best, _, ev, used, _ = N.pfsp_ig_core(p, d, tp, T, agent, fixed, lr, lr2, seed, lb, 0, 0, 0.0)
    return int(best), float(ev)


def seed_of(space, kind, ci, rep, method, s, extra=0):
    return (rep * 131 + ci * 15485863 + s * 7919 + zlib.crc32(method.encode()) % 100003 * 104729 + ["F5", "F15"].index(space) * 32452843 + {"tune": 1, "test": 2, "ta": 3, "calib": 4}[kind] * 49979687 + extra) % (2 ** 31 - 1)


def warm():
    p, lb = synth(20, 5, 0, 0, True)
    for sp in ("F5", "F15"):
        for ag in (0, 1, 2, 3, 4, 7, 8):
            run(sp, p, lb, ag, 0, 0.5, 0.1, 1, T=20)


def write(out, rows, new):
    with open(out, "a" if not new else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def choice_path():
    return R / "pfsp_choice_entrada197.json"


def k_path():
    return R / "pfsp_cpu_ratio_entrada197.json"


# ----------------------------------------------------------------------------------------------------------------------- ajuste
def tune(space, shard, nshards):
    warm()
    A = len(arms(space)[0])
    out = R / f"pfsp_tune_{space}_s{shard}of{nshards}_entrada197.csv"
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"{n}x{m}"
        for rep in range(N_TUNE):
            if (ci * N_TUNE + rep) % nshards != shard:
                continue
            p, lb = synth(n, m, ci, rep, True)
            base = dict(space=space, cls=cls, rep=rep, lb=lb, T=1000)
            for s in range(N_SEED):
                mk, ev = run(space, p, lb, N.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, "tune", ci, rep, "rand", s))
                rows.append(dict(base, method="rand", seed_i=s, makespan=mk, evals=ev))
                for a in range(A):
                    mk, ev = run(space, p, lb, N.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, "tune", ci, rep, f"arm{a}", s))
                    rows.append(dict(base, method=f"arm{a}", seed_i=s, makespan=mk, evals=ev))
            for k, grid in GRID.items():
                for (x, y) in grid:
                    mk, ev = run(space, p, lb, LEARNERS[k], 0, x, y, seed_of(space, "tune", ci, rep, f"{k}@{x},{y}", 0))
                    rows.append(dict(base, method=f"{k}@{x},{y}", seed_i=0, makespan=mk, evals=ev))
        print(f"[pfsp tune {space}] classe {cls} ok", flush=True)
    write(out, rows, True)


def choose():
    import glob

    import pandas as pd
    ch = {}
    for sp in ("F5", "F15"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pfsp_tune_{sp}_s*of*_entrada197.csv")))], ignore_index=True)
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
    for sp in ("F5", "F15"):
        res[sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            cls = f"{n}x{m}"
            ratios = {k: [] for k in list(LEARNERS) + ["val"]}
            for rep in range(2):
                p, lb = synth(n, m, ci, rep, False)

                def cpu(agent, fixed=0, x=0.1, y=0.1):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        run(sp, p, lb, agent, fixed, x, y, reps + 7)
                        reps += 1
                    return (time.process_time() - t0) / reps
                base = cpu(N.AGENT_RANDOM)
                for k, ag in LEARNERS.items():
                    x, y = ch[sp][cls]["hp"].get(k, [0.1, 0.1])
                    ratios[k].append(cpu(ag, 0, x, y) / base)
                ratios["val"].append(cpu(N.AGENT_FIXED, ch[sp][cls]["val"]) / base)
            res[sp][cls] = {k: float(np.median(v)) for k, v in ratios.items()}
            print(f"[pfsp-calib {sp}] {cls}: " + ", ".join(f"{k} {res[sp][cls][k]:.2f}" for k in res[sp][cls]), flush=True)
    k_path().write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------------------------------- teste
def test_items():
    items = []
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(N_TEST):
            items.append(("syn", ci, n, m, rep, f"syn|{n}x{m}|{rep}"))
    for idx in range(1, 91):
        n, m = [(20, 5), (20, 10), (20, 20), (50, 5), (50, 10), (50, 20), (100, 5), (100, 10), (100, 20)][(idx - 1) // 10]
        items.append(("ta", CLASSES.index((n, m)), n, m, idx, f"ta|{n}x{m}|ta{idx:03d}"))
    return items


def run_space(space, shard, nshards):
    ch = json.loads(choice_path().read_text())[space]
    K = json.loads(k_path().read_text())[space]
    out = R / f"pfsp_test_{space}_s{shard}of{nshards}_entrada197.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    warm()
    A = len(arms(space)[0])
    t00 = time.perf_counter()
    new = not out.exists()
    items = test_items()[shard::nshards]
    for i, (kind, ci, n, m, rep, cls_id) in enumerate(items):
        if cls_id in done:
            continue
        if kind == "syn":
            p, lb = synth(n, m, ci, rep, False)
        else:
            p, lb, _, _ = taillard(rep)
        cls = f"{n}x{m}"
        c = ch[cls]
        rows = []

        def add(method, s, mk, ev, TT):
            rows.append(dict(space=space, cls=cls_id, rep=rep, method=method, seed_i=s, makespan=mk, lb=lb, T=TT, evals=ev))
        kk = "ta" if kind == "ta" else "test"
        for s in range(N_SEED):
            mk, ev = run(space, p, lb, N.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, kk, ci, rep, "rand", s))
            add("rand", s, mk, ev, 1000)
            for k, ag in LEARNERS.items():
                x, y = c["hp"].get(k, [0.1, 0.1])
                mk, ev = run(space, p, lb, ag, 0, x, y, seed_of(space, kk, ci, rep, k, s))
                add(k, s, mk, ev, 1000)
                if k in ("ql", "alns", "ts"):
                    Teq = max(50, int(round(1000 * K[cls][k])))
                    mk, ev = run(space, p, lb, N.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, kk, ci, rep, "rand_eq" + k, s), T=Teq)
                    add(f"rand_eq_{k}", s, mk, ev, Teq)
                Tv = max(50, int(round(1000 * K[cls][k] / K[cls]["val"])))
                mk, ev = run(space, p, lb, N.AGENT_FIXED, c["val"], 0.1, 0.1, seed_of(space, kk, ci, rep, "valeq" + k, s), T=Tv)
                add(f"val_eq_{k}", s, mk, ev, Tv)
            mk, ev = run(space, p, lb, N.AGENT_FIXED, c["val"], 0.1, 0.1, seed_of(space, kk, ci, rep, "val", s))
            add("val", s, mk, ev, 1000)
        for a in range(A):
            mk, ev = run(space, p, lb, N.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, kk, ci, rep, f"arm{a}", 0))
            add(f"arm{a}", 0, mk, ev, 1000)
        write(out, rows, new)
        new = False
        if i % 20 == 0:
            print(f"[pfsp {space} shard {shard}] {i}/{len(items)} ({time.perf_counter() - t00:.0f}s)", flush=True)
    print(f"[pfsp {space} shard {shard}] concluido ({time.perf_counter() - t00:.0f}s)", flush=True)


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
