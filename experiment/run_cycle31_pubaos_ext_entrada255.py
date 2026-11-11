"""
experiment/run_cycle31_pubaos_ext_entrada255.py
================================================
Entrada 255 (Ciclo 31, frente B): os DOIS DESENHOS PUBLICADOS DE AOS da Entrada 242 (o Q-Learning episodico de estado binario, 'iq', e o bandido do tipo FRRMAB, 'fr'; pmsp/ur_numba_agents6.py) nas FAMILIAS PUBLICADAS de R||Cmax (Entrada 254). Como la', os hiperparametros dos artigos nao estao disponiveis e sao AJUSTADOS nas instancias de ajuste (as 20 por classe da Entrada 254, 75 classes), por espaco e agente, na mesma grade pequena (iq: episodio E em {10, 50, 100} x decaimento beta em {0,8; 0,95}; fr: constante C em {0,5; 2; 5} x decaimento D em {0,5; 1,0}); uma configuracao por espaco e agente. Teste: 20 instancias por classe x 5 sementes.
Fases (retomaveis): --tune ESPACO | --run ESPACO. Saidas: results/pubaos_ext_choice_ESPACO_entrada255.json, results/pubaos_ext_test_ESPACO_entrada255.csv. Pre-registro: analysis/report_entrada255_ext_competitors.py (docstring; hashes em results/entrada255_prereg_hashes.txt).
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

from experiment import run_cycle30_ur_external_entrada254 as X  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from pmsp import ur_numba_agents6 as U6  # noqa: E402

R = ROOT / "results"
GRID = {"iq": [(E, b) for E in (10, 50, 100) for b in (0.8, 0.95)], "fr": [(c, dd) for c in (0.5, 2.0, 5.0) for dd in (0.5, 1.0)]}
AGENT = {"iq": 11, "fr": 12}
FIELDS = ["host", "space", "cls", "rep", "method", "seed_i", "makespan"]
SPACES = {"S5": 0, "S15": 1}


def run1(space, ci, rep, tune, name, cfg, seed):
    f, n, m = X.CLASSES[ci]
    P, a0, od, lb = X.make(ci, rep, tune)
    st, d, ls = arm_table(space, n)
    if name == "iq":
        E, b = cfg
        lr, lr2, hp = 0.1, float(E), np.array([0.1, 0.9, b, 0.05, 0.5])
    else:
        lr, lr2, hp = cfg[0], cfg[1], U6.HP_DEFAULT
    best, _, ev, used, _ = U6.ur_agent_core6(P, a0, m, st, d, ls, 1000, AGENT[name], 0, lr, lr2, seed, od, lb, 0, 0, 0.0, hp, 0)
    return int(best), lb


def seed_of(space, kind, ci, rep, s, name, k=0):
    return 1_500_000_000 + 20_000_000 * SPACES[space] + (0 if kind == "test" else 7_000_000) + 3_000_000 * (name == "fr") + 1000 * ci + rep + 50_000 * s + 7_919 * k


def warm():
    for name in AGENT:
        run1("S5", 0, 0, True, name, GRID[name][0], 1)


def tune(space):
    warm()
    out = {}
    t0 = time.perf_counter()
    tot = {name: np.zeros(len(GRID[name])) for name in AGENT}
    cnt = 0
    for ci in range(len(X.CLASSES)):
        for rep in range(X.N_TUNE):
            for name in AGENT:
                for k, cfg in enumerate(GRID[name]):
                    for s in range(2):
                        mk, lb = run1(space, ci, rep, True, name, cfg, seed_of(space, "tune", ci, rep, s, name, k))
                        tot[name][k] += 100.0 * (mk - lb) / lb
            cnt += 2
    for name in AGENT:
        mean = tot[name] / cnt
        k = int(np.argmin(mean))
        out[name] = dict(cfg=list(GRID[name][k]), index=k, gaps={str(c): float(v) for c, v in zip(GRID[name], mean)})
        print(space, name, "->", GRID[name][k], {str(c): round(float(v), 3) for c, v in zip(GRID[name], mean)}, f"({time.perf_counter() - t0:.0f}s)")
    (R / f"pubaos_ext_choice_{space}_entrada255.json").write_text(json.dumps(out, indent=1))


def run_test(space):
    ch = json.loads((R / f"pubaos_ext_choice_{space}_entrada255.json").read_text())
    out = R / f"pubaos_ext_test_{space}_entrada255.csv"
    warm()
    rows = []
    t0 = time.perf_counter()
    for ci, (f, n, m) in enumerate(X.CLASSES):
        cls = X.cls_name(f, n, m)
        for rep in range(X.N_TEST):
            for name in AGENT:
                cfg = tuple(ch[name]["cfg"])
                for s in range(5):
                    mk, lb = run1(space, ci, rep, False, name, cfg, seed_of(space, "test", ci, rep, s, name))
                    rows.append(dict(host="ext", space=space, cls=cls, rep=rep, method=name, seed_i=s, makespan=mk))
        if ci % 15 == 14:
            print(f"[pubaos ext {space}] {ci + 1}/{len(X.CLASSES)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default=None, metavar="SPACE")
    ap.add_argument("--run", default=None, metavar="SPACE")
    a = ap.parse_args()
    if a.tune:
        tune(a.tune)
    if a.run:
        run_test(a.run)
