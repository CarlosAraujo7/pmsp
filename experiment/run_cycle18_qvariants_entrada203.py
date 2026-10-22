"""
experiment/run_cycle18_qvariants_entrada203.py
================================================
Entrada 203 (Ciclo 18, frente B'): as duas criticas da revisao cega ao aprendiz padrao, nas instancias PUBLICADAS (S5 e S15, T = 1000): o Q-Learning usava alfa = 0,1, gama = 0,9, epsilon 1,0 -> 0,05 por 0,995 e a recompensa max(melhora, 0). Aqui, com pmsp/ig_numba_agents4.py (hiperparametros ajustaveis; recompensa sensivel a platos):
  qt   Q-Learning com (alfa, gama, decaimento do epsilon) ajustados por FAMILIA nas 525 instancias de ajuste (grade 3 x 3 x 3 = 27: alfa {0,05; 0,1; 0,3}, gama {0,5; 0,9; 0,99}, decaimento {0,99; 0,995; 0,999}; piso 0,05), recompensa de sempre;
  qtp  Q-Learning com os tres hiperparametros E o bonus de plato ajustados por familia (grade 27 x 3 = 81: bonus {0,1; 0,3; 1,0}; rmode 1: r = max(melhora, 0) + bonus x max(maior melhora vista, 1) quando a candidata tem o mesmo makespan).
Criterio: menor lacuna media ao limite inferior em 2 sementes comuns (mesmas sementes para todas as configuracoes). Teste: as 2.550 instancias da Entrada 189 (discovery) e as 3.950 da 190 (held-out), 5 sementes por metodo; nada mais e' reajustado. Fases (retomaveis): --tune ESPACO --shard i --nshards n | --choose | --run ESPACO CONJUNTO --shard i --nshards n.
Saidas: results/qvar_tune_ESPACO_s*of*_entrada203.csv, qvar_choice_entrada203.json, qvar_ESPACO_CONJUNTO_s*of*_entrada203.csv. Pre-registro: analysis/report_entrada203_qvariants.py (docstring, congelado antes de rodar; hashes em results/entrada203_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from pmsp import ig_numba_agents4 as H4  # noqa: E402

R = ROOT / "results"
ALPHA, GAMMA, DECAY, BONUS = (0.05, 0.1, 0.3), (0.5, 0.9, 0.99), (0.99, 0.995, 0.999), (0.1, 0.3, 1.0)
GRID_T = [(a, g, d, 0.0, 0) for a in ALPHA for g in GAMMA for d in DECAY]
GRID_TP = [(a, g, d, b, 1) for a in ALPHA for g in GAMMA for d in DECAY for b in BONUS]
N_SEED_TUNE, N_SEED_TEST = 2, 5
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals", "cfg"]


def run4(space, rec, lb, a0, od, agent, fixed, lr, lr2, seed, cfg, T=1000):
    st, d, ls = arm_table(space, rec["n"])
    a, g, dc, b, rm = cfg
    hp = np.array([a, g, dc, 0.05, b], dtype=np.float64)
    best, _, ev, used, _ = H4.ig_agent_core4(rec["p"], a0, rec["m"], st, d, ls, T, agent, fixed, lr, lr2, seed, od, lb, 0, 0, 0.0, hp, rm)
    return int(best), float(ev)


def warm():
    rec = dict(p=np.arange(1, 21, dtype=np.int64), m=4, n=20, member="warm")
    lb, a0, od = P.prepare_inst(rec)
    for sp in ("S5", "S15"):
        run4(sp, rec, lb, a0, od, 2, 0, 0.1, 0.1, 1, (0.1, 0.9, 0.995, 0.0, 0), T=30)
        run4(sp, rec, lb, a0, od, 2, 0, 0.1, 0.1, 1, (0.1, 0.9, 0.995, 0.3, 1), T=30)


def _append(out, rows, new):
    S._append(out, FIELDS, rows, new)


def tune(space, shard, nshards):
    warm()
    info = json.loads(P.groups_path().read_text())
    cal = P.load_set(P.M_CAL)
    out = R / f"qvar_tune_{space}_s{shard}of{nshards}_entrada203.csv"
    rows, t0 = [], time.perf_counter()
    for i, rec in enumerate(cal):
        if i % nshards != shard:
            continue
        lb, a0, od = P.prepare_inst(rec)
        fam, g = info["calib"][rec["member"]]
        base = dict(space=space, cls=f"{fam}|{g}|{rec['member']}", rep=rec["inst_id"], lb=lb, T=1000)
        for ci, cfg in enumerate(GRID_T + GRID_TP):
            for s in range(N_SEED_TUNE):
                mk, ev = run4(space, rec, lb, a0, od, 2, 0, 0.1, 0.1, P.seed_of(space, "tune", fam, rec["inst_id"], "q", s), cfg)
                rows.append(dict(base, method="qt" if cfg[4] == 0 else "qtp", seed_i=s, makespan=mk, evals=ev, cfg=ci))
        if i % 50 == shard:
            print(f"[qvar tune {space} s{shard}] {i}/{len(cal)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    _append(out, rows, True)


def choose():
    import pandas as pd
    ch = {}
    cfgs = GRID_T + GRID_TP
    for sp in ("S5", "S15"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"qvar_tune_{sp}_s*of*_entrada203.csv")))], ignore_index=True)
        d["fam"] = d.cls.str.split("|").str[0]
        d["gap"] = 100 * (d.makespan - d.lb) / d.lb
        ch[sp] = {}
        for fam, x in d.groupby("fam"):
            ch[sp][fam] = {}
            for meth in ("qt", "qtp"):
                g = x[x.method == meth].groupby("cfg").gap.mean()
                best = int(g.idxmin())
                ch[sp][fam][meth] = dict(cfg=list(cfgs[best]), index=best, gap=float(g.min()), default_gap=float(x[(x.method == "qt") & (x.cfg == GRID_T.index((0.1, 0.9, 0.995, 0.0, 0)))].gap.mean()))
    (R / "qvar_choice_entrada203.json").write_text(json.dumps(ch, indent=1))
    for sp in ch:
        for fam in ch[sp]:
            print(sp, fam, {m: (v["cfg"], round(v["gap"], 4), "padrao", round(v["default_gap"], 4)) for m, v in ch[sp][fam].items()})


def run_space(space, conj, shard, nshards):
    ch = json.loads((R / "qvar_choice_entrada203.json").read_text())[space]
    recs, info = S.test_records(conj)
    out = R / f"qvar_{space}_{conj}_s{shard}of{nshards}_entrada203.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    warm()
    new = not out.exists()
    t0 = time.perf_counter()
    for i, rec in enumerate(recs[shard::nshards]):
        fam, g = info[rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        rows = []
        for meth in ("qt", "qtp"):
            cfg = tuple(ch[fam][meth]["cfg"])
            for s in range(N_SEED_TEST):
                mk, ev = run4(space, rec, lb, a0, od, 2, 0, 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], meth, s), cfg)
                rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=meth, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev, cfg=ch[fam][meth]["index"]))
        _append(out, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[qvar {space} {conj} s{shard}] {i} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[qvar {space} {conj} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--tune", default=None)
    ap.add_argument("--run", nargs=2, default=None, metavar=("ESPACO", "CONJUNTO"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        tune(a.tune, a.shard, a.nshards)
    if a.choose:
        choose()
    if a.run:
        run_space(a.run[0], a.run[1], a.shard, a.nshards)
