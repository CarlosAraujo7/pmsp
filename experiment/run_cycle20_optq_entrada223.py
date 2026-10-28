"""
experiment/run_cycle20_optq_entrada223.py
===========================================
Entrada 223 (Ciclo 20, frente D): a critica de desenho dos aprendizes da revisao cega #3 -- o epsilon cai a 0,05 em ~600 dos 1.000 passos, a tabela comeca em zero (sem otimismo) e a recompensa trunca em zero. O Q-LEARNING OTIMISTA (qo): tabela inicial otimista Q0 = c (todos os estados e acoes), recompensa invariante a escala
(pmsp/ig_numba_agents5.py, rmode 2: 100 melhora / LB), epsilon CONSTANTE de 0,05 (a exploracao vem do otimismo, nao do decaimento), gama 0,9. Hiperparametros (c em {0,02; 0,1; 0,5} x alfa em {0,1; 0,3}: 6 configuracoes) ajustados por FAMILIA e espaco nas 525 instancias de AJUSTE (2 sementes comuns; menor lacuna media ao limite inferior).
Teste: descoberta (2.550) e held-out (3.950), S5 e S15, T = 1000, 5 sementes. Fases (retomaveis): --tune ESPACO --shard i --nshards n | --choose | --run CONJUNTO ESPACO --shard i --nshards n. Saidas: results/optq_tune_ESPACO_s*of*_entrada223.csv, optq_choice_entrada223.json, optq_CONJUNTO_ESPACO_s*of*_entrada223.csv. Pre-registro: analysis/report_entrada223_optq.py (docstring; hashes em results/entrada223_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402
from experiment import run_cycle19_warmq_entrada211 as W  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402

R = ROOT / "results"
C0, ALPHA = (0.02, 0.1, 0.5), (0.1, 0.3)
GRID = [(c, a) for c in C0 for a in ALPHA]
FIELDS = W.FIELDS + ["cfg"]


def run_o(space, rec, lb, a0, od, seed, cfg, T=1000):
    c, a = cfg
    A = len(arm_table(space, rec["n"])[0])
    hp = np.array([a, 0.9, 1.0, 0.05, 0.0])
    mk, ev, _ = W.run5(space, rec, lb, a0, od, seed, hp, np.full((81, A), c), 0.05, T=T)
    return mk, ev


def _append(out, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def tune(space, shard, nshards):
    info = json.loads(P.groups_path().read_text())
    cal = P.load_set(P.M_CAL)
    P.warm()
    out = R / f"optq_tune_{space}_s{shard}of{nshards}_entrada223.csv"
    rows = []
    for i, rec in enumerate(cal):
        if i % nshards != shard:
            continue
        lb, a0, od = P.prepare_inst(rec)
        fam, g = info["calib"][rec["member"]]
        for ci, cfg in enumerate(GRID):
            for s in range(2):
                mk, ev = run_o(space, rec, lb, a0, od, P.seed_of(space, "tune", fam, rec["inst_id"], "qo", s), cfg)
                rows.append(dict(space=space, cls=f"{fam}|{g}|{rec['member']}", rep=rec["inst_id"], method="qo", seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev, cfg=ci))
    _append(out, rows, True)


def choose():
    import pandas as pd
    ch = {}
    for sp in ("S5", "S15"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"optq_tune_{sp}_s*of*_entrada223.csv")))], ignore_index=True)
        d["fam"] = d.cls.str.split("|").str[0]
        d["gap"] = 100 * (d.makespan - d.lb) / d.lb
        ch[sp] = {}
        for fam, x in d.groupby("fam"):
            g = x.groupby("cfg").gap.mean()
            best = int(g.idxmin())
            ch[sp][fam] = dict(cfg=list(GRID[best]), index=best, gap=float(g.min()), worst=float(g.max()))
    (R / "optq_choice_entrada223.json").write_text(json.dumps(ch, indent=1))
    for sp in ch:
        print(sp, {f: (v["cfg"], round(v["gap"], 4), "pior", round(v["worst"], 4)) for f, v in ch[sp].items()})


def run_space(conj, space, shard, nshards):
    ch = json.loads((R / "optq_choice_entrada223.json").read_text())[space]
    recs, info = S.test_records(conj)
    recs = recs[shard::nshards]
    out = R / f"optq_{conj}_{space}_s{shard}of{nshards}_entrada223.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        fam, g = info[rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        cfg = tuple(ch[fam]["cfg"])
        rows = []
        for s in range(S.N_SEED_TEST):
            mk, ev = run_o(space, rec, lb, a0, od, P.seed_of(space, "test", fam, rec["inst_id"], "qo", s), cfg)
            rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method="qo", seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev, cfg=ch[fam]["index"]))
        _append(out, rows, new)
        new = False
        if i % 300 == 0:
            print(f"[optq {conj} {space} s{shard}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[optq {conj} {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--tune", default=None)
    ap.add_argument("--run", nargs=2, default=None, metavar=("CONJUNTO", "ESPACO"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        tune(a.tune, a.shard, a.nshards)
    if a.choose:
        choose()
    if a.run:
        run_space(a.run[0], a.run[1], a.shard, a.nshards)
