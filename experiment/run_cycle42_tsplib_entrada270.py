"""
experiment/run_cycle42_tsplib_entrada270.py (Entrada 270, Ciclo 42)
O ILS do TSP (Entrada 267) em instancias PUBLICADAS da TSPLIB (37 instancias EUC_2D com 200 <= n <= 2103; external_benchmarks/tsplib; distancias arredondadas; valores otimos do arquivo `solutions` da TSPLIB). Nenhuma instancia da TSPLIB e' usada
para escolher acoes: as acoes ESTATICAS sao escolhidas nas instancias SINTETICAS de ajuste da Entrada 267 (results/tsp_tune_*_entrada267.csv): Gn_syn = a acao de menor razao media para o tamanho sintetico mais proximo (n em {200, 500, 1000, 2000};
proximidade em escala logaritmica; os dois tipos de instancia agrupados) e G0_syn = uma acao global; os aprendizes mantem os hiperparametros congelados. Metodos no teste: rand, ql, ql_perm, ucb, pg, alns, ts, qo (T5 e T15) e as 15 acoes fixas (T15),
T = 1.000, 3 sementes. Fases: --choose (results/tsplib_choice_entrada270.json) | --run --shard i --nshards n (results/tsplib_test_s{i}of{n}_entrada270.csv).
Uso: python -m experiment.run_cycle42_tsplib_entrada270 --choose   |   --run --shard 0 --nshards 12
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle40_tsp_entrada267 as E  # noqa: E402
from pmsp import tsp_numba as TS  # noqa: E402
from pmsp.instance_tsplib import TSPLIB, load_tsplib, optima  # noqa: E402

R = ROOT / "results"
FIELDS = ["space", "name", "n", "inst", "method", "seed_i", "length", "opt", "len0", "evals", "share0"]
SIZES = [200, 500, 1000, 2000]


def names():
    man = json.loads((TSPLIB / "manifest_tsplib.json").read_text())["instances"]
    return [(m["name"], int(m["n"])) for m in sorted(man, key=lambda m: m["n"])]


def nearest(n):
    return min(SIZES, key=lambda s: abs(math.log(n) - math.log(s)))


def choose():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "tsp_tune_s*of*_entrada267.csv")))], ignore_index=True)
    d["n"] = d.cls.str.split("|").str[1].astype(int)
    out = {}
    for sp, na in (("T15", 15), ("T5", 5)):
        x = d[d.arm < na]
        out[sp] = dict(G0=int(x.groupby("arm").ratio.mean().idxmin()), Gn={str(n): int(g.groupby("arm").ratio.mean().idxmin()) for n, g in x.groupby("n")})
    (R / "tsplib_choice_entrada270.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def run(shard, nshards):
    E.warm()
    qo = E.frozen_qo()
    opt = optima()
    out = R / f"tsplib_test_s{shard}of{nshards}_entrada270.csv"
    done = set()
    if out.exists():
        done = {r["name"] for r in csv.DictReader(open(out, newline=""))}
    new = not out.exists()
    t0, cnt = time.perf_counter(), 0
    for i, (nm, n) in enumerate(names()):
        if i % nshards != shard or nm in done:
            continue
        D, NL = load_tsplib(nm)
        tour0, _ = TS.construct(D, NL)
        len0 = float(TS.tour_len(tour0, D))
        inst = 5000 + i
        rows = []
        for sp in E.SPACES:
            for me in E.METHODS:
                for s in range(E.N_SEED):
                    ln, ev, sh0 = E.run_one(sp, me, D, NL, tour0, E.seed_of(sp, inst, me, s) + 90_000_000, qo)
                    rows.append(dict(space=sp, name=nm, n=n, inst=inst, method=me, seed_i=s, length=ln, opt=opt[nm], len0=len0, evals=ev, share0=sh0))
        for a in range(15):
            for s in range(E.N_SEED):
                ln, ev, _ = E.run_one("T15", f"arm{a}", D, NL, tour0, E.seed_of("T15", inst, f"arm{a}", s) + 90_000_000, None)
                rows.append(dict(space="T15", name=nm, n=n, inst=inst, method=f"arm{a}", seed_i=s, length=ln, opt=opt[nm], len0=len0, evals=ev, share0=0.0))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        print(f"[tsplib s{shard}] {nm} n={n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[tsplib s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.choose:
        choose()
    if a.run:
        run(a.shard, a.nshards)
