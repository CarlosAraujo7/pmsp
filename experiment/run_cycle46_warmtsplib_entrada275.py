"""
experiment/run_cycle46_warmtsplib_entrada275.py (Entrada 275, Ciclo 46)
A replica da Entrada 274 em instancias PUBLICADAS: as 37 instancias da TSPLIB da Entrada 270 (EUC_2D, 200 <= n <= 2103; distancias arredondadas; otimos do arquivo `solutions`). Nenhuma instancia da TSPLIB e' usada para construir a priori: ela vem so'
das instancias SINTETICAS de ajuste da Entrada 267 (results/tsp_tune_*_entrada267.csv), como as acoes estaticas da Entrada 270:
  syn_n = a priori do tamanho sintetico mais proximo (n em {200, 500, 1000, 2000}; escala logaritmica; os dois tipos agrupados): o par da acao estatica Gn_syn;
  syn_0 = a priori global (todas as classes sinteticas agrupadas): o par da acao estatica G0_syn.
Cada priori com aprendizagem (warm_*) e sem (frozen_*: alfa = 0). O Q-Learning otimista usa os hiperparametros congelados (c = 0,02, alfa = 0,1), Q0[s, a] = c * escore_a (pmsp/tsp_warm.py). Espacos T15 e T5; T = 1.000; 3 sementes; sementes proprias.
Fases: --prior (results/tsplibwarm_prior_entrada275.json) | --run --shard i --nshards n (results/tsplibwarm_test_s{i}of{n}_entrada275.csv).
Uso: python -m experiment.run_cycle46_warmtsplib_entrada275 --prior   |   --run --shard 0 --nshards 12
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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle40_tsp_entrada267 as E  # noqa: E402
from experiment import run_cycle42_tsplib_entrada270 as L  # noqa: E402
from experiment import run_cycle45_warmtsp_entrada274 as S  # noqa: E402
from pmsp import tsp_numba as TS  # noqa: E402
from pmsp.instance_tsplib import load_tsplib, optima  # noqa: E402

R = E.R
VARIANTS = [("warm", "synn"), ("frozen", "synn"), ("warm", "syn0"), ("frozen", "syn0")]
FIELDS = ["space", "name", "n", "inst", "method", "seed_i", "length", "opt", "len0", "evals", "share_top"]


def prior():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "tsp_tune_s*of*_entrada267.csv")))], ignore_index=True)
    d["n"] = d.cls.str.split("|").str[1].astype(int)
    out = {}
    for sp, na in S.NARM.items():
        x = d[d.arm < na]
        out[sp] = dict(syn0=S.score_of(x), synn={str(n): S.score_of(g) for n, g in x.groupby("n")})
    (R / "tsplibwarm_prior_entrada275.json").write_text(json.dumps(out, indent=1))
    print("priors:", {sp: len(v["synn"]) for sp, v in out.items()})


def run(shard, nshards):
    qo = E.frozen_qo()
    pri = json.loads((R / "tsplibwarm_prior_entrada275.json").read_text())
    opt = optima()
    out = R / f"tsplibwarm_test_s{shard}of{nshards}_entrada275.csv"
    done = set()
    if out.exists():
        done = {r["name"] for r in csv.DictReader(open(out, newline=""))}
    S.warm_up(qo)
    new = not out.exists()
    t0 = time.perf_counter()
    for i, (nm, n) in enumerate(L.names()):
        if i % nshards != shard or nm in done:
            continue
        D, NL = load_tsplib(nm)
        tour0, _ = TS.construct(D, NL)
        len0 = float(TS.tour_len(tour0, D))
        inst = 5000 + i
        rows = []
        for sp in S.NARM:
            for k, (mode, pn) in enumerate(VARIANTS):
                sc = np.asarray(pri[sp]["syn0"] if pn == "syn0" else pri[sp]["synn"][str(L.nearest(n))], dtype=np.float64)
                for s in range(E.N_SEED):
                    seed = (E.seed_of(sp, inst, "qo", s) + 90_000_000 + 75_000_000 + k * 1_000_000) % (2 ** 31 - 1)
                    ln, ev, st = S.run_one(sp, mode == "warm", D, NL, tour0, seed, sc, qo)
                    rows.append(dict(space=sp, name=nm, n=n, inst=inst, method=f"{mode}_{pn}", seed_i=s, length=ln, opt=opt[nm], len0=len0, evals=ev, share_top=st))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        print(f"[tsplibwarm s{shard}] {nm} n={n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[tsplibwarm s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.prior:
        prior()
    if a.run:
        run(a.shard, a.nshards)
