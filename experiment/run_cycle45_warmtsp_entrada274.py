"""
experiment/run_cycle45_warmtsp_entrada274.py (Entrada 274, Ciclo 45)
A REPLICA, NO TSP, do achado da Entrada 271: o conhecimento offline como ponto de partida do aprendiz online. O ILS do TSP (Entrada 267: 8 classes = tipo uniforme/agrupado x n em {200, 500, 1000, 2000}; 12 instancias de AJUSTE e 50 de TESTE por classe;
T = 1.000; 3 sementes) com o Q-Learning otimista (hiperparametros CONGELADOS de results/ur_choice_qo_S15_entrada234.json: c = 0,02, alfa = 0,1) cuja tabela Q inicial e' Q0[s, a] = c * escore_a em todos os 81 estados (nucleo pmsp/tsp_warm.py;
escore_a = (max - r_a) / (max - min) da razao media comprimento final / inicial nas instancias de AJUSTE). Tres prioris, cada uma COM aprendizagem (warm_*) e SEM (frozen_*: alfa = 0; a tabela nao muda, a acao de maior Q0 com probabilidade 0,95):
  full = a priori da classe inteira (tipo, n): o conhecimento da validada usado como ponto de partida;
  loso = a priori do MESMO tipo usando so' os dados de ajuste dos OUTROS tamanhos (a transferencia a um tamanho nao visto; no TSP os rankings de acoes das classes sao quase iguais, correlacao de Spearman >= 0,92, logo e' uma priori quase exata);
  rev  = o ranking INVERTIDO da priori da classe (escore 1 - escore_a): a priori deliberadamente ERRADA, um teste de esforco da correcao pela aprendizagem.
Espacos T15 (15 acoes) e T5 (5 acoes; a priori so' sobre as acoes 0 a 4). Fases: --prior (results/tspwarm_prior_entrada274.json) | --run --shard i --nshards n (results/tspwarm_test_s{i}of{n}_entrada274.csv). Sementes proprias.
Uso: python -m experiment.run_cycle45_warmtsp_entrada274 --prior   |   --run --shard 0 --nshards 12
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
from pmsp import tsp_numba as TS  # noqa: E402
from pmsp import tsp_warm as W  # noqa: E402
from pmsp.instance_tsp import build  # noqa: E402

R = E.R
VARIANTS = [("warm", "full"), ("frozen", "full"), ("warm", "loso"), ("frozen", "loso"), ("warm", "rev"), ("frozen", "rev")]
METHODS_W = [f"{k}_{p}" for k, p in VARIANTS]
FIELDS = ["space", "kind", "n", "rep", "inst", "method", "seed_i", "length", "len0", "evals", "share_top"]
NARM = {"T15": 15, "T5": 5}


def score_of(g):
    return W.prior_scores(g.groupby("arm").ratio.mean().to_numpy()).tolist()


def prior():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "tsp_tune_s*of*_entrada267.csv")))], ignore_index=True)
    d["kind"] = d.cls.str.split("|").str[0]
    d["n"] = d.cls.str.split("|").str[1].astype(int)
    out = {}
    for sp, na in NARM.items():
        x = d[d.arm < na]
        out[sp] = dict(full={c: score_of(g) for c, g in x.groupby("cls")},
                       loso={f"{k}|{n}": score_of(x[(x.kind == k) & (x.n != n)]) for k in E.KINDS for n in E.NS})
    (R / "tspwarm_prior_entrada274.json").write_text(json.dumps(out, indent=1))
    print("priors:", {sp: (len(v["full"]), len(v["loso"])) for sp, v in out.items()})


def vector(pri, space, cls, prior_name):
    """Escores da priori 'full', 'loso' ou 'rev' (1 - full) da classe."""
    if prior_name == "rev":
        return 1.0 - np.asarray(pri[space]["full"][cls], dtype=np.float64)
    return np.asarray(pri[space][prior_name][cls], dtype=np.float64)


def run_one(space, learn, D, NL, tour0, seed, sc, qo_cfg):
    k0, s0 = TS.arm_table(space)
    c, a = qo_cfg
    hp = np.array([a if learn else 0.0, 0.9, 1.0, 0.05, 0.0])
    r = W.tsp_qo_prior_core(D, NL, tour0, k0, s0, E.T, int(seed), hp, float(c) * np.asarray(sc, dtype=np.float64))
    return float(r[0]), float(r[2]), float(r[3][int(np.argmax(sc))]) / E.T


def warm_up(qo):
    D, NL = build("unif", 40, 1)
    tour0, _ = TS.construct(D, NL)
    for sp in NARM:
        run_one(sp, True, D, NL, tour0, 1, np.ones(NARM[sp]), qo)


def run(shard, nshards):
    qo = E.frozen_qo()
    pri = json.loads((R / "tspwarm_prior_entrada274.json").read_text())
    out = R / f"tspwarm_test_s{shard}of{nshards}_entrada274.csv"
    done = set()
    if out.exists():
        done = {int(r["inst"]) for r in csv.DictReader(open(out, newline=""))}
    warm_up(qo)
    new = not out.exists()
    t0, cnt = time.perf_counter(), 0
    for it in E.instances("test"):
        if it["inst"] % nshards != shard or it["inst"] in done:
            continue
        D, NL = build(it["kind"], it["n"], it["seed"])
        tour0, _ = TS.construct(D, NL)
        len0 = float(TS.tour_len(tour0, D))
        cls = f"{it['kind']}|{it['n']}"
        rows = []
        for sp in NARM:
            for k, (mode, pn) in enumerate(VARIANTS):
                sc = vector(pri, sp, cls, pn)
                for s in range(E.N_SEED):
                    seed = (E.seed_of(sp, it["inst"], "qo", s) + 74_000_000 + k * 1_000_000) % (2 ** 31 - 1)
                    ln, ev, st = run_one(sp, mode == "warm", D, NL, tour0, seed, sc, qo)
                    rows.append(dict(space=sp, kind=it["kind"], n=it["n"], rep=it["rep"], inst=it["inst"], method=f"{mode}_{pn}", seed_i=s, length=ln, len0=len0, evals=ev, share_top=st))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if cnt % 5 == 0:
            print(f"[tspwarm s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[tspwarm s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


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
