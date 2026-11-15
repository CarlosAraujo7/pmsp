"""
experiment/run_cycle43_warmqo_entrada271.py (Entrada 271, Ciclo 43)
Pode o conhecimento offline RESGATAR o aprendiz online? Q-Learning otimista (qo; hiperparametros congelados de results/ur_choice_qo_S15_entrada234.json) com a tabela Q inicial AQUECIDA por uma priori offline, no benchmark de Fanjul-Peyro e Ruiz (B30;
700 instancias de teste; T = 1.000; 3 sementes). A priori de cada acao a e' o escore (max - r_a) / (max - min) em [0, 1] da razao media makespan / melhor conhecido nas instancias de AJUSTE (a melhor acao tem 1; a pior, 0); a tabela inicial e'
Q0[s, a] = c * escore_a em todos os 81 estados (c = o fator otimista congelado do qo). Duas variantes: 'qo_warm_loso' (a priori do tipo da instancia usando so' os dados de ajuste dos OUTROS tamanhos: o regime de transferencia a um tamanho nao visto, LOSO)
e 'qo_warm_full' (a priori da classe (tipo, n) inteira: o conhecimento da validada, aqui usado como ponto de partida e nao como acao fixa). Fases: --prior (results/warmqo_prior_entrada271.json) | --run --shard i --nshards n
(results/warmqo_test_s{i}of{n}_entrada271.csv). Sementes proprias.
Uso: python -m experiment.run_cycle43_warmqo_entrada271 --prior   |   --run --shard 0 --nshards 12
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
from experiment import run_cycle32_fanjul_entrada256 as F  # noqa: E402

R = F.R
SPACE = "B30"
NSIZES = [100, 200, 500, 1000]
FIELDS = ["type", "instance", "m", "n", "idx", "method", "seed_i", "makespan", "bks", "evals"]


def score(df):
    r = df.groupby("arm").ratio.mean().to_numpy()
    return ((r.max() - r) / (r.max() - r.min())).tolist()


def prior():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "fr_tune_B30_s*of*_entrada256.csv")))], ignore_index=True)
    d["type"] = d.cls.str.split("|").str[0]
    d["n"] = d.cls.str.split("|").str[1].astype(int)
    out = dict(full={c: score(g) for c, g in d.groupby("cls")}, loso={f"{t}|{n}": score(d[(d.type == t) & (d.n != n)]) for t in sorted(d.type.unique()) for n in NSIZES})
    (R / "warmqo_prior_entrada271.json").write_text(json.dumps(out, indent=1))
    print("priors:", len(out["full"]), len(out["loso"]))


def run_one(P, a0, m, od, lb, n, seed, qo_cfg, sc):
    st, d, ls = F.arm_table_b(SPACE, n)
    A = len(st)
    c, a = qo_cfg
    hp = np.array([a, 0.9, 1.0, 0.05, 0.0])
    Q0 = np.tile(c * np.asarray(sc, dtype=np.float64), (81, 1))
    out = F.UA5.ur_agent_core5(P, a0, m, st, d, ls, F.T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 2, Q0, 0.05)
    return int(out[0]), float(out[2])


def run(shard, nshards):
    man = F.manifest()
    man = man[man.role == "test"]
    pri = json.loads((R / "warmqo_prior_entrada271.json").read_text())
    qo_cfg, _ = F.frozen(SPACE)
    out = R / f"warmqo_test_s{shard}of{nshards}_entrada271.csv"
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"]) for r in csv.DictReader(open(out, newline=""))}
    F.warm()
    new = not out.exists()
    t0, cnt = time.perf_counter(), 0
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard or (r["type"], str(int(r.instance))) in done:
            continue
        P, a0, od, lb = F.load(r)
        m_, n_ = int(r.m), int(r.n)
        cls = f"{r['type']}|{n_}"
        rows = []
        for k, (meth, sc) in enumerate((("qo_warm_loso", pri["loso"][cls]), ("qo_warm_full", pri["full"][cls]))):
            for s in range(F.N_SEED):
                seed = (F.seed_of(SPACE, int(r.ri), "qo", s, False) + 71_000_000 + k * 1_000_000) % (2 ** 31 - 1)
                mk, ev = run_one(P, a0, m_, od, lb, n_, seed, qo_cfg, sc)
                rows.append(dict(type=r["type"], instance=int(r.instance), m=m_, n=n_, idx=int(r.idx), method=meth, seed_i=s, makespan=mk, bks=int(r.best_known), evals=ev))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if cnt % 20 == 0:
            print(f"[warmqo s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[warmqo s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


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
