"""
experiment/run_cycle34_longT_entrada258.py (Entrada 258, Ciclo 34, frente 1)
O protocolo no benchmark padrao de Fanjul-Peyro e Ruiz com um HORIZONTE LONGO (T = 10.000 iteracoes em vez de 1.000), para perguntar se as conclusoes valem quando o IG chega mais perto dos melhores valores conhecidos
(a lacuna da validada a T = 1.000 era de 4,9% em B30). Mesmo desenho da Entrada 256 (mesmas 280 instancias de AJUSTE e 700 de TESTE, mesmos aprendizes com hiperparametros congelados), com a ACAO VALIDADA ESCOLHIDA DE NOVO
no horizonte longo (o competidor estatico recebe o mesmo horizonte do teste). Espaco B30 (o mais rico; B15 e B5 nao tem busca local). Fases (retomaveis):
  --tune --shard i --nshards n   (results/frL_tune_B30_T{T}_s{i}of{n}_entrada258.csv)
  --choose                       (results/frL_choice_B30_T{T}_entrada258.json)
  --run --shard i --nshards n    (results/frL_test_B30_T{T}_s{i}of{n}_entrada258.csv)
Sementes: as da Entrada 256 deslocadas de 10^9 (independentes das do T = 1.000).
"""
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
SEED_SHIFT = 1_000_000_000


def sd(ri, method, s, tune):
    return F.seed_of(SPACE, ri, method, s, tune) + SEED_SHIFT


def tune(T, shard, nshards):
    F.T = T
    man = F.manifest()
    man = man[man.role == "tune"]
    F.warm()
    A = len(F.arm_table_b(SPACE, 100)[0])
    out = R / f"frL_tune_{SPACE}_T{T}_s{shard}of{nshards}_entrada258.csv"
    rows, t0 = [], time.perf_counter()
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard:
            continue
        P, a0, od, lb = F.load(r)
        for a in range(A):
            for s in range(2):
                mk, _ = F.run_fixed_or_agent(SPACE, f"arm{a}", P, a0, int(r.m), od, lb, int(r.n), sd(int(r.ri), f"arm{a}", s, True))
                rows.append(dict(cls=f"{r['type']}|{int(r.n)}", arm=a, ratio=mk / float(r.best_known)))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[frL tune T{T} s{shard}] {len(rows)} linhas ({time.perf_counter() - t0:.0f}s)", flush=True)


def choose(T):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"frL_tune_{SPACE}_T{T}_s*of*_entrada258.csv")))], ignore_index=True)
    val = {c: int(g.groupby("arm").ratio.mean().idxmin()) for c, g in d.groupby("cls")}
    (R / f"frL_choice_{SPACE}_T{T}_entrada258.json").write_text(json.dumps(dict(val=val), indent=1))
    print(f"T{T}: {len(val)} classes validadas")


def run(T, shard, nshards):
    F.T = T
    man = F.manifest()
    man = man[man.role == "test"]
    val = json.loads((R / f"frL_choice_{SPACE}_T{T}_entrada258.json").read_text())["val"]
    qo, pub = F.frozen(SPACE)
    out = R / f"frL_test_{SPACE}_T{T}_s{shard}of{nshards}_entrada258.csv"
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"]) for r in csv.DictReader(open(out, newline=""))}
    F.warm()
    new = not out.exists()
    t0, cnt = time.perf_counter(), 0
    for _, r in man.iterrows():
        if (int(r.ri) % nshards != shard) or ((r["type"], str(int(r.instance))) in done):
            continue
        P, a0, od, lb = F.load(r)
        m_, n_ = int(r.m), int(r.n)
        rows = []
        for me in F.METHODS:
            for s in range(F.N_SEED):
                mk, ev = F.run_fixed_or_agent(SPACE, me, P, a0, m_, od, lb, n_, sd(int(r.ri), me, s, False), val[f"{r['type']}|{n_}"], qo, pub)
                rows.append(dict(space=SPACE, type=r["type"], instance=int(r.instance), m=m_, n=n_, idx=int(r.idx), method=me, seed_i=s, makespan=mk, lb=lb, bks=int(r.best_known), T=T, evals=ev))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=F.FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if cnt % 10 == 0:
            print(f"[frL T{T} s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[frL T{T} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        tune(a.T, a.shard, a.nshards)
    if a.choose:
        choose(a.T)
    if a.run:
        run(a.T, a.shard, a.nshards)
