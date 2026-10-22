"""
experiment/run_cycle18_pfsp_selector_entrada205.py
====================================================
Entrada 205 (Ciclo 18, frente D'): o SELETOR OFFLINE condicionado a instancia no FLOWSHOP DE PERMUTACAO (replica do ponto 2 de Carlos no segundo problema). Treino: 60 instancias sinteticas por classe (n em {20, 50, 100} x m em {5, 10, 20}; 540 no total; sementes 500000 + 1000 ci + rep, disjuntas das de ajuste 90000+ e de teste 0+),
cada acao fixa (F5 e F15; T = 1000), 3 sementes; atributos de pmsp/pfsp_features.py (razoes invariantes a escala, n, m); ExtraTreesRegressor (300 arvores, folha minima 10, max_features 0,5, semente 0) multi-saida sobre o desempenho relativo centrado por instancia de cada acao; politica = argmin da predicao.
Controle de dados: 'valbig' = a acao estatica por classe (n, m) escolhida nas MESMAS 540 instancias (no flowshop a classe e' (n, m), logo e' tambem a tabela de consulta por classe), contra a validada de 20 instancias por classe (Entrada 197).
Teste: as 360 instancias sinteticas e as 90 de Taillard da Entrada 197 (mesmas), 3 sementes: 'sel' e 'valbig'. Fases (retomaveis): --label ESPACO --shard i --nshards n | --fit | --run ESPACO --shard i --nshards n. Pre-registro: analysis/report_entrada205_pfsp_selector.py (docstring; hashes em results/entrada205_prereg_hashes.txt).
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

from experiment import run_cycle18_pfsp_entrada197 as F  # noqa: E402
from pmsp import pfsp_features as PF  # noqa: E402
from pmsp import pfsp_numba as N  # noqa: E402
from pmsp.pfsp import PfsInstance  # noqa: E402

R = ROOT / "results"
N_TRAIN, TRAIN_SEED0, N_SEED = 60, 500000, 3
TRAIN_FIELDS = ["space", "cls", "rep", "arm", "seed_i", "makespan", "lb"] + PF.NAMES
TEST_FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals", "arm"]


def _append(out, fields, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def train_items():
    return [(ci, n, m, rep) for ci, (n, m) in enumerate(F.CLASSES) for rep in range(N_TRAIN)]


def label(space, shard, nshards):
    F.warm()
    A = len(F.arms(space)[0])
    out = R / f"pfspsel_train_{space}_s{shard}of{nshards}_entrada205.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    new = not out.exists()
    t0 = time.perf_counter()
    for i, (ci, n, m, rep) in enumerate(train_items()[shard::nshards]):
        if (f"{n}x{m}", str(rep)) in done:
            continue
        inst = PfsInstance.random(n, m, TRAIN_SEED0 + 1000 * ci + rep)
        p, lb = np.ascontiguousarray(inst.p), inst.lower_bound
        fx = dict(zip(PF.NAMES, (float(v) for v in PF.features(p, lb))))
        rows = []
        for a in range(A):
            for s in range(N_SEED):
                mk, ev = F.run(space, p, lb, N.AGENT_FIXED, a, 0.1, 0.1, F.seed_of(space, "tune", ci, rep + 7000, f"sel_arm{a}", s))
                rows.append(dict(space=space, cls=f"{n}x{m}", rep=rep, arm=a, seed_i=s, makespan=mk, lb=lb, **fx))
        _append(out, TRAIN_FIELDS, rows, new)
        new = False
        if i % 20 == 0:
            print(f"[pfspsel label {space} s{shard}] {i} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[pfspsel label {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


def load_train(space):
    import pandas as pd
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pfspsel_train_{space}_s*of*_entrada205.csv")))], ignore_index=True)
    d["key"] = d.cls + "#" + d.rep.astype(str)
    m = d.groupby(["key", "arm"]).makespan.mean().unstack("arm")
    first = d.drop_duplicates("key").set_index("key").loc[m.index]
    X = first[PF.NAMES]
    Y = 100.0 * m.sub(m.mean(axis=1), axis=0).div(m.mean(axis=1), axis=0)
    return X, Y, first.cls


def make_model():
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=0, n_jobs=-1)


def fit():
    import joblib
    from sklearn.model_selection import KFold
    cv, big = {}, {}
    for sp in ("F5", "F15"):
        X, Y, cls = load_train(sp)
        Yn = Y.to_numpy()
        pred = np.zeros_like(Yn)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            pred[te] = make_model().fit(X.iloc[tr].to_numpy(), Yn[tr]).predict(X.iloc[te].to_numpy())
        sel = Yn[np.arange(len(Yn)), pred.argmin(axis=1)]
        cls_best = np.zeros(len(Yn))
        big[sp] = {}
        for c in sorted(set(cls)):
            idx = (cls == c).to_numpy()
            big[sp][c] = int(Yn[idx].mean(axis=0).argmin())
            cls_best[idx] = Yn[idx][:, Yn[idx].mean(axis=0).argmin()]
        cv[sp] = dict(n=int(len(Yn)), cv_selector=float(sel.mean()), cv_class_best_in_sample=float(cls_best.mean()), cv_global_best=float(Yn[:, Yn.mean(axis=0).argmin()].mean()), cv_oracle_biased=float(Yn.min(axis=1).mean()))
        joblib.dump(make_model().fit(X.to_numpy(), Yn), R / f"pfspsel_model_{sp}_entrada205.joblib")
        print(sp, json.dumps(cv[sp]))
    (R / "pfspsel_cv_entrada205.json").write_text(json.dumps(cv, indent=1))
    (R / "pfspsel_valbig_entrada205.json").write_text(json.dumps(big, indent=1))


def run_space(space, shard, nshards):
    import joblib
    mdl = joblib.load(R / f"pfspsel_model_{space}_entrada205.joblib")
    big = json.loads((R / "pfspsel_valbig_entrada205.json").read_text())[space]
    out = R / f"pfspsel_test_{space}_s{shard}of{nshards}_entrada205.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    F.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    items = F.test_items()[shard::nshards]
    for i, (kind, ci, n, m, rep, cls_id) in enumerate(items):
        if cls_id in done:
            continue
        if kind == "syn":
            p, lb = F.synth(n, m, ci, rep, False)
        else:
            p, lb, _, _ = F.taillard(rep)
        a_sel = int(mdl.predict(PF.features(p, lb).reshape(1, -1))[0].argmin())
        a_big = big[f"{n}x{m}"]
        kk = "ta" if kind == "ta" else "test"
        rows = []
        for meth, a in (("sel", a_sel), ("valbig", a_big)):
            for s in range(N_SEED):
                mk, ev = F.run(space, p, lb, N.AGENT_FIXED, a, 0.1, 0.1, F.seed_of(space, kk, ci, rep, meth, s))
                rows.append(dict(space=space, cls=cls_id, rep=rep, method=meth, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev, arm=a))
        _append(out, TEST_FIELDS, rows, new)
        new = False
        if i % 20 == 0:
            print(f"[pfspsel run {space} s{shard}] {i}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[pfspsel run {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.label:
        label(a.label, a.shard, a.nshards)
    if a.fit:
        fit()
    if a.run:
        run_space(a.run, a.shard, a.nshards)
