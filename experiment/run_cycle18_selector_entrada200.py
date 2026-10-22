"""
experiment/run_cycle18_selector_entrada200.py
===============================================
Entrada 200 (Ciclo 18, frente B): um SELETOR OFFLINE condicionado a INSTANCIA (a resposta ao ponto 2 de Carlos: 'um aprendiz que explore a folga'). A folga do oraculo (a melhor acao fixa por instancia contra o aleatorio) e' grande so' na familia C publicada; a acao validada ESTATICA (uma por familia e tercil de n) captura 2/3 a 9/10 dela e
um seletor ONLINE (Q-Learning etc.) captura menos a CPU igual. Aqui a politica e' aprendida OFFLINE, em instancias DISJUNTAS de todas as de ajuste e de teste: as 6.000 instancias B, C, E de manifest.csv (2.000 por familia, menos as de soma >= 2^62), com os atributos de instancia de pmsp/pcmax_features.py
(razoes invariantes a escala, n, m; nenhuma informacao de familia). Por instancia de treino: cada acao fixa, 3 sementes (T = 1000, espacos S5 e S15, partida LPT). Modelo: ExtraTreesRegressor (300 arvores, folha minima 10, max_features 0,5, semente 0) multi-saida sobre o desempenho relativo CENTRADO por instancia de cada acao,
y_a = 100 (mk_a - media_a' mk_a') / media_a' mk_a'; politica = argmin da predicao. No teste (as 2.550 instancias da Entrada 189 e as 3.950 held-out da Entrada 190), o seletor roda 5 sementes ('sel'); as demais colunas (aleatorio, aprendizes, val, oraculo) vem dos arquivos existentes.
Fases (retomaveis): --label ESPACO --shard i --nshards n | --fit | --run ESPACO CONJUNTO --shard i --nshards n (CONJUNTO = discovery | heldout). Saidas: results/selector_train_ESPACO_s*of*_entrada200.csv, selector_model_ESPACO_entrada200.joblib, selector_cv_entrada200.json,
results/sel_ESPACO_CONJUNTO_s*of*_entrada200.csv. Pre-registro: analysis/report_entrada200_selector.py (docstring, congelado antes de rodar o teste; hashes em results/entrada200_prereg_hashes.txt).
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

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle17_replication_entrada190 as RH  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402
from pmsp import pcmax_features as F  # noqa: E402

R = ROOT / "results"
M_TRAIN = P.EXT / "manifest.csv"
N_SEED_TRAIN, N_SEED_TEST = 3, 5
TRAIN_FIELDS = ["space", "member", "family", "lb", "arm", "seed_i", "makespan"] + F.NAMES
TEST_FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals", "arm"]


def train_set():
    recs = P.load_set(M_TRAIN)
    return [r for r in recs if r["family"] in ("B", "C", "E")]


def seed_train(space, rec, a, s):
    return (zlib.crc32(rec["member"].encode()) % 1000003 * 131 + s * 7919 + a * 104729 + ["S5", "S15"].index(space) * 32452843 + 5 * 49979687) % (2 ** 31 - 1)


def _append(out, fields, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def label(space, shard, nshards):
    P.warm()
    A = len(arm_table(space, 100)[0])
    out = R / f"selector_train_{space}_s{shard}of{nshards}_entrada200.csv"
    done = set()
    if out.exists():
        done = {r["member"] for r in csv.DictReader(open(out, newline=""))}
    recs = train_set()[shard::nshards]
    new = not out.exists()
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        if rec["member"] in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        fx = dict(zip(F.NAMES, (float(v) for v in F.features(rec["p"], rec["m"], lb))))
        rows = []
        for a in range(A):
            for s in range(N_SEED_TRAIN):
                mk, ev, _ = P.run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, seed_train(space, rec, a, s))
                rows.append(dict(space=space, member=rec["member"], family=rec["family"], lb=lb, arm=a, seed_i=s, makespan=mk, **fx))
        _append(out, TRAIN_FIELDS, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[sel label {space} s{shard}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[sel label {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


def load_train(space):
    import glob

    import pandas as pd
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"selector_train_{space}_s*of*_entrada200.csv")))], ignore_index=True)
    m = d.groupby(["member", "arm"]).makespan.mean().unstack("arm")
    X = d.drop_duplicates("member").set_index("member").loc[m.index, F.NAMES]
    Y = 100.0 * m.sub(m.mean(axis=1), axis=0).div(m.mean(axis=1), axis=0)
    fam = d.drop_duplicates("member").set_index("member").loc[m.index, "family"]
    return X, Y, fam


def make_model():
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=0, n_jobs=-1)


def fit():
    import joblib
    from sklearn.model_selection import KFold
    cv = {}
    for sp in ("S5", "S15"):
        X, Y, fam = load_train(sp)
        pred = np.zeros_like(Y.to_numpy())
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            mdl = make_model().fit(X.iloc[tr].to_numpy(), Y.iloc[tr].to_numpy())
            pred[te] = mdl.predict(X.iloc[te].to_numpy())
        Yn = Y.to_numpy()
        pick = pred.argmin(axis=1)
        sel = Yn[np.arange(len(Yn)), pick]
        glob_best = Yn[:, Yn.mean(axis=0).argmin()]
        fam_best = np.zeros(len(Yn))
        for f in ("B", "C", "E"):
            msk = (fam == f).to_numpy()
            fam_best[msk] = Yn[msk][:, Yn[msk].mean(axis=0).argmin()]
        cv[sp] = dict(n=int(len(Yn)), cv_selector=float(sel.mean()), cv_global_best_fixed=float(glob_best.mean()), cv_family_best_fixed=float(fam_best.mean()), cv_oracle_biased=float(Yn.min(axis=1).mean()), cv_mean_arm=float(Yn.mean()),
                      by_family={f: dict(selector=float(sel[(fam == f).to_numpy()].mean()), family_best=float(fam_best[(fam == f).to_numpy()].mean()), oracle=float(Yn[(fam == f).to_numpy()].min(axis=1).mean())) for f in ("B", "C", "E")},
                      pick_hist=np.bincount(pick, minlength=Yn.shape[1]).tolist())
        mdl = make_model().fit(X.to_numpy(), Y.to_numpy())
        joblib.dump(mdl, R / f"selector_model_{sp}_entrada200.joblib")
        cv[sp]["importance"] = dict(zip(F.NAMES, (float(v) for v in mdl.feature_importances_)))
        print(sp, json.dumps({k: v for k, v in cv[sp].items() if k not in ("importance",)}, indent=1))
    (R / "selector_cv_entrada200.json").write_text(json.dumps(cv, indent=1))


def test_records(conj):
    if conj == "discovery":
        info = json.loads(P.groups_path().read_text())["test"]
        return P.load_set(P.M_TEST), info
    info = json.loads(RH.groups_path().read_text())["test"]
    return P.load_set(RH.M_HOLD), info


def run_space(space, conj, shard, nshards):
    import joblib
    mdl = joblib.load(R / f"selector_model_{space}_entrada200.joblib")
    recs, info = test_records(conj)
    recs = recs[shard::nshards]
    out = R / f"sel_{space}_{conj}_s{shard}of{nshards}_entrada200.csv"
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
        x = F.features(rec["p"], rec["m"], lb).reshape(1, -1)
        a = int(mdl.predict(x)[0].argmin())
        rows = []
        for s in range(N_SEED_TEST):
            mk, ev, _ = P.run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], "sel", s))
            rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method="sel", seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev, arm=a))
        _append(out, TEST_FIELDS, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[sel run {space} {conj} s{shard}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[sel run {space} {conj} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--run", nargs=2, default=None, metavar=("ESPACO", "CONJUNTO"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.label:
        label(a.label, a.shard, a.nshards)
    if a.fit:
        fit()
    if a.run:
        run_space(a.run[0], a.run[1], a.shard, a.nshards)
