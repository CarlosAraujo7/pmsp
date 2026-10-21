"""
experiment/run_reserve_arms_entrada168.py
===========================================
Entrada 168 (Ciclo 13, frente G): MAIS DADOS DE TREINO OFFLINE ajudam o seletor instancia-condicional? Nas Entradas 159 e 163 os seletores offline (kNN, MLP) foram treinados em 280 instancias de ajuste e perderam para a validacao por
celula (tipo x n); a critica seguinte e' 'foram poucos dados'. Aqui os 5 bracos fixos do LP-IG (d/n in {0,01; 0,02; 0,05; 0,10; 0,20}, T = 1000; o mesmo codigo da Entrada 145) rodam nas 420 instancias RESERVADAS (idx 7-9) do benchmark padrao,
que passam a ser dados de treino: ajuste (280) + reserva (420) = 700 instancias de treino, disjuntas das 700 de teste. Saida: results/reserve_arms_entrada168.csv (type, instance, m, n, idx, method fix_k, makespan, cpu_time, best_known).
Execucao em fatias independentes: --run --shard i --nshards N ; --merge. Retomavel.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiment.run_dpolicy_entrada145 as D  # noqa: E402
import experiment.run_rcmax_benchmark_entrada137 as B  # noqa: E402

OUT = ROOT / "results" / "reserve_arms_entrada168.csv"
FIELDS = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "best_known"]
ARMS = [f"fix_{k}" for k in range(5)]


def _task(r):
    from pmsp.unrelated import _descent
    from pmsp.unrelated_lp import lp_start
    inst = B.load_instance(r["type"], r["instance"])
    start, _, _ = lp_start(inst)
    _descent(start, 10 ** 6)
    rows = []
    for arm in ARMS:
        ms, cpu, _ = D._solve(inst, arm, int(r["instance"]), start=start)
        rows.append(dict(type=r["type"], instance=r["instance"], m=inst.m, n=inst.n, idx=r["idx"], method=arm, makespan=int(ms), cpu_time=cpu, best_known=r["best_known"]))
    return rows


def run(shard, nshards):
    part = OUT.with_name(OUT.stem + f"_s{shard}.csv")
    done = set()
    for src in (OUT, part):
        if src.exists() and src.stat().st_size > 0:
            done |= {(r["type"], r["instance"]) for r in csv.DictReader(open(src, newline=""))}
    pend = [r for r in B.manifest("reserve") if (r["type"], r["instance"]) not in done]
    pend.sort(key=lambda r: -(int(r["n"]) * int(r["m"])))
    pend = [r for k, r in enumerate(pend) if k % nshards == shard]
    print(f"[fatia {shard}/{nshards}] {len(pend)} instancias pendentes -> {part.name}", flush=True)
    new = not part.exists()
    t0 = time.perf_counter()
    with open(part, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, r in enumerate(pend, 1):
            for row in _task(r):
                w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            if i % 10 == 0 or i == len(pend):
                print(f"  [fatia {shard} {i}/{len(pend)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido fatia", shard)


def merge():
    parts = sorted(OUT.parent.glob(OUT.stem + "_s*.csv"))
    frames = [pd.read_csv(OUT, dtype={"instance": str})] if OUT.exists() and OUT.stat().st_size > 0 else []
    frames += [pd.read_csv(p, dtype={"instance": str}) for p in parts]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True).drop_duplicates(["type", "instance", "method"])
    d.to_csv(OUT, index=False)
    print(f"merge -> {OUT.name}: {len(d)} linhas de {len(parts)} fatias")


def train():
    """Treina os seletores com tune (280) + reserva (420) = 700 instancias (so' dados de treino; nenhum resultado de teste) e grava os bracos previstos nas 700 instancias de teste + o braco validado por (tipo, n)."""
    import numpy as np
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from experiment.run_ctx_entrada159 import FEAT_COLS, FEATS, F
    import json
    ft = pd.read_csv(FEATS, dtype={"instance": str})
    tu = pd.read_csv(ROOT / "results" / "lpig_tune_d_entrada141.csv", dtype={"instance": str})
    tu["frac"] = tu.method.str.split("@").str[1].astype(float)
    tu["ratio"] = tu.makespan / tu.best_known
    Yt = tu.pivot(index=["type", "instance"], columns="frac", values="ratio")[F].reset_index()
    rs = pd.read_csv(OUT, dtype={"instance": str})
    rs["ratio"] = rs.makespan / rs.best_known
    rs["frac"] = rs.method.str[-1].astype(int).map(lambda k: F[k])
    Yr = rs.pivot(index=["type", "instance"], columns="frac", values="ratio")[F].reset_index()
    Y = pd.concat([Yt, Yr], ignore_index=True)
    D = Y.merge(ft[ft.role.isin(["tune", "reserve"])], on=["type", "instance"], how="left")
    assert len(D) == 700 and D[FEAT_COLS].notna().all().all(), len(D)
    X, Yv = D[FEAT_COLS].values, D[F].values
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights="distance")).fit(X, Yv)
    mu, sd = Yv.mean(axis=0), Yv.std(axis=0) + 1e-12
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 16), alpha=1e-2, max_iter=3000, random_state=0)).fit(X, (Yv - mu) / sd)
    te = ft[ft.role == "test"].reset_index(drop=True)
    p_knn = knn.predict(te[FEAT_COLS].values).argmin(axis=1)
    p_mlp = (mlp.predict(te[FEAT_COLS].values) * sd + mu).argmin(axis=1)
    cell = {f"{t}|{int(n)}": int(g[F].mean().values.argmin()) for (t, n), g in D.groupby(["type", "n"])}
    out = dict(n_train=len(D), val_tn700=cell, knn={f"{r.type}|{r.instance}": int(a) for r, a in zip(te.itertuples(), p_knn)}, mlp={f"{r.type}|{r.instance}": int(a) for r, a in zip(te.itertuples(), p_mlp)})
    (ROOT / "results" / "bigtrain_choice_entrada168.json").write_text(json.dumps(out, indent=1))
    print("treino em 700 instancias; bracos previstos kNN:", np.bincount(p_knn, minlength=5).tolist(), "MLP:", np.bincount(p_mlp, minlength=5).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.run:
        run(a.shard, a.nshards)
    if a.merge:
        merge()
    if a.train:
        train()


if __name__ == "__main__":
    main()
