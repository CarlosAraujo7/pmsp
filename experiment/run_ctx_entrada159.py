"""
experiment/run_ctx_entrada159.py
==================================
Entrada 159 (Ciclo 12, frente C): aprendizes de OUTRA CLASSE no espaco de acoes 'tamanho de destruicao d/n in {0,01; 0,02; 0,05; 0,10; 0,20}' do LP-IG em R||Cmax (benchmark padrao de Fanjul-Peyro e Ruiz;
as mesmas 700 instancias de teste da Entrada 145; os bracos fixos e o sorteio ja' rodaram la'). Pergunta de revisor: 'o aprendiz era um espantalho (Q-Learning tabular, janela curta)'.
  C-offline (instancia-condicional): seletor treinado OFFLINE nas 280 instancias de AJUSTE (bracos fixos de results/lpig_tune_d_entrada141.csv) a partir de features da MATRIZ de tempos (sem o tipo do gerador):
        log n, log m, log(n/m), CV de todos os p_ij, heterogeneidade entre maquinas (media sobre tarefas do CV de p_ij entre maquinas), heterogeneidade entre tarefas, razao media min_i p_ij / media_i p_ij,
        razao (media das cargas minimas / maior tarefa minima). Tres modelos candidatos (Ridge com padronizacao e alpha por validacao cruzada; floresta aleatoria; kNN ponderado) predizem a razao makespan/melhor conhecido de
        cada braco; o braco escolhido e' o de menor razao prevista. O modelo e' escolhido por validacao cruzada de 5 dobras SO NO AJUSTE (menor razao media do braco escolhido); depois treinado em todo o ajuste e aplicado ao teste.
  C-online (aproximacao de funcao): LinUCB contextual (pmsp/linucb.py) com contexto [1, r1, r2, progresso, estagnacao]; alpha in {0,25; 0,5; 1,0} escolhido nas 280 instancias de ajuste (menor razao media); 700 execucoes de teste (linucb_d).
Fases: --features ; --train ; --tune-linucb ; --run. Saidas: results/ctx_features_entrada159.csv, ctx_choice_entrada159.json, linucb_tune_entrada159.csv, linucb_choice_entrada159.json, ctx_test_entrada159.csv. Retomavel.
Uso: python -m experiment.run_ctx_entrada159 --features --train ; --tune-linucb --workers 10 ; --run --workers 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiment.run_rcmax_benchmark_entrada137 as B  # noqa: E402

F = [0.01, 0.02, 0.05, 0.10, 0.20]
R = ROOT / "results"
FEATS = R / "ctx_features_entrada159.csv"
CHOICE = R / "ctx_choice_entrada159.json"
TUNE_L = R / "linucb_tune_entrada159.csv"
CHOICE_L = R / "linucb_choice_entrada159.json"
OUT = R / "ctx_test_entrada159.csv"
FEAT_COLS = ["log_n", "log_m", "log_nm", "cv_all", "het_mach", "het_job", "min_ratio", "load_ratio"]
ALPHAS = [0.25, 0.5, 1.0]
FIELDS = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "T", "alpha", "best_known", "lb"]


def instance_features(p):
    m, n = p.shape
    p = p.astype(float)
    mean_i = p.mean(axis=0)                       # por tarefa (media entre maquinas)
    het_mach = float(np.mean(p.std(axis=0) / mean_i))
    het_job = float(np.mean(p.std(axis=1) / p.mean(axis=1)))
    pmin = p.min(axis=0)
    return dict(log_n=float(np.log(n)), log_m=float(np.log(m)), log_nm=float(np.log(n / m)), cv_all=float(p.std() / p.mean()), het_mach=het_mach, het_job=het_job,
                min_ratio=float(np.mean(pmin / mean_i)), load_ratio=float((pmin.sum() / m) / pmin.max()))


def make_features():
    rows = []
    for r in B.manifest():
        inst = B.load_instance(r["type"], r["instance"])
        rows.append(dict(type=r["type"], instance=r["instance"], role=r["role"], idx=r["idx"], n=r["n"], m=r["m"], **instance_features(inst.p)))
    import pandas as pd
    pd.DataFrame(rows).to_csv(FEATS, index=False)
    print("features ->", FEATS, len(rows))


def train():
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ft = pd.read_csv(FEATS, dtype={"instance": str})
    tune = pd.read_csv(R / "lpig_tune_d_entrada141.csv", dtype={"instance": str})
    tune["frac"] = tune.method.str.split("@").str[1].astype(float)
    tune["ratio"] = tune.makespan / tune.best_known
    Y = tune.pivot(index=["type", "instance"], columns="frac", values="ratio")[F].reset_index()
    D = Y.merge(ft[ft.role == "tune"], on=["type", "instance"], how="left")
    assert len(D) == 280 and D[FEAT_COLS].notna().all().all()
    X, Yv = D[FEAT_COLS].values, D[F].values
    models = {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        "forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=1),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights="distance")),
    }
    cv = {k: [] for k in list(models) + ["cell_best", "global_best", "oracle"]}
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        for k, mk in models.items():
            mdl = mk().fit(X[tr], Yv[tr])
            cv[k] += list(Yv[te][np.arange(len(te)), mdl.predict(X[te]).argmin(axis=1)])
        cell_arm = {c: Yv[tr][(D.iloc[tr][["type", "n"]].astype(str).agg("|".join, axis=1) == c).values].mean(axis=0).argmin()
                    for c in D.iloc[tr][["type", "n"]].astype(str).agg("|".join, axis=1).unique()}
        cells_te = D.iloc[te][["type", "n"]].astype(str).agg("|".join, axis=1).values
        cv["cell_best"] += [Yv[te][i, cell_arm.get(c, Yv[tr].mean(axis=0).argmin())] for i, c in enumerate(cells_te)]
        cv["global_best"] += list(Yv[te][:, Yv[tr].mean(axis=0).argmin()])
        cv["oracle"] += list(Yv[te].min(axis=1))
    cvm = {k: float(np.mean(v)) for k, v in cv.items()}
    best = min(models, key=lambda k: (cvm[k], 0 if k == "ridge" else 1))
    mdl = models[best]().fit(X, Yv)
    te = ft[ft.role == "test"].reset_index(drop=True)
    pred = mdl.predict(te[FEAT_COLS].values).argmin(axis=1)
    out = dict(cv_mean_ratio=cvm, model=best, n_train=len(D), features=FEAT_COLS, pred={f"{r.type}|{r.instance}": int(a) for r, a in zip(te.itertuples(), pred)})
    CHOICE.write_text(json.dumps(out, indent=1))
    print("CV razao media do braco escolhido:", {k: round(v, 5) for k, v in cvm.items()})
    print("modelo escolhido:", best, "| bracos previstos no teste:", np.bincount(pred, minlength=5).tolist(), "->", CHOICE)


def _solve(inst, alpha, seed):
    from pmsp.linucb import LinUCBSelector
    from pmsp.unrelated import _descent, ur_ig_run
    from pmsp.unrelated_lp import lp_start
    t0 = time.perf_counter()
    start, _, _ = lp_start(inst)
    _descent(start, 10 ** 6)
    t_start = time.perf_counter() - t0
    sel = LinUCBSelector(n_actions=len(F), alpha=alpha, seed=seed)
    r = ur_ig_run(inst, sel, max_iter=1000, d=None, seed=seed, initial_solution=start, state_fn=None, ls_steps=B.LS_STEPS, action_map=lambda a, n: (0, max(2, round(F[a] * n))))
    return r["makespan"], r["cpu_time"] + t_start


def _task(t):
    inst = B.load_instance(t["type"], t["instance"])
    ms, cpu = _solve(inst, t["alpha"], int(t["instance"]))
    return dict(type=t["type"], instance=t["instance"], m=inst.m, n=inst.n, idx=t["idx"], method=t["method"], makespan=int(ms), cpu_time=cpu, T=1000, alpha=t["alpha"], best_known=t["best_known"], lb=inst.lower_bound)


def _drive(tasks, out, workers, every):
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"], r["method"]) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if (t["type"], t["instance"], t["method"]) not in done]
    pend.sort(key=lambda t: -(int(t["n"]) * int(t["m"])))
    print(f"[tasks] {len(pend)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_task, t): t for t in pend}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut]["type"], futs[fut]["instance"], futs[fut]["method"], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % every == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def tune_linucb(workers):
    import pandas as pd
    tasks = [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], alpha=a, method=f"linucb@{a}", best_known=r["best_known"]) for r in B.manifest("tune") for a in ALPHAS]
    _drive(tasks, TUNE_L, workers, 100)
    d = pd.read_csv(TUNE_L)
    d["ratio"] = d.makespan / d.best_known
    mean = d.groupby("alpha").ratio.mean()
    alpha = float(mean.idxmin())
    CHOICE_L.write_text(json.dumps(dict(alpha=alpha, mean_ratio={str(k): float(v) for k, v in mean.items()}, n_tune=int(d.drop_duplicates(["type", "instance"]).shape[0])), indent=1))
    print("alpha escolhido:", alpha, mean.to_dict())


def run(workers):
    alpha = json.loads(CHOICE_L.read_text())["alpha"]
    tasks = [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], alpha=alpha, method="linucb_d", best_known=r["best_known"]) for r in B.manifest("test")]
    _drive(tasks, OUT, workers, 50)
    print("concluido ->", OUT)


def main():
    ap = argparse.ArgumentParser()
    for k in ("features", "train", "tune-linucb", "run"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.features:
        make_features()
    if a.train:
        train()
    if getattr(a, "tune_linucb"):
        tune_linucb(a.workers)
    if a.run:
        run(a.workers)


if __name__ == "__main__":
    main()
