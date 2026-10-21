"""
experiment/run_neural_entrada162.py
=====================================
Entrada 162 (Ciclo 13, frente E): aprendizes NEURAIS no espaco 'tamanho de destruicao d/n in {0,01; 0,02; 0,05; 0,10; 0,20}' do LP-IG em R||Cmax (as 700 instancias de teste da Entrada 145; bracos fixos e sorteio ja' rodaram la').
Resposta a 'o aprendiz nao e' neural' (a Entrada 159 ja' testou LinUCB e um seletor offline kNN):
  nq_d       : Q-learning com rede neural pequena ONLINE (pmsp/neuralq.py; contexto [1, r1, r2, progresso, estagnacao]; TD com gamma = 0,9; epsilon 1,0 -> 0,05 por 0,995); a taxa de aprendizado lr in {0,005; 0,02; 0,05}
               e' escolhida nas 280 instancias de AJUSTE (menor razao makespan/melhor conhecido); hidden = 16 fixo.
  val_mlp    : seletor OFFLINE: MLPRegressor (sklearn; padronizacao; 2 camadas de 16; alpha 1e-2; max_iter 3000; semente 0) treinado nas 280 instancias de ajuste para predizer a razao makespan/melhor conhecido de cada braco
               a partir das mesmas 8 features da matriz de tempos da Entrada 159 (sem o tipo); o braco previsto de menor razao e' aplicado a cada instancia de teste.
Fases: --train-mlp (usa so' dados de ajuste e features; grava results/neural_choice_entrada162.json) ; --tune --shard i --nshards N (840 execucoes de ajuste, em N processos independentes) ; --merge-tune ; --choose-lr ;
--run --shard i --nshards N (700 execucoes de teste) ; --merge-test. Retomavel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiment.run_rcmax_benchmark_entrada137 as B  # noqa: E402
from experiment.run_ctx_entrada159 import F, FEAT_COLS, FEATS  # noqa: E402

R = ROOT / "results"
CHOICE_MLP = R / "neural_choice_entrada162.json"
TUNE_N = R / "nq_tune_entrada162.csv"
CHOICE_N = R / "nq_choice_entrada162.json"
OUT = R / "nq_test_entrada162.csv"
LRS = [0.005, 0.02, 0.05]
FIELDS = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "T", "lr", "best_known", "lb"]


def train_mlp():
    import pandas as pd
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ft = pd.read_csv(FEATS, dtype={"instance": str})
    tune = pd.read_csv(R / "lpig_tune_d_entrada141.csv", dtype={"instance": str})
    tune["frac"] = tune.method.str.split("@").str[1].astype(float)
    tune["ratio"] = tune.makespan / tune.best_known
    Y = tune.pivot(index=["type", "instance"], columns="frac", values="ratio")[F].reset_index()
    D = Y.merge(ft[ft.role == "tune"], on=["type", "instance"], how="left")
    assert len(D) == 280 and D[FEAT_COLS].notna().all().all()
    from sklearn.model_selection import KFold
    X, Yv = D[FEAT_COLS].values, D[F].values
    mu, sd = Yv.mean(axis=0), Yv.std(axis=0) + 1e-12            # alvos padronizados por braco (as razoes ficam em torno de 1,01 e o MLP bruto prediz quase uma constante)

    def fit(Xa, Ya):
        m_ = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 16), alpha=1e-2, max_iter=3000, random_state=0)).fit(Xa, (Ya - mu) / sd)
        return lambda Xb: m_.predict(Xb) * sd + mu

    mdl = fit(X, Yv)
    cv = []
    for tr, te_ in KFold(5, shuffle=True, random_state=0).split(X):
        pr = fit(X[tr], Yv[tr])
        cv += list(Yv[te_][np.arange(len(te_)), pr(X[te_]).argmin(axis=1)])
    te = ft[ft.role == "test"].reset_index(drop=True)
    pred = mdl(te[FEAT_COLS].values).argmin(axis=1)
    in_sample = float(np.mean(Yv[np.arange(len(D)), mdl(X).argmin(axis=1)]))
    cv_mean = float(np.mean(cv))
    CHOICE_MLP.write_text(json.dumps(dict(n_train=len(D), in_sample_mean_ratio=in_sample, cv_mean_ratio=cv_mean, pred={f"{r.type}|{r.instance}": int(a) for r, a in zip(te.itertuples(), pred)}), indent=1))
    print("MLP offline: razao media no ajuste: dentro da amostra %.5f; validacao cruzada 5 dobras %.5f; bracos previstos no teste:" % (in_sample, cv_mean), np.bincount(pred, minlength=5).tolist(), "->", CHOICE_MLP)


def _solve(inst, lr, seed):
    from pmsp.neuralq import NeuralQSelector
    from pmsp.unrelated import _descent, ur_ig_run
    from pmsp.unrelated_lp import lp_start
    t0 = time.perf_counter()
    start, _, _ = lp_start(inst)
    _descent(start, 10 ** 6)
    t_start = time.perf_counter() - t0
    sel = NeuralQSelector(n_actions=len(F), hidden=16, lr=lr, seed=seed)
    r = ur_ig_run(inst, sel, max_iter=1000, d=None, seed=seed, initial_solution=start, state_fn=None, ls_steps=B.LS_STEPS, action_map=lambda a, n: (0, max(2, round(F[a] * n))))
    return r["makespan"], r["cpu_time"] + t_start


def _task(t):
    inst = B.load_instance(t["type"], t["instance"])
    ms, cpu = _solve(inst, t["lr"], int(t["instance"]))
    return dict(type=t["type"], instance=t["instance"], m=inst.m, n=inst.n, idx=t["idx"], method=t["method"], makespan=int(ms), cpu_time=cpu, T=1000, lr=t["lr"], best_known=t["best_known"], lb=inst.lower_bound)


def _tasks(kind):
    if kind == "tune":
        return [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], lr=lr, method=f"nq@{lr}", best_known=r["best_known"]) for r in B.manifest("tune") for lr in LRS]
    lr = json.loads(CHOICE_N.read_text())["lr"]
    return [dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], lr=lr, method="nq_d", best_known=r["best_known"]) for r in B.manifest("test")]


def run_shard(kind, shard, nshards):
    """Fatias independentes (sem ProcessPoolExecutor, que caiu com WinError 6 nesta maquina compartilhada); --merge junta os CSV das fatias."""
    out = TUNE_N if kind == "tune" else OUT
    part = out.with_name(out.stem + f"_s{shard}.csv")
    done = set()
    for src in (out, part):
        if src.exists() and src.stat().st_size > 0:
            done |= {(r["type"], r["instance"], r["method"]) for r in csv.DictReader(open(src, newline=""))}
    pend = [t for t in _tasks(kind) if (t["type"], t["instance"], t["method"]) not in done]
    pend.sort(key=lambda t: -(int(t["n"]) * int(t["m"])))
    pend = [t for k, t in enumerate(pend) if k % nshards == shard]
    print(f"[fatia {shard}/{nshards} {kind}] {len(pend)} pendentes -> {part.name}", flush=True)
    new = not part.exists()
    t0 = time.perf_counter()
    with open(part, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, t in enumerate(pend, 1):
            w.writerow(_task(t))
            f.flush()
            os.fsync(f.fileno())
            if i % 20 == 0 or i == len(pend):
                print(f"  [fatia {shard} {kind} {i}/{len(pend)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido fatia", shard, kind)


def merge(kind):
    import pandas as pd
    out = TUNE_N if kind == "tune" else OUT
    parts = sorted(out.parent.glob(out.stem + "_s*.csv"))
    frames = [pd.read_csv(out, dtype={"instance": str})] if out.exists() and out.stat().st_size > 0 else []
    frames += [pd.read_csv(q, dtype={"instance": str}) for q in parts]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True).drop_duplicates(["type", "instance", "method"])
    d.to_csv(out, index=False)
    print(f"merge {kind} -> {out.name}: {len(d)} linhas de {len(parts)} fatias")


def choose_lr():
    import pandas as pd
    d = pd.read_csv(TUNE_N)
    d["ratio"] = d.makespan / d.best_known
    mean = d.groupby("lr").ratio.mean()
    lr = float(mean.idxmin())
    CHOICE_N.write_text(json.dumps(dict(lr=lr, mean_ratio={str(k): float(v) for k, v in mean.items()}), indent=1))
    print("lr escolhido:", lr, mean.to_dict())


def main():
    ap = argparse.ArgumentParser()
    for k in ("train-mlp", "tune", "choose-lr", "run", "merge-tune", "merge-test"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if getattr(a, "train_mlp"):
        train_mlp()
    if a.tune:
        run_shard("tune", a.shard, a.nshards)
    if getattr(a, "merge_tune"):
        merge("tune")
    if getattr(a, "choose_lr"):
        choose_lr()
    if a.run:
        run_shard("test", a.shard, a.nshards)
    if getattr(a, "merge_test"):
        merge("test")


if __name__ == "__main__":
    main()
