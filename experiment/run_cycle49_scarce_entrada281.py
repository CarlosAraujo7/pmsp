"""
experiment/run_cycle49_scarce_entrada281.py (Entrada 281, Ciclo 49)
ONDE A APRENDIZAGEM ONLINE PAGA: O REGIME DE POUCOS DADOS DE AJUSTE. Roteiro de 16/11/2026, item P2. Com k instancias de ajuste por classe (k = 1, 2, 3, 5 das 10 disponiveis; D = 4 sorteios independentes do subconjunto por k, os mesmos para todos os metodos),
tres competidores usam EXATAMENTE os mesmos k dados: (a) a acao estatica (a de menor razao media makespan / melhor conhecido; avaliada nas execucoes ja existentes das 30 acoes, Entrada 266), (b) o Q-Learning otimista aquecido com a priori desses k dados
(qo congelado de results/ur_choice_qo_S15_entrada234.json, Q0[s, a] = c * escore_a, escore = (max - r) / (max - min) em [0, 1], como na Entrada 271; variante w{k}_{d}) e (c) o controle com taxa de aprendizagem zero (Entrada 272; variante f{k}_{d}).
Benchmark de Fanjul-Peyro e Ruiz, espaco B30, 700 instancias de teste, T = 1.000, 3 sementes; sementes proprias. Fases: --prior (results/scarce_prior_entrada281.json: subconjuntos, escores e acao estatica por classe) | --run --shard i --nshards n
(results/scarce_test_s{i}of{n}_entrada281.csv; 32 variantes x 3 sementes por instancia). --pilot roda 2 instancias de AJUSTE (nenhum resultado de teste e visto antes do registro).
Uso: python -m experiment.run_cycle49_scarce_entrada281 --prior   |   --run --shard 0 --nshards 12
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
KS = [1, 2, 3, 5]
D = 4
A = 30
FIELDS = ["type", "instance", "m", "n", "idx", "method", "seed_i", "makespan", "bks", "evals"]
OFFSET = 81_000_000
KEYS = [(k, d) for k in KS for d in range(D)]
VARIANTS = [(kind, k, d) for (k, d) in KEYS for kind in ("w", "f")]


def tuning_blocks():
    """razao[cls] = array (10 instancias, A acoes, 2 sementes), reconstruido da ordem de escrita (instancia > acao > semente) de cada fatia (como em analysis/report_entrada277_cost.py)."""
    out = {}
    for sh in range(4):
        d = pd.read_csv(R / f"fr_tune_B30_s{sh}of4_entrada256.csv")
        nb = len(d) // (2 * A)
        assert nb * 2 * A == len(d)
        cls = d.cls.to_numpy(dtype=object).reshape(nb, 2 * A)
        assert (cls == cls[:, [0]]).all()
        ratio = d.ratio.to_numpy().reshape(nb, A, 2)
        for b in range(nb):
            out.setdefault(cls[b, 0], []).append(ratio[b])
    return {c: np.stack(v) for c, v in out.items()}


def score_of(r):
    assert r.max() > r.min()
    return (r.max() - r) / (r.max() - r.min())


def prior():
    T = tuning_blocks()
    classes = sorted(T)
    assert len(classes) == 28 and all(T[c].shape[0] == 10 for c in classes)
    out = {}
    for k, d in KEYS:
        rng = np.random.default_rng(281_000 + 100 * k + d)
        e = {}
        for c in classes:
            sel = sorted(int(x) for x in rng.permutation(10)[:k])
            r = T[c][sel].mean(axis=(0, 2))
            e[c] = dict(sel=sel, score=score_of(r).tolist(), static=int(r.argmin()))
        out[f"{k}|{d}"] = e
    (R / "scarce_prior_entrada281.json").write_text(json.dumps(out, indent=1))
    print("priors:", len(out), "chaves x", len(classes), "classes")


def run_one(P, a0, m, od, lb, n, seed, qo_cfg, sc, alpha):
    st, d, ls = F.arm_table_b(SPACE, n)
    c, a = qo_cfg
    hp = np.array([alpha if alpha is not None else a, 0.9, 1.0, 0.05, 0.0])
    Q0 = np.tile(c * np.asarray(sc, dtype=np.float64), (81, 1))
    out = F.UA5.ur_agent_core5(P, a0, m, st, d, ls, F.T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 2, Q0, 0.05)
    return int(out[0]), float(out[2])


def run(shard, nshards, pilot):
    man = F.manifest()
    man = man[man.role == ("tune" if pilot else "test")]
    pri = json.loads((R / "scarce_prior_entrada281.json").read_text())
    qo_cfg, _ = F.frozen(SPACE)
    out = R / (f"scarce_pilot_entrada281.csv" if pilot else f"scarce_test_s{shard}of{nshards}_entrada281.csv")
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
        for vi, (kind, k, d) in enumerate(VARIANTS):
            sc = pri[f"{k}|{d}"][cls]["score"]
            for s in range(F.N_SEED):
                seed = (F.seed_of(SPACE, int(r.ri), "qo", s, False) + OFFSET + vi * 1_000_000) % (2 ** 31 - 1)
                mk, ev = run_one(P, a0, m_, od, lb, n_, seed, qo_cfg, sc, 0.0 if kind == "f" else None)
                rows.append(dict(type=r["type"], instance=int(r.instance), m=m_, n=n_, idx=int(r.idx), method=f"{kind}{k}_{d}", seed_i=s, makespan=mk, bks=int(r.best_known), evals=ev))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if pilot and cnt >= 2:
            break
        if cnt % 10 == 0:
            print(f"[scarce s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[scarce s{shard}] concluido, {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.prior:
        prior()
    if a.run:
        run(a.shard, a.nshards, a.pilot)
