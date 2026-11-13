"""
experiment/run_cycle41_eqcpu_fanjul_entrada268.py (Entrada 268, Ciclo 41)
O protocolo a CPU IGUAL no benchmark de Fanjul-Peyro e Ruiz (B30; os mesmos 700 de teste e 280 de ajuste da Entrada 256): a Entrada 256 comparou a iteracoes iguais; aqui cada aprendiz e' comparado com (a) o aleatorio e (b) a acao validada, ambos
com o numero de iteracoes que iguala o tempo de CPU do aprendiz (T = 1.000). Razao de CPU K de uma execucao de 1.000 iteracoes de cada metodo sobre a do aleatorio, por classe (tipo, n), medida em sequencia (process_time, ao menos 1 s por medida)
em duas instancias de AJUSTE da classe (a mediana); K_val e' a da acao validada. Aleatorio a CPU igual do aprendiz k: T = max(50, round(1000 K_k)); validada a CPU igual do aprendiz k: T = max(50, round(1000 K_k / K_val)).
Fases: --calibrate (results/eqcpu_K_entrada268.json) | --run --shard i --nshards n (results/eqcpu_test_s{i}of{n}_entrada268.csv; metodos rand_eq_<k> e val_eq_<k>, k nos sete aprendizes, 3 sementes; sementes proprias)
Uso: python -m experiment.run_cycle41_eqcpu_fanjul_entrada268 --calibrate   |   --run --shard 0 --nshards 12
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
from experiment import run_cycle32_fanjul_entrada256 as F  # noqa: E402

R = F.R
SPACE = "B30"
LEARN = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr"]
SEED_SHIFT = 2_000_000_000 % (2 ** 31)
FIELDS = ["type", "instance", "m", "n", "idx", "method", "seed_i", "makespan", "bks", "T", "evals"]


def cpu_per_run(name, P, a0, m, od, lb, n, val_arm, qo, pub, seed0):
    reps, t0 = 0, time.process_time()
    while time.process_time() - t0 < 1.0:
        F.run_fixed_or_agent(SPACE, name, P, a0, m, od, lb, n, seed0 + reps, val_arm, qo, pub)
        reps += 1
    return (time.process_time() - t0) / reps


def calibrate():
    man = F.manifest()
    tune = man[man.role == "tune"]
    val = json.loads((R / f"fr_choice_{SPACE}_entrada256.json").read_text())["val"]
    qo, pub = F.frozen(SPACE)
    F.warm()
    res = {}
    for ty in sorted(tune["type"].unique()):
        for n in sorted(tune[tune["type"] == ty].n.unique()):
            sub = tune[(tune["type"] == ty) & (tune.n == n)].sort_values("instance").head(2)
            cls = f"{ty}|{int(n)}"
            ratios = {k: [] for k in LEARN + ["val"]}
            for _, r in sub.iterrows():
                P, a0, od, lb = F.load(r)
                m_ = int(r.m)
                base = cpu_per_run("rand", P, a0, m_, od, lb, int(n), 0, qo, pub, 7)
                for k in LEARN + ["val"]:
                    ratios[k].append(cpu_per_run(k, P, a0, m_, od, lb, int(n), val[cls], qo, pub, 7) / base)
            res[cls] = {k: float(np.median(v)) for k, v in ratios.items()}
            print(f"[eqcpu calib] {cls}: " + ", ".join(f"{k} {res[cls][k]:.2f}" for k in LEARN + ["val"]), flush=True)
    (R / "eqcpu_K_entrada268.json").write_text(json.dumps(res, indent=1))


def run(shard, nshards):
    man = F.manifest()
    man = man[man.role == "test"]
    val = json.loads((R / f"fr_choice_{SPACE}_entrada256.json").read_text())["val"]
    K = json.loads((R / "eqcpu_K_entrada268.json").read_text())
    qo, pub = F.frozen(SPACE)
    out = R / f"eqcpu_test_s{shard}of{nshards}_entrada268.csv"
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
        cls = f"{r['type']}|{n_}"
        rows = []
        for k in LEARN:
            Te = max(50, int(round(1000 * K[cls][k])))
            Tv = max(50, int(round(1000 * K[cls][k] / K[cls]["val"])))
            for meth, name, Tt in ((f"rand_eq_{k}", "rand", Te), (f"val_eq_{k}", "val", Tv)):
                F.T = Tt
                for s in range(F.N_SEED):
                    seed = (F.seed_of(SPACE, int(r.ri), name if name != "val" else "val", s, False) + SEED_SHIFT + (1 if name == "val" else 0) * 7_000_000 + LEARN.index(k) * 100_000) % (2 ** 31 - 1)
                    mk, ev = F.run_fixed_or_agent(SPACE, name, P, a0, m_, od, lb, n_, seed, val[cls], qo, pub)
                    rows.append(dict(type=r["type"], instance=int(r.instance), m=m_, n=n_, idx=int(r.idx), method=meth, seed_i=s, makespan=mk, bks=int(r.best_known), T=Tt, evals=ev))
        F.T = 1000
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if cnt % 10 == 0:
            print(f"[eqcpu s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[eqcpu s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.run:
        run(a.shard, a.nshards)
