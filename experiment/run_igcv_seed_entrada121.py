"""
experiment/run_igcv_seed_entrada121.py
========================================
Entrada 121 (Ciclo 3, frente 4): o IG com tamanho de destruicao AJUSTADO POR VALIDACAO e partida pela melhor construtiva (`igcv_seed`) bate MULTIFIT, LDM
e o IG+ padrao? Metodo: IG, estrategia de destruicao `Random` fixa, d/n escolhido por validacao, partida = min(LPT, MULTIFIT, LDM), T=1000.
  sintetico : 12 classes x 100 reps (mesmas instancias de results.csv); d/n por classe = melhor d fixo do AJUSTE da Entrada 114 (dsel_tuning_entrada114.csv)
  externo   : subamostra de 1.500 (manifest_ilsseed_entrada117.csv); d/n por FAMILIA escolhido nas 553 instancias de CALIBRACAO da Entrada 115 (nao vistas;
              B 196, C 225, E 132), com o mesmo IG semeado e aleatorio, T=1000, criterio = menor makespan/LB medio; a familia F (sem instancias nao vistas) usa o d/n
              escolhido pooled sobre todas as instancias de calibracao.
Saidas: results/igcv_seed_entrada121.csv, results/igcv_ext_tuning_entrada121.csv, results/igcv_ext_choice_entrada121.json,
external_benchmarks/pcmax_symmetry2025/results_external_igcv_seed_entrada121.csv. Uso: python -m experiment.run_igcv_seed_entrada121 --tune-ext --synthetic --external --workers 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))

from experiment.run_dsel_entrada114 import FIX, FRACS  # noqa: E402

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
OUT_S = ROOT / "results" / "igcv_seed_entrada121.csv"
OUT_X = EXT / "results_external_igcv_seed_entrada121.csv"
TUNE_X = ROOT / "results" / "igcv_ext_tuning_entrada121.csv"
CHOICE_X = ROOT / "results" / "igcv_ext_choice_entrada121.json"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "member", "frac"]


def _solve(inst, frac, seed):
    from experiment import config as CFG
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ig import ig_run
    from pmsp.seeding import best_constructive
    k = FRACS.index(frac)
    return ig_run(instance=inst, selector=FixedActionSelector(k), max_iter=1000, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW,
                  action_map=lambda a, n: (0, max(2, round(FRACS[a] * n))), initial_solution=best_constructive(inst))


def _run_s(t):
    from experiment import config as CFG
    from pmsp.instance import PMSPInstance
    n, m, rep = t["n"], t["m"], t["rep"]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    r = _solve(inst, t["frac"], CFG.BASE_SEED + rep)
    return dict(instance=inst.name, method="igcv_seed", rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], member="", frac=t["frac"])


def _run_x(t):
    from pmsp.instance import PMSPInstance
    inst = PMSPInstance(p=t["p"], m=t["m"], name=f"ext_{t['member']}")
    r = _solve(inst, t["frac"], t["inst_id"])
    return dict(instance=inst.name, method=t.get("method", "igcv_seed"), rep=t["inst_id"], n=inst.n, m=inst.m, lb=inst.lower_bound, makespan=r["makespan"],
                gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"], member=t["member"], frac=t["frac"])


def _drive(fn, tasks, out, keyf, workers, every=200):
    done = set()
    if out.exists():
        done = {keyf(r) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if keyf(t) not in done]
    pend.sort(key=lambda t: -t.get("n", 0))
    print(f"[tasks] {len(pend)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in pend}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % every == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def _key_s(x):
    if "lb" in x:
        return (x["instance"], x["method"], x["rep"])
    return (f"rand_{x['n']}j_{x['m']}m", "igcv_seed", str(x["rep"]))


def _load_manifest(path):
    from loader import load_many
    rows = list(csv.DictReader(open(path, newline="")))
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r["member"])
    data = {}
    for fam, mem in by.items():
        data.update(load_many(fam, mem))
    out = []
    for r in rows:
        m_, n_, p_ = data[r["member"]]
        out.append(dict(family=r["family"], member=r["member"], m=m_, n=n_, p=p_, inst_id=int(r["inst_id"])))
    return out


def tune_ext(workers):
    items = _load_manifest(EXT / "manifest_calib_entrada115.csv")
    tasks = [dict(it, frac=f, method=f"tune_{f}") for it in items for f in FRACS]
    _drive(_run_x, tasks, TUNE_X, lambda x: (x["member"], x["method"]), workers, every=500)
    d = pd.read_csv(TUNE_X)
    d["ratio"] = d.makespan / d.lb
    d["family"] = d.member.str[0]
    d["frac"] = d.frac.astype(float)
    ch = {}
    for fam, g in d.groupby("family"):
        mean = g.groupby("frac").ratio.mean()
        ch[fam] = float(mean.idxmin())
        print(f"[tune-ext] {fam}: " + ", ".join(f"{k:.2f}:{v:.5f}" for k, v in mean.items()) + f" -> {ch[fam]}")
    mean = d.groupby("frac").ratio.mean()
    ch["F"] = float(mean.idxmin())
    print("[tune-ext] pooled ->", ch["F"])
    CHOICE_X.write_text(json.dumps(ch, indent=1))


def main():
    ap = argparse.ArgumentParser()
    for k in ("tune-ext", "synthetic", "external"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if getattr(a, "tune_ext"):
        tune_ext(a.workers)
    if a.synthetic:
        tune = pd.read_csv(ROOT / "results" / "dsel_tuning_entrada114.csv")
        ch = {c: FRACS[FIX.index(min(FIX, key=lambda k: g[g.method == k].makespan.mean()))] for c, g in tune.groupby("instance")}
        tasks = [dict(n=n, m=m, rep=r, frac=ch[f"rand_{n}j_{m}m"]) for n, m in CLASSES for r in range(100)]
        _drive(_run_s, tasks, OUT_S, _key_s, a.workers)
    if a.external:
        ch = json.loads(CHOICE_X.read_text())
        items = _load_manifest(EXT / "manifest_ilsseed_entrada117.csv")
        tasks = [dict(it, frac=ch[it["family"]], method="igcv_seed") for it in items]
        _drive(_run_x, tasks, OUT_X, lambda x: (x["member"], x["method"]), a.workers)


if __name__ == "__main__":
    main()
