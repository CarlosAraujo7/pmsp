"""
experiment/run_igcvls_entrada127.py
=====================================
Entrada 127 (Ciclo 5, frente 1): a receita pratica (d ajustado + partida pela melhor construtiva; `igcv_seed`, Entrada 121) ganha com a busca local do hibrido
(Entrada 124: a busca local acrescenta 0,15% ao IG)? `igcvls_seed` = igcv_seed + a mesma descida (inter-troca e realocacao, ate 2 passos) apos cada reconstrucao.
Para a comparacao a CPU igual, `igcv_seed_eq` roda o igcv_seed com T_eq = round(1000 K) iteracoes, K = razao mediana de CPU igcvls_seed/igcv_seed medida por classe
em 3 instancias (reps 0-2) numa fase de CALIBRACAO sequencial com a maquina em repouso (results/igcvls_cpu_ratio_entrada127.json).
  --calibrate : mede K por classe (sequencial)
  --synthetic : 12 classes x 100 reps (d/n por classe = ajuste da Entrada 114): igcvls_seed (T=1000) e igcv_seed_eq (T=T_eq)
  --external  : subamostra de 1.500 da Entrada 117: igcvls_seed (d/n por familia da Entrada 121)
Saidas: results/igcvls_seed_entrada127.csv, external_benchmarks/pcmax_symmetry2025/results_external_igcvls_seed_entrada127.csv. Retomavel.
Uso: python -m experiment.run_igcvls_entrada127 --calibrate ; --synthetic --external --workers 6
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

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))

from experiment.run_dsel_entrada114 import FIX, FRACS  # noqa: E402
from experiment.run_igls_p_entrada124 import make_ls  # noqa: E402

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
RATIO = ROOT / "results" / "igcvls_cpu_ratio_entrada127.json"
OUT_S = ROOT / "results" / "igcvls_seed_entrada127.csv"
OUT_X = EXT / "results_external_igcvls_seed_entrada127.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "member", "frac", "T"]


def d_choice():
    tune = pd.read_csv(ROOT / "results" / "dsel_tuning_entrada114.csv")
    return {c: FRACS[FIX.index(min(FIX, key=lambda k: g[g.method == k].makespan.mean()))] for c, g in tune.groupby("instance")}


def _solve(inst, frac, seed, T, ls):
    from experiment import config as CFG
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ig import ig_run
    from pmsp.seeding import best_constructive
    k = FRACS.index(frac)
    return ig_run(instance=inst, selector=FixedActionSelector(k), max_iter=T, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW,
                  action_map=lambda a, n: (0, max(2, round(FRACS[a] * n))), initial_solution=best_constructive(inst), ls_fn=make_ls() if ls else None)


def calibrate():
    from experiment import config as CFG
    from pmsp.instance import PMSPInstance
    ch = d_choice()
    out = {}
    for n, m in CLASSES:
        rs = []
        for rep in range(3):
            inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name="cal")
            a = _solve(inst, ch[f"rand_{n}j_{m}m"], rep, 1000, False)["cpu_time"]
            b = _solve(inst, ch[f"rand_{n}j_{m}m"], rep, 1000, True)["cpu_time"]
            rs.append(b / a)
        k = sorted(rs)[1]
        out[f"{n}x{m}"] = dict(K=k, T_eq=max(1000, int(round(1000 * k))))
        print(f"[calib] {n}x{m}: K = {k:.2f} -> T_eq = {out[f'{n}x{m}']['T_eq']}", flush=True)
    RATIO.write_text(json.dumps(out, indent=1))


def _run_s(t):
    from experiment import config as CFG
    from pmsp.instance import PMSPInstance
    n, m, rep, mt = t["n"], t["m"], t["rep"], t["method"]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    r = _solve(inst, t["frac"], CFG.BASE_SEED + rep, t["T"], mt == "igcvls_seed")
    return dict(instance=inst.name, method=mt, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], member="", frac=t["frac"], T=t["T"])


def _run_x(t):
    from pmsp.instance import PMSPInstance
    inst = PMSPInstance(p=t["p"], m=t["m"], name=f"ext_{t['member']}")
    r = _solve(inst, t["frac"], t["inst_id"], 1000, True)
    return dict(instance=inst.name, method="igcvls_seed", rep=t["inst_id"], n=inst.n, m=inst.m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"],
                cpu_time=r["cpu_time"], n_improvements=r["n_improvements"], member=t["member"], frac=t["frac"], T=1000)


def _key(x):
    return (x["member"], x["method"]) if x.get("member") else (x["instance"] if "lb" in x else f"rand_{x['n']}j_{x['m']}m", x["method"], str(x["rep"]))


def _drive(fn, tasks, out, keyf, workers):
    done = set()
    if out.exists():
        done = {keyf(r) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if keyf(t) not in done]
    pend.sort(key=lambda t: -t.get("n", 0) * t.get("T", 1000))
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
                if i % 100 == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("calibrate", "synthetic", "external"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.synthetic:
        ch, ratio = d_choice(), json.loads(RATIO.read_text())
        tasks = []
        for n, m in CLASSES:
            f = ch[f"rand_{n}j_{m}m"]
            for r in range(100):
                tasks.append(dict(n=n, m=m, rep=r, frac=f, T=1000, method="igcvls_seed"))
                tasks.append(dict(n=n, m=m, rep=r, frac=f, T=ratio[f"{n}x{m}"]["T_eq"], method="igcv_seed_eq"))
        _drive(_run_s, tasks, OUT_S, _key, a.workers)
    if a.external:
        from build_sample import evenly_spaced_indices  # noqa: F401
        from experiment.run_igcv_seed_entrada121 import CHOICE_X, _load_manifest
        ch = json.loads(CHOICE_X.read_text())
        items = _load_manifest(EXT / "manifest_ilsseed_entrada117.csv")
        tasks = [dict(it, frac=ch[it["family"]], method="igcvls_seed") for it in items]
        _drive(_run_x, tasks, OUT_X, lambda x: (x["member"], x["method"]) if "member" in x else None, a.workers)


if __name__ == "__main__":
    main()
