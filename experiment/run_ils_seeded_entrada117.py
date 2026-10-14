"""
experiment/run_ils_seeded_entrada117.py
=========================================
Entrada 117: ILS tambem partindo da melhor construtiva (min de LPT, MULTIFIT, LDM), para a comparacao IG+ vs ILS+ ser SIMETRICA
(na Entrada 102 so' o IG foi semeado). Metodos: ils_ql_seed, ils_random_seed (T=1000; limiar de perturbacao 15 se n>=150, senao 50 nas
sinteticas -- como results.csv --; 50 no externo -- como run_single/bateria externa).
  --synthetic : 12 classes x 100 reps (mesmas instancias/seeds de results.csv)
  --external  : subamostra estratificada deterministica de 1.500 instancias do manifest.csv (B 400, C 400, E 400, F 300; indices
                igualmente espacados dentro da familia), seed = inst_id
Saidas: results/ils_seeded_entrada117.csv; external_benchmarks/pcmax_symmetry2025/manifest_ilsseed_entrada117.csv e
results_external_ils_seeded_entrada117.csv. Uso: python -m experiment.run_ils_seeded_entrada117 --synthetic --external --workers 12
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
METHODS = ["ils_ql_seed", "ils_random_seed"]
OUT_S = ROOT / "results" / "ils_seeded_entrada117.csv"
M_X = EXT / "manifest_ilsseed_entrada117.csv"
OUT_X = EXT / "results_external_ils_seeded_entrada117.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "member"]
QUOTA = {"B": 400, "C": 400, "E": 400, "F": 300}


def _solve(inst, method, seed, thr):
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.ils import ils_run
    from pmsp.seeding import best_constructive
    return ils_run(instance=inst, selector=make_ils_selector(method.replace("_seed", ""), seed), max_iter=1000, perturbation_threshold=thr,
                   n_perturb=CFG.N_PERTURB, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW, initial_solution=best_constructive(inst))


def _run_s(t):
    from experiment import config as CFG
    from pmsp.instance import PMSPInstance
    n, m, rep, mt = t["n"], t["m"], t["rep"], t["method"]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    r = _solve(inst, mt, CFG.BASE_SEED + rep, 15 if n >= 150 else 50)
    return dict(instance=inst.name, method=mt, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"],
                cpu_time=r["cpu_time"], n_improvements=r["n_improvements"], member="")


def _run_x(t):
    from pmsp.instance import PMSPInstance
    inst = PMSPInstance(p=t["p"], m=t["m"], name=f"ext_{t['member']}")
    r = _solve(inst, t["method"], t["inst_id"], 50)
    return dict(instance=inst.name, method=t["method"], rep=t["inst_id"], n=inst.n, m=inst.m, lb=inst.lower_bound, makespan=r["makespan"],
                gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"], member=t["member"])


def _drive(fn, tasks, out, key, workers):
    done = set()
    if out.exists():
        done = {key(r) for r in csv.DictReader(open(out, newline=""))}
    tasks = [t for t in tasks if key(t) not in done]
    tasks.sort(key=lambda t: -(t.get("n", 0)))
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 200 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.synthetic:
        tasks = [dict(n=n, m=m, rep=r, method=mt) for n, m in CLASSES for r in range(100) for mt in METHODS]
        _drive(_run_s, tasks, OUT_S, lambda x: (f"rand_{x['n']}j_{x['m']}m" if "n" in x and "lb" not in x else x["instance"], x["method"], str(x["rep"])), a.workers)
    if a.external:
        from build_sample import evenly_spaced_indices
        from loader import load_many
        if not M_X.exists():
            rows = list(csv.DictReader(open(EXT / "manifest.csv", newline="")))
            sel = []
            for fam, q in QUOTA.items():
                fr = [r for r in rows if r["family"] == fam]
                sel += [fr[i] for i in evenly_spaced_indices(len(fr), q)]
            with open(M_X, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["family", "member", "subfolder", "m", "n", "inst_id"])
                wr.writeheader()
                wr.writerows(sel)
        sel = list(csv.DictReader(open(M_X, newline="")))
        by = defaultdict(list)
        for r in sel:
            by[r["family"]].append(r["member"])
        data = {}
        for fam, mem in by.items():
            data.update(load_many(fam, mem))
        tasks = []
        for r in sel:
            m_, n_, p_ = data[r["member"]]
            for mt in METHODS:
                tasks.append(dict(member=r["member"], m=m_, n=n_, p=p_, inst_id=int(r["inst_id"]), method=mt))
        _drive(_run_x, tasks, OUT_X, lambda x: (x["member"], x["method"]) if "p" in x else (x["member"], x["method"]), a.workers)


if __name__ == "__main__":
    main()
