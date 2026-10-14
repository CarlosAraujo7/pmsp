"""
experiment/run_new_distributions_entrada108.py
================================================
WP6 do plano de revisao (diario, Entrada 108): generalidade das conclusoes fora de U[1,100] e na regiao dificil n/m = 2-3.
6 distribuicoes (pmsp/generators.py) x 5 tamanhos (n,m) in {(25,10),(60,20),(50,10),(100,10),(200,20)} x 30 repeticoes =
900 instancias (seeds 70000 + 1000*cell + rep, disjuntas das demais baterias). T = 1000.
Metodos: lpt, multifit, ldm, ig_random, ig_ql, ils_random, ils_ql (limiar de perturbacao 15 se n >= 150, senao 50, como
as baterias do artigo), ig_random_seed, ig_ql_seed. Fase --exact: LB arc-flow certificado (pmsp.exact) com UB = melhor
makespan entre os metodos.
Saidas: results/new_distributions_entrada108.csv, results/new_distributions_entrada108_exact.csv.
Uso: python -m experiment.run_new_distributions_entrada108 --run --workers 12 | --exact --workers 12 | --smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIZES = [(25, 10), (60, 20), (50, 10), (100, 10), (200, 20)]
N_REPS = 30
METHODS = ["lpt", "multifit", "ldm", "ig_random", "ig_ql", "ils_random", "ils_ql", "ig_random_seed", "ig_ql_seed"]
OUT = ROOT / "results" / "new_distributions_entrada108.csv"
OUT_EXACT = ROOT / "results" / "new_distributions_entrada108_exact.csv"
FIELDS = ["dist", "n", "m", "rep", "method", "lb", "makespan", "gap_pct", "cpu_time", "dup_ratio"]


def cells():
    from pmsp.generators import DISTRIBUTIONS
    return [(d, n, m) for d in DISTRIBUTIONS for (n, m) in SIZES]


def seed_of(dist: str, n: int, m: int, rep: int) -> int:
    return 70000 + 1000 * cells().index((dist, n, m)) + rep


def _instance(dist, n, m, rep):
    from pmsp.generators import make_instance
    return make_instance(dist, n, m, seed_of(dist, n, m, rep))


def _run(task: dict) -> dict:
    from experiment import config as CFG
    from experiment.runner import make_ils_selector, run_single
    from pmsp.apriori_select import dup_ratio
    from pmsp.ils import ils_run

    dist, n, m, rep, method = task["dist"], task["n"], task["m"], task["rep"], task["method"]
    inst = _instance(dist, n, m, rep)
    seed = CFG.BASE_SEED + rep
    if method in ("ils_random", "ils_ql"):
        sel = make_ils_selector(method, seed)
        r = ils_run(instance=inst, selector=sel, max_iter=1000, perturbation_threshold=15 if n >= 150 else 50,
                    n_perturb=CFG.N_PERTURB, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW)
    else:
        r = run_single(inst, method, rep, max_iter=1000)
    return dict(dist=dist, n=n, m=m, rep=rep, method=method, lb=inst.lower_bound, makespan=r["makespan"],
                gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], dup_ratio=dup_ratio(inst))


def _exact(task: dict) -> dict:
    from pmsp.exact import lp_lower_bound
    from pmsp.exact import improved_lower_bound
    inst = _instance(task["dist"], task["n"], task["m"], task["rep"])
    # EMENDA 1 (Entrada 108, antes de ler qualquer resultado): o grafo de arcos tem ~ (#tamanhos distintos) x UB arcos; acima de
    # 4e5 (inicialmente 3e6, rebaixado ao ver instancias u1_10000 n=50 levarem >10 min) o LP nao resolve em tempo razoavel (u1_10000: ate 580 s por instancia, n>=100 inviavel). Nesses casos usa-se o LB
    # classico melhorado (max(pmax, ceil(sum/m), p_(m)+p_(m+1)), tambem valido): lb_lp = esse LB, skipped = 1.
    cost = len(set(inst.p.tolist())) * task["ub"]
    if cost > 400_000:
        lb = improved_lower_bound(inst.p.tolist(), inst.m)
        return dict(dist=task["dist"], n=task["n"], m=task["m"], rep=task["rep"], ub=task["ub"], lb_lp=lb, proven=int(lb >= task["ub"]),
                    t=0.0, lb_classic=inst.lower_bound, skipped=1)
    lb, proven, t = lp_lower_bound(inst.p.tolist(), inst.m, task["ub"])
    return dict(dist=task["dist"], n=task["n"], m=task["m"], rep=task["rep"], ub=task["ub"], lb_lp=lb, proven=int(proven), t=t,
                lb_classic=inst.lower_bound, skipped=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.run or args.smoke:
        cl = cells()[:1] if args.smoke else cells()
        out = OUT.with_name("new_distributions_entrada108_SMOKE.csv") if args.smoke else OUT
        done = set()
        if out.exists():
            with open(out, newline="") as f:
                done = {(r["dist"], r["n"], r["m"], r["rep"], r["method"]) for r in csv.DictReader(f)}
        tasks = [dict(dist=d, n=n, m=m, rep=rep, method=mt) for d, n, m in cl for rep in range(2 if args.smoke else N_REPS)
                 for mt in METHODS if (d, str(n), str(m), str(rep), mt) not in done]
        print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)")
        new = not out.exists()
        t0 = time.perf_counter()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_run, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futs), 1):
                    try:
                        w.writerow(fut.result())
                    except Exception as e:  # noqa: BLE001
                        print("[ERRO]", futs[fut], e)
                        continue
                    f.flush()
                    os.fsync(f.fileno())
                    if i % 200 == 0 or i == len(tasks):
                        el = time.perf_counter() - t0
                        print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
        print("concluido ->", out)
    if args.exact:
        import pandas as pd
        d = pd.read_csv(OUT)
        ub = d.groupby(["dist", "n", "m", "rep"]).makespan.min().reset_index()
        done = set()
        if OUT_EXACT.exists():
            e = pd.read_csv(OUT_EXACT)
            done = set(zip(e.dist, e.n, e.m, e.rep))
        tasks = [dict(dist=r.dist, n=int(r.n), m=int(r.m), rep=int(r.rep), ub=int(r.makespan)) for r in ub.itertuples()
                 if (r.dist, r.n, r.m, r.rep) not in done]
        print(f"[exact] {len(tasks)} pendentes")
        t0 = time.perf_counter()
        new = not OUT_EXACT.exists()
        with open(OUT_EXACT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dist", "n", "m", "rep", "ub", "lb_lp", "proven", "t", "lb_classic", "skipped"])
            if new:
                w.writeheader()
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_exact, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futs), 1):
                    try:
                        w.writerow(fut.result())
                    except Exception as e:  # noqa: BLE001
                        print("[ERRO]", futs[fut], e)
                        continue
                    f.flush()
                    os.fsync(f.fileno())
                    if i % 50 == 0 or i == len(tasks):
                        print(f"  [{i}/{len(tasks)}] {time.perf_counter() - t0:.0f}s", flush=True)
        print("concluido ->", OUT_EXACT)


if __name__ == "__main__":
    main()
