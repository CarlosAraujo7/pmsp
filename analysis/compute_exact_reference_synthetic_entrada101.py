"""
analysis/compute_exact_reference_synthetic_entrada101.py (Entrada 101)
Referencia exata das 12 classes sinteticas: para cada (classe, rep) com UB = melhor makespan dos 23
metodos de results.csv, calcula LB3 (classico melhorado) e o LB certificado do arc-flow (pmsp.exact).
Saida: results/exact_reference_synthetic_entrada101.csv (instance, rep, n, m, ub, lb1, lb3, lb_lp, proven).
Uso: python -m analysis.compute_exact_reference_synthetic_entrada101 [workers]
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _one(a):
    inst, rep, n, m, ub = a
    from pmsp.exact import improved_lower_bound, lp_lower_bound
    from pmsp.instance import PMSPInstance
    p = [int(x) for x in PMSPInstance.random(n=n, m=m, seed=int(rep), name=inst).p]
    lb1 = max(max(p), -(-sum(p) // m))
    lb3 = improved_lower_bound(p, m)
    lb, proven, t = lp_lower_bound(p, m, ub)
    return dict(instance=inst, rep=rep, n=n, m=m, ub=ub, lb1=lb1, lb3=lb3, lb_lp=lb, proven=proven, t=t)


def main():
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    df = pd.read_csv(ROOT / "results" / "results.csv")
    best = df.groupby(["instance", "rep"]).makespan.min().reset_index()
    nm = df.groupby("instance")[["n", "m"]].first()
    jobs = [(r.instance, int(r.rep), int(nm.loc[r.instance, "n"]), int(nm.loc[r.instance, "m"]), int(r.makespan))
            for r in best.itertuples()]
    jobs.sort(key=lambda j: -j[2])                  # classes grandes primeiro
    with ProcessPoolExecutor(max_workers=W) as ex:
        out = pd.DataFrame(list(ex.map(_one, jobs, chunksize=4)))
    out.to_csv(ROOT / "results" / "exact_reference_synthetic_entrada101.csv", index=False)
    g = out.groupby("instance").agg(N=("rep", "count"), proven=("proven", "mean"), t_max=("t", "max"), slack_lp=("lb_lp", "mean"))
    print(g.round(3).to_string())
    print("provados:", int(out.proven.sum()), "de", len(out), "| tempo total s: %.0f" % out.t.sum())


if __name__ == "__main__":
    main()
