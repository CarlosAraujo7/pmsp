"""
analysis/audit_entrada101_exact_vs_iaf.py (Entrada 101)
Valida pmsp.exact.lp_lower_bound contra os otimos do iAF (Z, linhas com otimalidade provada):
(1) LB certificado NUNCA pode exceder Z; (2) mede a fracao de casos em que LB == Z (justo) e o
tempo. UB = melhor makespan dos 13 metodos ja executados na instancia.
Uso: python -m analysis.audit_entrada101_exact_vs_iaf [n_amostra] [n_max] [workers]
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_benchmarks" / "pcmax_symmetry2025"))


def _one(args):
    member, p, m, ub, z = args
    from pmsp.exact import lp_lower_bound
    lb, proven, t = lp_lower_bound(p, m, ub)
    return dict(member=member, n=len(p), m=m, Z=z, ub=ub, lb=lb, proven=proven, t=t)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    from loader import load_many
    EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
    ref = pd.read_csv(EXT / "iaf_reference.csv")
    res = pd.read_csv(EXT / "results_external_symmetry2025.csv").drop_duplicates(["member", "method"])
    best = res.groupby("member").makespan.min()
    ref = ref[(ref.gap_sheet == 0) & ref.member.isin(best.index) & (ref.n <= NMAX) & (ref.family == "C")]
    s = ref.sample(min(N, len(ref)), random_state=3)
    data = load_many("C", s.member.tolist())
    jobs = [(r.member, data[r.member][2], data[r.member][0], int(best[r.member]), int(r.Z)) for r in s.itertuples()]
    with ProcessPoolExecutor(max_workers=W) as ex:
        d = pd.DataFrame(list(ex.map(_one, jobs, chunksize=2)))
    print(f"instancias: {len(d)} (n<= {NMAX})")
    print("VIOLACOES (LB > Z):", int((d.lb > d.Z).sum()))
    print("LB == Z (justo):", int((d.lb == d.Z).sum()), f"({100 * (d.lb == d.Z).mean():.1f}%) | LB < Z:", int((d.lb < d.Z).sum()),
          "| UB == Z:", int((d.ub == d.Z).sum()))
    print("otimo provado por LB==UB:", int(d.proven.sum()), "| tempo s: mediana %.2f, p90 %.2f, max %.1f" % (d.t.median(), d.t.quantile(.9), d.t.max()))
    print("folga media (Z-LB)/Z entre os LB < Z: %.4f%%" % (100 * ((d.Z - d.lb) / d.Z)[d.lb < d.Z].mean() if (d.lb < d.Z).any() else 0))
    d.to_csv(ROOT / "results" / "entrada101_exact_vs_iaf.csv", index=False)


if __name__ == "__main__":
    main()
