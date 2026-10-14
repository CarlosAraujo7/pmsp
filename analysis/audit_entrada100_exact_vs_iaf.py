"""
analysis/audit_entrada100_exact_vs_iaf.py (Entrada 100)
Valida pmsp/exact.py contra os valores ótimos do iAF (Results_iAF.xlsx, join verificado
em iaf_reference.csv) em instancias reais da familia C/F ja executadas: para cada
instancia amostrada, UB = melhor makespan dos 13 metodos; compara o makespan
retornado com Z (so' linhas com otimalidade provada pelo iAF).
Uso: python -m analysis.audit_entrada100_exact_vs_iaf [n_amostra] [limite_n]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_benchmarks" / "pcmax_symmetry2025"))
from loader import load_many  # noqa: E402
from pmsp.exact import optimal_makespan  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 100
EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
ref = pd.read_csv(EXT / "iaf_reference.csv")
res = pd.read_csv(EXT / "results_external_symmetry2025.csv").drop_duplicates(["member", "method"])
best = res.groupby("member").makespan.min()
ref = ref[(ref.gap_sheet == 0) & (ref.member.isin(best.index)) & (ref.n <= NMAX) & (ref.family == "C")]
sample = ref.sample(min(N, len(ref)), random_state=0)
data = load_many("C", sample.member.tolist())
rows = []
for r in sample.itertuples():
    m, n, p = data[r.member]
    opt, proven, t = optimal_makespan(p, m, int(best[r.member]), time_limit=60)
    rows.append(dict(member=r.member, n=n, m=m, Z=r.Z, opt=opt, proven=proven, t=t, ub=int(best[r.member])))
d = pd.DataFrame(rows)
print(d.groupby("proven").size().to_dict(), "| proven AND equal to iAF Z:", int(((d.opt == d.Z) & d.proven).sum()), "of", len(d))
print("mismatches:", d[(d.opt != d.Z) & d.proven].to_string(index=False) if ((d.opt != d.Z) & d.proven).any() else "none")
print("unproven (time limit):", int((~d.proven).sum()), "| time s: median %.2f, p90 %.2f, max %.1f" % (d.t.median(), d.t.quantile(.9), d.t.max()))
print("opt < Z (would contradict iAF optimality):", int((d.opt < d.Z).sum()))
