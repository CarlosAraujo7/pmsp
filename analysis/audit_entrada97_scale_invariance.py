"""
analysis/audit_entrada97_scale_invariance.py (Entrada 97)
Invariancia a escala por MAKESPAN: se todos os p_j sao multiplicados por 1000, um
metodo invariante a escala produz makespan_escalado == 1000 * makespan_base
(a comparacao por gap_pct e' confundida pelo arredondamento do LB inteiro).
Uso: python -m analysis.audit_entrada97_scale_invariance
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment.runner import run_single  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

for method in ["lpt", "ig_random", "ils_random", "ig_ql", "ils_ql", "ils_ucb", "ig_ucb"]:
    same, tot = 0, 0
    for rep in range(6):
        base = PMSPInstance.random(n=60, m=12, seed=rep, name="b")
        scaled = PMSPInstance(p=np.asarray(base.p, dtype=np.int64) * 1000, m=12, name="s")
        a = run_single(base, method, rep, max_iter=300)["makespan"]
        b = run_single(scaled, method, rep, max_iter=300)["makespan"]
        same += int(b == 1000 * a); tot += 1
    print(f"{method:11s}: makespan(1000p) == 1000*makespan(p) em {same}/{tot} instancias")
