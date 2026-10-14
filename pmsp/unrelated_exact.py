"""
pmsp/unrelated_exact.py (Entrada 118)
=====================================
Referencias exatas para R||Cmax com o HiGHS do SciPy (sem solver comercial):
  lp_bound(inst)            relaxacao linear do modelo de atribuicao -> limite inferior VALIDO (ceil do valor otimo do LP)
  milp(inst, time_limit)    MILP de atribuicao (x_ij binario; sum_i x_ij = 1; sum_j p_ij x_ij <= C; min C) com limite de tempo;
                            devolve (incumbente, limite_dual, otimo_provado, tempo). O limite dual do branch-and-bound tambem e' um LB valido.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, csr_matrix

from .unrelated import UnrelatedInstance


def _model(inst: UnrelatedInstance):
    m, n = inst.m, inst.n
    nv = m * n + 1                                   # x[i, j] em i*n + j ; C na ultima posicao
    rows, cols, vals = [], [], []
    for j in range(n):                               # sum_i x_ij = 1
        for i in range(m):
            rows.append(j)
            cols.append(i * n + j)
            vals.append(1.0)
    A_eq = coo_matrix((vals, (rows, cols)), shape=(n, nv)).tocsr()
    rows, cols, vals = [], [], []
    for i in range(m):                               # sum_j p_ij x_ij - C <= 0
        for j in range(n):
            rows.append(i)
            cols.append(i * n + j)
            vals.append(float(inst.p[i, j]))
        rows.append(i)
        cols.append(m * n)
        vals.append(-1.0)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(m, nv)).tocsr()
    c = np.zeros(nv)
    c[-1] = 1.0
    return c, A_eq, A_ub


def lp_bound(inst: UnrelatedInstance) -> int:
    c, A_eq, A_ub = _model(inst)
    nv = len(c)
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(inst.m), A_eq=A_eq, b_eq=np.ones(inst.n), bounds=[(0, 1)] * (nv - 1) + [(0, None)], method="highs-ipm")
    if r.status != 0:
        return inst.lower_bound
    return int(max(inst.lower_bound, math.ceil(r.fun - 1e-6)))


def milp_solve(inst: UnrelatedInstance, time_limit: float = 60.0) -> Tuple[Optional[int], int, bool, float]:
    c, A_eq, A_ub = _model(inst)
    nv = len(c)
    cons = [LinearConstraint(A_eq, np.ones(inst.n), np.ones(inst.n)), LinearConstraint(A_ub, -np.inf, np.zeros(inst.m))]
    integrality = np.r_[np.ones(nv - 1), 0.0]
    lb = np.r_[np.zeros(nv - 1), float(inst.lower_bound)]
    ub = np.r_[np.ones(nv - 1), float(inst.p.max() * inst.n)]
    t0 = time.perf_counter()
    r = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub), options={"time_limit": float(time_limit), "mip_rel_gap": 0.0, "disp": False})
    t = time.perf_counter() - t0
    inc = None
    if r.x is not None:
        x = r.x[:-1].reshape(inst.m, inst.n)
        a = x.argmax(axis=0)
        inc = _ms(inst, a)
    dual = getattr(r, "mip_dual_bound", None)
    dual_i = int(max(inst.lower_bound, math.ceil(dual - 1e-6))) if dual is not None and np.isfinite(dual) else inst.lower_bound
    proven = bool(inc is not None and inc <= dual_i)
    return inc, dual_i, proven, t


def _ms(inst: UnrelatedInstance, a: np.ndarray) -> int:
    loads = np.zeros(inst.m, dtype=np.int64)
    np.add.at(loads, a, inst.p[a, np.arange(inst.n)])
    return int(loads.max())
