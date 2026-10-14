"""
pmsp/unrelated_srmilp.py (Entrada 148)
=======================================
MILP RESTRITO ("size reduction", ideia de Fanjul-Peyro e Ruiz 2011) para polir uma solucao de R||Cmax: cada tarefa so' pode ir para as `k` maquinas de menor p_ij e para a maquina que ocupa na solucao
corrente (o problema restrito e' viavel e contem a solucao corrente). Resolve com o HiGHS do SciPy com limite de tempo e devolve a melhor das duas (corrente ou do MILP restrito).
  restricted_milp(inst, sol, k, time_limit) -> (UrSol, makespan_milp or None, cpu)
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .unrelated import UnrelatedInstance, UrSol


def restricted_milp(inst: UnrelatedInstance, sol: UrSol, k: int = 4, time_limit: float = 20.0, rank: Optional[np.ndarray] = None,
                    allowed_extra: Optional[np.ndarray] = None) -> Tuple[UrSol, Optional[int], float]:
    t0 = time.perf_counter()
    m, n = inst.m, inst.n
    kk = min(k, m)
    order = np.argsort(inst.p if rank is None else rank, axis=0, kind="stable")[:kk]   # (kk, n): maquinas de menor p_ij (ou de menor `rank`, ex.: custo reduzido do LP) de cada tarefa
    allowed = np.zeros((m, n), dtype=bool)
    allowed[order, np.arange(n)[None, :]] = True
    if allowed_extra is not None:
        allowed |= allowed_extra
    allowed[sol.a, np.arange(n)] = True
    ii, jj = np.nonzero(allowed)
    nv = len(ii)
    # variaveis: x_(i,j) para pares permitidos, depois C
    rows_eq = jj
    cols_eq = np.arange(nv)
    A_eq = coo_matrix((np.ones(nv), (rows_eq, cols_eq)), shape=(n, nv + 1)).tocsr()
    rows_ub = np.r_[ii, np.arange(m)]
    cols_ub = np.r_[np.arange(nv), np.full(m, nv)]
    vals_ub = np.r_[inst.p[ii, jj].astype(float), -np.ones(m)]
    A_ub = coo_matrix((vals_ub, (rows_ub, cols_ub)), shape=(m, nv + 1)).tocsr()
    c = np.zeros(nv + 1)
    c[-1] = 1.0
    cons = [LinearConstraint(A_eq, np.ones(n), np.ones(n)), LinearConstraint(A_ub, -np.inf, np.zeros(m))]
    integrality = np.r_[np.ones(nv), 0.0]
    lb = np.r_[np.zeros(nv), float(inst.lower_bound)]
    ub = np.r_[np.ones(nv), float(sol.makespan)]                                   # C <= makespan corrente
    r = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub), options={"time_limit": float(time_limit), "mip_rel_gap": 0.0, "disp": False})
    cpu = time.perf_counter() - t0
    if r.x is None:
        return sol, None, cpu
    x = r.x[:-1]
    a = np.full(n, -1, dtype=np.int64)
    best = np.full(n, -1.0)
    for idx in range(nv):
        j = jj[idx]
        if x[idx] > best[j]:
            best[j] = x[idx]
            a[j] = ii[idx]
    new = UrSol(inst, a)
    if new.makespan < sol.makespan:
        return new, int(new.makespan), cpu
    return sol, int(new.makespan), cpu
