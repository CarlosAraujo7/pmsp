"""
pmsp/unrelated_lp.py (Entrada 141)
===================================
Partida por RELAXACAO LINEAR para R||Cmax (Lenstra, Shmoys e Tardos 1990): resolve a relaxacao do modelo de atribuicao (min C; sum_i x_ij = 1; sum_j p_ij x_ij <= C; 0 <= x <= 1) com o HiGHS
e arredonda a solucao basica: as tarefas com x_ij ~ 1 ficam na maquina i; as (poucas: no maximo uma por maquina na solucao basica) fracionarias vao, em ordem decrescente do maior p,
para a maquina de menor conclusao entre as do seu suporte (x_ij > 0). Deterministico dado a instancia. O arredondamento tem makespan <= C_LP + max p (garantia de 2-aproximacao do LP).
  lp_solution(inst) -> (C_LP, x)         x: (m, n) ou None se o solver falhar
  lp_round(inst, x)  -> UrSol
  lp_start(inst)     -> (UrSol, C_LP, tempo)   cai para mct_greedy se o LP falhar
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from .unrelated import UnrelatedInstance, UrSol, mct_greedy
from .unrelated_exact import _model


def lp_solution(inst: UnrelatedInstance) -> Tuple[Optional[float], Optional[np.ndarray]]:
    c, A_eq, A_ub = _model(inst)
    nv = len(c)
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(inst.m), A_eq=A_eq, b_eq=np.ones(inst.n), bounds=[(0, 1)] * (nv - 1) + [(0, None)], method="highs-ipm")
    if r.status != 0 or r.x is None:
        return None, None
    return float(r.fun), r.x[:-1].reshape(inst.m, inst.n)


def lp_round(inst: UnrelatedInstance, x: np.ndarray, tol: float = 1e-6) -> UrSol:
    m, n = inst.m, inst.n
    a = x.argmax(axis=0).astype(np.int64)
    frac = np.where(x.max(axis=0) < 1.0 - tol)[0]
    sol = UrSol(inst, a)
    if len(frac):
        # remove as fracionarias e as reinsere na maquina de menor conclusao dentro do suporte da solucao LP
        loads = np.zeros(m, dtype=np.int64)
        keep = np.ones(n, dtype=bool)
        keep[frac] = False
        np.add.at(loads, a[keep], inst.p[a[keep], np.arange(n)[keep]])
        for j in frac[np.argsort(-inst.p[:, frac].max(axis=0), kind="stable")]:
            sup = np.where(x[:, j] > tol)[0]
            i = int(sup[np.argmin(loads[sup] + inst.p[sup, j])])
            a[j] = i
            loads[i] += inst.p[i, j]
        sol = UrSol(inst, a)
    return sol


def lp_start(inst: UnrelatedInstance) -> Tuple[UrSol, Optional[float], float]:
    t0 = time.perf_counter()
    z, x = lp_solution(inst)
    if x is None:
        return mct_greedy(inst), None, time.perf_counter() - t0
    sol = lp_round(inst, x)
    return sol, z, time.perf_counter() - t0


def lp_duals(inst: UnrelatedInstance):
    """Resolve o LP e devolve (C_LP, x, custo_reduzido) com custo_reduzido[i, j] = y_i p_ij - u_j >= 0 (y = precos das maquinas, soma 1; u = valores das tarefas)."""
    c, A_eq, A_ub = _model(inst)
    nv = len(c)
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(inst.m), A_eq=A_eq, b_eq=np.ones(inst.n), bounds=[(0, 1)] * (nv - 1) + [(0, None)], method="highs")
    if r.status != 0 or r.x is None:
        return None, None, None
    y = -np.asarray(r.ineqlin.marginals)            # (m,)
    u = np.asarray(r.eqlin.marginals)               # (n,)
    rc = y[:, None] * inst.p - u[None, :]
    return float(r.fun), r.x[:-1].reshape(inst.m, inst.n), rc
