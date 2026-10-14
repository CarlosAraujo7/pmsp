"""
pmsp/exact.py
=============
Limite inferior CERTIFICADO e referencia de otimalidade para P||Cmax (Entradas 100-101).

Modelo de fluxo em arcos para o empacotamento associado (Valerio de Carvalho, 1999;
aplicado a P||Cmax por Mrad & Souayah, 2018, e Gharbi & Bamatraf, 2022): dada a
capacidade C, os n tempos cabem em m bins de capacidade C se e somente se existe um
fluxo inteiro no grafo cujos nos sao as somas parciais alcancaveis em [0, C]. A
RELAXACAO LINEAR desse modelo e' um limite inferior valido (e, na pratica, quase
sempre justo para o empacotamento): se o numero minimo fracionario de bins z*(C)
excede m, nenhum escalonamento tem makespan <= C.

    lp_lower_bound(p, m, upper_bound) -> (LB, provado_otimo)

LB e' a menor capacidade C em [LB3, UB] cujo LP e' viavel (z*(C) <= m + tol); como z*(C)
e' nao crescente em C, toda capacidade menor e' inviavel. Se LB == UB, o UB e' ótimo
PROVADO. O LP e' resolvido com pontos interiores do HiGHS (0,3 s contra ~50 s do
simplex dual numa instancia de 10.000 arcos; Entrada 101).

A tolerancia e' conservadora: so' se declara inviavel quando z* > m + 1e-4.
Validado contra forca bruta (tests/test_exact.py) e contra os otimos do iAF
(analysis/audit_entrada101_exact_vs_iaf.py): LB nunca excede o otimo do iAF.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

TOL = 1e-4


def improved_lower_bound(p: Sequence[int], m: int) -> int:
    """max( pmax, ceil(sum/m), p_(m) + p_(m+1) ) com p ordenado de forma decrescente."""
    q = sorted((int(x) for x in p), reverse=True)
    lb = max(q[0], -(-sum(q) // m))
    if len(q) > m:
        lb = max(lb, q[m - 1] + q[m])
    return lb


def _lp_bins(sizes: dict, C: int) -> float:
    """Numero minimo FRACIONARIO de bins de capacidade C (relaxacao linear do arc-flow)."""
    reach = np.zeros(C + 1, dtype=bool)
    reach[0] = True
    for s, d in sizes.items():
        for _ in range(min(d, C // s)):
            sh = np.zeros_like(reach)
            sh[s:] = reach[:C + 1 - s]
            new = reach | sh
            if (new == reach).all():
                break
            reach = new
    nodes = np.flatnonzero(reach).tolist()
    if nodes[-1] != C:
        nodes.append(C)
    idx = {v: k for k, v in enumerate(nodes)}
    arcs = [(idx[u], idx[u + s], s) for s in sizes for u in nodes if u + s <= C and (u + s) in idx]
    arcs += [(idx[a], idx[C], 0) for a in nodes[:-1]]                 # perda direta ao terminal
    nA, nN = len(arcs) + 1, len(nodes)
    z = nA - 1
    rows, cols, vals = [], [], []
    for j, (u, v, _) in enumerate(arcs):
        rows += [v, u]; cols += [j, j]; vals += [1.0, -1.0]
    rows += [idx[0], idx[C]]; cols += [z, z]; vals += [1.0, -1.0]
    sl = sorted(sizes)
    si = {s: k for k, s in enumerate(sl)}
    for j, (_, _, s) in enumerate(arcs):
        if s > 0:
            rows.append(nN + si[s]); cols.append(j); vals.append(1.0)
    A = coo_matrix((vals, (rows, cols)), shape=(nN + len(sl), nA)).tocsr()
    b = np.r_[np.zeros(nN), [sizes[s] for s in sl]]
    c = np.zeros(nA)
    c[z] = 1.0
    r = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs-ipm", options={"time_limit": 90.0})
    if r.status != 0:                       # nao resolveu: nao declarar inviavel
        return 0.0
    return float(r.fun)


def lp_lower_bound(p: Sequence[int], m: int, upper_bound: int) -> Tuple[int, bool, float]:
    """(LB certificado, provado_otimo, tempo_s). `upper_bound` = makespan de uma solucao viavel."""
    t0 = time.perf_counter()
    sizes = dict(Counter(int(x) for x in p))
    lo = improved_lower_bound(p, m)
    hi = int(upper_bound)
    if lo >= hi:
        return hi, True, 0.0
    # menor C em [lo, hi-1] com LP viavel; se nenhum, LB = hi (otimo provado)
    a, b = lo, hi                           # invariante: todo C < a e' inviavel; b e' viavel (UB)
    while a < b:
        mid = (a + b) // 2
        if mid >= hi:
            break
        if _lp_bins(sizes, mid) > m + TOL:
            a = mid + 1                      # inviavel
        else:
            b = mid                          # LP viavel: nao se pode provar inviabilidade
    return a, a >= hi, time.perf_counter() - t0


def optimal_makespan(p: Sequence[int], m: int, upper_bound: int, time_limit: float = 0.0):
    """Compatibilidade: devolve (LB certificado, provado). `time_limit` e' ignorado."""
    lb, proven, t = lp_lower_bound(p, m, upper_bound)
    return lb, proven, t
