"""
pmsp/instance_ur.py (Entrada 234, Ciclo 22)
=============================================
Instancias de MAQUINAS NAO RELACIONADAS (R||Cmax) para o hospedeiro novo: tempos P[j, k] inteiros independentes U[1, 100] (nao correlacionados), partida gulosa e limite inferior.
  * random_ur(n, m, seed): P (n x m, int64).
  * lower_bound_ur(P): max( max_j min_k P[j, k], ceil( sum_j min_k P[j, k] / m ) ) (limite simples; so' alimenta o estado do Q-Learning e a lacuna relativa, nao ha' alegacao de otimalidade).
  * constructive_ur(P): tarefas em ordem decrescente de min_k P[j, k] (estavel), cada uma na maquina de menor termino (carga + P[j, k]); devolve (atribuicao, order_desc), onde order_desc e' a mesma ordem (a passada a estrategia 'as d maiores').
  * makespan_ur(P, assign).
"""
from __future__ import annotations

import numpy as np


def random_ur(n, m, seed, low=1, high=100):
    rng = np.random.default_rng(seed)
    return rng.integers(low, high + 1, size=(n, m)).astype(np.int64)


def lower_bound_ur(P):
    mn = P.min(axis=1)
    m = P.shape[1]
    return int(max(mn.max(), -(-int(mn.sum()) // m)))


def constructive_ur(P):
    n, m = P.shape
    mn = P.min(axis=1)
    order = np.argsort(-mn, kind="stable")
    loads = np.zeros(m, dtype=np.int64)
    assign = np.zeros(n, dtype=np.int64)
    for j in order:
        k = int(np.argmin(loads + P[j]))
        assign[j] = k
        loads[k] += P[j, k]
    return assign, order.astype(np.int64)


def makespan_ur(P, assign):
    m = P.shape[1]
    loads = np.zeros(m, dtype=np.int64)
    for j in range(P.shape[0]):
        loads[assign[j]] += P[j, assign[j]]
    return int(loads.max())
