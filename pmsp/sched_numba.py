"""
pmsp/sched_numba.py (Entrada 238, Ciclo 22)
=============================================
IG COMPILADO com um CRONOGRAMA de acoes: a acao da iteracao t e' seq[t] (um vetor de indices da tabela de acoes), isto e', uma politica ABERTA no tempo (varia com a iteracao, nao com o estado). E' o oraculo/competidor 'dinamico' possivel sem olhar o resultado antes de escolher:
a melhor sequencia de acoes por classe (escolhida em instancias separadas) limita o que ADAPTAR AO LONGO DA EXECUCAO pode acrescentar a' acao estatica validada (a revisao cega #7 pediu um oraculo dinamico; a folga de acao unica nao limita quem mistura acoes).
Duas versoes do mesmo laco: ig_sched_core (P||Cmax; tempos p[j]) e ur_sched_core (R||Cmax; tempos P[j, k]); o corpo e' o de ig_agent_core4 (pmsp/ig_numba_agents4.py) com a escolha da acao trocada por seq[t]: com uma sequencia constante reproduz o agente FIXO EXATAMENTE com a mesma semente (tests/test_sched_numba.py).
Retorna (melhor makespan, avaliacoes).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _destroy
from .ig_numba import init_state as init_p
from .ig_numba import local_search as ls_p
from .ig_numba import reconstruct as rec_p
from .ur_numba import init_state as init_u
from .ur_numba import local_search as ls_u
from .ur_numba import reconstruct as rec_u


@njit(cache=True)
def ig_sched_core(p, assign0, m, arm_strat, arm_d, arm_ls, seq, seed, order_desc):
    np.random.seed(seed)
    n = len(p)
    T = len(seq)
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = init_p(assign, p, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    best = loads.max()
    ev = 0.0
    for t in range(T):
        a = seq[t]
        for k in range(m):
            c_loads[k] = loads[k]
            c_cnt[k] = cnt[k]
            for q in range(cnt[k]):
                c_mj[k, q] = mj[k, q]
        for j in range(n):
            c_assign[j] = assign[j]
        d = arm_d[a]
        _destroy(arm_strat[a], c_assign, p, d, order_desc, perm, removed, m)
        ev += rec_p(c_assign, c_loads, c_mj, c_cnt, p, removed, d)
        if arm_ls[a] == 1:
            ev += ls_p(c_assign, c_loads, c_mj, c_cnt, p)
        cm = c_loads.max()
        cur = loads.max()
        if cm <= cur:
            for k in range(m):
                loads[k] = c_loads[k]
                cnt[k] = c_cnt[k]
                for q in range(c_cnt[k]):
                    mj[k, q] = c_mj[k, q]
            for j in range(n):
                assign[j] = c_assign[j]
            if cm < best:
                best = cm
    return best, ev


@njit(cache=True)
def ur_sched_core(P, assign0, m, arm_strat, arm_d, arm_ls, seq, seed, order_desc):
    np.random.seed(seed)
    n = P.shape[0]
    T = len(seq)
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = init_u(assign, P, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    best = loads.max()
    ev = 0.0
    for t in range(T):
        a = seq[t]
        for k in range(m):
            c_loads[k] = loads[k]
            c_cnt[k] = cnt[k]
            for q in range(cnt[k]):
                c_mj[k, q] = mj[k, q]
        for j in range(n):
            c_assign[j] = assign[j]
        d = arm_d[a]
        _destroy(arm_strat[a], c_assign, P, d, order_desc, perm, removed, m)
        ev += rec_u(c_assign, c_loads, c_mj, c_cnt, P, removed, d)
        if arm_ls[a] == 1:
            ev += ls_u(c_assign, c_loads, c_mj, c_cnt, P)
        cm = c_loads.max()
        cur = loads.max()
        if cm <= cur:
            for k in range(m):
                loads[k] = c_loads[k]
                cnt[k] = c_cnt[k]
                for q in range(c_cnt[k]):
                    mj[k, q] = c_mj[k, q]
            for j in range(n):
                assign[j] = c_assign[j]
            if cm < best:
                best = cm
    return best, ev
