"""
pmsp/closedloop_numba.py (Entrada 249, Ciclo 27)
=================================================
Oraculo de MALHA FECHADA (dependente do estado) para o IG de P||Cmax e de R||Cmax: a cada iteracao, a partir da solucao ATUAL, cada acao e' avaliada por `m_roll` execucoes independentes (destruicao + reconstrucao + busca local, a partir do mesmo estado; a solucao atual nao muda); a acao escolhida e' a de maior melhora media esperada do makespan (a mesma recompensa que os aprendizes recebem, max(melhora, 0)), com sorteio entre as empatadas, e ela e' EXECUTADA com uma execucao nova e independente (a candidata aceita nao e' a da avaliacao: nao ha' selecao sobre o ruido da execucao). Aceitacao como no IG: aceita se o makespan nao piora. NAO e' justo no orcamento (cada iteracao custa A * m_roll avaliacoes a mais; 'ev_roll' conta o custo extra): e' um LIMITE para politicas que decidem pelo estado, comparavel ao oraculo estatico (a melhor acao FIXA por instancia).
oracle = 0: controle com o mesmo caminho de codigo e sem avaliacoes (a acao e' sorteada: reproduz o aleatorio em distribuicao).
Retorna (melhor makespan, avaliacoes da trajetoria, avaliacoes das execucoes de avaliacao, uso por acao).
pc_lookahead (p: vetor) e ur_lookahead (P: matriz n x m) tem o mesmo texto, com as rotinas de pmsp/ig_numba.py e pmsp/ur_numba.py.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _destroy
from .ig_numba import init_state as pc_init
from .ig_numba import local_search as pc_ls
from .ig_numba import reconstruct as pc_rec
from .ur_numba import init_state as ur_init
from .ur_numba import local_search as ur_ls
from .ur_numba import reconstruct as ur_rec


@njit(cache=True)
def ur_lookahead(P, assign0, m, arm_strat, arm_d, arm_ls, T, m_roll, seed, order_desc, oracle):
    np.random.seed(seed)
    A = len(arm_strat)
    n = P.shape[0]
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = ur_init(assign, P, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    best = loads.max()
    ev = 0.0
    ev_roll = 0.0
    gains = np.zeros(A)
    used = np.zeros(A, dtype=np.int64)
    for t in range(T):
        cur = loads.max()
        for a in range(A):
            tot = 0.0
            if oracle == 1:
                for q in range(m_roll):
                    for k in range(m):
                        c_loads[k] = loads[k]
                        c_cnt[k] = cnt[k]
                        for z in range(cnt[k]):
                            c_mj[k, z] = mj[k, z]
                    for j in range(n):
                        c_assign[j] = assign[j]
                    d = arm_d[a]
                    _destroy(arm_strat[a], c_assign, P, d, order_desc, perm, removed, m)
                    ev_roll += ur_rec(c_assign, c_loads, c_mj, c_cnt, P, removed, d)
                    if arm_ls[a] == 1:
                        ev_roll += ur_ls(c_assign, c_loads, c_mj, c_cnt, P)
                    cm = c_loads.max()
                    if cur > cm:
                        tot += float(cur - cm)
            gains[a] = tot / m_roll
        gmax = gains.max()
        nt = 0
        for k in range(A):
            if gains[k] == gmax:
                nt += 1
        pick = np.random.randint(0, nt)
        a = 0
        for k in range(A):
            if gains[k] == gmax:
                if pick == 0:
                    a = k
                    break
                pick -= 1
        used[a] += 1
        for k in range(m):
            c_loads[k] = loads[k]
            c_cnt[k] = cnt[k]
            for z in range(cnt[k]):
                c_mj[k, z] = mj[k, z]
        for j in range(n):
            c_assign[j] = assign[j]
        d = arm_d[a]
        _destroy(arm_strat[a], c_assign, P, d, order_desc, perm, removed, m)
        ev += ur_rec(c_assign, c_loads, c_mj, c_cnt, P, removed, d)
        if arm_ls[a] == 1:
            ev += ur_ls(c_assign, c_loads, c_mj, c_cnt, P)
        cm = c_loads.max()
        if cm <= cur:
            for k in range(m):
                loads[k] = c_loads[k]
                cnt[k] = c_cnt[k]
                for z in range(c_cnt[k]):
                    mj[k, z] = c_mj[k, z]
            for j in range(n):
                assign[j] = c_assign[j]
            if cm < best:
                best = cm
    return best, ev, ev_roll, used


@njit(cache=True)
def pc_lookahead(p, assign0, m, arm_strat, arm_d, arm_ls, T, m_roll, seed, order_desc, oracle):
    np.random.seed(seed)
    A = len(arm_strat)
    n = len(p)
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = pc_init(assign, p, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    best = loads.max()
    ev = 0.0
    ev_roll = 0.0
    gains = np.zeros(A)
    used = np.zeros(A, dtype=np.int64)
    for t in range(T):
        cur = loads.max()
        for a in range(A):
            tot = 0.0
            if oracle == 1:
                for q in range(m_roll):
                    for k in range(m):
                        c_loads[k] = loads[k]
                        c_cnt[k] = cnt[k]
                        for z in range(cnt[k]):
                            c_mj[k, z] = mj[k, z]
                    for j in range(n):
                        c_assign[j] = assign[j]
                    d = arm_d[a]
                    _destroy(arm_strat[a], c_assign, p, d, order_desc, perm, removed, m)
                    ev_roll += pc_rec(c_assign, c_loads, c_mj, c_cnt, p, removed, d)
                    if arm_ls[a] == 1:
                        ev_roll += pc_ls(c_assign, c_loads, c_mj, c_cnt, p)
                    cm = c_loads.max()
                    if cur > cm:
                        tot += float(cur - cm)
            gains[a] = tot / m_roll
        gmax = gains.max()
        nt = 0
        for k in range(A):
            if gains[k] == gmax:
                nt += 1
        pick = np.random.randint(0, nt)
        a = 0
        for k in range(A):
            if gains[k] == gmax:
                if pick == 0:
                    a = k
                    break
                pick -= 1
        used[a] += 1
        for k in range(m):
            c_loads[k] = loads[k]
            c_cnt[k] = cnt[k]
            for z in range(cnt[k]):
                c_mj[k, z] = mj[k, z]
        for j in range(n):
            c_assign[j] = assign[j]
        d = arm_d[a]
        _destroy(arm_strat[a], c_assign, p, d, order_desc, perm, removed, m)
        ev += pc_rec(c_assign, c_loads, c_mj, c_cnt, p, removed, d)
        if arm_ls[a] == 1:
            ev += pc_ls(c_assign, c_loads, c_mj, c_cnt, p)
        cm = c_loads.max()
        if cm <= cur:
            for k in range(m):
                loads[k] = c_loads[k]
                cnt[k] = c_cnt[k]
                for z in range(c_cnt[k]):
                    mj[k, z] = c_mj[k, z]
            for j in range(n):
                assign[j] = c_assign[j]
            if cm < best:
                best = cm
    return best, ev, ev_roll, used


# ----------------------------------------------------------------------------------------------------------------------
# Oraculo de ROLLOUT (melhoria de politica sobre a acao validada; nao miope): em cada decisao (a cada K_dec iteracoes), cada acao a e'
# avaliada por `m_roll` execucoes independentes a partir da solucao atual: a acao a por K_dec iteracoes e, depois, a acao base (a validada)
# por H_tail iteracoes (H_tail < 0: ate' o fim da execucao de T iteracoes); o escore e' a melhora media do makespan ao fim da execucao. Escolhe-se a de maior escore (sorteio entre empates),
# aplicada por K_dec iteracoes na trajetoria real. Como a acao base e' candidata, a politica melhora a base em esperanca (teorema da
# melhoria de politica, com estimativas exatas). Nao e' justa no orcamento; ev_roll conta as iteracoes das avaliacoes.
@njit(cache=True)
def ur_iter(a, assign, loads, mj, cnt, c_assign, c_loads, c_mj, c_cnt, P, arm_strat, arm_d, arm_ls, order_desc, perm, removed, m):
    n = P.shape[0]
    for k in range(m):
        c_loads[k] = loads[k]
        c_cnt[k] = cnt[k]
        for z in range(cnt[k]):
            c_mj[k, z] = mj[k, z]
    for j in range(n):
        c_assign[j] = assign[j]
    d = arm_d[a]
    _destroy(arm_strat[a], c_assign, P, d, order_desc, perm, removed, m)
    ev = ur_rec(c_assign, c_loads, c_mj, c_cnt, P, removed, d)
    if arm_ls[a] == 1:
        ev += ur_ls(c_assign, c_loads, c_mj, c_cnt, P)
    if c_loads.max() <= loads.max():
        for k in range(m):
            loads[k] = c_loads[k]
            cnt[k] = c_cnt[k]
            for z in range(c_cnt[k]):
                mj[k, z] = c_mj[k, z]
        for j in range(n):
            assign[j] = c_assign[j]
    return ev


@njit(cache=True)
def pc_iter(a, assign, loads, mj, cnt, c_assign, c_loads, c_mj, c_cnt, p, arm_strat, arm_d, arm_ls, order_desc, perm, removed, m):
    n = len(p)
    for k in range(m):
        c_loads[k] = loads[k]
        c_cnt[k] = cnt[k]
        for z in range(cnt[k]):
            c_mj[k, z] = mj[k, z]
    for j in range(n):
        c_assign[j] = assign[j]
    d = arm_d[a]
    _destroy(arm_strat[a], c_assign, p, d, order_desc, perm, removed, m)
    ev = pc_rec(c_assign, c_loads, c_mj, c_cnt, p, removed, d)
    if arm_ls[a] == 1:
        ev += pc_ls(c_assign, c_loads, c_mj, c_cnt, p)
    if c_loads.max() <= loads.max():
        for k in range(m):
            loads[k] = c_loads[k]
            cnt[k] = c_cnt[k]
            for z in range(c_cnt[k]):
                mj[k, z] = c_mj[k, z]
        for j in range(n):
            assign[j] = c_assign[j]
    return ev


@njit(cache=True)
def ur_rollout(P, assign0, m, arm_strat, arm_d, arm_ls, T, m_roll, k_dec, h_tail, base, seed, order_desc):
    np.random.seed(seed)
    A = len(arm_strat)
    n = P.shape[0]
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = ur_init(assign, P, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    r_assign = assign.copy()
    r_loads = loads.copy()
    r_mj = mj.copy()
    r_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    ev = 0.0
    ev_roll = 0.0
    sc = np.zeros(A)
    used = np.zeros(A, dtype=np.int64)
    t = 0
    while t < T:
        kk = min(k_dec, T - t)
        ht = h_tail if h_tail >= 0 else T - t - kk
        cur = loads.max()
        for a in range(A):
            tot = 0.0
            for q in range(m_roll):
                for k in range(m):
                    r_loads[k] = loads[k]
                    r_cnt[k] = cnt[k]
                    for z in range(cnt[k]):
                        r_mj[k, z] = mj[k, z]
                for j in range(n):
                    r_assign[j] = assign[j]
                for i in range(kk):
                    ev_roll += ur_iter(a, r_assign, r_loads, r_mj, r_cnt, c_assign, c_loads, c_mj, c_cnt, P, arm_strat, arm_d, arm_ls, order_desc, perm, removed, m)
                for i in range(ht):
                    ev_roll += ur_iter(base, r_assign, r_loads, r_mj, r_cnt, c_assign, c_loads, c_mj, c_cnt, P, arm_strat, arm_d, arm_ls, order_desc, perm, removed, m)
                tot += float(cur - r_loads.max())
            sc[a] = tot / m_roll
        smax = sc.max()
        nt = 0
        for k in range(A):
            if sc[k] == smax:
                nt += 1
        pick = np.random.randint(0, nt)
        a = 0
        for k in range(A):
            if sc[k] == smax:
                if pick == 0:
                    a = k
                    break
                pick -= 1
        used[a] += kk
        for i in range(kk):
            ev += ur_iter(a, assign, loads, mj, cnt, c_assign, c_loads, c_mj, c_cnt, P, arm_strat, arm_d, arm_ls, order_desc, perm, removed, m)
        t += kk
    return int(loads.max()), ev, ev_roll, used


@njit(cache=True)
def pc_rollout(p, assign0, m, arm_strat, arm_d, arm_ls, T, m_roll, k_dec, h_tail, base, seed, order_desc):
    np.random.seed(seed)
    A = len(arm_strat)
    n = len(p)
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = pc_init(assign, p, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    r_assign = assign.copy()
    r_loads = loads.copy()
    r_mj = mj.copy()
    r_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    ev = 0.0
    ev_roll = 0.0
    sc = np.zeros(A)
    used = np.zeros(A, dtype=np.int64)
    t = 0
    while t < T:
        kk = min(k_dec, T - t)
        ht = h_tail if h_tail >= 0 else T - t - kk
        cur = loads.max()
        for a in range(A):
            tot = 0.0
            for q in range(m_roll):
                for k in range(m):
                    r_loads[k] = loads[k]
                    r_cnt[k] = cnt[k]
                    for z in range(cnt[k]):
                        r_mj[k, z] = mj[k, z]
                for j in range(n):
                    r_assign[j] = assign[j]
                for i in range(kk):
                    ev_roll += pc_iter(a, r_assign, r_loads, r_mj, r_cnt, c_assign, c_loads, c_mj, c_cnt, p, arm_strat, arm_d, arm_ls, order_desc, perm, removed, m)
                for i in range(ht):
                    ev_roll += pc_iter(base, r_assign, r_loads, r_mj, r_cnt, c_assign, c_loads, c_mj, c_cnt, p, arm_strat, arm_d, arm_ls, order_desc, perm, removed, m)
                tot += float(cur - r_loads.max())
            sc[a] = tot / m_roll
        smax = sc.max()
        nt = 0
        for k in range(A):
            if sc[k] == smax:
                nt += 1
        pick = np.random.randint(0, nt)
        a = 0
        for k in range(A):
            if sc[k] == smax:
                if pick == 0:
                    a = k
                    break
                pick -= 1
        used[a] += kk
        for i in range(kk):
            ev += pc_iter(a, assign, loads, mj, cnt, c_assign, c_loads, c_mj, c_cnt, p, arm_strat, arm_d, arm_ls, order_desc, perm, removed, m)
        t += kk
    return int(loads.max()), ev, ev_roll, used
