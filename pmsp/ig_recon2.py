"""
pmsp/ig_recon2.py (Entrada 285, Ciclo 52)
==========================================
Extensao de pmsp/ig_recon.py (Entrada 284, congelado) com uma DESCIDA COMPLETA: `ls_mode` = 1 aplica a descida curta do IG (no maximo duas melhorias: reproduz bit a bit ig_recon.ig_agent_core_recon, tests/test_ig_recon2_entrada285.py);
`ls_mode` = 2 repete a melhor troca entre maquinas (e, quando nao melhora, a melhor reinsercao) a partir da maquina gargalo ate' que nenhuma das duas melhore o makespan (otimo local das duas vizinhancas) ou ate' `max_moves`
movimentos. Cada movimento aceito reduz estritamente o makespan (inteiro), de modo que a descida termina; o tampo `max_moves` e' so' uma salvaguarda e os movimentos e os tampos atingidos sao contados. Com recon_mode = 1 e ls_mode = 2 o laco e' o
ILS classico para P||Cmax: perturbacao (d tarefas reinseridas ao acaso), descida ate' o otimo local, aceitacao se nao piora. Retorna (melhor makespan, melhorias, avaliacoes, uso por acao, atribuicao final, movimentos, descidas truncadas, descidas).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _destroy, init_state, local_search, op1_best, op2_best, reconstruct
from .ig_recon import reconstruct_random


@njit(cache=True)
def local_search_full(assign, loads, mj, cnt, p, max_moves):
    """Descida ate' o otimo local (troca; senao reinsercao; a partir da maquina gargalo). Retorna (avaliacoes, movimentos, truncada)."""
    ev = 0
    moves = 0
    for _ in range(max_moves):
        r1, e1 = op1_best(assign, loads, mj, cnt, p)
        ev += e1
        if r1 <= 0:
            r2, e2 = op2_best(assign, loads, mj, cnt, p)
            ev += e2
            if r2 <= 0:
                return ev, moves, 0
        moves += 1
    return ev, moves, 1


@njit(cache=True)
def ig_agent_core_recon2(p, assign0, m, arm_strat, arm_d, arm_ls, T, agent, fixed_arm, seed, order_desc, recon_mode, ls_mode, max_moves):
    np.random.seed(seed)
    A = len(arm_strat)
    n = len(p)
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = init_state(assign, p, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    best = loads.max()
    n_imp = 0
    ev = 0.0
    used = np.zeros(A, dtype=np.int64)
    moves = 0
    capped = 0
    n_desc = 0
    for t in range(T):
        if agent == 0:
            a = fixed_arm
        else:
            a = np.random.randint(0, A)
        used[a] += 1
        for k in range(m):                      # copia da solucao corrente
            c_loads[k] = loads[k]
            c_cnt[k] = cnt[k]
            for q in range(cnt[k]):
                c_mj[k, q] = mj[k, q]
        for j in range(n):
            c_assign[j] = assign[j]
        d = arm_d[a]
        _destroy(arm_strat[a], c_assign, p, d, order_desc, perm, removed, m)
        if recon_mode == 0:
            ev += reconstruct(c_assign, c_loads, c_mj, c_cnt, p, removed, d)
        else:
            ev += reconstruct_random(c_assign, c_loads, c_mj, c_cnt, p, removed, d)
        if arm_ls[a] == 1:
            if ls_mode == 1:
                ev += local_search(c_assign, c_loads, c_mj, c_cnt, p)
            else:
                e_, mv_, cp_ = local_search_full(c_assign, c_loads, c_mj, c_cnt, p, max_moves)
                ev += e_
                moves += mv_
                capped += cp_
                n_desc += 1
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
                n_imp += 1
    return best, n_imp, ev, used, assign, moves, capped, n_desc
