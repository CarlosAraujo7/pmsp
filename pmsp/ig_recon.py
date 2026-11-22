"""
pmsp/ig_recon.py (Entrada 284, Ciclo 51)
=========================================
Nucleo compilado do IG de maquinas identicas com UMA escolha a mais, `recon_mode`: 0 = reconstrucao gulosa (as tarefas removidas, em ordem nao crescente de tempo, vao para a maquina de menor carga: reproduz bit a bit
pmsp/ig_numba_agents.ig_agent_core para o agente fixo e para o aleatorio, tests/test_ig_recon_entrada284.py); 1 = reinsercao ALEATORIA (cada tarefa removida vai, na ordem de remocao, para uma maquina sorteada uniformemente entre as m,
inclusive a de origem). Com `arm_ls = 1` aplica-se depois a mesma busca local curta do IG (ate' 2 passadas de troca entre maquinas e reinsercao); com recon_mode = 1 e arm_ls = 1 o laco e' um ILS: perturbacao forte (d tarefas
reinseridas ao acaso), busca local ate' o otimo local curto, aceitacao se nao piora. So' os agentes 0 (acao fixa) e 1 (acao uniforme ao acaso) sao implementados: a ablacao compara ESTRUTURAS, nao aprendizes. Os sorteios usam o gerador
do numba. Retorna (melhor makespan, melhorias, avaliacoes, contagem de uso por acao, atribuicao final).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _destroy, init_state, local_search, move_job, reconstruct


@njit(cache=True)
def reconstruct_random(assign, loads, mj, cnt, p, removed, d):
    """Reinsercao aleatoria: cada tarefa removida vai, na ordem de remocao, para uma maquina uniforme entre as m (a de origem inclusive). Nenhuma maquina candidata e' avaliada (0 avaliacoes)."""
    m = len(loads)
    for i in range(d):
        move_job(assign, loads, mj, cnt, p, removed[i], np.random.randint(0, m))
    return 0


@njit(cache=True)
def ig_agent_core_recon(p, assign0, m, arm_strat, arm_d, arm_ls, T, agent, fixed_arm, seed, order_desc, recon_mode):
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
            ev += local_search(c_assign, c_loads, c_mj, c_cnt, p)
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
    return best, n_imp, ev, used, assign
