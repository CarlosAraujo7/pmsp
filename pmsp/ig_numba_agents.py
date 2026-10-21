"""
pmsp/ig_numba_agents.py (Entrada 180, Ciclo 15)
=================================================
Agentes de selecao de acao COMPILADOS (numba) dentro do IG compilado de pmsp/ig_numba.py: o custo do agente deixa de ser o de um objeto Python (limite do artigo: 'o custo do agente foi medido num agente Python') e
as execucoes ficam rapidas o bastante para repeticoes em massa e para espacos de acao grandes. Mesma semantica do ig_run de pmsp/ig.py:
  estado s_t = (f1 folga relativa ao limite inferior, f2 CV das cargas, f3 progresso, f4 estagnacao) discretizado em 3 bins por feature (limiares 0,10/0,35; 0,15/0,40; 0,33/0,67; 0,25/0,60; janela 50) = 81 estados (exatamente pmsp/bounds.compute_state);
  acao a escolhida pelo agente; candidata = destruir (estrategia e tamanho da acao) + reconstruir + (se a acao pede) busca local de pmsp/ig_numba; recompensa = max(melhora do makespan, 0); aceita se nao piora.
Agentes: 0 fixo (acao dada), 1 aleatorio uniforme, 2 Q-Learning tabular (alpha 0,1; gamma 0,9; epsilon 1,0 -> 0,05 por 0,995; Q inicial 0; como QLearningAgent), 3 UCB1 com recompensa normalizada pelo maximo corrente (como NormalisedUCBSelector; C = raiz de 2),
4 GRADIENTE DE POLITICA: preferencias H[estado, acao], politica softmax, recompensa normalizada pelo maximo corrente, linha de base = media corrente da recompensa normalizada, H[s,k] += lr (r - b) (1[k = a] - pi(k|s)) (gradiente de bandit de Sutton e Barto, condicionado ao estado).
Tabela de acoes: arm_strat[a] in {0,1,2} (estrategia de destruicao), arm_d[a] (numero de tarefas destruidas), arm_ls[a] in {0,1} (aplica a busca local). Os sorteios usam o gerador do numba (equivalencia estatistica com a referencia Python, nao por trajetoria;
a equivalencia EXATA vale entre este nucleo e ig_core quando o agente e fixo ou aleatorio com as mesmas acoes: tests/test_ig_numba_agents.py).
Retorna (melhor makespan, melhorias, avaliacoes (peso 1), contagem de uso por acao).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _argmax, _destroy, init_state, local_search, reconstruct

AGENT_FIXED, AGENT_RANDOM, AGENT_Q, AGENT_UCB, AGENT_PG = 0, 1, 2, 3, 4


@njit(cache=True)
def state_id(loads, lb, t, T, ino, window):
    """Estado de 0 a 80 (pmsp/bounds.compute_state): bins de f1..f4 em base 3."""
    m = len(loads)
    mx = loads[0]
    tot = 0.0
    for k in range(m):
        tot += loads[k]
        if loads[k] > mx:
            mx = loads[k]
    mu = tot / m
    var = 0.0
    for k in range(m):
        var += (loads[k] - mu) ** 2
    sd = np.sqrt(var / m)
    f1 = (mx - lb) / lb if lb > 0 else 0.0
    f2 = sd / mu if mu > 0 else 0.0
    f3 = 1.0 if T <= 0 else min(t / T, 1.0)
    f4 = min(ino / window, 1.0)
    b1 = 0 if f1 < 0.10 else (1 if f1 < 0.35 else 2)
    b2 = 0 if f2 < 0.15 else (1 if f2 < 0.40 else 2)
    b3 = 0 if f3 < 0.33 else (1 if f3 < 0.67 else 2)
    b4 = 0 if f4 < 0.25 else (1 if f4 < 0.60 else 2)
    return b1 * 27 + b2 * 9 + b3 * 3 + b4


@njit(cache=True)
def ig_agent_core(p, assign0, m, arm_strat, arm_d, arm_ls, T, agent, fixed_arm, lr, seed, order_desc, lb):
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
    ino = 0
    Q = np.zeros((81, A))
    eps = 1.0
    ucb_n = np.zeros(A)
    ucb_r = np.zeros(A)
    ucb_t = 0
    rmax = 0.0
    H = np.zeros((81, A))
    pi = np.zeros(A)
    rbar = 0.0
    used = np.zeros(A, dtype=np.int64)
    for t in range(T):
        s = state_id(loads, lb, t, T, ino, 50)
        if agent == 0:
            a = fixed_arm
        elif agent == 1:
            a = np.random.randint(0, A)
        elif agent == 2:
            if np.random.random() < eps:
                a = np.random.randint(0, A)
            else:
                a = 0
                for k in range(1, A):
                    if Q[s, k] > Q[s, a]:
                        a = k
            eps = max(0.05, eps * 0.995)
        elif agent == 3:
            ucb_t += 1
            a = -1
            for k in range(A):
                if ucb_n[k] == 0:
                    a = k
                    break
            if a < 0:
                a = 0
                bestv = -1e18
                for k in range(A):
                    v = ucb_r[k] / ucb_n[k] + 1.4142135623730951 * np.sqrt(np.log(ucb_t) / ucb_n[k])
                    if v > bestv:
                        bestv = v
                        a = k
        else:
            mxh = H[s, 0]
            for k in range(1, A):
                if H[s, k] > mxh:
                    mxh = H[s, k]
            z = 0.0
            for k in range(A):
                pi[k] = np.exp(H[s, k] - mxh)
                z += pi[k]
            u = np.random.random() * z
            a = A - 1
            acc = 0.0
            for k in range(A):
                acc += pi[k]
                if u < acc:
                    a = k
                    break
            for k in range(A):
                pi[k] = pi[k] / z
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
        ev += reconstruct(c_assign, c_loads, c_mj, c_cnt, p, removed, d)
        if arm_ls[a] == 1:
            ev += local_search(c_assign, c_loads, c_mj, c_cnt, p)
        cm = c_loads.max()
        cur = loads.max()
        r = float(cur - cm) if cur > cm else 0.0
        if agent == 2:
            s2 = state_id(c_loads, lb, t + 1, T, ino, 50)
            bn = Q[s2, 0]
            for k in range(1, A):
                if Q[s2, k] > bn:
                    bn = Q[s2, k]
            Q[s, a] += 0.1 * (r + 0.9 * bn - Q[s, a])
        elif agent == 3:
            if r > rmax:
                rmax = r
            ucb_n[a] += 1.0
            ucb_r[a] += (r / rmax if rmax > 0 else 0.0)
        elif agent == 4:
            if r > rmax:
                rmax = r
            rn = r / rmax if rmax > 0 else 0.0
            for k in range(A):
                ind = 1.0 if k == a else 0.0
                H[s, k] += lr * (rn - rbar) * (ind - pi[k])
            rbar += (rn - rbar) / (t + 1.0)
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
                ino = 0
            else:
                ino += 1
        else:
            ino += 1
    return best, n_imp, ev, used
