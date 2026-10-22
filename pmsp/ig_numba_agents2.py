"""
pmsp/ig_numba_agents2.py (Entrada 184, Ciclo 16)
==================================================
Extensao COMPILADA de pmsp/ig_numba_agents.py (que fica congelado, com os hashes de Entrada 180): (i) tres seletores classicos de operadores adaptativos (AOS) que faltavam,
(ii) um CONTROLE POSITIVO: um bonus de recompensa plantado numa acao conhecida, para verificar que os aprendizes e o teste detectam estrutura quando ela existe.
Mesma semantica do ig_agent_core (estado de 81 bins, candidata = destruir + reconstruir + busca local opcional, recompensa = max(melhora, 0), aceita se nao piora); os agentes 0 a 4 consomem os sorteios na MESMA ordem,
logo reproduzem ig_agent_core EXATAMENTE com a mesma semente (tests/test_ig_numba_agents2.py).
Agentes novos (nao condicionados ao estado, como na literatura de AOS; recompensa normalizada pelo maximo corrente rn = r / rmax):
  5 PROBABILITY MATCHING (Goldberg 1990; Thierens 2005): q[a] <- (1 - alfa) q[a] + alfa rn (q inicial 1); p[a] = pmin + (1 - A pmin) q[a] / soma(q); pmin = 0,1 / A. alfa = lr.
  6 ADAPTIVE PURSUIT (Thierens 2005): q como no PM; a* = argmax q; p[a*] += beta (pmax - p[a*]), p[outros] += beta (pmin - p[a]); pmax = 1 - (A - 1) pmin. alfa = lr, beta = lr2.
  7 ROLETA TIPO ALNS (Pisinger e Ropke 2007): pesos w[a] (inicial 1), selecao proporcional a w; ao fim de cada segmento de 100 iteracoes w[a] <- (1 - rho) w[a] + rho (pontos[a] / usos[a]) para as acoes usadas; pontos 33 se a candidata melhora a corrente (r > 0),
     13 se e' aceita sem melhorar (empate), 0 caso contrario (com aceitacao gulosa o 'melhor que a corrente' e o 'novo melhor' coincidem). rho = lr.
Controle positivo (plant_mode): 0 nenhum; 1 a acao plantada e' plant_arm (a mesma em todo estado); 2 a acao plantada depende do estado (s mod A). Quando a acao escolhida e' a plantada, r += plant_bonus (unidades de makespan) ANTES da atualizacao do agente;
a solucao NAO muda (a aceitacao usa o makespan, nao r): o teste mede se o aprendiz CAPTURA a estrutura de recompensa. Retorna (melhor makespan, melhorias, avaliacoes, uso por acao, acertos da acao plantada na segunda metade).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _destroy, init_state, local_search, reconstruct
from .ig_numba_agents import state_id

AGENT_FIXED, AGENT_RANDOM, AGENT_Q, AGENT_UCB, AGENT_PG, AGENT_PM, AGENT_AP, AGENT_ALNS = 0, 1, 2, 3, 4, 5, 6, 7


@njit(cache=True)
def _roulette(w, A, total):
    u = np.random.random() * total
    acc = 0.0
    for k in range(A):
        acc += w[k]
        if u < acc:
            return k
    return A - 1


@njit(cache=True)
def ig_agent_core2(p, assign0, m, arm_strat, arm_d, arm_ls, T, agent, fixed_arm, lr, lr2, seed, order_desc, lb, plant_mode, plant_arm, plant_bonus):
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
    hit2 = 0
    pmin = 0.1 / A
    pmax = 1.0 - (A - 1) * pmin
    qa = np.ones(A)
    pa = np.full(A, 1.0 / A)
    w = np.ones(A)
    seg_pts = np.zeros(A)
    seg_cnt = np.zeros(A)
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
        elif agent == 4:
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
        elif agent == 5:
            sq = 0.0
            for k in range(A):
                sq += qa[k]
            for k in range(A):
                pa[k] = pmin + (1.0 - A * pmin) * qa[k] / sq
            a = _roulette(pa, A, 1.0)
        elif agent == 6:
            a = _roulette(pa, A, 1.0)
        else:
            tw = 0.0
            for k in range(A):
                tw += w[k]
            a = _roulette(w, A, tw)
        used[a] += 1
        for k in range(m):
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
        if plant_mode > 0:
            pl = plant_arm if plant_mode == 1 else s % A
            if a == pl:
                r += plant_bonus
                if t >= T // 2:
                    hit2 += 1
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
        elif agent == 5 or agent == 6:
            if r > rmax:
                rmax = r
            rn = r / rmax if rmax > 0 else 0.0
            qa[a] = (1.0 - lr) * qa[a] + lr * rn
            if agent == 6:
                bs = 0
                for k in range(1, A):
                    if qa[k] > qa[bs]:
                        bs = k
                for k in range(A):
                    tgt = pmax if k == bs else pmin
                    pa[k] += lr2 * (tgt - pa[k])
        elif agent == 7:
            seg_cnt[a] += 1.0
            if r > 0:
                seg_pts[a] += 33.0
            elif cm == cur:
                seg_pts[a] += 13.0
            if (t + 1) % 100 == 0:
                for k in range(A):
                    if seg_cnt[k] > 0:
                        w[k] = (1.0 - lr) * w[k] + lr * seg_pts[k] / seg_cnt[k]
                    seg_pts[k] = 0.0
                    seg_cnt[k] = 0.0
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
    return best, n_imp, ev, used, hit2
