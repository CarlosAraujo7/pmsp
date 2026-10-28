"""
pmsp/ga_numba.py (Entrada 219, Ciclo 19, frente F)
====================================================
HOSPEDEIRO POPULACIONAL para P||Cmax: um algoritmo genetico de estado estacionario compilado (numba) cujo operador de variacao e' escolhido por um seletor -- os mesmos onze agentes de pmsp/ig_numba_agents4.py (o codigo de selecao e de atualizacao dos agentes e' o MESMO texto, gerado de ig_numba_agents4.py por scratch_mkga.py).
Populacao de mu = 20 individuos (atribuicoes tarefa -> maquina): o 0 e' a solucao LPT e os demais sao copias do LPT com ~i n / mu tarefas realocadas ao acaso. A cada iteracao (uma avaliacao de filho): torneio binario escolhe dois pais; a ACAO a = (cruzamento, mutacao, intensidade) gera um filho; o filho substitui o pior individuo se nao for pior que ele.
Tabelas de acao (as mesmas posicoes de arm_strat, arm_d, arm_ls do IG): arm_strat = cruzamento (0 uniforme, 1 por maquinas: herda de um pai as tarefas de um subconjunto aleatorio de maquinas e do outro as demais, 2 clone do primeiro pai), arm_d = mutacao (0 realoca r tarefas ao acaso, 1 gulosa: r vezes move uma tarefa
aleatoria da maquina mais carregada para a menos carregada, 2 troca r pares de tarefas entre maquinas), arm_ls = intensidade (0: r = 1 + n // 50; 1: r = 1 + n // 10). Recompensa (a mesma definicao do IG): max(melhor pai - filho, 0) em unidades de makespan. Estado de 81 bins: o do melhor individuo (pmsp/ig_numba_agents.state_id).
Retorna (melhor makespan, melhorias do melhor, avaliacoes, uso por acao, acertos da acao plantada na segunda metade, melhor atribuicao).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _destroy, init_state, local_search, reconstruct
from .ig_numba_agents import state_id
from .ig_numba_agents2 import _roulette

AGENT_FIXED, AGENT_RANDOM, AGENT_Q, AGENT_UCB, AGENT_PG, AGENT_PM, AGENT_AP, AGENT_ALNS = 0, 1, 2, 3, 4, 5, 6, 7
AGENT_TS, AGENT_DMAB, AGENT_RACE = 8, 9, 10
HP_DEFAULT = np.array([0.1, 0.9, 0.995, 0.05, 0.0], dtype=np.float64)


@njit(cache=True)
def _loads_of(assign, p, m, out):
    for k in range(m):
        out[k] = 0
    for j in range(len(p)):
        out[assign[j]] += p[j]


@njit(cache=True)
def _variate(xo, mu_kind, inten, a1, a2, p, m, child, ld):
    n = len(p)
    if xo == 0:
        for j in range(n):
            child[j] = a1[j] if np.random.random() < 0.5 else a2[j]
    elif xo == 1:
        for j in range(n):
            child[j] = a1[j]
        take = np.random.random(m) < 0.5
        for j in range(n):
            if not take[a1[j]]:
                child[j] = a2[j]
    else:
        for j in range(n):
            child[j] = a1[j]
    r = 1 + n // 50 if inten == 0 else 1 + n // 10
    _loads_of(child, p, m, ld)
    for _ in range(r):
        if mu_kind == 0:
            j = np.random.randint(0, n)
            k = np.random.randint(0, m)
            ld[child[j]] -= p[j]
            child[j] = k
            ld[k] += p[j]
        elif mu_kind == 1:
            kmax = 0
            kmin = 0
            for k in range(1, m):
                if ld[k] > ld[kmax]:
                    kmax = k
                if ld[k] < ld[kmin]:
                    kmin = k
            if kmax != kmin:
                cnt = 0
                for j in range(n):
                    if child[j] == kmax:
                        cnt += 1
                if cnt > 0:
                    pick = np.random.randint(0, cnt)
                    c = 0
                    for j in range(n):
                        if child[j] == kmax:
                            if c == pick:
                                child[j] = kmin
                                ld[kmax] -= p[j]
                                ld[kmin] += p[j]
                                break
                            c += 1
        else:
            j1 = np.random.randint(0, n)
            j2 = np.random.randint(0, n)
            k1 = child[j1]
            k2 = child[j2]
            if k1 != k2:
                child[j1] = k2
                child[j2] = k1
                ld[k1] += p[j2] - p[j1]
                ld[k2] += p[j1] - p[j2]
    return 0


@njit(cache=True)
def ga_agent_core(p, assign0, m, arm_strat, arm_d, arm_ls, T, agent, fixed_arm, lr, lr2, seed, order_desc, lb, plant_mode, plant_arm, plant_bonus, hp, rmode):
    np.random.seed(seed)
    A = len(arm_strat)
    n = len(p)
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = init_state(assign, p, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    MU = 20
    pop = np.empty((MU, n), dtype=np.int64)
    fit = np.empty(MU)
    tmpl = np.empty(m, dtype=np.int64)
    for i in range(MU):
        for j in range(n):
            pop[i, j] = assign[j]
        kk = (i * n) // MU
        for _ in range(kk):
            jj = np.random.randint(0, n)
            pop[i, jj] = np.random.randint(0, m)
        _loads_of(pop[i], p, m, tmpl)
        fit[i] = tmpl.max()
    ib = 0
    for i in range(1, MU):
        if fit[i] < fit[ib]:
            ib = i
    for j in range(n):
        assign[j] = pop[ib, j]
    _loads_of(assign, p, m, loads)
    best = loads.max()
    child = np.empty(n, dtype=np.int64)
    ld = np.empty(m, dtype=np.int64)
    n_imp = 0
    ev = 0.0
    ino = 0
    Q = np.zeros((81, A))
    eps = 1.0
    ucb_n = np.zeros(A)
    ucb_r = np.zeros(A)
    ucb_t = 0
    rmax = 0.0
    rmax_q = 0.0
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
    # Thompson
    ts_s = np.zeros(A)
    ts_f = np.zeros(A)
    # DMAB
    dm_n = np.zeros(A)
    dm_s = np.zeros(A)
    dm_t = 0
    ph_g = np.zeros(A)
    ph_max = np.zeros(A)
    # corrida
    alive = np.ones(A, dtype=np.int64)
    n_alive = A
    rc_sum = np.zeros(A)
    rc_n = np.zeros(A)
    rounds = 0
    ra = A
    while ra > 1:
        ra = (ra + 1) // 2
        rounds += 1
    rc_round = 0
    rc_ptr = 0
    rc_pulled = 0
    rc_budget = lr * T / max(rounds, 1)
    rc_quota = 0
    if n_alive > 0:
        rc_quota = max(1, int(rc_budget / n_alive))
    rc_winner = -1
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
            eps = max(hp[3], eps * hp[2])
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
        elif agent == 7:
            tw = 0.0
            for k in range(A):
                tw += w[k]
            a = _roulette(w, A, tw)
        elif agent == 8:
            a = 0
            bv = -1.0
            for k in range(A):
                g1 = np.random.gamma(1.0 + ts_s[k], 1.0)
                g2 = np.random.gamma(1.0 + ts_f[k], 1.0)
                th = g1 / (g1 + g2)
                if th > bv:
                    bv = th
                    a = k
        elif agent == 9:
            dm_t += 1
            a = -1
            for k in range(A):
                if dm_n[k] == 0:
                    a = k
                    break
            if a < 0:
                a = 0
                bestv = -1e18
                for k in range(A):
                    v = dm_s[k] / dm_n[k] + 1.4142135623730951 * np.sqrt(np.log(dm_t) / dm_n[k])
                    if v > bestv:
                        bestv = v
                        a = k
        else:
            if rc_winner >= 0:
                a = rc_winner
            else:
                while alive[rc_ptr] == 0:
                    rc_ptr = (rc_ptr + 1) % A
                a = rc_ptr
        used[a] += 1
        i1 = np.random.randint(0, MU)
        i2 = np.random.randint(0, MU)
        pa1 = i1 if fit[i1] <= fit[i2] else i2
        i3 = np.random.randint(0, MU)
        i4 = np.random.randint(0, MU)
        pa2 = i3 if fit[i3] <= fit[i4] else i4
        _variate(arm_strat[a], arm_d[a], arm_ls[a], pop[pa1], pop[pa2], p, m, child, ld)
        for k in range(m):
            c_loads[k] = ld[k]
        ev += 1.0
        cm = c_loads.max()
        cur = fit[pa1] if fit[pa1] <= fit[pa2] else fit[pa2]
        r = float(cur - cm) if cur > cm else 0.0
        if rmode == 1:
            if r > rmax_q:
                rmax_q = r
            if cm == cur:
                r += hp[4] * max(rmax_q, 1.0)
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
            Q[s, a] += hp[0] * (r + hp[1] * bn - Q[s, a])
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
        elif agent == 8:
            for k in range(A):
                ts_s[k] *= lr
                ts_f[k] *= lr
            if r > 0:
                ts_s[a] += 1.0
            else:
                ts_f[a] += 1.0
        elif agent == 9:
            if r > rmax:
                rmax = r
            rn = r / rmax if rmax > 0 else 0.0
            dm_n[a] += 1.0
            dm_s[a] += rn
            mean_a = dm_s[a] / dm_n[a]
            ph_g[a] += rn - mean_a + lr2
            if ph_g[a] > ph_max[a]:
                ph_max[a] = ph_g[a]
            if ph_max[a] - ph_g[a] > lr:
                for k in range(A):
                    dm_n[k] = 0.0
                    dm_s[k] = 0.0
                    ph_g[k] = 0.0
                    ph_max[k] = 0.0
                dm_t = 0
        elif agent == 10:
            if r > rmax:
                rmax = r
            rn = r / rmax if rmax > 0 else 0.0
            if rc_winner < 0:
                rc_sum[a] += rn
                rc_n[a] += 1.0
                rc_ptr = (rc_ptr + 1) % A
                rc_pulled += 1
                if rc_pulled >= rc_quota * n_alive:
                    # fim da rodada: mantem as ceil(n_vivas / 2) de maior media acumulada
                    keep = (n_alive + 1) // 2
                    if n_alive <= 1:
                        keep = 1
                    for _ in range(n_alive - keep):
                        worst = -1
                        wv = 1e18
                        for k in range(A):
                            if alive[k] == 1:
                                v = rc_sum[k] / max(rc_n[k], 1.0) + np.random.random() * 1e-9
                                if v < wv:
                                    wv = v
                                    worst = k
                        alive[worst] = 0
                    n_alive = keep
                    rc_round += 1
                    rc_pulled = 0
                    if n_alive == 1 or rc_round >= rounds:
                        for k in range(A):
                            if alive[k] == 1:
                                rc_winner = k
                                break
                    else:
                        rc_quota = max(1, int(rc_budget / n_alive))
        wi = 0
        for i in range(1, MU):
            if fit[i] > fit[wi]:
                wi = i
        if cm <= fit[wi]:
            for j in range(n):
                pop[wi, j] = child[j]
            fit[wi] = cm
        if cm < best:
            best = cm
            n_imp += 1
            ino = 0
            for k in range(m):
                loads[k] = c_loads[k]
            for j in range(n):
                assign[j] = child[j]
        else:
            ino += 1
    return best, n_imp, ev, used, hit2, assign
