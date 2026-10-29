"""
pmsp/ig_numba_agents6.py (Entrada 242, Ciclo 23)
==================================================
Extensao COMPILADA de pmsp/ig_numba_agents4.py (que fica congelado) com DOIS PROJETOS DE AOS DA LITERATURA como agentes adicionais (as revisoes cegas #4-#8 pediram 'um AOS estabelecido'); os agentes 0 a 10 sao os mesmos (com as mesmas sementes reproduzem ig_agent_core4 EXATAMENTE: tests/test_ig_numba_agents6.py):
  agente 11  Q-Learning EPISODICO de estado BINARIO (o projeto do IG com Q-Learning para o flowshop de permutacao: Karimi-Mamaghan et al., EJOR 2023, e a extensao arXiv 2408.14864): o estado e' 1 se o melhor makespan melhorou no episodio anterior e 0 caso contrario; uma acao (operador) por episodio de E iteracoes (E = int(lr2));
            recompensa do episodio = eta * (melhora relativa do makespan CORRENTE no episodio) + (1 - eta) * (melhora relativa do MELHOR makespan), eta = hp[4]; Q inicial zero, estado inicial 0, primeira acao aleatoria; epsilon-guloso com epsilon inicial 1, multiplicado por beta = hp[2] ao fim de cada episodio, piso hp[3]; alfa = hp[0], gama = hp[1].
  agente 12  FRRMAB (Fialho, Schoenauer, Sebag, 2010): janela deslizante das ultimas W = 50 aplicacoes (operador, taxa de melhora relativa do makespan: (cur - cm)/cur se melhorou, 0 caso contrario); credito = D^posto * soma das recompensas do operador na janela (posto 0 = a maior), normalizado (a taxa-posto-ajustada, FRR);
            selecao = argmax FRR_i + C sqrt(2 ln(N) / n_i) (N = aplicacoes na janela; operadores ainda ausentes da janela primeiro); C = lr, D = lr2.
Os hiperparametros dos dois agentes NAO sao os dos artigos (nao ha' valores completos disponiveis); sao ajustados nas instancias de ajuste (experiment/run_cycle23_pubaos_entrada242.py). Mesma assinatura e retorno de ig_agent_core4.
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
def ig_agent_core6(p, assign0, m, arm_strat, arm_d, arm_ls, T, agent, fixed_arm, lr, lr2, seed, order_desc, lb, plant_mode, plant_arm, plant_bonus, hp, rmode):
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
    ep_len = max(1, int(lr2)) if agent == 11 else 1
    ep_a = 0
    ep_eps = 1.0
    ep_s = 0
    ep_cur0 = loads.max()
    ep_best0 = best
    W = 50
    win_op = np.full(W, -1, dtype=np.int64)
    win_rew = np.zeros(W)
    win_ptr = 0
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
        elif agent == 11:
            if t % ep_len == 0:
                ep_cur0 = loads.max()
                ep_best0 = best
                if np.random.random() < ep_eps:
                    ep_a = np.random.randint(0, A)
                else:
                    ep_a = 0
                    for k in range(1, A):
                        if Q[ep_s, k] > Q[ep_s, ep_a]:
                            ep_a = k
            a = ep_a
        elif agent == 12:
            cn = np.zeros(A)
            cr = np.zeros(A)
            nw = 0
            for q in range(W):
                if win_op[q] >= 0:
                    cn[win_op[q]] += 1.0
                    cr[win_op[q]] += win_rew[q]
                    nw += 1
            a = -1
            for k in range(A):
                if cn[k] == 0:
                    a = k
                    break
            if a < 0:
                dec = np.zeros(A)
                tot = 0.0
                for k in range(A):
                    rk = 0
                    for k2 in range(A):
                        if cr[k2] > cr[k] or (cr[k2] == cr[k] and k2 < k):
                            rk += 1
                    dec[k] = (lr2 ** rk) * cr[k]
                    tot += dec[k]
                bestv = -1e18
                a = 0
                for k in range(A):
                    frr = dec[k] / tot if tot > 0 else 1.0 / A
                    v = frr + lr * np.sqrt(2.0 * np.log(nw) / cn[k])
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
        if agent == 12:
            win_op[win_ptr] = a
            win_rew[win_ptr] = (float(cur - cm) / float(cur)) if cur > cm else 0.0
            win_ptr = (win_ptr + 1) % W
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
        if agent == 11 and ((t + 1) % ep_len == 0 or t == T - 1):
            cur_now = loads.max()
            r_loc = (ep_cur0 - cur_now) / ep_cur0 if ep_cur0 > 0 else 0.0
            r_glo = (ep_best0 - best) / ep_best0 if ep_best0 > 0 else 0.0
            rq = hp[4] * r_loc + (1.0 - hp[4]) * r_glo
            ns = 1 if best < ep_best0 else 0
            bn = Q[ns, 0]
            for k in range(1, A):
                if Q[ns, k] > bn:
                    bn = Q[ns, k]
            Q[ep_s, ep_a] += hp[0] * (rq + hp[1] * bn - Q[ep_s, ep_a])
            ep_s = ns
            ep_eps = max(hp[3], ep_eps * hp[2])
    return best, n_imp, ev, used, hit2
