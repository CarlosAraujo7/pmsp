"""
pmsp/tsp_ties.py (Entrada 280, Ciclo 48)
=========================================
O nucleo de agentes do ILS do TSP (a ultima funcao de pmsp/tsp_numba.py, com os agentes 2 e 11) com UM argumento a mais, `tie_mode`: 0 = o primeiro maximo (reproduz tsp_agent_core bit a bit,
tests/test_ties_more_entrada280.py), 1 = um maximo escolhido ao acaso a cada decisao. Gerado por transformacao de texto; pmsp/tsp_numba.py mantem o seu hash congelado.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .tsp_numba import EPS, _perturb, _push, _roulette, _state, local_search, reverse_path, tour_len  # noqa: F401


@njit(cache=True)
def tsp_agent_core_ties(D, NL, tour0, arm_kind, arm_size, T, agent, fixed_arm, lr, lr2, seed, hp, q0, tie_mode):
    """ILS do TSP com o seletor 'agent'. hp = [alfa, gama, decaimento do epsilon, piso do epsilon, (livre)]; q0 = Q inicial do agente 11."""
    np.random.seed(seed)
    A = len(arm_kind)
    n = D.shape[0]
    tour = tour0.copy()
    pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos[tour[i]] = i
    ctour = tour.copy()
    cpos = pos.copy()
    stk = np.empty(n, dtype=np.int64)
    inq = np.zeros(n, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    seg = np.empty(3, dtype=np.int64)
    buf = np.empty(n, dtype=np.int64)
    cur = tour_len(tour, D)
    len0 = cur
    best = cur
    n_imp = 0
    ev = 0.0
    ino = 0
    ema = 0.0
    Q = np.zeros((81, A))
    if agent == 11:
        for s_ in range(81):
            for k in range(A):
                Q[s_, k] = q0
    eps = 1.0
    if agent == 11:
        eps = hp[3]
    ucb_n = np.zeros(A)
    ucb_r = np.zeros(A)
    ucb_t = 0
    rmax = 0.0
    H = np.zeros((81, A))
    pi = np.zeros(A)
    rbar = 0.0
    used = np.zeros(A, dtype=np.int64)
    w = np.ones(A)
    seg_pts = np.zeros(A)
    seg_cnt = np.zeros(A)
    ts_s = np.zeros(A)
    ts_f = np.zeros(A)
    for t in range(T):
        gain = (len0 - cur) / len0
        s = _state(t, T, ino, ema, gain)
        if agent == 0:
            a = fixed_arm
        elif agent == 1:
            a = np.random.randint(0, A)
        elif agent == 2 or agent == 11:
            if np.random.random() < eps:
                a = np.random.randint(0, A)
            else:
                if tie_mode == 1:
                    mxq = Q[s, 0]
                    for k in range(1, A):
                        if Q[s, k] > mxq:
                            mxq = Q[s, k]
                    ntie = 0
                    for k in range(A):
                        if Q[s, k] == mxq:
                            ntie += 1
                    pick = np.random.randint(0, ntie)
                    a = 0
                    for k in range(A):
                        if Q[s, k] == mxq:
                            if pick == 0:
                                a = k
                                break
                            pick -= 1
                else:
                    a = 0
                    for k in range(1, A):
                        if Q[s, k] > Q[s, a]:
                            a = k
            if agent == 2:
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
        elif agent == 7:
            tw = 0.0
            for k in range(A):
                tw += w[k]
            a = _roulette(w, A, tw)
        else:
            a = 0
            bv = -1.0
            for k in range(A):
                g1 = np.random.gamma(1.0 + ts_s[k], 1.0)
                g2 = np.random.gamma(1.0 + ts_f[k], 1.0)
                th = g1 / (g1 + g2)
                if th > bv:
                    bv = th
                    a = k
        used[a] += 1
        for i in range(n):
            ctour[i] = tour[i]
            cpos[i] = pos[i]
        sp = _perturb(arm_kind[a], arm_size[a], ctour, cpos, stk, inq, 0, buf)
        ev += local_search(ctour, cpos, D, NL, stk, inq, sp, out, seg)
        cm = tour_len(ctour, D)
        r = cur - cm if cur > cm + EPS else 0.0
        if agent == 2:
            s2 = _state(t + 1, T, ino, ema, (len0 - min(cur, cm)) / len0)
            bn = Q[s2, 0]
            for k in range(1, A):
                if Q[s2, k] > bn:
                    bn = Q[s2, k]
            Q[s, a] += hp[0] * (r + hp[1] * bn - Q[s, a])
        elif agent == 11:
            if r > rmax:
                rmax = r
            rn = r / rmax if rmax > 0 else 0.0
            s2 = _state(t + 1, T, ino, ema, (len0 - min(cur, cm)) / len0)
            bn = Q[s2, 0]
            for k in range(1, A):
                if Q[s2, k] > bn:
                    bn = Q[s2, k]
            Q[s, a] += hp[0] * (rn + hp[1] * bn - Q[s, a])
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
        elif agent == 7:
            seg_cnt[a] += 1.0
            if r > 0:
                seg_pts[a] += 33.0
            elif cm <= cur + EPS:
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
        ema = 0.9 * ema + 0.1 * (1.0 if r > 0 else 0.0)
        if cm <= cur + EPS:
            for i in range(n):
                tour[i] = ctour[i]
                pos[i] = cpos[i]
            cur = cm
            if cm < best - EPS:
                best = cm
                n_imp += 1
                ino = 0
            else:
                ino += 1
        else:
            ino += 1
    return best, n_imp, ev, used
