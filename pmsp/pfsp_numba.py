"""
pmsp/pfsp_numba.py (Entrada 194, Ciclo 18)
============================================
SEGUNDO PROBLEMA do protocolo, COMPILADO: o flowshop de permutacao (Fm|prmu|Cmax) com o IG de Ruiz e Stutzle (2007) e os mesmos seletores de acao do P||Cmax (aleatorio, Q-Learning, UCB1, gradiente de politica, roleta tipo ALNS, Thompson, DMAB, corrida), para repetir
a atribuicao 'mecanismo x politica x orcamento' e o pre-teste de folga num problema diferente. Mesma semantica de pmsp/pfsp.py (que fica como referencia em Python; tests/test_pfsp_numba.py confere o makespan, a melhor insercao, o NEH e a busca local contra ele):
instancia p de forma (n, m) (tempo da tarefa j na maquina i); NEH inicial; a cada iteracao o seletor escolhe uma acao a = (tamanho de destruicao d[a], fator de temperatura tp[a]); remove d tarefas aleatorias, reinsere cada uma na melhor posicao (aceleracao de Taillard, empates: primeira
posicao), busca local de insercao (ordem aleatoria, ate' nao melhorar), aceitacao com temperatura constante T = tp * soma(p) / (n m 10) (aceita se nao piora, ou com probabilidade exp(-delta / T)); recompensa = max(melhora do makespan corrente, 0).
Estado (81 = 3^4): f1 = (corrente - LB) / LB em {< 0,02; < 0,06}, f2 = (inicial - corrente) / inicial em {< 0,01; < 0,03}, f3 = progresso em {0,33; 0,67}, f4 = estagnacao (iteracoes sem melhorar o melhor, janela 50) em {0,25; 0,60}.
Agentes 0 a 10 com a mesma numeracao e os mesmos hiperparametros de pmsp/ig_numba_agents3.py (lr, lr2). Retorna (melhor makespan, melhorias do melhor, avaliacoes de posicao de insercao, uso por acao, acertos da acao plantada na segunda metade).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba_agents2 import _roulette

AGENT_FIXED, AGENT_RANDOM, AGENT_Q, AGENT_UCB, AGENT_PG, AGENT_PM, AGENT_AP, AGENT_ALNS, AGENT_TS, AGENT_DMAB, AGENT_RACE = range(11)


@njit(cache=True)
def heads_tails(p, perm, K, e, t):
    """e[k, i]: conclusao da k-esima tarefa (k = 1..K) na maquina i, e[0, :] = 0; t[k, i]: caminho mais longo a partir da tarefa k (0..K-1) ate' o fim, t[K, :] = 0."""
    m = p.shape[1]
    for i in range(m):
        e[0, i] = 0
    for k in range(1, K + 1):
        j = perm[k - 1]
        e[k, 0] = e[k - 1, 0] + p[j, 0]
        for i in range(1, m):
            a = e[k - 1, i]
            b = e[k, i - 1]
            e[k, i] = (a if a > b else b) + p[j, i]
    for i in range(m):
        t[K, i] = 0
    for k in range(K - 1, -1, -1):
        j = perm[k]
        t[k, m - 1] = t[k + 1, m - 1] + p[j, m - 1]
        for i in range(m - 2, -1, -1):
            a = t[k + 1, i]
            b = t[k, i + 1]
            t[k, i] = (a if a > b else b) + p[j, i]


@njit(cache=True)
def makespan(p, perm, K, e):
    m = p.shape[1]
    for i in range(m):
        e[0, i] = 0
    for k in range(1, K + 1):
        j = perm[k - 1]
        e[k, 0] = e[k - 1, 0] + p[j, 0]
        for i in range(1, m):
            a = e[k - 1, i]
            b = e[k, i - 1]
            e[k, i] = (a if a > b else b) + p[j, i]
    return e[K, m - 1]


@njit(cache=True)
def best_insertion(p, perm, K, job, e, t, f):
    """Melhor posicao (0..K) para inserir `job` em perm[:K]; empates: a primeira. Devolve (posicao, makespan)."""
    m = p.shape[1]
    heads_tails(p, perm, K, e, t)
    best = 1 << 60
    bpos = 0
    for k in range(K + 1):
        f0 = e[k, 0] + p[job, 0]
        ms = f0 + t[k, 0]
        prev = f0
        for i in range(1, m):
            a = prev
            b = e[k, i]
            prev = (a if a > b else b) + p[job, i]
            v = prev + t[k, i]
            if v > ms:
                ms = v
        if ms < best:
            best = ms
            bpos = k
    return bpos, best


@njit(cache=True)
def _insert_at(perm, K, pos, job):
    for k in range(K, pos, -1):
        perm[k] = perm[k - 1]
    perm[pos] = job


@njit(cache=True)
def _remove_at(perm, K, pos):
    for k in range(pos, K - 1):
        perm[k] = perm[k + 1]


@njit(cache=True)
def neh(p, perm_out, e, t, f):
    n = p.shape[0]
    tot = np.zeros(n, dtype=np.int64)
    for j in range(n):
        tot[j] = p[j].sum()
    order = np.argsort(-tot, kind="mergesort")
    perm_out[0] = order[0]
    K = 1
    for idx in range(1, n):
        job = order[idx]
        pos, _ = best_insertion(p, perm_out, K, job, e, t, f)
        _insert_at(perm_out, K, pos, job)
        K += 1


@njit(cache=True)
def local_search(p, perm, n, cur, e, t, f):
    """Descida de insercao (Ruiz e Stutzle): ate' nenhuma tarefa, em ordem aleatoria, melhorar o makespan. Altera perm; devolve (makespan, avaliacoes de posicao)."""
    ev = 0.0
    improved = True
    while improved:
        improved = False
        order = np.random.permutation(n)
        for jj in range(n):
            job = order[jj]
            idx = 0
            while perm[idx] != job:
                idx += 1
            _remove_at(perm, n, idx)
            pos, ms = best_insertion(p, perm, n - 1, job, e, t, f)
            ev += n
            if ms < cur:
                _insert_at(perm, n - 1, pos, job)
                cur = ms
                improved = True
            else:
                _insert_at(perm, n - 1, idx, job)
    return cur, ev


@njit(cache=True)
def state_id(cur, lb, init, t, T, ino):
    f1 = (cur - lb) / lb if lb > 0 else 0.0
    f2 = (init - cur) / init if init > 0 else 0.0
    f3 = 1.0 if T <= 0 else min(t / T, 1.0)
    f4 = min(ino / 50.0, 1.0)
    b1 = 0 if f1 < 0.02 else (1 if f1 < 0.06 else 2)
    b2 = 0 if f2 < 0.01 else (1 if f2 < 0.03 else 2)
    b3 = 0 if f3 < 0.33 else (1 if f3 < 0.67 else 2)
    b4 = 0 if f4 < 0.25 else (1 if f4 < 0.60 else 2)
    return b1 * 27 + b2 * 9 + b3 * 3 + b4


@njit(cache=True)
def pfsp_ig_core(p, arm_d, arm_tp, T, agent, fixed_arm, lr, lr2, seed, lb, plant_mode, plant_arm, plant_bonus):
    np.random.seed(seed)
    n = p.shape[0]
    m = p.shape[1]
    A = len(arm_d)
    e = np.zeros((n + 2, m), dtype=np.int64)
    tt = np.zeros((n + 2, m), dtype=np.int64)
    f = np.zeros((n + 2, m), dtype=np.int64)
    perm = np.zeros(n, dtype=np.int64)
    neh(p, perm, e, tt, f)
    cur = makespan(p, perm, n, e)
    init = cur
    best = cur
    n_imp = 0
    ev = 0.0
    ino = 0
    base_temp = p.sum() / (n * m * 10.0)
    cand = np.zeros(n, dtype=np.int64)
    removed = np.zeros(n, dtype=np.int64)
    is_rem = np.zeros(n, dtype=np.int64)
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
    ts_s = np.zeros(A)
    ts_f = np.zeros(A)
    dm_n = np.zeros(A)
    dm_s = np.zeros(A)
    dm_t = 0
    ph_g = np.zeros(A)
    ph_max = np.zeros(A)
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
    rc_quota = max(1, int(rc_budget / n_alive))
    rc_winner = -1
    for t in range(T):
        s = state_id(cur, lb, init, t, T, ino)
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
        # candidata: destruir d tarefas aleatorias, reinserir pela melhor posicao, busca local
        d = arm_d[a]
        if d > n - 1:
            d = n - 1
        rp = np.random.permutation(n)
        for j in range(n):
            is_rem[j] = 0
        for q in range(d):
            is_rem[rp[q]] = 1
            removed[q] = rp[q]
        K = 0
        for k in range(n):
            if is_rem[perm[k]] == 0:
                cand[K] = perm[k]
                K += 1
        order = np.random.permutation(d)
        for qq in range(d):
            job = removed[order[qq]]
            pos, _ = best_insertion(p, cand, K, job, e, tt, f)
            ev += K + 1
            _insert_at(cand, K, pos, job)
            K += 1
        cm = makespan(p, cand, n, e)
        cm, ev_ls = local_search(p, cand, n, cm, e, tt, f)
        ev += ev_ls
        r = float(cur - cm) if cur > cm else 0.0
        if plant_mode > 0:
            pl = plant_arm if plant_mode == 1 else s % A
            if a == pl:
                r += plant_bonus
                if t >= T // 2:
                    hit2 += 1
        if agent == 2:
            s2 = state_id(cm if cm < cur else cur, lb, init, t + 1, T, ino)
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
        # aceitacao (temperatura constante da acao)
        temp = arm_tp[a] * base_temp
        accept = cm <= cur
        if not accept:
            accept = np.random.random() < np.exp(-(cm - cur) / temp)
        if accept:
            for k in range(n):
                perm[k] = cand[k]
            cur = cm
        if cur < best:
            best = cur
            n_imp += 1
            ino = 0
        else:
            ino += 1
    return best, n_imp, ev, used, hit2
