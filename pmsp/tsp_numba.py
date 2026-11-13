"""
pmsp/tsp_numba.py (Entrada 267, Ciclo 40)
==========================================
Um SEGUNDO DOMINIO, fora do escalonamento: a busca local iterada (ILS) para o caixeiro-viajante simetrico (TSP), compilada (numba), com o MESMO conjunto de seletores online do restante do estudo escolhendo a perturbacao a cada iteracao.
Busca local: 2-opt e Or-opt (segmentos de 1 a 3 cidades, nos dois sentidos) com listas de K vizinhos mais proximos e fila de cidades ativas (equivalente aos 'don't look bits'). Perturbacao (as ACOES), aplicada a uma copia do tour corrente:
  tipo 0: troca de dois segmentos adjacentes (double-bridge local) numa janela de w posicoes, w em {8, 16, 32, 64, 128};
  tipo 1: k inversoes aleatorias de segmentos curtos (2 a 10 cidades), k em {2, 4, 8, 16, 32};
  tipo 2: embaralhamento de um segmento de L cidades, L em {4, 8, 16, 32, 64}.
As 15 acoes sao os tres tipos cruzados com os cinco tamanhos (indice = 5 * tipo + tamanho); o espaco de 5 acoes e' o tipo 0. Os tamanhos sao limitados a n // 2. Depois da perturbacao, a busca local roda so' a partir das cidades tocadas
(fila); aceita-se o candidato se o comprimento nao piora (como no IG dos outros hospedeiros). Recompensa = melhora do comprimento do tour corrente, truncada em zero. O estado do Q-Learning e' discretizado em 81 estados (3 niveis de quatro
caracteristicas: fracao do orcamento, estagnacao, taxa de sucesso recente e melhora acumulada). Agentes: 0 fixa, 1 aleatorio, 2 Q-Learning (tabela zerada, epsilon-guloso de 1,0 a 0,05), 3 UCB1, 4 gradiente de politica, 7 roleta ALNS,
8 Thompson com desconto, 11 Q-Learning otimista (recompensa normalizada, Q inicial c, epsilon constante), copiados na logica dos nucleos de pmsp/ur_numba_agents*.py. 'ev' conta as avaliacoes de movimentos da busca local.
Retorno de tsp_agent_core: (melhor comprimento, melhorias, avaliacoes, uso por acao).
"""
from __future__ import annotations

import numpy as np
from numba import njit

EPS = 1e-9
AGENT_FIXED, AGENT_RANDOM, AGENT_Q, AGENT_UCB, AGENT_PG, AGENT_ALNS, AGENT_TS, AGENT_QO = 0, 1, 2, 3, 4, 7, 8, 11
KINDS = 3
SIZES = np.array([[8, 16, 32, 64, 128], [2, 4, 8, 16, 32], [4, 8, 16, 32, 64]], dtype=np.int64)


def arm_table(space: str):
    """(tipo, tamanho) das acoes: 'T5' = tipo 0 (cinco tamanhos), 'T15' = tres tipos x cinco tamanhos."""
    if space == "T5":
        return np.zeros(5, dtype=np.int64), SIZES[0].copy()
    kinds = np.array([k for k in range(3) for _ in range(5)], dtype=np.int64)
    sizes = np.array([SIZES[k][j] for k in range(3) for j in range(5)], dtype=np.int64)
    return kinds, sizes


@njit(cache=True)
def tour_len(tour, D):
    n = len(tour)
    s = 0.0
    for i in range(n):
        s += D[tour[i], tour[(i + 1) % n]]
    return s


@njit(cache=True)
def reverse_path(tour, pos, i, j):
    """Inverte o caminho que vai da posicao i a posicao j (ciclicamente, para a frente); inverte o complemento se for menor."""
    n = len(tour)
    L = (j - i) % n + 1
    if 2 * L > n:
        i2 = (j + 1) % n
        j2 = (i - 1) % n
        i = i2
        j = j2
        L = n - L
    for k in range(L // 2):
        a = (i + k) % n
        b = (j - k) % n
        ca = tour[a]
        cb = tour[b]
        tour[a] = cb
        tour[b] = ca
        pos[cb] = a
        pos[ca] = b


@njit(cache=True)
def _push(stk, inq, sp, c):
    if inq[c] == 0:
        inq[c] = 1
        stk[sp] = c
        sp += 1
    return sp


@njit(cache=True)
def local_search(tour, pos, D, NL, stk, inq, sp, out, seg):
    n = len(tour)
    K = NL.shape[1]
    ev = 0.0
    while sp > 0:
        sp -= 1
        c1 = stk[sp]
        inq[c1] = 0
        again = True
        while again:
            again = False
            # ---------------- 2-opt (sucessor e predecessor)
            for dirn in range(2):
                p1 = pos[c1]
                if dirn == 0:
                    c2 = tour[(p1 + 1) % n]
                else:
                    c2 = tour[(p1 - 1) % n]
                d12 = D[c1, c2]
                for k in range(K):
                    c3 = NL[c1, k]
                    d13 = D[c1, c3]
                    if d13 >= d12:
                        break
                    p3 = pos[c3]
                    if dirn == 0:
                        c4 = tour[(p3 + 1) % n]
                    else:
                        c4 = tour[(p3 - 1) % n]
                    if c3 == c2 or c4 == c1:
                        continue
                    ev += 1.0
                    delta = d13 + D[c2, c4] - d12 - D[c3, c4]
                    if delta < -EPS:
                        if dirn == 0:
                            reverse_path(tour, pos, (p1 + 1) % n, p3)
                        else:
                            reverse_path(tour, pos, p1, (p3 - 1) % n)
                        sp = _push(stk, inq, sp, c1)
                        sp = _push(stk, inq, sp, c2)
                        sp = _push(stk, inq, sp, c3)
                        sp = _push(stk, inq, sp, c4)
                        again = True
                        break
                if again:
                    break
            if again:
                continue
            # ---------------- Or-opt (segmentos de 1 a 3 cidades com c1 numa ponta)
            for L in range(1, 4):
                if L > n - 3:
                    break
                for end in range(2):
                    p1 = pos[c1]
                    if end == 0:
                        i0 = p1
                    else:
                        i0 = (p1 - L + 1) % n
                    a1 = tour[i0]
                    aL = tour[(i0 + L - 1) % n]
                    p = tour[(i0 - 1) % n]
                    q = tour[(i0 + L) % n]
                    g = D[p, a1] + D[aL, q] - D[p, q]
                    if g <= EPS:
                        continue
                    for side in range(2):
                        if side == 0:
                            e = a1
                        else:
                            e = aL
                        for k in range(K):
                            c = NL[e, k]
                            if D[e, c] >= g:
                                break
                            pc = pos[c]
                            if (pc - i0) % n < L:
                                continue
                            for opt in range(2):
                                ev += 1.0
                                if side == 0:
                                    if opt == 0:
                                        u = c
                                        v = tour[(pc + 1) % n]
                                        rev = 0
                                        add = D[c, a1] + D[aL, v] - D[u, v]
                                    else:
                                        v = c
                                        u = tour[(pc - 1) % n]
                                        rev = 1
                                        add = D[u, aL] + D[a1, c] - D[u, v]
                                else:
                                    if opt == 0:
                                        v = c
                                        u = tour[(pc - 1) % n]
                                        rev = 0
                                        add = D[u, a1] + D[aL, c] - D[u, v]
                                    else:
                                        u = c
                                        v = tour[(pc + 1) % n]
                                        rev = 1
                                        add = D[c, aL] + D[a1, v] - D[u, v]
                                if (pos[u] - i0) % n < L or (pos[v] - i0) % n < L:
                                    continue
                                if add - g < -EPS:
                                    for kk in range(L):
                                        seg[kk] = tour[(i0 + kk) % n]
                                    cnt = 0
                                    cp = pos[v]
                                    while cnt < n - L:
                                        city = tour[cp % n]
                                        if (pos[city] - i0) % n >= L:
                                            out[cnt] = city
                                            cnt += 1
                                        cp += 1
                                    for kk in range(L):
                                        if rev == 0:
                                            out[n - L + kk] = seg[kk]
                                        else:
                                            out[n - L + kk] = seg[L - 1 - kk]
                                    for kk in range(n):
                                        tour[kk] = out[kk]
                                        pos[out[kk]] = kk
                                    sp = _push(stk, inq, sp, p)
                                    sp = _push(stk, inq, sp, q)
                                    sp = _push(stk, inq, sp, a1)
                                    sp = _push(stk, inq, sp, aL)
                                    sp = _push(stk, inq, sp, u)
                                    sp = _push(stk, inq, sp, v)
                                    again = True
                                    break
                            if again:
                                break
                        if again:
                            break
                    if again:
                        break
                if again:
                    break
    return ev


@njit(cache=True)
def construct(D, NL):
    """Vizinho mais proximo a partir da cidade 0, depois busca local completa (determinístico)."""
    n = D.shape[0]
    tour = np.empty(n, dtype=np.int64)
    pos = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=np.int64)
    tour[0] = 0
    used[0] = 1
    for i in range(1, n):
        c = tour[i - 1]
        bd = 1e300
        bj = -1
        for j in range(n):
            if used[j] == 0 and D[c, j] < bd:
                bd = D[c, j]
                bj = j
        tour[i] = bj
        used[bj] = 1
    for i in range(n):
        pos[tour[i]] = i
    stk = np.empty(n, dtype=np.int64)
    inq = np.zeros(n, dtype=np.int64)
    sp = 0
    for i in range(n - 1, -1, -1):
        sp = _push(stk, inq, sp, tour[i])
    out = np.empty(n, dtype=np.int64)
    seg = np.empty(3, dtype=np.int64)
    local_search(tour, pos, D, NL, stk, inq, sp, out, seg)
    return tour, pos


@njit(cache=True)
def _perturb(kind, size, tour, pos, stk, inq, sp, buf):
    n = len(tour)
    if kind == 0:
        w = min(size, n // 2)
        if w < 2:
            w = 2
        p0 = np.random.randint(0, n)
        l1 = 1 + np.random.randint(0, max(w // 2, 1))
        l2 = 1 + np.random.randint(0, max(w // 2, 1))
        if l1 + l2 > n - 2:
            l1 = max(1, (n - 2) // 2)
            l2 = max(1, (n - 2) - l1)
        tot = l1 + l2
        for k in range(tot):
            buf[k] = tour[(p0 + 1 + k) % n]
        for k in range(l2):
            c = buf[l1 + k]
            tour[(p0 + 1 + k) % n] = c
            pos[c] = (p0 + 1 + k) % n
        for k in range(l1):
            c = buf[k]
            tour[(p0 + 1 + l2 + k) % n] = c
            pos[c] = (p0 + 1 + l2 + k) % n
        sp = _push(stk, inq, sp, tour[p0])
        sp = _push(stk, inq, sp, tour[(p0 + 1) % n])
        sp = _push(stk, inq, sp, tour[(p0 + l2) % n])
        sp = _push(stk, inq, sp, tour[(p0 + l2 + 1) % n])
        sp = _push(stk, inq, sp, tour[(p0 + tot) % n])
        sp = _push(stk, inq, sp, tour[(p0 + tot + 1) % n])
    elif kind == 1:
        k_ = min(size, n // 2)
        for _ in range(k_):
            i = np.random.randint(0, n)
            lr = 2 + np.random.randint(0, max(min(9, n // 2 - 2), 1))
            j = (i + lr - 1) % n
            reverse_path(tour, pos, i, j)
            sp = _push(stk, inq, sp, tour[(i - 1) % n])
            sp = _push(stk, inq, sp, tour[i])
            sp = _push(stk, inq, sp, tour[j])
            sp = _push(stk, inq, sp, tour[(j + 1) % n])
    else:
        ls_ = min(size, n // 2)
        if ls_ < 2:
            ls_ = 2
        i = np.random.randint(0, n)
        for k in range(ls_ - 1, 0, -1):
            r = np.random.randint(0, k + 1)
            a = (i + k) % n
            b = (i + r) % n
            ca = tour[a]
            cb = tour[b]
            tour[a] = cb
            tour[b] = ca
            pos[cb] = a
            pos[ca] = b
        for k in range(-1, ls_ + 1):
            sp = _push(stk, inq, sp, tour[(i + k) % n])
    return sp


@njit(cache=True)
def _roulette(p, A, total):
    u = np.random.random() * total
    acc = 0.0
    for k in range(A):
        acc += p[k]
        if u < acc:
            return k
    return A - 1


@njit(cache=True)
def _state(t, T, ino, ema, gain):
    f0 = min(2, int(3.0 * t / T))
    f1 = 0 if ino == 0 else (1 if ino < 10 else 2)
    f2 = 0 if ema < 0.02 else (1 if ema < 0.1 else 2)
    f3 = 0 if gain < 0.005 else (1 if gain < 0.02 else 2)
    return f0 + 3 * f1 + 9 * f2 + 27 * f3


@njit(cache=True)
def tsp_agent_core(D, NL, tour0, arm_kind, arm_size, T, agent, fixed_arm, lr, lr2, seed, hp, q0):
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
