"""
pmsp/tsp_warm.py (Entradas 274 e 275, Ciclos 45 e 46)
=====================================================
O Q-Learning otimista (agente 11 de pmsp/tsp_numba.py) com a tabela Q inicial AQUECIDA por uma priori offline POR ACAO: Q0[s, a] = c * escore_a em todos os 81 estados (o ILS do TSP; mesma construcao do benchmark de
Fanjul-Peyro e Ruiz da Entrada 271). O modulo de origem (pmsp/tsp_numba.py) tem hash congelado desde a Entrada 267 e aceita so' um Q inicial escalar; este nucleo repete a logica do agente 11, na mesma ordem de consumo do gerador
aleatorio, com um vetor de Q inicial. Com um vetor constante o resultado e' IGUAL ao do nucleo original (tests/test_tsp_warm_entrada274.py). A taxa de aprendizagem hp[0] = 0 congela a tabela (o controle sem aprendizagem: a acao de
maior Q0 com probabilidade 1 - epsilon, epsilon = hp[3]).
Retorno: (melhor comprimento, melhorias, avaliacoes, uso por acao).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from pmsp.tsp_numba import EPS, _perturb, _state, local_search, tour_len


def prior_scores(ratios):
    """Escore em [0, 1] de cada acao a partir da razao media (comprimento final / inicial): 1 a melhor, 0 a pior; vetor constante (empate) -> zeros."""
    r = np.asarray(ratios, dtype=np.float64)
    span = r.max() - r.min()
    return np.zeros_like(r) if span <= 0 else (r.max() - r) / span


@njit(cache=True)
def tsp_qo_prior_core(D, NL, tour0, arm_kind, arm_size, T, seed, hp, q0vec):
    """ILS do TSP com o Q-Learning otimista cuja tabela inicial e' q0vec[a] em todos os estados. hp = [alfa, gama, (livre), piso do epsilon = epsilon constante, (livre)]."""
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
    for s_ in range(81):
        for k in range(A):
            Q[s_, k] = q0vec[k]
    eps = hp[3]
    rmax = 0.0
    used = np.zeros(A, dtype=np.int64)
    for t in range(T):
        gain = (len0 - cur) / len0
        s = _state(t, T, ino, ema, gain)
        if np.random.random() < eps:
            a = np.random.randint(0, A)
        else:
            a = 0
            for k in range(1, A):
                if Q[s, k] > Q[s, a]:
                    a = k
        used[a] += 1
        for i in range(n):
            ctour[i] = tour[i]
            cpos[i] = pos[i]
        sp = _perturb(arm_kind[a], arm_size[a], ctour, cpos, stk, inq, 0, buf)
        ev += local_search(ctour, cpos, D, NL, stk, inq, sp, out, seg)
        cm = tour_len(ctour, D)
        r = cur - cm if cur > cm + EPS else 0.0
        if r > rmax:
            rmax = r
        rn = r / rmax if rmax > 0 else 0.0
        s2 = _state(t + 1, T, ino, ema, (len0 - min(cur, cm)) / len0)
        bn = Q[s2, 0]
        for k in range(1, A):
            if Q[s2, k] > bn:
                bn = Q[s2, k]
        Q[s, a] += hp[0] * (rn + hp[1] * bn - Q[s, a])
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
