"""
pmsp/ur_numba_neural.py (Entrada 259, Ciclo 34, frente 4)
===========================================================
Politica NEURAL treinada OFFLINE (estilo configuracao dinamica de algoritmos, DAC) sobre o IG de maquinas nao relacionadas: uma rede com uma camada oculta (tanh) mapeia as caracteristicas do estado do IG e da
instancia para a distribuicao (softmax) sobre as acoes; os pesos sao fixados (treinados antes, nas instancias de ajuste) e a politica NAO aprende durante a execucao. O laco do IG e' o de pmsp/ur_numba_agents.py
(ur_agent_core): copia da solucao corrente, destruicao, reconstrucao, busca local opcional, aceitacao de makespan nao pior; so' a escolha da acao muda.
Caracteristicas (todas sem escala): f0 = t/T; f1 = min(iteracoes sem melhora/50, 1); f2 = min((makespan - lb)/lb, 1); f3 = min(desvio/media das cargas, 1); f4 = ln n / ln 1000; f5 = ln m / ln 50;
f6 = media movel exponencial (0,9) do indicador de melhora do makespan; f7 = 1 (vies). Com use_feat = 0 todas as caracteristicas (exceto o vies) sao zeradas: a politica vira uma MISTURA constante (controle).
Retorno: (melhor makespan, avaliacoes, uso por acao).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _destroy
from .ur_numba import init_state, local_search, reconstruct

NF = 8


@njit(cache=True)
def ur_neural_core(P, assign0, m, arm_strat, arm_d, arm_ls, T, W1, b1, W2, b2, use_feat, seed, order_desc, lb):
    np.random.seed(seed)
    A = len(arm_strat)
    n = P.shape[0]
    H = W1.shape[0]
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = init_state(assign, P, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    best = loads.max()
    ino = 0
    ev = 0.0
    used = np.zeros(A, dtype=np.int64)
    f = np.zeros(NF)
    hid = np.zeros(H)
    logit = np.zeros(A)
    pr = np.zeros(A)
    ema = 0.0
    lnn = np.log(n) / np.log(1000.0)
    lnm = np.log(m) / np.log(50.0)
    for t in range(T):
        # --- caracteristicas
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
        f[0] = t / T
        f[1] = min(ino / 50.0, 1.0)
        f[2] = min(max((mx - lb) / lb, 0.0), 1.0) if lb > 0 else 0.0
        f[3] = min(sd / mu, 1.0) if mu > 0 else 0.0
        f[4] = lnn
        f[5] = lnm
        f[6] = ema
        f[7] = 1.0
        if use_feat == 0:
            for k in range(NF - 1):
                f[k] = 0.0
        # --- rede: oculta tanh, saida softmax
        for h in range(H):
            s = b1[h]
            for k in range(NF):
                s += W1[h, k] * f[k]
            hid[h] = np.tanh(s)
        mxl = -1e300
        for a in range(A):
            s = b2[a]
            for h in range(H):
                s += W2[a, h] * hid[h]
            logit[a] = s
            if s > mxl:
                mxl = s
        z = 0.0
        for a in range(A):
            pr[a] = np.exp(logit[a] - mxl)
            z += pr[a]
        u = np.random.random() * z
        a = A - 1
        acc = 0.0
        for k in range(A):
            acc += pr[k]
            if u < acc:
                a = k
                break
        used[a] += 1
        # --- passo do IG (identico ao de ur_agent_core)
        for k in range(m):
            c_loads[k] = loads[k]
            c_cnt[k] = cnt[k]
            for q in range(cnt[k]):
                c_mj[k, q] = mj[k, q]
        for j in range(n):
            c_assign[j] = assign[j]
        d = arm_d[a]
        _destroy(arm_strat[a], c_assign, P, d, order_desc, perm, removed, m)
        ev += reconstruct(c_assign, c_loads, c_mj, c_cnt, P, removed, d)
        if arm_ls[a] == 1:
            ev += local_search(c_assign, c_loads, c_mj, c_cnt, P)
        cm = c_loads.max()
        cur = loads.max()
        ema = 0.9 * ema + (0.1 if cm < cur else 0.0)
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
                ino = 0
            else:
                ino += 1
        else:
            ino += 1
    return best, ev, used


def n_params(A, H=16):
    return H * NF + H + A * H + A


def unpack(theta, A, H=16):
    i = 0
    W1 = theta[i:i + H * NF].reshape(H, NF); i += H * NF
    b1 = theta[i:i + H]; i += H
    W2 = theta[i:i + A * H].reshape(A, H); i += A * H
    b2 = theta[i:i + A]
    return np.ascontiguousarray(W1), np.ascontiguousarray(b1), np.ascontiguousarray(W2), np.ascontiguousarray(b2)
