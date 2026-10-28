"""
pmsp/ur_numba.py (Entrada 234, Ciclo 22)
==========================================
Auxiliares COMPILADOS do Iterated Greedy para MAQUINAS NAO RELACIONADAS (R||Cmax): o tempo da tarefa j na maquina k e' P[j, k] (matriz n x m) e o resto do codigo e' o do P||Cmax (pmsp/ig_numba.py), cujas rotinas
sem dependencia de P (_destroy, _remove, _append, _other_max, _argmax, _argmin) sao reaproveitadas. Aqui ficam as que leem os tempos: init_state, move_job, swap_jobs, op1_best, op2_best, local_search e reconstruct.
Motivacao (revisoes cegas #5 a #7): o P||Cmax sintetico fica perto do teto (a folga do oraculo e' de centesimos de ponto) e a regra do limiar foi achada nos dois primeiros hospedeiros; um hospedeiro NOVO, com outra estrutura de decisao,
testa a regra congelada fora da amostra. Diferencas em relacao ao P||Cmax: (i) o custo de colocar uma tarefa depende da maquina; (ii) a reconstrucao poe cada tarefa removida na maquina que minimiza o seu TERMINO (carga + P[j, k]) e a ordena por
min_k P[j, k] decrescente (a melhor tempo possivel da tarefa; empates: ordem de remocao); (iii) a estrategia de destruicao 1 ('as d maiores') usa a mesma ordem min_k P[j, k] decrescente, passada pronta em order_desc.
Unidade de trabalho (avaliacoes): igual a do P||Cmax (d x m na reconstrucao; os candidatos de movimento da busca local).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ig_numba import _append, _argmax, _other_max, _remove


@njit(cache=True)
def init_state(assign, P, m):
    """loads, machine_jobs (matriz m x n, preenchida ate' cnt[k]) e cnt a partir de assign, na ordem dos indices."""
    n = len(assign)
    loads = np.zeros(m, dtype=np.int64)
    mj = np.zeros((m, n), dtype=np.int32)
    cnt = np.zeros(m, dtype=np.int64)
    for j in range(n):
        k = assign[j]
        loads[k] += P[j, k]
        mj[k, cnt[k]] = j
        cnt[k] += 1
    return loads, mj, cnt


@njit(cache=True)
def move_job(assign, loads, mj, cnt, P, j, tgt):
    src = assign[j]
    if src == tgt:
        return
    loads[src] -= P[j, src]
    loads[tgt] += P[j, tgt]
    _remove(mj, cnt, src, j)
    _append(mj, cnt, tgt, j)
    assign[j] = tgt


@njit(cache=True)
def swap_jobs(assign, loads, mj, cnt, P, ja, jb):
    """Troca inter-maquinas: ja (na maquina ma) e jb (na maquina mb) trocam de maquina."""
    ma = assign[ja]
    mb = assign[jb]
    loads[ma] += P[jb, ma] - P[ja, ma]
    loads[mb] += P[ja, mb] - P[jb, mb]
    _remove(mj, cnt, ma, ja)
    _append(mj, cnt, ma, jb)
    _remove(mj, cnt, mb, jb)
    _append(mj, cnt, mb, ja)
    assign[ja] = mb
    assign[jb] = ma


@njit(cache=True)
def op1_best(assign, loads, mj, cnt, P):
    """Melhor troca entre a maquina gargalo e as demais; retorna (melhora, avaliacoes)."""
    m = len(loads)
    a = _argmax(loads)
    old = loads[a]
    if cnt[a] == 0:
        return 0, 0
    other = _other_max(loads, a)
    la = loads[a]
    best = 0
    bja = -1
    bjb = -1
    ev = 0
    for ia in range(cnt[a]):
        ja = mj[a, ia]
        paa = P[ja, a]
        for b in range(m):
            if b == a:
                continue
            lb = loads[b]
            om = other[b]
            pab = P[ja, b]
            for ib in range(cnt[b]):
                jb = mj[b, ib]
                nc = max(la - paa + P[jb, a], lb - P[jb, b] + pab, om)
                ev += 1
                if old - nc > best:
                    best = old - nc
                    bja = ja
                    bjb = jb
    if best <= 0:
        return 0, ev
    swap_jobs(assign, loads, mj, cnt, P, bja, bjb)
    return best, ev


@njit(cache=True)
def op2_best(assign, loads, mj, cnt, P):
    """Melhor reinsercao de uma tarefa da maquina gargalo; retorna (melhora, avaliacoes)."""
    m = len(loads)
    a = _argmax(loads)
    old = loads[a]
    if cnt[a] == 0 or m < 2:
        return 0, 0
    other = _other_max(loads, a)
    la = loads[a]
    best = 0
    bj = -1
    bt = -1
    ev = 0
    for ia in range(cnt[a]):
        j = mj[a, ia]
        pja = P[j, a]
        for b in range(m):
            if b == a:
                continue
            nc = max(la - pja, loads[b] + P[j, b], other[b])
            ev += 1
            if old - nc > best:
                best = old - nc
                bj = j
                bt = b
    if best <= 0:
        return 0, ev
    move_job(assign, loads, mj, cnt, P, bj, bt)
    return best, ev


@njit(cache=True)
def local_search(assign, loads, mj, cnt, P):
    """ate' 2 passes de (inter-troca; se nao melhorou, reinsercao); para quando as duas nao melhoram. Retorna as avaliacoes."""
    ev = 0
    for _ in range(2):
        r1, e1 = op1_best(assign, loads, mj, cnt, P)
        ev += e1
        if r1 <= 0:
            r2, e2 = op2_best(assign, loads, mj, cnt, P)
            ev += e2
            if r2 <= 0:
                break
    return ev


@njit(cache=True)
def row_min(P, j):
    mn = P[j, 0]
    for k in range(1, P.shape[1]):
        if P[j, k] < mn:
            mn = P[j, k]
    return mn


@njit(cache=True)
def reconstruct(assign, loads, mj, cnt, P, removed, d):
    """Reconstrucao gulosa: desconta as removidas, ordena por min_k P[j, k] decrescente (estavel) e poe cada uma na maquina de menor TERMINO (carga + P[j, k]; primeiro minimo). Retorna as avaliacoes (d x m)."""
    m = len(loads)
    tracker = loads.copy()
    for i in range(d):
        j = removed[i]
        tracker[assign[j]] -= P[j, assign[j]]
    order = removed[:d].copy()
    for i in range(1, d):
        x = order[i]
        px = row_min(P, x)
        q = i - 1
        while q >= 0 and row_min(P, order[q]) < px:
            order[q + 1] = order[q]
            q -= 1
        order[q + 1] = x
    for i in range(d):
        j = order[i]
        k = 0
        bv = tracker[0] + P[j, 0]
        for kk in range(1, m):
            v = tracker[kk] + P[j, kk]
            if v < bv:
                bv = v
                k = kk
        tracker[k] += P[j, k]
        move_job(assign, loads, mj, cnt, P, j, k)
    return d * m
