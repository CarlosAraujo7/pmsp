"""
analysis/learnability_dense.py (Entrada 289, Ciclo 55)
========================================================
A versao do GEMEO DIGITAL para recompensa DENSA (o hospedeiro de evolucao diferencial), em modulo novo para nao alterar analysis/learnability_model.py (congelado na Entrada 287). Mesma ideia: o bandido oblivio por NB = 5 faixas
de tempo, agora com a distribuicao EMPIRICA completa das recompensas por (acao, faixa) (as recompensas sao taxas de sucesso em [0, 1], nao esparsas), sorteada com reposicao. Os seletores sem estado do hospedeiro (UCB1, PM, AP, ETC) sao
simulados com as MESMAS regras de pmsp/de_host.py; o Q-Learning sem estados (alfa 0,1; gama 0; epsilon 1 -> 0,05 por 0,995; desempate aleatorio) e' um limite superior do Q-Learning de 81 estados. O ganho e' a recompensa ESPERADA
da celula (acao, faixa) escolhida (Rao-Blackwell). Captura de recompensa kappa_R = (G_pi - G_U) / (G_max - G_U), com G_a = (T / NB) soma_b mu_ab (acao fixa, bandido oblivio), G_U = media das acoes, G_max = maximo.
"""
from __future__ import annotations

import numpy as np
from numba import njit

NB = 5
UCB, PM, AP, ETC, QL1 = 0, 1, 2, 3, 4
NAMES = {UCB: "ucb", PM: "pm", AP: "ap", ETC: "etc", QL1: "ql1"}


def empirical_tables(A, Rw, K, T):
    """flat/off por (acao, faixa) e mu_ab, a partir dos registros do controle aleatorio (A, Rw: execucoes x T)."""
    b = (np.arange(T) * NB) // T
    bb = np.broadcast_to(b, A.shape)
    idx = A.astype(np.int64) * NB + bb
    parts, mu = [], np.zeros((K, NB))
    for k in range(K):
        for j in range(NB):
            v = Rw[idx == k * NB + j].astype(np.float64)
            if len(v) == 0:
                v = np.zeros(1)
            parts.append(v)
            mu[k, j] = v.mean()
    off = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    return np.concatenate(parts), off, mu


def static_gains(mu_ab, T):
    G = mu_ab.sum(axis=1) * (T / NB)
    return G, float(G.mean()), float(G.max())


@njit(cache=True)
def twin_dense(flat, off, mu_ab, T, n_sim, seed, which):
    np.random.seed(seed)
    K, nb = mu_ab.shape
    tot_all = 0.0
    for _ in range(n_sim):
        tot = 0.0
        cn = np.zeros(K)
        cr = np.zeros(K)
        q = np.ones(K)
        p = np.full(K, 1.0 / K)
        Q = np.zeros(K)
        rmax = 0.0
        eps = 1.0
        commit = -1
        for t in range(T):
            b = (t * nb) // T
            # ---- escolha
            a = 0
            if which == 0:                                    # UCB1
                a = -1
                for k in range(K):
                    if cn[k] == 0:
                        a = k
                        break
                if a < 0:
                    a = 0
                    bv = -1e18
                    lt = np.log(t + 1.0)
                    for k in range(K):
                        v = cr[k] / cn[k] + 1.4142135623730951 * np.sqrt(lt / cn[k])
                        if v > bv:
                            bv = v
                            a = k
            elif which == 1:                                  # PM
                z = 0.0
                for k in range(K):
                    z += q[k]
                u = np.random.random()
                acc = 0.0
                a = K - 1
                for k in range(K):
                    acc += 0.05 + (1.0 - K * 0.05) * q[k] / z
                    if u < acc:
                        a = k
                        break
            elif which == 2:                                  # AP
                z = 0.0
                for k in range(K):
                    z += p[k]
                u = np.random.random() * z
                acc = 0.0
                a = K - 1
                for k in range(K):
                    acc += p[k]
                    if u < acc:
                        a = k
                        break
            elif which == 3:                                  # ETC (m = 20)
                if commit >= 0:
                    a = commit
                else:
                    a = 0
                    for k in range(1, K):
                        if cn[k] < cn[a]:
                            a = k
            else:                                             # Q-Learning sem estados, desempate aleatorio
                if np.random.random() < eps:
                    a = np.random.randint(0, K)
                else:
                    mx = Q[0]
                    for k in range(1, K):
                        if Q[k] > mx:
                            mx = Q[k]
                    nt = 0
                    for k in range(K):
                        if Q[k] == mx:
                            nt += 1
                    pick = np.random.randint(0, nt)
                    c = 0
                    for k in range(K):
                        if Q[k] == mx:
                            if c == pick:
                                a = k
                                break
                            c += 1
                eps = max(0.05, eps * 0.995)
            # ---- recompensa sorteada da distribuicao empirica de (a, b)
            j = a * nb + b
            r = flat[off[j] + np.random.randint(0, off[j + 1] - off[j])]
            tot += mu_ab[a, b]
            # ---- atualizacao
            if which == 0:
                if r > rmax:
                    rmax = r
                cn[a] += 1.0
                cr[a] += (r / rmax if rmax > 0 else 0.0)
            elif which == 1:
                q[a] = 0.7 * q[a] + 0.3 * r
            elif which == 2:
                q[a] = 0.7 * q[a] + 0.3 * r
                ab = 0
                for k in range(1, K):
                    if q[k] > q[ab]:
                        ab = k
                pmax = 1.0 - (K - 1) * 0.05
                for k in range(K):
                    tgt = pmax if k == ab else 0.05
                    p[k] += 0.1 * (tgt - p[k])
            elif which == 3:
                cn[a] += 1.0
                cr[a] += r
                if commit < 0:
                    ok = True
                    for k in range(K):
                        if cn[k] < 20:
                            ok = False
                    if ok:
                        commit = 0
                        for k in range(1, K):
                            if cr[k] / cn[k] > cr[commit] / cn[commit]:
                                commit = k
            else:
                Q[a] += 0.1 * (r - Q[a])
        tot_all += tot
    return tot_all / n_sim


def predict(A, Rw, K, T, n_sim=3000, seed=289):
    """Capturas de recompensa previstas pelo gemeo para UCB1, PM, AP, ETC e Q-Learning sem estados."""
    flat, off, mu = empirical_tables(A, Rw, K, T)
    G, GU, Gmax = static_gains(mu, T)
    den = Gmax - GU
    out = dict(G_static=G, G_uniform=GU, G_max=Gmax)
    for w, nm in NAMES.items():
        g = twin_dense(flat, off, mu, T, n_sim, seed + 17 * w, w)
        out[f"gain_{nm}"] = float(g - GU)
        out[f"kappa_{nm}"] = float((g - GU) / den) if den > 1e-12 else float("nan")
    return out
