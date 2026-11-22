"""
analysis/learnability_model.py (Entrada 287, Ciclo 54)
=======================================================
O MODELO DE APRENDIBILIDADE, fixado antes de abrir os registros de results/logs287. Cada celula e' vista como um bandido de K bracos com recompensa esparsa: a cada decisao t o agente escolhe a acao a e recebe r = max(melhora do makespan, 0),
que e' ZERO na enorme maioria das decisoes. O ganho total de uma execucao e' a soma das recompensas (o IG so' aceita candidatas que nao pioram, entao makespan inicial - makespan final = soma das recompensas; conferido nos testes).
Unidade: r~ = 100 r / limite inferior (pontos do limite inferior).

Quantidades (por celula, sobre as decisoes do controle aleatorio):
  p_a   fracao das decisoes da acao a com recompensa positiva;   m_a  media das recompensas positivas da acao a;   mu_a = p_a m_a  recompensa esperada por decisao;  G_a = T mu_a  ganho esperado da acao fixa a.
  p_ab  o mesmo por faixa de tempo (NB = 5 faixas de 200 decisoes): o modelo 'T1' do bandido OBLIVIO nao estacionario (a recompensa de (a, faixa b) nao depende da trajetoria do agente).
  indice de aprendibilidade  L = T Delta_cf^2 / (K sigma^2): Delta_cf e' a vantagem POR DECISAO da melhor acao sobre a media das acoes, escolhida na metade A das instancias e medida na metade B (validacao cruzada; so' L = 0 se
  Delta_cf <= 0), sigma^2 a variancia de r~ por decisao. L << 1: a melhor acao nao e' identificavel dentro de T decisoes.
  captura 'ganha-fica' (win-stay) kappa_WS: com recompensas esparsas o aprendiz de desempate aleatorio escolhe, na pratica, a acao que deu o PRIMEIRO sucesso (probabilidade proporcional a p_a) e a repete; seu ganho esperado
  e' T sum_a p_a mu_a / sum_a p_a e kappa_WS = (T sum p_a mu_a / sum p_a - G_U) / (G_max - G_U), G_U = T mean(mu_a), G_max = T max(mu_a). Nao tem parametro ajustavel.
  gemeo digital: simulacao de UCB1 (como o do IG: recompensa normalizada pelo maximo corrente, C = raiz de 2, uma puxada inicial por acao) e de um Q-Learning SEM estados (alfa 0,1; gama 0; epsilon 1 -> 0,05 por 0,995; desempate
  aleatorio) no bandido T1, com o ganho medido pela recompensa ESPERADA da acao escolhida (Rao-Blackwell: menos variancia que somar as recompensas sorteadas). O Q-Learning sem estados aprende mais depressa que o de 81 estados: e' um LIMITE SUPERIOR do real.
  folga por fase e por estado, com validacao cruzada de instancias (escolhe na metade de treino, mede na outra, e troca): folga de fase = melhor acao por faixa de tempo contra a melhor acao unica; folga de estado = melhor acao por
  estado (so' nos estados com pelo menos 30 decisoes de cada acao no treino; nos demais, a acao unica) contra a acao unica; ambas MIOPES (recompensa imediata) e obliviias; razao rho = folga / (vantagem da acao unica sobre a media).
"""
from __future__ import annotations

import numpy as np
from numba import njit

NB = 5
NS = 81


def normalise(Rw, lb):
    return 100.0 * np.asarray(Rw, dtype=np.float64) / np.asarray(lb, dtype=np.float64)[:, None]


def bin_index(T):
    return (np.arange(T) * NB) // T


def tables_ab(A, rt, K, mask=None):
    """cnt, pos, sum por (acao, faixa) sobre as execucoes de mask."""
    if mask is not None:
        A, rt = A[mask], rt[mask]
    b = np.broadcast_to(bin_index(A.shape[1]), A.shape)
    idx = (A.astype(np.int64) * NB + b).ravel()
    cnt = np.bincount(idx, minlength=K * NB).reshape(K, NB).astype(float)
    pos = np.bincount(idx, weights=(rt > 0).ravel().astype(float), minlength=K * NB).reshape(K, NB)
    sm = np.bincount(idx, weights=rt.ravel(), minlength=K * NB).reshape(K, NB)
    return cnt, pos, sm


def tables_sa(A, rt, S, K, mask):
    A, rt, S = A[mask], rt[mask], S[mask]
    idx = (S.astype(np.int64) * K + A.astype(np.int64)).ravel()
    cnt = np.bincount(idx, minlength=NS * K).reshape(NS, K).astype(float)
    sm = np.bincount(idx, weights=rt.ravel(), minlength=NS * K).reshape(NS, K)
    return cnt, sm


def pos_values(A, rt, K):
    """Recompensas positivas de cada acao (proprias) e de todas as acoes."""
    a, r = A.ravel(), rt.ravel()
    pos = r > 0
    return [r[pos & (a == k)] for k in range(K)], r[pos]


def arm_params(A, rt, K):
    """p_ab (K x NB), m_a (K), mu_ab = p_ab m_a, p_a agregado, e os valores positivos achatados (flat) com os deslocamentos (off) para o sorteio da magnitude."""
    cnt, pos, sm = tables_ab(A, rt, K)
    p_ab = np.divide(pos, cnt, out=np.zeros_like(pos), where=cnt > 0)
    own, allp = pos_values(A, rt, K)
    m_a = np.array([v.mean() if len(v) else 0.0 for v in own])
    p_a = pos.sum(axis=1) / np.maximum(cnt.sum(axis=1), 1.0)
    fallback = allp if len(allp) else np.zeros(1)
    sample = [v if len(v) else fallback for v in own]          # um braco sem sucessos usa a distribuicao global (p_ab = 0 nele, so' para o sorteio ser definido)
    flat = np.concatenate(sample).astype(np.float64)
    off = np.concatenate([[0], np.cumsum([len(v) for v in sample])]).astype(np.int64)
    return dict(p_ab=p_ab, m_a=m_a, mu_ab=p_ab * m_a[:, None], p_a=p_a, flat=flat, off=off, cnt=cnt, pos=pos, sm=sm)


def static_gains(par, T):
    """G_a = (T / NB) sum_b mu_ab ; G_U = media sobre as acoes; o melhor da amostra."""
    G = par["mu_ab"].sum(axis=1) * (T / NB)
    return G, float(G.mean()), float(G.max())


def kappa_win_stay(par, T):
    p, mu = par["p_a"], par["p_a"] * par["m_a"]
    if p.sum() <= 0:
        return float("nan")
    gmax, gu = T * mu.max(), T * mu.mean()
    if gmax - gu <= 1e-15:
        return float("nan")
    return float((T * (p * mu).sum() / p.sum() - gu) / (gmax - gu))


def index_L(A, rt, K, T, half):
    """L = T Delta_cf^2 / (K sigma^2); Delta_cf: melhor acao escolhida na metade `half` (True) e medida na outra."""
    a, r = A.astype(np.int64), rt
    cA = np.bincount(a[half].ravel(), minlength=K).astype(float)
    sA = np.bincount(a[half].ravel(), weights=r[half].ravel(), minlength=K)
    cB = np.bincount(a[~half].ravel(), minlength=K).astype(float)
    sB = np.bincount(a[~half].ravel(), weights=r[~half].ravel(), minlength=K)
    mA = np.divide(sA, cA, out=np.zeros(K), where=cA > 0)
    mB = np.divide(sB, cB, out=np.zeros(K), where=cB > 0)
    ak = int(np.argmax(mA))
    delta = float(mB[ak] - r[~half].mean())
    sigma2 = float(r.var())
    L = T * delta * delta / (K * sigma2) if (delta > 0 and sigma2 > 0) else 0.0
    return L, delta, sigma2


@njit(cache=True)
def twin_sim(p_ab, mu_ab, flat, off, T, n_sim, seed):
    """UCB1 e Q-Learning sem estados no bandido oblivio T1; devolve os ganhos esperados medios (pontos do limite inferior)."""
    np.random.seed(seed)
    K, nb = p_ab.shape
    g_ucb = 0.0
    g_ql = 0.0
    for _ in range(n_sim):
        # ---- UCB1 (como o do IG)
        cn = np.zeros(K)
        cr = np.zeros(K)
        rmax = 0.0
        tot = 0.0
        for t in range(T):
            b = (t * nb) // T
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
            r = 0.0
            if np.random.random() < p_ab[a, b]:
                r = flat[off[a] + np.random.randint(0, off[a + 1] - off[a])]
            if r > rmax:
                rmax = r
            cn[a] += 1.0
            cr[a] += (r / rmax if rmax > 0 else 0.0)
            tot += mu_ab[a, b]
        g_ucb += tot
        # ---- Q-Learning sem estados (alfa 0,1; gama 0; epsilon 1 -> 0,05 por 0,995; desempate aleatorio)
        Q = np.zeros(K)
        eps = 1.0
        tot = 0.0
        for t in range(T):
            b = (t * nb) // T
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
                a = 0
                c = 0
                for k in range(K):
                    if Q[k] == mx:
                        if c == pick:
                            a = k
                            break
                        c += 1
            eps = max(0.05, eps * 0.995)
            r = 0.0
            if np.random.random() < p_ab[a, b]:
                r = flat[off[a] + np.random.randint(0, off[a + 1] - off[a])]
            Q[a] += 0.1 * (r - Q[a])
            tot += mu_ab[a, b]
        g_ql += tot
    return g_ucb / n_sim, g_ql / n_sim


def twin_predictions(par, T, n_sim=2000, seed=287):
    G, GU, Gmax = static_gains(par, T)
    gu, gq = twin_sim(par["p_ab"], par["mu_ab"], par["flat"], par["off"], T, n_sim, seed)
    den = Gmax - GU
    return dict(G_static=G, G_uniform=GU, G_max=Gmax, gain_ucb=float(gu - GU), gain_ql=float(gq - GU),
                kappa_ucb=float((gu - GU) / den) if den > 1e-15 else float("nan"), kappa_ql=float((gq - GU) / den) if den > 1e-15 else float("nan"))


def headroom_crossfit(A, rt, S, K, T, half):
    """Folga de fase e de estado (miopes, obliviias) com validacao cruzada de instancias; media das duas direcoes."""
    out = {"phase": [], "state": [], "adv_static": [], "n_state_used": []}
    for tr, ev in ((half, ~half), (~half, half)):
        c_tr, _, s_tr = tables_ab(A, rt, K, tr)
        c_ev, _, s_ev = tables_ab(A, rt, K, ev)
        mu_tr = np.divide(s_tr, c_tr, out=np.zeros_like(s_tr), where=c_tr > 0)
        mu_ev = np.divide(s_ev, c_ev, out=np.zeros_like(s_ev), where=c_ev > 0)
        a_s = int(np.argmax(mu_tr.sum(axis=1)))
        a_b = mu_tr.argmax(axis=0)
        h_phase = (T / NB) * float(sum(mu_ev[a_b[b], b] - mu_ev[a_s, b] for b in range(NB)))
        adv = (T / NB) * float((mu_ev[a_s, :] - mu_ev.mean(axis=0)).sum())
        # estado
        cs_tr, ss_tr = tables_sa(A, rt, S, K, tr)
        cs_ev, ss_ev = tables_sa(A, rt, S, K, ev)
        m_tr = np.divide(ss_tr, cs_tr, out=np.zeros_like(ss_tr), where=cs_tr > 0)
        m_ev = np.divide(ss_ev, cs_ev, out=np.zeros_like(ss_ev), where=cs_ev > 0)
        w_ev = cs_ev.sum(axis=1) / cs_ev.sum()
        tot_tr = ss_tr.sum(axis=0) / np.maximum(cs_tr.sum(axis=0), 1.0)
        a_static = int(np.argmax(tot_tr))
        used = 0
        h_state = 0.0
        for s in range(NS):
            if w_ev[s] > 0 and (cs_tr[s] >= 30).all():
                a_star = int(np.argmax(m_tr[s]))
                used += 1
            else:
                a_star = a_static
            h_state += w_ev[s] * (m_ev[s, a_star] - m_ev[s, a_static])
        out["phase"].append(h_phase)
        out["state"].append(T * h_state)
        out["adv_static"].append(adv)
        out["n_state_used"].append(used)
    return dict(H_phase=float(np.mean(out["phase"])), H_state=float(np.mean(out["state"])), adv_static=float(np.mean(out["adv_static"])), n_state_used=float(np.mean(out["n_state_used"])))
