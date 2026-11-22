"""Testes do modelo de aprendibilidade (analysis/learnability_model.py) com bandidos SINTETICOS de parametros conhecidos."""
from __future__ import annotations

import numpy as np
import pytest

from analysis import learnability_model as LM

T = 1000


def synth_logs(p_ab, m_a, n_runs=400, seed=0, K=None):
    """Registros de um controle aleatorio num bandido oblivio com p_ab (K x NB) e magnitudes m_a (exponenciais de media m_a)."""
    rng = np.random.default_rng(seed)
    K = p_ab.shape[0]
    A = rng.integers(0, K, size=(n_runs, T))
    b = (np.arange(T) * LM.NB) // T
    p = p_ab[A, b[None, :]]
    pos = rng.random((n_runs, T)) < p
    mag = rng.exponential(1.0, size=(n_runs, T)) * np.asarray(m_a)[A]
    rt = np.where(pos, mag, 0.0)
    S = rng.integers(0, 81, size=(n_runs, T))
    return A.astype(np.int8), rt, S.astype(np.int8)


def test_arm_params_recover_the_generating_bandit():
    p_ab = np.array([[0.02] * 5, [0.01] * 5, [0.005] * 5])
    m_a = np.array([1.0, 2.0, 3.0])
    A, rt, S = synth_logs(p_ab, m_a, n_runs=600)
    par = LM.arm_params(A, rt, 3)
    assert np.allclose(par["p_ab"].mean(axis=1), p_ab.mean(axis=1), rtol=0.2)
    assert np.allclose(par["m_a"], m_a, rtol=0.2)
    G, GU, Gmax = LM.static_gains(par, T)
    assert GU == pytest.approx(G.mean()) and Gmax == pytest.approx(G.max())
    assert np.allclose(G, T * p_ab.mean(axis=1) * m_a, rtol=0.2)


def test_kappa_win_stay_known_value():
    par = dict(p_a=np.array([0.5, 0.05, 0.05]), m_a=np.array([1.0, 1.0, 1.0]))
    # mu = p m; G_U = T 0,2; G_max = T 0,5; win-stay = T sum p mu / sum p = T 0,425 -> kappa = (0,425 - 0,2)/(0,5 - 0,2) = 0,75
    assert LM.kappa_win_stay(par, T) == pytest.approx(0.75)
    flat = dict(p_a=np.array([0.1, 0.1, 0.1]), m_a=np.array([1.0, 1.0, 1.0]))
    assert np.isnan(LM.kappa_win_stay(flat, T))


def _py_ucb(p_ab, mu_ab, flat, off, T, n_sim, seed):
    rng = np.random.default_rng(seed)
    K = p_ab.shape[0]
    tot_all = 0.0
    for _ in range(n_sim):
        cn, cr, rmax, tot = np.zeros(K), np.zeros(K), 0.0, 0.0
        for t in range(T):
            b = (t * LM.NB) // T
            zero = np.where(cn == 0)[0]
            if len(zero):
                a = int(zero[0])
            else:
                a = int(np.argmax(cr / cn + np.sqrt(2.0) * np.sqrt(np.log(t + 1.0) / cn)))
            r = 0.0
            if rng.random() < p_ab[a, b]:
                r = flat[off[a] + rng.integers(0, off[a + 1] - off[a])]
            rmax = max(rmax, r)
            cn[a] += 1
            cr[a] += (r / rmax if rmax > 0 else 0.0)
            tot += mu_ab[a, b]
        tot_all += tot
    return tot_all / n_sim


def test_twin_ucb_matches_a_plain_python_simulation():
    p_ab = np.array([[0.05] * 5, [0.02] * 5, [0.01] * 5])
    m_a = np.array([1.0, 1.0, 1.0])
    A, rt, S = synth_logs(p_ab, m_a, n_runs=300, seed=3)
    par = LM.arm_params(A, rt, 3)
    g_numba, g_ql = LM.twin_sim(par["p_ab"], par["mu_ab"], par["flat"], par["off"], 300, 400, 5)
    g_py = _py_ucb(par["p_ab"], par["mu_ab"], par["flat"], par["off"], 300, 150, 7)
    assert g_numba == pytest.approx(g_py, rel=0.06)


def test_twin_learners_beat_uniform_when_the_signal_is_strong_and_not_when_absent():
    strong = np.array([[0.30] * 5, [0.02] * 5, [0.02] * 5, [0.02] * 5])
    flat_ = np.full((4, 5), 0.05)
    for p_ab, expect_gain in ((strong, True), (flat_, False)):
        A, rt, S = synth_logs(p_ab, np.ones(4), n_runs=300, seed=1)
        par = LM.arm_params(A, rt, 4)
        tw = LM.twin_predictions(par, T, n_sim=300, seed=1)
        if expect_gain:
            # o UCB1 normalizado pelo maximo corrente quase nao explota com recompensa esparsa em T = 1000 (o bonus de exploracao domina);
            # o Q-Learning com epsilon decrescente explota; os dois ganham alguma coisa
            assert tw["kappa_ql"] > 0.3 and tw["kappa_ucb"] > 0.05
        else:
            assert abs(tw["gain_ucb"]) < 0.05 * tw["G_uniform"] and abs(tw["gain_ql"]) < 0.05 * tw["G_uniform"]


def test_index_L_grows_with_the_gap():
    Ls = []
    for gap in (0.0, 0.02, 0.08):
        p_ab = np.array([[0.05 + gap] * 5, [0.05] * 5, [0.05] * 5])
        A, rt, S = synth_logs(p_ab, np.ones(3), n_runs=400, seed=2)
        half = (np.arange(len(A)) % 2) == 0
        Ls.append(LM.index_L(A, rt, 3, T, half)[0])
    assert Ls[0] < Ls[1] < Ls[2]


def test_headroom_detects_planted_phase_structure_and_not_its_absence():
    # braco 0 melhor nas 2 primeiras faixas, braco 1 melhor nas 3 ultimas: ha' folga de fase
    p_ab = np.array([[0.10, 0.10, 0.02, 0.02, 0.02], [0.02, 0.02, 0.10, 0.10, 0.10]])
    A, rt, S = synth_logs(p_ab, np.ones(2), n_runs=400, seed=4)
    half = (np.arange(len(A)) % 2) == 0
    h = LM.headroom_crossfit(A, rt, S, 2, T, half)
    assert h["H_phase"] > 0.5 * h["adv_static"] and h["H_phase"] > 0
    # sem estrutura de fase: a folga com validacao cruzada nao e' positiva em media
    vals = []
    for seed in range(8):
        p0 = np.array([[0.05] * 5, [0.05] * 5, [0.05] * 5])
        A, rt, S = synth_logs(p0, np.ones(3), n_runs=200, seed=10 + seed)
        half = (np.arange(len(A)) % 2) == 0
        vals.append(LM.headroom_crossfit(A, rt, S, 3, T, half)["H_phase"])
    assert np.mean(vals) <= 0.02
