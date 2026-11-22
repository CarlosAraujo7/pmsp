"""Entrada 289: o gemeo digital denso simula os seletores sem estado com as mesmas regras de pmsp/de_host.py (conferido contra as classes reais num bandido sintetico)."""
from __future__ import annotations

import numpy as np
import pytest

from analysis import learnability_dense as LD
from pmsp import de_host as DH

T, K = 400, 4


def synth(seed=0, runs=60):
    """Registros de um controle aleatorio num bandido com taxa de sucesso por (acao, faixa), reforcos em [0, 1] (Beta)."""
    rng = np.random.default_rng(seed)
    base = np.array([0.30, 0.18, 0.10, 0.05])
    decay = np.linspace(1.0, 0.4, LD.NB)
    mean = base[:, None] * decay[None, :]
    A = rng.integers(0, K, size=(runs, T))
    b = (np.arange(T) * LD.NB) // T
    m = mean[A, b[None, :]]
    Rw = np.clip(rng.normal(m, 0.05), 0, 1)
    return A.astype(np.int8), Rw.astype(np.float32), mean


def python_payoff(sel_factory, mean, n_sim, seed):
    rng = np.random.default_rng(seed)
    tot_all = 0.0
    for sim in range(n_sim):
        sel = sel_factory(sim)
        tot = 0.0
        for t in range(T):
            b = (t * LD.NB) // T
            a = sel.select(0, t)
            r = float(np.clip(rng.normal(mean[a, b], 0.05), 0, 1))
            sel.update(a, r, 0, 0)
            tot += mean[a, b]
        tot_all += tot
    return tot_all / n_sim


def test_empirical_tables_roundtrip():
    A, Rw, mean = synth()
    flat, off, mu = LD.empirical_tables(A, Rw, K, T)
    assert np.allclose(mu, mean, atol=0.03)
    assert off[-1] == len(flat) and len(off) == K * LD.NB + 1


@pytest.mark.parametrize("which,factory", [(LD.UCB, lambda s: DH.UCB1(K, s)), (LD.PM, lambda s: DH.PM(K, s)), (LD.AP, lambda s: DH.AP(K, s)), (LD.ETC, lambda s: DH.ETC(K, s))])
def test_twin_matches_the_real_selector_classes(which, factory):
    A, Rw, mean = synth()
    flat, off, mu = LD.empirical_tables(A, Rw, K, T)
    g_twin = LD.twin_dense(flat, off, mu, T, 300, 5, which)
    g_real = python_payoff(factory, mu, 120, 7)
    assert g_twin == pytest.approx(g_real, rel=0.04)


def test_predict_orders_learners_sensibly():
    A, Rw, mean = synth()
    out = LD.predict(A, Rw, K, T, n_sim=300)
    assert out["G_max"] > out["G_uniform"]
    # com um sinal forte e denso, o Q-Learning sem estados e o ETC capturam bastante; a captura nunca passa de ~1
    assert out["kappa_ql1"] > 0.4 and out["kappa_etc"] > 0.3
    for nm in ("ucb", "pm", "ap", "etc", "ql1"):
        assert -0.2 < out[f"kappa_{nm}"] < 1.05
