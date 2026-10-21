"""Testes do DQN maior (Entrada 173)."""
import numpy as np

from pmsp.dqn import DQNSelector


def _f(t=0.5, st=0.1):
    return np.array([0.04, 0.2, t, st])


def test_gradient_matches_finite_differences():
    """O gradiente da perda (sem recorte) calculado a mao coincide com diferencas finitas em uma rede pequena."""
    s = DQNSelector(n_actions=3, hidden=(5, 4), seed=0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, s.in_dim))
    A = rng.integers(0, 3, size=6)
    T = rng.normal(size=6) * 0.1                                  # alvos pequenos: o erro TD fica dentro de [-1, 1] (sem recorte)

    def loss():
        out = s._forward(s.W, s.b, X)[-1]
        return 0.5 * np.sum((out[np.arange(6), A] - T) ** 2) / 6

    acts = s._forward(s.W, s.b, X)
    delta = np.zeros_like(acts[-1])
    delta[np.arange(6), A] = (acts[-1][np.arange(6), A] - T) / 6
    gW = [None] * len(s.W)
    for i in range(len(s.W) - 1, -1, -1):
        gW[i] = acts[i].T @ delta
        if i > 0:
            delta = (delta @ s.W[i].T) * (1.0 - acts[i] ** 2)
    for i in range(len(s.W)):
        idx = (0, 0)
        old = s.W[i][idx]
        s.W[i][idx] = old + 1e-6
        up = loss()
        s.W[i][idx] = old - 1e-6
        dn = loss()
        s.W[i][idx] = old
        assert abs((up - dn) / 2e-6 - gW[i][idx]) < 1e-6


def test_learns_best_arm_in_a_bandit():
    rng = np.random.default_rng(0)
    sel = DQNSelector(n_actions=4, hidden=(16, 16), lr=0.01, seed=1, epsilon_decay=0.99)
    p = [0.05, 0.1, 0.7, 0.1]
    picks = []
    for k in range(2000):
        f = _f(t=k / 2000)
        a = sel.select(0, f)
        sel.update(0, f, a, float(rng.random() < p[a]), 0)
        picks.append(a)
    assert np.mean(np.array(picks[-400:]) == 2) > 0.5


def test_deterministic_and_reset():
    def run(sel):
        out = []
        for k in range(80):
            f = _f(t=k / 80)
            a = sel.select(0, f)
            sel.update(0, f, a, float(a == 1), 0)
            out.append(a)
        return out
    assert run(DQNSelector(3, seed=7)) == run(DQNSelector(3, seed=7))
    s = DQNSelector(3, seed=7)
    first = run(s)
    s.reset()
    assert run(s) == first


def test_finite_weights_and_actions_in_range():
    sel = DQNSelector(n_actions=15, seed=3)
    rng = np.random.default_rng(2)
    for k in range(600):
        f = _f(t=k / 600, st=rng.random())
        a = sel.select(0, f)
        assert 0 <= a < 15
        sel.update(0, f, a, float(rng.random() < 0.3), 0)
    assert all(np.isfinite(w).all() for w in sel.W) and all(np.isfinite(b).all() for b in sel.b)
    assert sel._buf_n == 256                                       # o buffer enche e circula
    assert sel.in_dim == 5 + 2 * 15


def test_target_network_lags_the_online_network():
    sel = DQNSelector(n_actions=3, seed=0, target_every=1000)
    rng = np.random.default_rng(0)
    for k in range(100):
        f = _f(t=k / 100)
        a = sel.select(0, f)
        sel.update(0, f, a, float(rng.random() < 0.5), 0)
    assert not np.allclose(sel.W[0], sel.W_t[0])                   # a rede online aprendeu; a alvo ainda nao foi copiada
