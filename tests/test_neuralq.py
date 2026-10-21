"""Testes do seletor neural (Entrada 162)."""
import numpy as np

from pmsp.neuralq import NeuralQSelector


def _f(t=0.5, st=0.1):
    return np.array([0.04, 0.2, t, st])


def test_learns_best_arm_in_a_bandit():
    rng = np.random.default_rng(0)
    sel = NeuralQSelector(n_actions=4, hidden=8, lr=0.05, seed=1, epsilon_decay=0.99)
    p = [0.05, 0.1, 0.7, 0.1]
    picks = []
    for k in range(1500):
        f = _f(t=k / 1500)
        a = sel.select(0, f)
        sel.update(0, f, a, float(rng.random() < p[a]), 0)
        picks.append(a)
    assert np.mean(np.array(picks[-400:]) == 2) > 0.5


def test_deterministic_and_reset():
    def run(sel):
        out = []
        for k in range(60):
            f = _f(t=k / 60)
            a = sel.select(0, f)
            sel.update(0, f, a, float(a == 1), 0)
            out.append(a)
        return out
    assert run(NeuralQSelector(3, seed=7)) == run(NeuralQSelector(3, seed=7))
    s = NeuralQSelector(3, seed=7)
    first = run(s)
    s.reset()
    assert run(s) == first


def test_reward_binary_and_finite_weights():
    sel = NeuralQSelector(3, seed=0)
    for k in range(200):
        f = _f(t=k / 200)
        a = sel.select(0, f)
        sel.update(0, f, a, 1e6 if k % 3 == 0 else 0.0, 0)      # recompensas enormes nao explodem os pesos (binarizadas e gradiente recortado)
    assert np.all(np.isfinite(sel.W1)) and np.all(np.isfinite(sel.W2)) and np.abs(sel.W2).max() < 50
