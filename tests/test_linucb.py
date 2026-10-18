"""Testes do LinUCB (Entrada 159)."""
import numpy as np

from pmsp.linucb import LinUCBSelector


def _feats(t=0.5, st=0.1):
    return np.array([0.04, 0.2, t, st])


def test_learns_best_arm_when_reward_depends_only_on_arm():
    rng = np.random.default_rng(0)
    sel = LinUCBSelector(n_actions=4, alpha=0.5, seed=1)
    p = [0.05, 0.1, 0.6, 0.1]
    picks = []
    for k in range(600):
        f = _feats(t=k / 600)
        a = sel.select(0, f)
        r = float(rng.random() < p[a])
        sel.update(0, f, a, r, 0)
        picks.append(a)
    assert np.mean(np.array(picks[300:]) == 2) > 0.6


def test_deterministic_given_seed_and_reset():
    def run(sel):
        out = []
        for k in range(50):
            f = _feats(t=k / 50)
            a = sel.select(0, f)
            sel.update(0, f, a, float(a == 1), 0)
            out.append(a)
        return out
    s1, s2 = LinUCBSelector(3, seed=7), LinUCBSelector(3, seed=7)
    assert run(s1) == run(s2)
    s1.reset()
    assert run(s1) == run(LinUCBSelector(3, seed=7))


def test_context_handles_zero_reference_and_clips():
    sel = LinUCBSelector(2, seed=0)
    x = sel._context(np.array([0.0, 0.0, 0.3, 0.2]))
    assert x.shape == (5,) and x[1] == 0.0 and x[2] == 0.0
    x2 = sel._context(np.array([5.0, 5.0, 0.3, 0.2]))      # relativo ao primeiro valor (0.0 -> referencia nula)
    assert np.all(np.isfinite(x2))


def test_reward_is_binary_success_signal():
    sel = LinUCBSelector(2, seed=0)
    f = _feats()
    sel.update(0, f, 1, 123.0, 0)
    sel2 = LinUCBSelector(2, seed=0)
    sel2.update(0, f, 1, 1.0, 0)
    assert np.allclose(sel.b, sel2.b)
