"""Testes do NormalisedUCBSelector (Entrada 98)."""
import numpy as np

from pmsp.selectors import NormalisedUCBSelector, UCBSelector


def _drive(sel, rewards, n_actions):
    feats = np.zeros(4)
    chosen = []
    for r in rewards:
        a = sel.select(0, feats)
        chosen.append(a)
        sel.update(0, feats, a, r, 0)
    return chosen


def test_inits_every_arm_once():
    sel = NormalisedUCBSelector(seed=1, n_actions=3)
    assert _drive(sel, [0.0, 0.0, 0.0], 3) == [0, 1, 2]


def test_rewards_fed_to_base_are_in_unit_interval():
    sel = NormalisedUCBSelector(seed=0, n_actions=4)
    _drive(sel, [5.0, 0.0, 40.0, 2.0, 100.0, 7.0, 0.0, 33.0], 4)
    mean = sel._rewards / np.maximum(sel._counts, 1)
    assert (mean >= 0).all() and (mean <= 1.0 + 1e-12).all()


def test_scale_invariant_decisions():
    """Multiplicar todas as recompensas por uma constante nao muda as escolhas."""
    rng = np.random.default_rng(3)
    rewards = rng.integers(0, 50, size=200).astype(float)
    a = _drive(NormalisedUCBSelector(seed=2), rewards, 4)
    b = _drive(NormalisedUCBSelector(seed=2), rewards * 1000.0, 4)
    assert a == b


def test_base_ucb_is_not_scale_invariant():
    """Documenta a diferenca: o UCBSelector original (recompensa bruta) depende da escala."""
    # Recompensas esparsas, como nas execucoes reais (poucos passos com melhoria):
    # com escala pequena o bonus de exploracao domina; com escala x1000 a media domina.
    rng = np.random.default_rng(5)
    rewards = np.where(rng.random(600) < 0.02, rng.integers(1, 50, size=600), 0).astype(float)
    a = _drive(UCBSelector(seed=2), rewards, 4)
    b = _drive(UCBSelector(seed=2), rewards * 1000.0, 4)
    assert a != b


def test_reset_clears_running_max():
    sel = NormalisedUCBSelector(seed=0, n_actions=3)
    _drive(sel, [10.0, 20.0, 30.0], 3)
    sel.reset()
    assert sel._rmax == 0.0 and sel._counts.sum() == 0
