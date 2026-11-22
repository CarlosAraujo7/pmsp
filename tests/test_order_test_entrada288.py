"""Entrada 288: testes do teste de invariancia a ordem (pmsp/order_test.py) com dados sinteticos."""
from __future__ import annotations

import numpy as np

from pmsp.order_test import order_invariance_test


def test_detects_a_real_order_effect():
    rng = np.random.default_rng(1)
    n, r = 40, 8
    inst = rng.normal(0, 3, size=(n, 1))
    eff = np.linspace(-0.5, 0.5, r)[None, :]                 # a ordem desloca o resultado em +-0,5
    y = inst + eff + rng.normal(0, 0.6, size=(n, r))
    out = order_invariance_test(y, n_perm=500, seed=2)
    assert out["p"] < 0.01 and out["sd_between"] > 0.2


def test_p_values_are_roughly_uniform_under_the_null():
    ps = []
    for s in range(150):
        rng = np.random.default_rng(100 + s)
        y = rng.normal(0, 3, size=(30, 1)) + rng.normal(0, 0.6, size=(30, 6))
        ps.append(order_invariance_test(y, n_perm=150, seed=s)["p"])
    ps = np.array(ps)
    assert 0.01 <= (ps < 0.05).mean() <= 0.12                # taxa de erro tipo I perto de 5% (150 replicas: faixa larga o bastante para nao ser instavel)
    assert 0.35 <= ps.mean() <= 0.65


def test_instance_effect_is_removed():
    # um efeito de instancia enorme, sem efeito de ordem: nao deve ser detectado
    rng = np.random.default_rng(5)
    y = rng.normal(0, 100, size=(25, 1)) + rng.normal(0, 0.5, size=(25, 5))
    assert order_invariance_test(y, n_perm=300, seed=1)["p"] > 0.05
