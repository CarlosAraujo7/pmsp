"""Testes do limite inferior certificado (pmsp/exact.py, Entradas 100-101)."""
import itertools

import numpy as np

from pmsp.constructive import lpt
from pmsp.exact import improved_lower_bound, lp_lower_bound
from pmsp.instance import PMSPInstance


def brute_force(p, m):
    best = sum(p)
    for assign in itertools.product(range(m), repeat=len(p)):
        loads = [0] * m
        for j, a in enumerate(assign):
            loads[a] += p[j]
        best = min(best, max(loads))
    return best


def test_lp_lower_bound_is_valid_and_usually_tight():
    rng = np.random.default_rng(7)
    tight = 0
    total = 40
    for _ in range(total):
        n = int(rng.integers(4, 9)); m = int(rng.integers(2, 4))
        p = [int(x) for x in rng.integers(1, 20, size=n)]
        ub = lpt(PMSPInstance(p=p, m=m, name="t")).makespan
        lb, proven, _ = lp_lower_bound(p, m, ub)
        opt = brute_force(p, m)
        assert lb <= opt <= ub, (p, m, lb, opt, ub)      # nunca excede o otimo
        if proven:
            assert lb == opt
        tight += (lb == opt)
    assert tight >= 0.85 * total


def test_improved_lower_bound_is_valid_and_not_weaker():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = int(rng.integers(4, 9)); m = int(rng.integers(2, 4))
        p = [int(x) for x in rng.integers(1, 15, size=n)]
        lb = improved_lower_bound(p, m)
        assert lb <= brute_force(p, m)
        assert lb >= max(max(p), -(-sum(p) // m))


def test_returns_upper_bound_when_lb_equals_it():
    lb, proven, _ = lp_lower_bound([5, 5, 5, 5], 2, 10)
    assert lb == 10 and proven
