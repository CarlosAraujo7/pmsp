"""Entrada 288: a contabilidade da linha zerada. Com desempate pelo primeiro maximo, a probabilidade de escolher o indice 0 e' pelo menos eps_t / K + (1 - eps_t) z_t; com desempate aleatorio, numa linha zerada ela e' 1 / K."""
from __future__ import annotations

import numpy as np

from pmsp import ig_logged as IL
from pmsp.constructive import lpt
from pmsp.instance import PMSPInstance

HP = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
T = 400


def setup(n=60, m=8, seed=4):
    inst = PMSPInstance.random(n=n, m=m, seed=seed, name="t")
    p = np.asarray(inst.p, dtype=np.int64)
    a0 = lpt(inst).assignment.astype(np.int64)
    od = np.argsort(inst.p)[::-1].astype(np.int64)
    d = np.array([max(2, round(f * n)) for f in (0.05, 0.1, 0.2, 0.3, 0.4)], dtype=np.int64)
    return p, a0, m, np.zeros(5, dtype=np.int64), d, np.zeros(5, dtype=np.int64), od, float(inst.lower_bound)


def shares(tie, seeds=60):
    p, a0, m, st, d, ls, od, lb = setup()
    eps = np.maximum(0.05, 0.995 ** np.arange(T))
    first, bound, zero = [], [], []
    for s in range(seeds):
        res = IL.ig_agent_core_logged(p, a0, m, st, d, ls, T, 2, 0, 0.1, 0.1, 1000 + s, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, 5)), 1.0, tie)
        la, lz = res[6], res[10]
        first.append(np.mean(la == 0))
        zero.append(np.mean(lz == 1))
        bound.append(np.mean(eps / 5 + (1 - eps) * lz))
    return float(np.mean(first)), float(np.mean(bound)), float(np.mean(zero))


def test_first_index_share_respects_the_zero_row_bound_and_lock_in():
    f, b, z = shares(tie=0)
    assert f >= b - 0.02                      # a cota nunca e' violada (tolerancia de amostragem)
    assert f > 0.35 and z > 0.5               # ha' aprisionamento no primeiro indice e a maioria das linhas esta' zerada


def test_random_ties_remove_the_default_to_the_first_index():
    f, b, z = shares(tie=1)
    assert abs(f - 0.2) < 0.05                # perto de 1/K
