"""Entrada 289: testes do hospedeiro de evolucao diferencial (pmsp/de_host.py)."""
from __future__ import annotations

import numpy as np
import pytest

from pmsp import de_host as DH


def test_run_is_deterministic_and_bounded():
    a = DH.run_de(1, 6, "arm0", 5, T=60, log=True)
    b = DH.run_de(1, 6, "arm0", 5, T=60, log=True)
    assert a["err"] == b["err"] and np.array_equal(a["Rw"], b["Rw"])
    assert (a["Rw"] >= 0).all() and (a["Rw"] <= 1).all()
    assert (a["A"] == 0).all()                                     # a acao fixa so' usa a propria estrategia
    assert a["S"].min() >= 0 and a["S"].max() <= 80


def test_rand1_solves_the_sphere_and_random_control_logs_all_arms():
    r = DH.run_de(1, 7, "arm0", 3, T=1000)
    assert r["err"] < 1e-5
    q = DH.run_de(1, 7, "rand", 3, T=400, log=True)
    assert set(np.unique(q["A"])) == {0, 1, 2, 3, 4}


def test_generation_keeps_population_inside_the_domain():
    rng = np.random.default_rng(0)
    f, _ = DH.bbob(8, 6)
    pop = rng.uniform(DH.LOW, DH.HIGH, size=(100, 10))
    fit = np.asarray(f(pop), float)
    for a in range(5):
        before = fit.copy()
        s = DH.generation(pop, fit, a, rng, f)
        assert 0.0 <= s <= 1.0
        assert (pop >= DH.LOW).all() and (pop <= DH.HIGH).all()
        assert (fit <= before + 1e-12).all()                       # a selecao so' melhora
        assert np.allclose(fit, f(pop))


def test_selectors():
    # UCB1 puxa cada acao uma vez antes de qualquer outra coisa
    u = DH.UCB1(5, 0)
    first = []
    for t in range(5):
        a = u.select(0, t)
        first.append(a)
        u.update(a, 0.1, 0, 0)
    assert sorted(first) == [0, 1, 2, 3, 4]
    # Q-Learning com tabela zerada e epsilon 0: desempate pelo primeiro maximo escolhe sempre o indice 0; aleatorio escolhe todos
    q0, q1 = DH.QL(5, 1, 0), DH.QL(5, 1, 1)
    c0, c1 = set(), set()
    for t in range(200):
        q0.eps = q1.eps = 0.0                                      # o piso de epsilon e' reaplicado a cada chamada: zerar antes de cada uma
        c0.add(q0.select(3, t))
        c1.add(q1.select(3, t))
    assert c0 == {0} and len(c1) == 5
    # ETC compromete-se depois de m puxadas por acao, na melhor media
    e = DH.ETC(3, 0, m=4)
    for t in range(12):
        a = e.select(0, t)
        e.update(a, 1.0 if a == 2 else 0.1, 0, 0)
    assert e.commit == 2 and e.select(0, 99) == 2
    # PM e AP: distribuicoes validas
    pm, ap = DH.PM(5, 0), DH.AP(5, 0)
    for sel in (pm, ap):
        for t in range(40):
            a = sel.select(0, t)
            sel.update(a, 0.3 if a == 1 else 0.05, 0, 0)
    assert np.isclose((pm.pmin + (1 - 5 * pm.pmin) * pm.q / pm.q.sum()).sum(), 1.0)
    assert np.isclose(ap.p.sum(), 1.0, atol=1e-6) and ap.p.argmax() == 1


def test_strategies_differ_on_a_deceptive_function():
    errs = [DH.run_de(8, 6, f"arm{a}", 123, T=1000)["err"] for a in range(5)]
    assert max(errs) > 5 * min(errs)                               # as estrategias NAO sao redundantes
