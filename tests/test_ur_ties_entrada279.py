"""Entrada 279: o nucleo com desempate (pmsp/ur_ties.py) reproduz o nucleo original com tie_mode = 0 e desempata ao acaso a cada decisao com tie_mode = 1."""
import numpy as np

from pmsp import ur_numba_agents5 as U5
from pmsp import ur_ties as UT
from pmsp.instance_ur import constructive_ur, lower_bound_ur

HP = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
ST = np.array([s for s in range(3) for _ in range(5)], dtype=np.int64)
D = np.array([2, 3, 4, 6, 8] * 3, dtype=np.int64)
LS = np.zeros(15, dtype=np.int64)


def _inst(seed=1):
    P = np.random.default_rng(seed).integers(1, 50, size=(30, 4)).astype(np.int64)
    a0, od = constructive_ur(P)
    return P, a0, od, float(lower_bound_ur(P))


def test_tie_mode_zero_reproduces_the_original_core_for_the_standard_learner():
    P, a0, od, lb = _inst()
    for seed in (5, 6):
        a = U5.ur_agent_core5(P, a0, 4, ST, D, LS, 200, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, 15)), 1.0)
        b = UT.ur_agent_core_ties(P, a0, 4, ST, D, LS, 200, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, 15)), 1.0, 0)
        assert float(a[0]) == float(b[0]) and float(a[2]) == float(b[2]) and a[3].tolist() == b[3].tolist()


def test_tie_mode_zero_reproduces_the_original_core_for_the_optimistic_learner():
    P, a0, od, lb = _inst(2)
    hp = np.array([0.3, 0.9, 1.0, 0.05, 0.0])
    a = U5.ur_agent_core5(P, a0, 4, ST, D, LS, 200, 2, 0, 0.1, 0.1, 9, od, lb, 0, 0, 0.0, hp, 2, np.full((81, 15), 0.5), 0.05)
    b = UT.ur_agent_core_ties(P, a0, 4, ST, D, LS, 200, 2, 0, 0.1, 0.1, 9, od, lb, 0, 0, 0.0, hp, 2, np.full((81, 15), 0.5), 0.05, 0)
    assert float(a[0]) == float(b[0]) and float(a[2]) == float(b[2]) and a[3].tolist() == b[3].tolist()


def test_constant_table_without_learning_first_maximum_repeats_arm_zero_random_ties_spread():
    P, a0, od, lb = _inst(3)
    hp = np.array([0.0, 0.9, 1.0, 0.0, 0.0])           # sem aprendizagem, epsilon 0: so' o desempate decide
    z = np.zeros((81, 15))
    first = UT.ur_agent_core_ties(P, a0, 4, ST, D, LS, 400, 2, 0, 0.1, 0.1, 4, od, lb, 0, 0, 0.0, hp, 0, z, 0.0, 0)[3]
    rand = UT.ur_agent_core_ties(P, a0, 4, ST, D, LS, 400, 2, 0, 0.1, 0.1, 4, od, lb, 0, 0, 0.0, hp, 0, z, 0.0, 1)[3]
    assert first[0] == 400 and first.sum() == 400
    assert rand.sum() == 400 and rand[0] < 100 and (rand > 0).sum() == 15       # todas as acoes usadas, nenhuma domina


def test_random_ties_are_deterministic_given_the_seed_and_a_strict_maximum_is_still_followed():
    P, a0, od, lb = _inst(4)
    hp = np.array([0.0, 0.9, 1.0, 0.0, 0.0])
    q0 = np.zeros((81, 15))
    q0[:, 11] = 1.0                                      # maximo estrito na acao 11 em todos os estados
    a = UT.ur_agent_core_ties(P, a0, 4, ST, D, LS, 300, 2, 0, 0.1, 0.1, 8, od, lb, 0, 0, 0.0, hp, 0, q0, 0.0, 1)
    b = UT.ur_agent_core_ties(P, a0, 4, ST, D, LS, 300, 2, 0, 0.1, 0.1, 8, od, lb, 0, 0, 0.0, hp, 0, q0, 0.0, 1)
    assert a[3].tolist() == b[3].tolist() and float(a[0]) == float(b[0])
    assert a[3][11] == 300
