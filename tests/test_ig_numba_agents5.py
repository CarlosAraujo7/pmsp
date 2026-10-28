"""Equivalencia, escala e aquecimento do Q-Learning (pmsp/ig_numba_agents5.py)."""
import numpy as np
import pytest

from pmsp import ig_numba_agents4 as H4
from pmsp import ig_numba_agents5 as H5


def _lpt_assign(p, m):
    order = np.argsort(p)[::-1]
    loads = np.zeros(m, dtype=np.int64)
    a = np.empty(len(p), dtype=np.int64)
    for j in order:
        k = int(loads.argmin())
        a[j] = k
        loads[k] += p[j]
    return a


def _setup(n, m, seed, scale=1):
    rng = np.random.default_rng(seed)
    p = (rng.integers(1, 101, size=n) * scale).astype(np.int64)
    lb = max(int(np.ceil(p.sum() / m)), int(p.max()))
    return p, _lpt_assign(p, m), np.argsort(p)[::-1].astype(np.int64), lb


def _table(A, n, ls=0):
    strat = np.array([k % 3 for k in range(A)], dtype=np.int64)
    d = np.array([max(2, round(f * n)) for f in np.linspace(0.05, 0.4, A)], dtype=np.int64)
    return strat, d, np.full(A, ls, dtype=np.int64)


def _run5(p, a0, m, od, lb, tab, T, agent, seed, lr=0.1, lr2=0.1, fixed=0, hp=None, rmode=0, Q0=None, eps0=1.0):
    hp = H5.HP_DEFAULT if hp is None else np.asarray(hp, dtype=np.float64)
    A = len(tab[0])
    Q0 = np.zeros((81, A)) if Q0 is None else Q0
    return H5.ig_agent_core5(p, a0, m, tab[0], tab[1], tab[2], T, agent, fixed, lr, lr2, seed, od, lb, 0, 0, 0.0, hp, rmode, Q0, eps0)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("agent", range(11))
@pytest.mark.parametrize("rmode", [0, 1])
def test_zero_q_and_eps1_reproduce_core4_exactly(seed, agent, rmode):
    n, m = 50, 5
    p, a0, od, lb = _setup(n, m, seed)
    tab = _table(6, n)
    hp = np.array([0.1, 0.9, 0.995, 0.05, 0.3])
    lr, lr2 = (0.4, 0.0) if agent == 10 else ((8.0, 0.02) if agent == 9 else (0.1, 0.3))
    ref = H4.ig_agent_core4(p, a0, m, tab[0], tab[1], tab[2], 250, agent, 2, lr, lr2, seed, od, lb, 0, 0, 0.0, hp, rmode)
    got = _run5(p, a0, m, od, lb, tab, 250, agent, seed, lr=lr, lr2=lr2, fixed=2, hp=hp, rmode=rmode)
    assert ref[0] == got[0] and ref[1] == got[1] and ref[2] == got[2] and np.array_equal(ref[3], got[3]) and ref[4] == got[4]


@pytest.mark.parametrize("seed", range(3))
def test_scale_free_reward_gives_the_same_decisions_for_q(seed):
    n, m = 60, 6
    p, a0, od, lb = _setup(n, m, seed)
    tab = _table(6, n)
    a = _run5(p, a0, m, od, lb, tab, 400, 2, seed, rmode=0)
    b = _run5(p, a0, m, od, lb, tab, 400, 2, seed, rmode=2)
    assert a[0] == b[0] and np.array_equal(a[3], b[3])


def test_scale_free_reward_is_invariant_to_instance_scale():
    """A mesma instancia com tempos x1000 da' a mesma tabela Q final (rmode 2)."""
    n, m = 40, 4
    p1, a1, od1, lb1 = _setup(n, m, 5, 1)
    p2, a2, od2, lb2 = _setup(n, m, 5, 1000)
    tab = _table(5, n)
    q1 = _run5(p1, a1, m, od1, lb1, tab, 300, 2, 3, rmode=2)[5]
    q2 = _run5(p2, a2, m, od2, lb1 * 1000, tab, 300, 2, 3, rmode=2)[5]                  # LB exatamente 1000 vezes (o teto da media muda com a escala)
    assert np.allclose(q1, q2, rtol=1e-9, atol=1e-9)


def test_warm_greedy_policy_follows_the_initial_table():
    n, m, A = 60, 6, 6
    p, a0, od, lb = _setup(n, m, 2)
    tab = _table(A, n)
    Q0 = np.zeros((81, A))
    Q0[:, 3] = 5.0                                              # acao 3 domina em todos os estados
    out = _run5(p, a0, m, od, lb, tab, 300, 2, 1, hp=[0.0, 0.9, 1.0, 0.0, 0.0], rmode=2, Q0=Q0, eps0=0.0)
    assert out[3][3] == 300                                      # sem aprendizado e sem exploracao: so' a acao 3


def test_final_table_is_returned_and_learns():
    n, m, A = 60, 6, 6
    p, a0, od, lb = _setup(n, m, 4)
    tab = _table(A, n)
    q = _run5(p, a0, m, od, lb, tab, 400, 2, 2, rmode=2)[5]
    assert q.shape == (81, A) and np.abs(q).sum() > 0
