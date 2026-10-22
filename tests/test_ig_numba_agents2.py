"""Equivalencia, invariantes e controle positivo dos agentes compilados da Entrada 184 (pmsp/ig_numba_agents2.py)."""
import numpy as np
import pytest

from pmsp import ig_numba_agents as G
from pmsp import ig_numba_agents2 as H


def _lpt_assign(p, m):
    order = np.argsort(p)[::-1]
    loads = np.zeros(m, dtype=np.int64)
    a = np.empty(len(p), dtype=np.int64)
    for j in order:
        k = int(loads.argmin())
        a[j] = k
        loads[k] += p[j]
    return a


def _setup(n, m, seed):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 101, size=n).astype(np.int64)
    lb = max(int(np.ceil(p.sum() / m)), int(p.max()))
    return p, _lpt_assign(p, m), np.argsort(p)[::-1].astype(np.int64), lb


def _table(A, n, ls=0):
    strat = np.array([k % 3 for k in range(A)], dtype=np.int64)
    d = np.array([max(2, round(f * n)) for f in np.linspace(0.05, 0.4, A)], dtype=np.int64)
    return strat, d, np.full(A, ls, dtype=np.int64)


def _run2(p, a0, m, od, lb, tab, T, agent, seed, lr=0.1, lr2=0.1, fixed=0, mode=0, parm=0, bonus=0.0):
    return H.ig_agent_core2(p, a0, m, tab[0], tab[1], tab[2], T, agent, fixed, lr, lr2, seed, od, lb, mode, parm, bonus)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("agent", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("ls", [0, 1])
def test_old_agents_reproduce_the_frozen_core_exactly(seed, agent, ls):
    n, m = 50, 5
    p, a0, od, lb = _setup(n, m, seed)
    tab = _table(6, n, ls)
    ref = G.ig_agent_core(p, a0, m, tab[0], tab[1], tab[2], 250, agent, 2, 0.1, seed, od, lb)
    got = _run2(p, a0, m, od, lb, tab, 250, agent, seed, fixed=2)
    assert ref[0] == got[0] and ref[1] == got[1] and ref[2] == got[2] and np.array_equal(ref[3], got[3])


def test_zero_bonus_changes_nothing_even_with_a_planted_arm():
    n, m = 50, 5
    p, a0, od, lb = _setup(n, m, 3)
    tab = _table(6, n)
    base = _run2(p, a0, m, od, lb, tab, 250, 2, 11)
    pl = _run2(p, a0, m, od, lb, tab, 250, 2, 11, mode=1, parm=3, bonus=0.0)
    assert base[0] == pl[0] and np.array_equal(base[3], pl[3])


@pytest.mark.parametrize("agent", [5, 6, 7])
def test_frozen_learner_is_uniform(agent):
    """alfa = 0 (PM, AP) ou rho = 0 (ALNS) e beta = 0: os pesos nao mudam e a escolha e' uniforme."""
    n, m, A = 60, 6, 5
    p, a0, od, lb = _setup(n, m, 1)
    tab = _table(A, n)
    used = np.zeros(A)
    for sd in range(20):
        used += _run2(p, a0, m, od, lb, tab, 500, agent, sd, lr=0.0, lr2=0.0)[3]
    share = used / used.sum()
    assert np.all(np.abs(share - 1.0 / A) < 0.03), share


@pytest.mark.parametrize("agent", [2, 3, 4, 5, 6, 7])
def test_learners_capture_a_planted_global_reward(agent):
    """Controle positivo: com um bonus grande numa acao fixa, todo aprendiz passa a escolhe-la muito mais que 1/A na segunda metade."""
    n, m, A, T = 60, 6, 6, 1000
    p, a0, od, lb = _setup(n, m, 2)
    tab = _table(A, n)
    hits = 0
    for sd in range(10):
        hits += _run2(p, a0, m, od, lb, tab, T, agent, sd, lr=0.1 if agent != 4 else 0.3, lr2=0.3, mode=1, parm=4, bonus=5.0)[4]
    share = hits / (10 * (T - T // 2))
    assert share > 2.0 / A, (agent, share)


@pytest.mark.parametrize("agent", [5, 6, 7])
def test_probability_learners_never_zero_out_an_arm(agent):
    n, m, A = 60, 6, 6
    p, a0, od, lb = _setup(n, m, 5)
    tab = _table(A, n)
    out = _run2(p, a0, m, od, lb, tab, 1000, agent, 9, lr=0.5, lr2=0.8, mode=1, parm=0, bonus=5.0)
    assert out[3].sum() == 1000
    assert np.all(out[3] > 0)


def test_acceptance_is_greedy_so_best_never_exceeds_the_start():
    n, m = 50, 5
    p, a0, od, lb = _setup(n, m, 7)
    start = np.zeros(m, dtype=np.int64)
    for j in range(n):
        start[a0[j]] += p[j]
    tab = _table(5, n)
    for agent in range(8):
        assert _run2(p, a0, m, od, lb, tab, 300, agent, 4)[0] <= start.max()
