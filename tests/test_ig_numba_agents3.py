"""Equivalencia, invariantes e controle positivo dos agentes Thompson, DMAB e corrida (pmsp/ig_numba_agents3.py)."""
import numpy as np
import pytest

from pmsp import ig_numba_agents2 as H2
from pmsp import ig_numba_agents3 as H3


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


def _run3(p, a0, m, od, lb, tab, T, agent, seed, lr=0.1, lr2=0.1, fixed=0, mode=0, parm=0, bonus=0.0):
    return H3.ig_agent_core3(p, a0, m, tab[0], tab[1], tab[2], T, agent, fixed, lr, lr2, seed, od, lb, mode, parm, bonus)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("agent", range(8))
@pytest.mark.parametrize("ls", [0, 1])
def test_agents_0_to_7_reproduce_core2_exactly(seed, agent, ls):
    n, m = 50, 5
    p, a0, od, lb = _setup(n, m, seed)
    tab = _table(6, n, ls)
    ref = H2.ig_agent_core2(p, a0, m, tab[0], tab[1], tab[2], 250, agent, 2, 0.1, 0.3, seed, od, lb, 1, 3, 2.0)
    got = _run3(p, a0, m, od, lb, tab, 250, agent, seed, lr=0.1, lr2=0.3, fixed=2, mode=1, parm=3, bonus=2.0)
    assert ref[0] == got[0] and ref[1] == got[1] and ref[2] == got[2] and np.array_equal(ref[3], got[3]) and ref[4] == got[4]


@pytest.mark.parametrize("agent,lr,lr2", [(8, 1.0, 0.0), (8, 0.95, 0.0), (9, 8.0, 0.02), (10, 0.5, 0.0)])
def test_new_agents_capture_a_planted_global_reward(agent, lr, lr2):
    n, m, A, T = 60, 6, 6, 1000
    p, a0, od, lb = _setup(n, m, 2)
    tab = _table(A, n)
    hits = 0
    for sd in range(10):
        hits += _run3(p, a0, m, od, lb, tab, T, agent, sd, lr=lr, lr2=lr2, mode=1, parm=4, bonus=5.0)[4]
    assert hits / (10 * (T - T // 2)) > 2.0 / A


def test_race_visits_every_arm_then_commits_to_one():
    n, m, A, T = 60, 6, 5, 1000
    p, a0, od, lb = _setup(n, m, 4)
    tab = _table(A, n)
    out = _run3(p, a0, m, od, lb, tab, T, 10, 7, lr=0.4)
    used = out[3]
    assert used.sum() == T and np.all(used > 0)
    assert used.max() >= 0.55 * T                       # a vencedora joga a fracao exploratoria restante (>= 1 - rho) do orcamento


def test_race_commits_to_the_planted_arm():
    n, m, A, T = 60, 6, 5, 1000
    p, a0, od, lb = _setup(n, m, 4)
    tab = _table(A, n)
    out = _run3(p, a0, m, od, lb, tab, T, 10, 3, lr=0.4, mode=1, parm=2, bonus=5.0)
    assert int(out[3].argmax()) == 2 and out[3][2] >= 0.55 * T


def test_dmab_with_hair_trigger_restarts_stays_near_uniform():
    n, m, A, T = 60, 6, 5, 1000
    p, a0, od, lb = _setup(n, m, 6)
    tab = _table(A, n)
    used = np.zeros(A)
    for sd in range(10):
        used += _run3(p, a0, m, od, lb, tab, T, 9, sd, lr=0.0005, lr2=0.0)[3]
    share = used / used.sum()
    assert np.all(share > 0.5 / A), share


def test_thompson_exploits_when_one_arm_always_succeeds():
    n, m, A, T = 60, 6, 6, 1000
    p, a0, od, lb = _setup(n, m, 8)
    tab = _table(A, n)
    out = _run3(p, a0, m, od, lb, tab, T, 8, 5, lr=1.0, mode=1, parm=1, bonus=5.0)
    assert out[3][1] > 0.6 * T


@pytest.mark.parametrize("agent", [8, 9, 10])
def test_new_agents_keep_greedy_acceptance_and_count_every_iteration(agent):
    n, m = 50, 5
    p, a0, od, lb = _setup(n, m, 7)
    start = np.zeros(m, dtype=np.int64)
    for j in range(n):
        start[a0[j]] += p[j]
    tab = _table(5, n)
    out = _run3(p, a0, m, od, lb, tab, 300, agent, 4, lr=0.5 if agent == 10 else (0.99 if agent == 8 else 8.0))
    assert out[0] <= start.max() and out[3].sum() == 300
