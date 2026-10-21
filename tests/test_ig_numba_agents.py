"""Equivalencia e invariantes dos agentes compilados (Entrada 180)."""
import numpy as np
import pytest

from pmsp import ig_numba as N
from pmsp import ig_numba_agents as G
from pmsp.bounds import compute_state
from pmsp.instance import PMSPInstance
from pmsp.solution import PMSPSolution


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


@pytest.mark.parametrize("seed", range(6))
def test_state_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(10, 80)), int(rng.integers(2, 10))
    p = rng.integers(1, 101, size=n)
    inst = PMSPInstance(p=p, m=m)
    sol = PMSPSolution(inst, rng.integers(0, m, size=n))
    for t, ino in [(0, 0), (120, 7), (500, 30), (999, 80), (1000, 0)]:
        want = compute_state(sol, t, 1000, ino, 50)
        got = G.state_id(sol.loads.astype(np.int64), inst.lower_bound, t, 1000, ino, 50)
        assert want == got


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ls", [0, 1])
def test_fixed_agent_is_ig_core(seed, ls):
    n, m = 60, 6
    p, a0, od, lb = _setup(n, m, seed)
    d = 18
    ref = N.ig_core(p, a0, m, d, 0, 300, 0.0, bool(ls), seed, od, 1.0)
    got = G.ig_agent_core(p, a0, m, np.array([0, 1], dtype=np.int64), np.array([d, 9], dtype=np.int64), np.array([ls, ls], dtype=np.int64), 300, G.AGENT_FIXED, 0, 0.1, seed, od, lb)
    assert ref[0] == got[0] and ref[1] == got[1] and ref[2] == got[2]


@pytest.mark.parametrize("seed", range(5))
def test_random_agent_is_ig_core_with_random_strategy(seed):
    n, m = 60, 6
    p, a0, od, lb = _setup(n, m, seed)
    d = 18
    ref = N.ig_core(p, a0, m, d, -1, 300, 0.0, False, seed, od, 1.0)
    got = G.ig_agent_core(p, a0, m, np.array([0, 1, 2], dtype=np.int64), np.full(3, d, dtype=np.int64), np.zeros(3, dtype=np.int64), 300, G.AGENT_RANDOM, 0, 0.1, seed, od, lb)
    assert ref[0] == got[0] and ref[1] == got[1] and ref[2] == got[2]


@pytest.mark.parametrize("agent", [G.AGENT_Q, G.AGENT_UCB, G.AGENT_PG])
@pytest.mark.parametrize("A", [3, 5, 15, 30])
def test_learners_invariants(agent, A):
    n, m = 80, 8
    p, a0, od, lb = _setup(n, m, 1)
    strat, d, ls = _table(A, n, 0)
    ls[A // 2:] = 1
    start = np.bincount(a0, weights=p, minlength=m).max()
    r1 = G.ig_agent_core(p, a0, m, strat, d, ls, 400, agent, 0, 0.1, 5, od, lb)
    r2 = G.ig_agent_core(p, a0, m, strat, d, ls, 400, agent, 0, 0.1, 5, od, lb)
    assert r1[0] == r2[0] and r1[2] == r2[2] and np.array_equal(r1[3], r2[3])      # deterministico dada a semente
    assert lb <= r1[0] <= start
    assert int(r1[3].sum()) == 400 and r1[3].min() >= 0                              # uma acao por iteracao
    assert r1[2] > 0


def test_ucb_tries_every_arm_first():
    n, m = 60, 6
    p, a0, od, lb = _setup(n, m, 2)
    strat, d, ls = _table(5, n)
    used = G.ig_agent_core(p, a0, m, strat, d, ls, 5, G.AGENT_UCB, 0, 0.1, 3, od, lb)[3]
    assert np.array_equal(used, np.ones(5, dtype=np.int64))


def test_policy_gradient_with_zero_learning_rate_is_uniform_and_learning_changes_the_policy():
    n, m = 100, 10
    p, a0, od, lb = _setup(n, m, 4)
    strat = np.array([0, 0], dtype=np.int64)
    d = np.array([2, 40], dtype=np.int64)
    ls = np.zeros(2, dtype=np.int64)
    u0 = sum(G.ig_agent_core(p, a0, m, strat, d, ls, 600, G.AGENT_PG, 0, 0.0, sd, od, lb)[3] for sd in range(20))
    u1 = sum(G.ig_agent_core(p, a0, m, strat, d, ls, 600, G.AGENT_PG, 0, 0.3, sd, od, lb)[3] for sd in range(20))
    assert u0.sum() == 20 * 600 and 0.45 < u0[0] / u0.sum() < 0.55                    # lr = 0: politica uniforme
    assert not np.array_equal(u0, u1)                                                   # lr > 0: a politica se move
