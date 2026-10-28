"""Invariantes, determinismo e controle positivo do hospedeiro populacional (pmsp/ga_numba.py)."""
import numpy as np
import pytest

from pmsp import ga_numba as G


def _setup(n, m, seed, high=10000):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, high + 1, size=n).astype(np.int64)
    lb = max(int(np.ceil(p.sum() / m)), int(p.max()))
    order = np.argsort(p)[::-1]
    loads = np.zeros(m, dtype=np.int64)
    a0 = np.empty(n, dtype=np.int64)
    for j in order:
        k = int(loads.argmin())
        a0[j] = k
        loads[k] += p[j]
    return p, a0, np.argsort(p)[::-1].astype(np.int64), lb, int(loads.max())


def _tab(A=9):
    xo, mu = np.repeat([0, 1, 2], 3), np.tile([0, 1, 2], 3)
    if A == 9:
        return xo.astype(np.int64), mu.astype(np.int64), np.zeros(9, dtype=np.int64)
    return np.concatenate([xo, xo]).astype(np.int64), np.concatenate([mu, mu]).astype(np.int64), np.concatenate([np.zeros(9), np.ones(9)]).astype(np.int64)


def _run(p, a0, m, od, lb, tab, T, agent, seed, fixed=0, lr=0.1, lr2=0.1, mode=0, parm=0, bonus=0.0):
    return G.ga_agent_core(p, a0, m, tab[0], tab[1], tab[2], T, agent, fixed, lr, lr2, seed, od, lb, mode, parm, bonus, G.HP_DEFAULT, 0)


@pytest.mark.parametrize("agent", range(11))
def test_runs_use_the_whole_budget_and_the_best_is_consistent(agent):
    n, m = 80, 8
    p, a0, od, lb, lpt = _setup(n, m, agent)
    tab = _tab(18)
    lr, lr2 = (0.4, 0.0) if agent == 10 else ((8.0, 0.02) if agent == 9 else (0.1, 0.3))
    out = _run(p, a0, m, od, lb, tab, 1500, agent, 3, fixed=4, lr=lr, lr2=lr2)
    assert out[3].sum() == 1500 and out[2] == 1500
    ms = int(np.bincount(out[5], weights=p, minlength=m).max())
    assert ms == out[0] and lb <= out[0] <= lpt


def test_same_seed_same_result_different_seed_may_differ():
    p, a0, od, lb, _ = _setup(100, 10, 1)
    tab = _tab(9)
    a = _run(p, a0, 10, od, lb, tab, 3000, 1, 5)
    b = _run(p, a0, 10, od, lb, tab, 3000, 1, 5)
    c = _run(p, a0, 10, od, lb, tab, 3000, 1, 6)
    assert a[0] == b[0] and np.array_equal(a[3], b[3]) and np.array_equal(a[5], b[5])
    assert not np.array_equal(a[3], c[3])


def test_greedy_mutation_beats_blind_reassignment_on_average():
    """Acoes com mutacao gulosa (indice 1, 4, 7) valem mais que a realocacao cega (0, 3, 6) em tempos finos."""
    tab = _tab(9)
    good, blind = [], []
    for sd in range(6):
        p, a0, od, lb, _ = _setup(100, 10, 10 + sd)
        for a in (1, 4, 7):
            good.append(_run(p, a0, 10, od, lb, tab, 4000, 0, sd, fixed=a)[0] / lb)
        for a in (0, 3, 6):
            blind.append(_run(p, a0, 10, od, lb, tab, 4000, 0, sd, fixed=a)[0] / lb)
    assert np.mean(good) < np.mean(blind)


@pytest.mark.parametrize("agent,lr,lr2", [(2, 0.1, 0.1), (3, 0.1, 0.1), (8, 1.0, 0.0), (7, 0.3, 0.0)])
def test_agents_capture_a_planted_reward(agent, lr, lr2):
    p, a0, od, lb, _ = _setup(80, 8, 2)
    tab = _tab(9)
    hits = 0
    T = 1500
    for sd in range(6):
        hits += _run(p, a0, 8, od, lb, tab, T, agent, sd, lr=lr, lr2=lr2, mode=1, parm=5, bonus=5000.0)[4]
    assert hits / (6 * (T - T // 2)) > 1.5 / 9
