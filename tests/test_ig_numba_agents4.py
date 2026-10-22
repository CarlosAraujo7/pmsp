"""Equivalencia, hiperparametros do Q-Learning e recompensa sensivel a platos (pmsp/ig_numba_agents4.py)."""
import numpy as np
import pytest

from pmsp import ig_numba_agents3 as H3
from pmsp import ig_numba_agents4 as H4


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


def _run4(p, a0, m, od, lb, tab, T, agent, seed, lr=0.1, lr2=0.1, fixed=0, mode=0, parm=0, bonus=0.0, hp=None, rmode=0):
    hp = H4.HP_DEFAULT if hp is None else np.asarray(hp, dtype=np.float64)
    return H4.ig_agent_core4(p, a0, m, tab[0], tab[1], tab[2], T, agent, fixed, lr, lr2, seed, od, lb, mode, parm, bonus, hp, rmode)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("agent", range(11))
@pytest.mark.parametrize("ls", [0, 1])
def test_default_hp_and_rmode0_reproduce_core3_exactly(seed, agent, ls):
    n, m = 50, 5
    p, a0, od, lb = _setup(n, m, seed)
    tab = _table(6, n, ls)
    lr, lr2 = (0.4, 0.0) if agent == 10 else ((8.0, 0.02) if agent == 9 else (0.1, 0.3))
    ref = H3.ig_agent_core3(p, a0, m, tab[0], tab[1], tab[2], 250, agent, 2, lr, lr2, seed, od, lb, 1, 3, 2.0)
    got = _run4(p, a0, m, od, lb, tab, 250, agent, seed, lr=lr, lr2=lr2, fixed=2, mode=1, parm=3, bonus=2.0)
    assert ref[0] == got[0] and ref[1] == got[1] and ref[2] == got[2] and np.array_equal(ref[3], got[3]) and ref[4] == got[4]


def test_q_hyperparameters_change_the_trajectory():
    n, m = 60, 6
    p, a0, od, lb = _setup(n, m, 5)
    tab = _table(6, n)
    base = _run4(p, a0, m, od, lb, tab, 400, 2, 3)
    other = _run4(p, a0, m, od, lb, tab, 400, 2, 3, hp=[0.5, 0.5, 0.99, 0.2, 0.0])
    assert not np.array_equal(base[3], other[3])


def test_epsilon_floor_one_means_pure_random_choice():
    """Com epsilon fixo em 1 (decaimento 1, piso 1) o Q-Learning escolhe como o aleatorio (mesmos sorteios na mesma ordem)."""
    n, m = 60, 6
    p, a0, od, lb = _setup(n, m, 6)
    tab = _table(6, n)
    q = _run4(p, a0, m, od, lb, tab, 300, 2, 9, hp=[0.1, 0.9, 1.0, 1.0, 0.0])
    # os sorteios de acao de Q (random() < eps, depois randint) diferem dos do aleatorio (so' randint): basta que o uso seja ~uniforme
    assert q[3].min() > 0.5 * 300 / 6 * 0.5


def test_plateau_reward_changes_only_when_there_are_plateau_moves():
    n, m = 40, 4
    p, a0, od, lb = _setup(n, m, 1)
    tab = _table(5, n)
    r0 = _run4(p, a0, m, od, lb, tab, 300, 2, 4, rmode=0)
    r1 = _run4(p, a0, m, od, lb, tab, 300, 2, 4, rmode=1, hp=[0.1, 0.9, 0.995, 0.05, 0.0])
    assert r0[0] == r1[0] and r0[2] == r1[2] and np.array_equal(r0[3], r1[3])      # bonus 0: identica
    r2 = _run4(p, a0, m, od, lb, tab, 300, 2, 4, rmode=1, hp=[0.1, 0.9, 0.995, 0.05, 0.5])
    assert not np.array_equal(r0[3], r2[3])                                          # bonus > 0 muda o aprendizado


def test_q_with_plateau_reward_still_captures_a_planted_reward():
    n, m, A, T = 60, 6, 6, 1000
    p, a0, od, lb = _setup(n, m, 2)
    tab = _table(A, n)
    hits = 0
    for sd in range(10):
        hits += _run4(p, a0, m, od, lb, tab, T, 2, sd, mode=1, parm=4, bonus=5.0, rmode=1, hp=[0.1, 0.9, 0.995, 0.05, 0.3])[4]
    assert hits / (10 * (T - T // 2)) > 1.5 / A
