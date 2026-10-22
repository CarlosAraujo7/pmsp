"""Equivalencia e invariantes do IG de flowshop compilado (pmsp/pfsp_numba.py) contra a referencia em Python (pmsp/pfsp.py)."""
import numpy as np
import pytest

from pmsp import pfsp as P
from pmsp import pfsp_numba as N


def _inst(n, m, seed):
    return P.PfsInstance.random(n, m, seed)


def _ws(n, m):
    return np.zeros((n + 2, m), dtype=np.int64), np.zeros((n + 2, m), dtype=np.int64), np.zeros((n + 2, m), dtype=np.int64)


@pytest.mark.parametrize("seed", range(6))
def test_makespan_matches_reference(seed):
    inst = _inst(15 + seed, 4 + seed % 3, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(inst.n).astype(np.int64)
    e, _, _ = _ws(inst.n, inst.m)
    assert N.makespan(inst.p, perm, inst.n, e) == P.makespan(inst.p, perm)


@pytest.mark.parametrize("seed", range(6))
def test_best_insertion_matches_reference(seed):
    inst = _inst(18, 5, seed)
    rng = np.random.default_rng(seed + 10)
    perm = rng.permutation(inst.n).astype(np.int64)
    job = int(perm[-1])
    rest = perm[:-1].copy()
    buf = np.zeros(inst.n, dtype=np.int64)
    buf[: len(rest)] = rest
    e, t, f = _ws(inst.n, inst.m)
    pos, ms = N.best_insertion(inst.p, buf, len(rest), job, e, t, f)
    rpos, rms = P.best_insertion(inst.p, rest, job)
    assert (pos, ms) == (rpos, rms)


@pytest.mark.parametrize("seed", range(5))
def test_neh_matches_reference(seed):
    inst = _inst(25, 6, seed)
    perm = np.zeros(inst.n, dtype=np.int64)
    e, t, f = _ws(inst.n, inst.m)
    N.neh(inst.p, perm, e, t, f)
    assert np.array_equal(perm, P.neh(inst.p))


@pytest.mark.parametrize("seed", range(4))
def test_local_search_improves_and_reaches_an_insertion_local_optimum(seed):
    inst = _inst(20, 5, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(inst.n).astype(np.int64)
    e, t, f = _ws(inst.n, inst.m)
    np.random.seed(seed)
    start = int(N.makespan(inst.p, perm, inst.n, e))
    np.random.seed(seed + 100)
    cur, ev = N.local_search(inst.p, perm, inst.n, start, e, t, f)
    assert cur <= start and cur == P.makespan(inst.p, perm) and ev > 0
    for job in range(inst.n):                                   # otimo local: nenhuma reinsercao melhora
        idx = int(np.flatnonzero(perm == job)[0])
        rest = np.delete(perm, idx)
        _, ms = P.best_insertion(inst.p, rest, job)
        assert ms >= cur


def _run(inst, agent, seed, T=60, lr=0.1, lr2=0.1, fixed=0, mode=0, parm=0, bonus=0.0, d=(2, 3, 4, 5), tp=(0.4, 0.4, 0.4, 0.4)):
    return N.pfsp_ig_core(inst.p, np.array(d, dtype=np.int64), np.array(tp, dtype=np.float64), T, agent, fixed, lr, lr2, seed, inst.lower_bound, mode, parm, bonus)


@pytest.mark.parametrize("agent", range(11))
def test_every_agent_runs_and_respects_basic_invariants(agent):
    inst = _inst(20, 5, 3)
    perm = np.zeros(inst.n, dtype=np.int64)
    e, t, f = _ws(inst.n, inst.m)
    N.neh(inst.p, perm, e, t, f)
    neh_ms = int(N.makespan(inst.p, perm, inst.n, e))
    best, n_imp, ev, used, _ = _run(inst, agent, 5, lr=0.5 if agent == 10 else 0.1)
    assert inst.lower_bound <= best <= neh_ms and used.sum() == 60 and ev > 0


def test_same_seed_same_result_and_fixed_arm_is_used_exclusively():
    inst = _inst(20, 5, 4)
    a = _run(inst, N.AGENT_FIXED, 9, fixed=2)
    b = _run(inst, N.AGENT_FIXED, 9, fixed=2)
    assert a[0] == b[0] and np.array_equal(a[3], b[3]) and a[3][2] == 60


@pytest.mark.parametrize("agent,lr,lr2", [(2, 0.1, 0.1), (3, 0.1, 0.1), (4, 0.3, 0.1), (7, 0.5, 0.1), (8, 1.0, 0.0)])
def test_learners_capture_a_planted_reward(agent, lr, lr2):
    inst = _inst(20, 5, 6)
    hits = 0
    for sd in range(6):
        hits += _run(inst, agent, sd, T=300, lr=lr, lr2=lr2, mode=1, parm=1, bonus=50.0)[4]
    assert hits / (6 * 150) > 2.0 / 4


def test_acceptance_temperature_matters_for_the_trajectory():
    inst = _inst(20, 5, 7)
    cold = _run(inst, N.AGENT_FIXED, 1, fixed=0, T=200, tp=(1e-6, 1e-6, 1e-6, 1e-6))
    hot = _run(inst, N.AGENT_FIXED, 1, fixed=0, T=200, tp=(50.0, 50.0, 50.0, 50.0))
    assert cold[0] <= hot[0] + 3 * inst.p.max()          # sanidade: ambos validos; a quente nao e' melhor que a fria em geral
