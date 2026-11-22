"""Entrada 285: pmsp/ig_recon2.py reproduz o nucleo da Entrada 284 com a descida curta (ls_mode = 1) e oferece a descida completa (ls_mode = 2) ate' o otimo local, sem quebrar as invariantes."""
import numpy as np

from pmsp import ig_recon as RC
from pmsp import ig_recon2 as R2
from pmsp.ig_numba import init_state, local_search, op1_best, op2_best

BIG = 10 ** 6


def _lpt_assign(p, m):
    order = np.argsort(p)[::-1]
    loads = np.zeros(m, dtype=np.int64)
    a = np.empty(len(p), dtype=np.int64)
    for j in order:
        k = int(loads.argmin())
        a[j] = k
        loads[k] += p[j]
    return a


def _inst(seed=1, n=60, m=5, A=6, ls_all=False):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 101, size=n).astype(np.int64)
    st = np.zeros(A, dtype=np.int64)
    d = np.array([max(2, round(f * n)) for f in np.linspace(0.05, 0.4, A)], dtype=np.int64)
    ls = np.ones(A, dtype=np.int64) if ls_all else np.array([k % 2 for k in range(A)], dtype=np.int64)
    return p, _lpt_assign(p, m), m, st, d, ls, np.argsort(p)[::-1].astype(np.int64)


def _makespan(p, assign, m):
    return int(np.bincount(assign, weights=p, minlength=m).max())


def test_short_descent_mode_reproduces_the_entry_284_core():
    p, a0, m, st, d, ls, od = _inst()
    A = len(st)
    for seed in (3, 4):
        for rc in (0, 1):
            for agent, fixed in ((1, 0), (0, 3), (0, A - 1)):
                a = RC.ig_agent_core_recon(p, a0, m, st, d, ls, 150, agent, fixed, seed, od, rc)
                b = R2.ig_agent_core_recon2(p, a0, m, st, d, ls, 150, agent, fixed, seed, od, rc, 1, BIG)
                assert float(a[0]) == float(b[0]) and a[1] == b[1] and float(a[2]) == float(b[2]) and a[3].tolist() == b[3].tolist() and a[4].tolist() == b[4].tolist()
                assert b[5] == 0 and b[6] == 0 and b[7] == 0


def test_full_descent_reaches_a_local_optimum_of_both_neighbourhoods():
    rng = np.random.default_rng(5)
    for trial in range(6):
        n, m = 50 + 10 * trial, 4 + trial
        p = rng.integers(1, 101, size=n).astype(np.int64)
        assign = rng.integers(0, m, size=n).astype(np.int64)
        loads, mj, cnt = init_state(assign, p, m)
        before = int(loads.max())
        ev, moves, capped = R2.local_search_full(assign, loads, mj, cnt, p, BIG)
        assert capped == 0 and moves >= 1 and int(loads.max()) <= before
        assert int(loads.max()) == _makespan(p, assign, m)
        snap = assign.copy()
        r1, _ = op1_best(assign, loads, mj, cnt, p)
        r2, _ = op2_best(assign, loads, mj, cnt, p)
        assert r1 <= 0 and r2 <= 0 and (assign == snap).all()


def test_full_descent_is_never_worse_than_the_short_descent_and_often_better():
    rng = np.random.default_rng(6)
    better = 0
    for trial in range(8):
        n, m = 60, 6
        p = rng.integers(1, 101, size=n).astype(np.int64)
        assign = rng.integers(0, m, size=n).astype(np.int64)
        a1, a2 = assign.copy(), assign.copy()
        l1, mj1, c1 = init_state(a1, p, m)
        l2, mj2, c2 = init_state(a2, p, m)
        local_search(a1, l1, mj1, c1, p)
        R2.local_search_full(a2, l2, mj2, c2, p, BIG)
        assert int(l2.max()) <= int(l1.max())
        better += int(l2.max()) < int(l1.max())
    assert better >= 4


def test_full_descent_core_invariants_and_the_cap_counter():
    p, a0, m, st, d, ls, od = _inst(2, ls_all=True)
    for rc in (0, 1):
        best, n_imp, ev, used, assign, moves, capped, n_desc = R2.ig_agent_core_recon2(p, a0, m, st, d, ls, 120, 1, 0, 7, od, rc, 2, BIG)
        assert used.sum() == 120 and n_desc == 120 and capped == 0 and moves >= 0
        assert _makespan(p, assign, m) == int(best) and best <= _makespan(p, a0, m)
    # com max_moves = 1 a descida e' truncada sempre que ainda ha' melhoria
    out = R2.ig_agent_core_recon2(p, a0, m, st, d, ls, 120, 1, 0, 7, od, 1, 2, 1)
    assert out[6] > 0 and out[5] <= 120
