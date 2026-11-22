"""Entrada 284: o nucleo pmsp/ig_recon.py reproduz o IG compilado com a reconstrucao gulosa (recon_mode = 0) e oferece a reinsercao aleatoria (recon_mode = 1) sem quebrar as invariantes."""
import numpy as np

from pmsp import ig_numba_agents as G
from pmsp import ig_recon as RC
from pmsp.ig_numba import init_state, reconstruct


def _lpt_assign(p, m):
    order = np.argsort(p)[::-1]
    loads = np.zeros(m, dtype=np.int64)
    a = np.empty(len(p), dtype=np.int64)
    for j in order:
        k = int(loads.argmin())
        a[j] = k
        loads[k] += p[j]
    return a


def _inst(seed=1, n=60, m=5, A=6):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 101, size=n).astype(np.int64)
    lb = max(int(np.ceil(p.sum() / m)), int(p.max()))
    st = np.array([k % 3 for k in range(A)], dtype=np.int64)
    d = np.array([max(2, round(f * n)) for f in np.linspace(0.05, 0.4, A)], dtype=np.int64)
    ls = np.array([k % 2 for k in range(A)], dtype=np.int64)
    return p, _lpt_assign(p, m), m, st, d, ls, np.argsort(p)[::-1].astype(np.int64), lb


def _makespan(p, assign, m):
    return int(np.bincount(assign, weights=p, minlength=m).max())


def test_greedy_mode_reproduces_the_original_core_for_fixed_and_random_agents():
    p, a0, m, st, d, ls, od, lb = _inst()
    A = len(st)
    for seed in (3, 4, 5):
        for agent, fixed in ((1, 0), (0, 0), (0, 3), (0, A - 1)):
            a = G.ig_agent_core(p, a0, m, st, d, ls, 200, agent, fixed, 0.1, seed, od, lb)
            b = RC.ig_agent_core_recon(p, a0, m, st, d, ls, 200, agent, fixed, seed, od, 0)
            assert float(a[0]) == float(b[0]) and float(a[2]) == float(b[2]) and a[3].tolist() == b[3].tolist() and a[1] == b[1]


def test_random_reinsertion_keeps_the_invariants():
    p, a0, m, st, d, ls, od, lb = _inst(2)
    for seed in (1, 2):
        best, n_imp, ev, used, assign = RC.ig_agent_core_recon(p, a0, m, st, d, ls, 300, 1, 0, seed, od, 1)
        assert used.sum() == 300
        assert assign.min() >= 0 and assign.max() < m
        assert _makespan(p, assign, m) == int(best)
        assert best <= _makespan(p, a0, m)


def test_random_reinsertion_without_descent_evaluates_nothing_and_never_worsens_the_start():
    p, a0, m, st, d, ls, od, lb = _inst(3)
    zero = np.zeros(len(st), dtype=np.int64)
    best, n_imp, ev, used, assign = RC.ig_agent_core_recon(p, a0, m, st, d, zero, 200, 1, 0, 7, od, 1)
    assert ev == 0.0 and best <= _makespan(p, a0, m) and _makespan(p, assign, m) == int(best)


def test_random_reinsertion_spreads_jobs_over_all_machines_and_greedy_is_better_on_average():
    p, a0, m, st, d, ls, od, lb = _inst(4, n=80, m=6)
    zero = np.zeros(len(st), dtype=np.int64)
    g = [RC.ig_agent_core_recon(p, a0, m, st, d, zero, 300, 1, 0, s, od, 0)[0] for s in range(20)]
    r = [RC.ig_agent_core_recon(p, a0, m, st, d, zero, 300, 1, 0, s, od, 1)[0] for s in range(20)]
    assert np.mean(g) <= np.mean(r)
    # a reinsercao aleatoria usa as m maquinas: sorteio direto sobre um estado
    assign = a0.copy()
    loads, mj, cnt = init_state(assign, p, m)
    removed = np.arange(len(p), dtype=np.int64)
    np.random.seed(11)
    RC.reconstruct_random(assign, loads, mj, cnt, p, removed, len(p))
    assert len(np.unique(assign)) == m
    assert int(loads.sum()) == int(p.sum())
