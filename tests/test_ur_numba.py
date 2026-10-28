"""Testes do IG de maquinas nao relacionadas compilado (pmsp/ur_numba.py, pmsp/ur_numba_agents.py, pmsp/instance_ur.py)."""
import numpy as np
import pytest

from pmsp import ur_numba as U
from pmsp import ur_numba_agents as UA
from pmsp.instance_ur import constructive_ur, lower_bound_ur, makespan_ur, random_ur


def loads_of(P, assign, m):
    out = np.zeros(m, dtype=np.int64)
    for j in range(P.shape[0]):
        out[assign[j]] += P[j, assign[j]]
    return out


def check_state(P, assign, loads, mj, cnt):
    m = P.shape[1]
    assert (loads == loads_of(P, assign, m)).all()
    seen = []
    for k in range(m):
        for q in range(cnt[k]):
            assert assign[mj[k, q]] == k
            seen.append(int(mj[k, q]))
    assert sorted(seen) == list(range(P.shape[0]))


def test_instance_and_bound():
    P = random_ur(30, 5, 7)
    assert P.shape == (30, 5) and P.min() >= 1 and P.max() <= 100
    assert (P == random_ur(30, 5, 7)).all() and not (P == random_ur(30, 5, 8)).all()
    lb = lower_bound_ur(P)
    a, od = constructive_ur(P)
    assert makespan_ur(P, a) >= lb
    assert sorted(od.tolist()) == list(range(30))
    mn = P.min(axis=1)
    assert all(mn[od[i]] >= mn[od[i + 1]] for i in range(29))


@pytest.mark.parametrize("seed", range(5))
def test_state_move_swap(seed):
    P = random_ur(25, 4, seed)
    rng = np.random.default_rng(seed)
    assign = rng.integers(0, 4, size=25).astype(np.int64)
    loads, mj, cnt = U.init_state(assign, P, 4)
    check_state(P, assign, loads, mj, cnt)
    for _ in range(30):
        j = int(rng.integers(0, 25))
        U.move_job(assign, loads, mj, cnt, P, j, int(rng.integers(0, 4)))
        check_state(P, assign, loads, mj, cnt)
    for _ in range(30):
        ja, jb = (int(x) for x in rng.integers(0, 25, size=2))
        if assign[ja] != assign[jb]:
            U.swap_jobs(assign, loads, mj, cnt, P, ja, jb)
            check_state(P, assign, loads, mj, cnt)


def py_reconstruct(P, assign, removed):
    """referencia em Python: ordena por min_k P decrescente (estavel) e poe na maquina de menor termino (primeiro minimo)"""
    m = P.shape[1]
    assign = assign.copy()
    loads = loads_of(P, assign, m)
    for j in removed:
        loads[assign[j]] -= P[j, assign[j]]
    order = sorted(removed, key=lambda j: -P[j].min())
    for j in order:
        k = int(np.argmin(loads + P[j]))
        loads[k] += P[j, k]
        assign[j] = k
    return assign


@pytest.mark.parametrize("seed", range(6))
def test_reconstruct_matches_reference(seed):
    n, m, d = 30, 5, 7
    P = random_ur(n, m, seed)
    rng = np.random.default_rng(100 + seed)
    assign = rng.integers(0, m, size=n).astype(np.int64)
    removed = rng.choice(n, size=d, replace=False).astype(np.int64)
    loads, mj, cnt = U.init_state(assign, P, m)
    ev = U.reconstruct(assign, loads, mj, cnt, P, removed, d)
    assert ev == d * m
    check_state(P, assign, loads, mj, cnt)
    # referencia aplicada ao estado inicial (antes da reconstrucao): reconstroi a mesma atribuicao
    rng2 = np.random.default_rng(100 + seed)
    a0 = rng2.integers(0, m, size=n).astype(np.int64)
    rem2 = rng2.choice(n, size=d, replace=False).astype(np.int64)
    assert (py_reconstruct(P, a0, rem2.tolist()) == assign).all()


def py_local_search(P, assign):
    """referencia: ate' 2 passes de (melhor inter-troca; se nao melhorou, melhor reinsercao), mesma ordem de varredura e desempate"""
    m = P.shape[1]
    assign = assign.copy()

    def mjobs(a):
        return [[j for j in range(len(a)) if a[j] == k] for k in range(m)]

    for _ in range(2):
        loads = loads_of(P, assign, m)
        a = int(np.argmax(loads))
        old = loads[a]
        mj = mjobs(assign)
        best, bja, bjb = 0, -1, -1
        for ja in mj[a]:
            for b in range(m):
                if b == a:
                    continue
                for jb in mj[b]:
                    nl = loads.copy()
                    nl[a] += P[jb, a] - P[ja, a]
                    nl[b] += P[ja, b] - P[jb, b]
                    nc = nl.max()
                    if old - nc > best:
                        best, bja, bjb = old - nc, ja, jb
        improved = best > 0
        if improved:
            ma, mb = assign[bja], assign[bjb]
            assign[bja], assign[bjb] = mb, ma
        else:
            best, bj, bt = 0, -1, -1
            for j in mj[a]:
                for b in range(m):
                    if b == a:
                        continue
                    nl = loads.copy()
                    nl[a] -= P[j, a]
                    nl[b] += P[j, b]
                    nc = nl.max()
                    if old - nc > best:
                        best, bj, bt = old - nc, j, b
            if best > 0:
                assign[bj] = bt
                improved = True
            else:
                break
    return assign


@pytest.mark.parametrize("seed", range(8))
def test_local_search_matches_reference(seed):
    n, m = 24, 4
    P = random_ur(n, m, seed)
    rng = np.random.default_rng(seed)
    assign = rng.integers(0, m, size=n).astype(np.int64)
    ref = py_local_search(P, assign)
    loads, mj, cnt = U.init_state(assign, P, m)
    before = loads.max()
    ev = U.local_search(assign, loads, mj, cnt, P)
    check_state(P, assign, loads, mj, cnt)
    assert loads.max() <= before and ev > 0
    assert (assign == ref).all()


def test_agent_core_basic():
    n, m = 40, 6
    P = random_ur(n, m, 3)
    a0, od = constructive_ur(P)
    lb = lower_bound_ur(P)
    st = np.zeros(5, dtype=np.int64)
    d = np.array([2, 4, 8, 12, 16], dtype=np.int64)
    ls = np.zeros(5, dtype=np.int64)
    hp = UA.HP_DEFAULT
    start = makespan_ur(P, a0)
    for agent in (0, 1, 2, 3, 4, 7, 8):
        best, n_imp, ev, used, _ = UA.ur_agent_core(P, a0, m, st, d, ls, 200, agent, 1, 0.1, 0.1, 11, od, lb, 0, 0, 0.0, hp, 0)
        assert best <= start and int(used.sum()) == 200 and ev > 0
        again = UA.ur_agent_core(P, a0, m, st, d, ls, 200, agent, 1, 0.1, 0.1, 11, od, lb, 0, 0, 0.0, hp, 0)
        assert again[0] == best and (again[3] == used).all()
    best0, _, _, used0, _ = UA.ur_agent_core(P, a0, m, st, d, ls, 100, 0, 3, 0.1, 0.1, 5, od, lb, 0, 0, 0.0, hp, 0)
    assert used0[3] == 100 and used0.sum() == 100
    assert best0 >= lb


def test_core5_reproduces_core_and_optimistic_q_runs():
    from pmsp import ur_numba_agents5 as U5
    n, m = 40, 6
    P = random_ur(n, m, 4)
    a0, od = constructive_ur(P)
    lb = lower_bound_ur(P)
    st = np.zeros(5, dtype=np.int64)
    d = np.array([2, 4, 8, 12, 16], dtype=np.int64)
    ls = np.zeros(5, dtype=np.int64)
    for agent in (1, 2, 3, 4):
        a = UA.ur_agent_core(P, a0, m, st, d, ls, 150, agent, 0, 0.1, 0.1, 21, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
        b = U5.ur_agent_core5(P, a0, m, st, d, ls, 150, agent, 0, 0.1, 0.1, 21, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0, np.zeros((81, 5)), 1.0)
        assert a[0] == b[0] and (a[3] == b[3]).all()
    hp = np.array([0.1, 0.9, 1.0, 0.05, 0.0])
    out = U5.ur_agent_core5(P, a0, m, st, d, ls, 150, 2, 0, 0.1, 0.1, 21, od, lb, 0, 0, 0.0, hp, 2, np.full((81, 5), 0.1), 0.05)
    assert out[0] <= makespan_ur(P, a0) and int(out[3].sum()) == 150 and out[5].shape == (81, 5)
