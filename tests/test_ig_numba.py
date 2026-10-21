"""Equivalencia da busca local e da reconstrucao compiladas (numba) com a referencia, e invariantes do nucleo do IG (Entrada 170)."""
import numpy as np
import pytest

from pmsp import ig_numba as N
from pmsp.ig import apply_destroy_reconstruct
from pmsp.instance import PMSPInstance
from pmsp.operators_fast import make_ls_fast, op1_best_fast, op2_best_fast
from pmsp.solution import PMSPSolution


def _solution(n, m, seed, pmax, scramble=True):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, pmax + 1, size=n)
    inst = PMSPInstance(p=p, m=m)
    sol = PMSPSolution(inst, rng.integers(0, m, size=n))
    if scramble:                                   # embaralha a ordem de machine_jobs (como apos varias iteracoes)
        for _ in range(3 * n):
            sol.move_job(int(rng.integers(0, n)), int(rng.integers(0, m)))
    return sol


def _state(sol):
    p = sol.instance.p.astype(np.int64)
    assign = sol.assignment.astype(np.int64).copy()
    m, n = sol.instance.m, sol.instance.n
    loads = sol.loads.astype(np.int64).copy()
    mj = np.zeros((m, n), dtype=np.int32)
    cnt = np.zeros(m, dtype=np.int64)
    for k, jobs in enumerate(sol.machine_jobs):    # copia a ORDEM de machine_jobs da referencia
        mj[k, :len(jobs)] = jobs
        cnt[k] = len(jobs)
    return p, assign, loads, mj, cnt


def _same(sol, assign, loads, mj, cnt):
    assert np.array_equal(sol.assignment, assign)
    assert np.array_equal(sol.loads, loads)
    for k, jobs in enumerate(sol.machine_jobs):
        assert list(jobs) == list(mj[k, :cnt[k]])


CASES = [(n, m, s, pmax) for (n, m) in [(2, 2), (6, 3), (10, 2), (20, 4), (30, 8), (60, 20), (100, 20)] for s in range(8) for pmax in (3, 100)]


@pytest.mark.parametrize("n,m,seed,pmax", CASES)
def test_ops_exact(n, m, seed, pmax):
    for ref, new in [(op1_best_fast, N.op1_best), (op2_best_fast, N.op2_best)]:
        sol = _solution(n, m, seed, pmax)
        p, assign, loads, mj, cnt = _state(sol)
        r_ref = ref(sol, None)
        r_new, ev = new(assign, loads, mj, cnt, p)
        assert r_ref == r_new
        assert ev >= 0
        _same(sol, assign, loads, mj, cnt)


@pytest.mark.parametrize("n,m,seed,pmax", CASES)
def test_local_search_exact(n, m, seed, pmax):
    sol = _solution(n, m, seed, pmax)
    p, assign, loads, mj, cnt = _state(sol)
    make_ls_fast()(sol, None)
    ev = N.local_search(assign, loads, mj, cnt, p)
    assert ev >= 0
    _same(sol, assign, loads, mj, cnt)


@pytest.mark.parametrize("n,m,seed,pmax", [c for c in CASES if c[0] >= 10])
def test_reconstruct_exact(n, m, seed, pmax):
    sol = _solution(n, m, seed, pmax)
    rng = np.random.default_rng(seed + 7)
    d = int(rng.integers(2, max(3, n // 2)))
    removed = rng.choice(n, size=d, replace=False)
    # referencia: o mesmo corpo de apply_destroy_reconstruct, com a lista de removidas dada
    ref = sol.copy()
    p_ref = ref.instance.p
    tracker = ref.loads.copy()
    for j in removed:
        tracker[ref.assignment[j]] -= p_ref[j]
    for j in sorted(removed, key=lambda j: -p_ref[j]):
        k = int(tracker.argmin())
        tracker[k] += p_ref[j]
        ref.move_job(j, k)
    p, assign, loads, mj, cnt = _state(sol)
    ev = N.reconstruct(assign, loads, mj, cnt, p, removed.astype(np.int64), d)
    assert ev == d * m
    _same(ref, assign, loads, mj, cnt)


@pytest.mark.parametrize("n,m,seed,pmax", [c for c in CASES if c[0] >= 20])
def test_strategy1_full_step_exact(n, m, seed, pmax):
    """Estrategia 'mais caras' e' deterministica: destruir + reconstruir da referencia == destroy + reconstruct compilados."""
    sol = _solution(n, m, seed, pmax)
    d = max(2, round(0.3 * n))
    ref, _ = apply_destroy_reconstruct(1, sol, d, np.random.default_rng(0))
    p, assign, loads, mj, cnt = _state(sol)
    order_desc = np.argsort(sol.instance.p)[::-1].astype(np.int64)
    removed = np.empty(n, dtype=np.int64)
    N._destroy(1, assign, p, d, order_desc, np.arange(n), removed, m)
    N.reconstruct(assign, loads, mj, cnt, p, removed, d)
    _same(ref, assign, loads, mj, cnt)


def _lpt_assign(p, m):
    order = np.argsort(p)[::-1]
    loads = np.zeros(m, dtype=np.int64)
    a = np.empty(len(p), dtype=np.int64)
    for j in order:
        k = int(loads.argmin())
        a[j] = k
        loads[k] += p[j]
    return a


def _run(n, m, seed, T=200, ls=True, strat=-1, eval_budget=0, d=None, w_rec=1.0):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 101, size=n).astype(np.int64)
    a0 = _lpt_assign(p, m)
    d = d or max(2, round(0.3 * n))
    return p, a0, N.ig_core(p, a0, m, d, strat, T, eval_budget, ls, seed, np.argsort(p)[::-1].astype(np.int64), w_rec)


@pytest.mark.parametrize("n,m,seed", [(20, 4, 0), (50, 10, 1), (100, 20, 2), (150, 15, 3)])
@pytest.mark.parametrize("ls", [False, True])
@pytest.mark.parametrize("strat", [-1, 0, 1, 2])
def test_ig_core_invariants(n, m, seed, ls, strat):
    p, a0, (best, n_imp, ev, done, final) = _run(n, m, seed, ls=ls, strat=strat)
    start = np.bincount(a0, weights=p, minlength=m).max()
    loads = np.bincount(final, weights=p, minlength=m)
    assert done == 200
    assert best <= start                           # nunca piora
    assert loads.max() == best                     # a atribuicao final e' a melhor (aceita-se so' se nao piora)
    assert np.all((final >= 0) & (final < m))
    assert n_imp >= 0 and ev > 0
    lb = max(int(np.ceil(p.sum() / m)), int(p.max()))
    assert best >= lb


def test_ig_core_deterministic_and_eval_accounting():
    _, _, r1 = _run(60, 6, 5)
    _, _, r2 = _run(60, 6, 5)
    assert r1[:4] == r2[:4] and np.array_equal(r1[4], r2[4])
    # sem busca local, cada iteracao custa exatamente d x m avaliacoes
    n, m = 60, 6
    d = max(2, round(0.3 * n))
    _, _, r = _run(n, m, 5, ls=False, T=100)
    assert r[2] == 100 * d * m


def test_eval_budget_stops_the_run():
    n, m = 60, 6
    _, _, ref = _run(n, m, 3, ls=True, T=300)
    budget = ref[2]
    _, _, r = _run(n, m, 3, ls=False, T=10 ** 6, eval_budget=budget)
    d = max(2, round(0.3 * n))
    assert budget <= r[2] < budget + d * m          # para na primeira iteracao que atinge o orcamento
    assert r[3] == -(-budget // (d * m))


def test_reconstruction_weight_scales_the_count():
    n, m = 60, 6
    d = max(2, round(0.3 * n))
    _, _, r = _run(n, m, 5, ls=False, T=100, w_rec=0.25)
    assert r[2] == 100 * d * m * 0.25
    _, _, r1 = _run(n, m, 5, ls=True, T=100, w_rec=1.0)
    _, _, r4 = _run(n, m, 5, ls=True, T=100, w_rec=4.0)
    assert r4[2] - r1[2] == pytest.approx(3.0 * 100 * d * m)    # so' a parte da reconstrucao muda; a busca local conta o mesmo (mesma trajetoria: mesma semente)
