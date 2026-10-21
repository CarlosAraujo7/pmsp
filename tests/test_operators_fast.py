"""Equivalencia exata entre os operadores vetorizados e os originais (Entrada 162)."""
import numpy as np
import pytest

from pmsp.instance import PMSPInstance
from pmsp.operators import op1_best, op2_best
from pmsp.operators_fast import make_ls_fast, op1_best_fast, op2_best_fast
from pmsp.solution import PMSPSolution


def _random_solution(n, m, seed, pmax):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, pmax + 1, size=n)
    inst = PMSPInstance(p=p, m=m, name=f"t{n}x{m}s{seed}") if "name" in PMSPInstance.__init__.__code__.co_varnames else PMSPInstance(p=p, m=m)
    asg = rng.integers(0, m, size=n)
    return PMSPSolution(inst, asg)


CASES = [(n, m, s, pmax) for (n, m) in [(2, 2), (6, 3), (10, 2), (20, 4), (30, 8), (60, 20), (100, 20)] for s in range(12) for pmax in (3, 100)]


@pytest.mark.parametrize("n,m,seed,pmax", CASES)
def test_ops_equivalent(n, m, seed, pmax):
    for orig, fast in [(op1_best, op1_best_fast), (op2_best, op2_best_fast)]:
        a = _random_solution(n, m, seed, pmax)
        b = a.copy()
        ra, rb = orig(a, None), fast(b, None)
        assert ra == rb
        assert np.array_equal(a.assignment, b.assignment)
        assert np.array_equal(a.loads, b.loads)


@pytest.mark.parametrize("n,m,seed,pmax", CASES)
def test_op4_and_dispatch_equivalent(n, m, seed, pmax):
    from pmsp.operators import apply_best
    from pmsp.operators_fast import apply_best_fast, op4_best_fast
    from pmsp.operators import op4_best
    a = _random_solution(n, m, seed, pmax)
    b = a.copy()
    assert op4_best(a, None) == op4_best_fast(b, None)
    assert np.array_equal(a.assignment, b.assignment) and np.array_equal(a.loads, b.loads)
    for op in range(4):
        a2 = _random_solution(n, m, seed + 100, pmax)
        b2 = a2.copy()
        assert apply_best(op, a2, np.random.default_rng(0)) == apply_best_fast(op, b2, np.random.default_rng(0))
        assert np.array_equal(a2.assignment, b2.assignment)


def test_ils_best_fn_default_unchanged():
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.ils import ils_run
    from pmsp.operators_fast import apply_best_fast
    inst = PMSPInstance.random(n=30, m=8, seed=CFG.BASE_SEED, name="t")
    r0 = ils_run(instance=inst, selector=make_ils_selector("ils_random", CFG.BASE_SEED), max_iter=200, seed=CFG.BASE_SEED)
    r1 = ils_run(instance=inst, selector=make_ils_selector("ils_random", CFG.BASE_SEED), max_iter=200, seed=CFG.BASE_SEED, best_fn=apply_best_fast)
    assert r0["makespan"] == r1["makespan"] and r0["n_improvements"] == r1["n_improvements"]


@pytest.mark.parametrize("n,m,seed,pmax", [(30, 8, s, pmax) for s in range(10) for pmax in (3, 100)])
def test_local_search_equivalent(n, m, seed, pmax):
    from experiment.run_igls_p_entrada124 import make_ls
    a = _random_solution(n, m, seed, pmax)
    b = a.copy()
    make_ls()(a, None)
    make_ls_fast()(b, None)
    assert np.array_equal(a.assignment, b.assignment)
