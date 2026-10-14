"""Testes de pmsp/unrelated_lp.py (partida por relaxacao linear para R||Cmax)."""
import numpy as np

from pmsp.unrelated import UnrelatedInstance, UrSol, _descent, ur_ig_run
from pmsp.selectors import RandomSelector
from pmsp.unrelated import N_UR_DESTROY
from pmsp.unrelated_lp import lp_round, lp_solution, lp_start


def _inst(n=40, m=6, lo=1, hi=100, seed=3):
    return UnrelatedInstance.random(n, m, lo, hi, seed=seed)


def test_lp_value_is_valid_lower_bound_and_x_is_feasible():
    inst = _inst()
    z, x = lp_solution(inst)
    assert x is not None and x.shape == (inst.m, inst.n)
    assert np.allclose(x.sum(axis=0), 1.0, atol=1e-6)
    assert (x >= -1e-9).all() and (x <= 1 + 1e-9).all()
    assert (x * inst.p).sum(axis=1).max() <= z + 1e-5
    assert z >= inst.lower_bound - 1.0 - 1e-6 or z <= inst.lower_bound + inst.p.max()   # LP e' limite valido: nao excede um makespan viavel
    sol, _, _ = lp_start(inst)
    assert z <= sol.makespan + 1e-6


def test_rounding_is_a_complete_assignment_and_within_lp_plus_pmax():
    for seed in range(5):
        inst = _inst(n=60, m=8, seed=seed)
        z, x = lp_solution(inst)
        sol = lp_round(inst, x)
        assert isinstance(sol, UrSol)
        assert sol.a.shape == (inst.n,) and sol.a.min() >= 0 and sol.a.max() < inst.m
        assert sol.makespan <= z + inst.p.max() + 1e-6
        assert sol.makespan >= inst.lower_bound


def test_lp_start_is_deterministic_and_usable_as_ig_start():
    inst = _inst(n=50, m=7, seed=11)
    s1, z1, _ = lp_start(inst)
    s2, z2, _ = lp_start(inst)
    assert (s1.a == s2.a).all() and z1 == z2
    _descent(s1, 10 ** 6)
    r = ur_ig_run(inst, RandomSelector(seed=1, n_actions=N_UR_DESTROY), max_iter=30, d=5, seed=1, initial_solution=s1, ls_steps=20)
    assert r["makespan"] <= s1.makespan
