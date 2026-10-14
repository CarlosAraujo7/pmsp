"""Testes do LDM e do IG inicializado pela melhor construtiva (Entrada 102)."""
import itertools

import numpy as np

from experiment.runner import run_single
from pmsp.constructive import lpt
from pmsp.instance import PMSPInstance
from pmsp.ldm import ldm
from pmsp.multifit import multifit
from pmsp.seeding import best_constructive


def _brute(p, m):
    best = sum(p)
    for a in itertools.product(range(m), repeat=len(p)):
        loads = [0] * m
        for j, x in enumerate(a):
            loads[x] += p[j]
        best = min(best, max(loads))
    return best


def test_ldm_is_a_valid_complete_assignment_and_respects_lb():
    for seed in range(5):
        inst = PMSPInstance.random(n=40, m=7, seed=seed, name="t")
        sol = ldm(inst)
        asg = np.asarray(sol.assignment)
        assert asg.shape == (40,) and asg.min() >= 0 and asg.max() < 7
        loads = np.zeros(7, dtype=np.int64)
        for j, a in enumerate(asg):
            loads[a] += inst.p[j]
        assert loads.sum() == int(np.sum(inst.p)) and int(loads.max()) == sol.makespan
        assert sol.makespan >= inst.lower_bound


def test_ldm_never_below_optimum_on_tiny_instances():
    rng = np.random.default_rng(3)
    for _ in range(25):
        n = int(rng.integers(4, 9)); m = int(rng.integers(2, 4))
        p = [int(x) for x in rng.integers(1, 20, size=n)]
        inst = PMSPInstance(p=p, m=m, name="t")
        assert ldm(inst).makespan >= _brute(p, m)


def test_ldm_known_example_two_machines():
    # Karmarkar-Karp classico: {8,7,6,5,4} em 2 conjuntos -> diferenca 2 (soma 30 -> 16/14)
    inst = PMSPInstance(p=[8, 7, 6, 5, 4], m=2, name="kk")
    assert ldm(inst).makespan == 16


def test_best_constructive_is_the_minimum_of_the_three():
    for seed in range(5):
        inst = PMSPInstance.random(n=60, m=9, seed=seed, name="t")
        best = best_constructive(inst).makespan
        assert best == min(lpt(inst).makespan, multifit(inst).makespan, ldm(inst).makespan)


def test_seeded_ig_never_worse_than_its_seed_and_is_deterministic():
    for seed in range(3):
        inst = PMSPInstance.random(n=50, m=8, seed=seed, name="t")
        r1 = run_single(inst, "ig_ql_seed", seed, max_iter=200)
        r2 = run_single(inst, "ig_ql_seed", seed, max_iter=200)
        assert r1["makespan"] <= best_constructive(inst).makespan
        assert r1["makespan"] == r2["makespan"]
        assert run_single(inst, "ldm", seed)["makespan"] == ldm(inst).makespan
