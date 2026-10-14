"""Testes de pmsp/pfsp.py: cabecas/caudas/insercao contra a avaliacao ingenua; NEH; busca local; IG e ILS."""
import itertools

import numpy as np

from pmsp.pfsp import PfsInstance, best_insertion, heads, local_search, makespan, neh, pfsp_ig_run, pfsp_ils_run, tails
from pmsp.selectors import RandomSelector


def naive_makespan(p, perm):
    m = p.shape[1]
    c = np.zeros(m, dtype=np.int64)
    for j in perm:
        c[0] += p[j, 0]
        for i in range(1, m):
            c[i] = max(c[i], c[i - 1]) + p[j, i]
    return int(c[-1])


def test_makespan_matches_naive_recursion():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, m = int(rng.integers(2, 12)), int(rng.integers(2, 7))
        p = rng.integers(1, 100, size=(n, m))
        perm = rng.permutation(n)
        assert makespan(p, perm) == naive_makespan(p, perm)


def test_best_insertion_matches_bruteforce_and_tails_are_consistent():
    rng = np.random.default_rng(1)
    for _ in range(15):
        n, m = int(rng.integers(3, 10)), int(rng.integers(2, 6))
        p = rng.integers(1, 100, size=(n, m))
        perm = rng.permutation(n)
        job, rest = int(perm[0]), perm[1:]
        pos, ms = best_insertion(p, rest, job)
        brute = min(naive_makespan(p, np.insert(rest, k, job)) for k in range(len(rest) + 1))
        assert ms == brute
        assert naive_makespan(p, np.insert(rest, pos, job)) == ms
        t = tails(p, rest)
        e = heads(p, rest)
        # makespan da sequencia = max_i (cabeca[k, i] + cauda[k, i]) para qualquer k
        for k in range(len(rest)):
            assert int((e[k] + t[k]).max()) == makespan(p, rest)


def test_neh_is_a_permutation_and_local_search_never_worsens():
    inst = PfsInstance.random(20, 5, seed=3)
    perm = neh(inst.p)
    assert sorted(perm.tolist()) == list(range(inst.n))
    assert makespan(inst.p, perm) >= inst.lower_bound
    rng = np.random.default_rng(0)
    new, ms = local_search(inst.p, rng.permutation(inst.n), rng)
    assert ms == makespan(inst.p, new) and sorted(new.tolist()) == list(range(inst.n))
    perm2, ms2 = local_search(inst.p, perm, rng)
    assert ms2 <= makespan(inst.p, perm)


def test_ig_and_ils_return_valid_solutions_not_worse_than_neh():
    inst = PfsInstance.random(15, 5, seed=5)
    base = makespan(inst.p, neh(inst.p))
    r = pfsp_ig_run(inst, RandomSelector(seed=1, n_actions=3), [2, 4, 6], max_iter=20, seed=1)
    assert sorted(r["perm"].tolist()) == list(range(inst.n)) and r["makespan"] == makespan(inst.p, r["perm"]) and r["makespan"] <= base
    r2 = pfsp_ils_run(inst, max_iter=10, seed=1)
    assert sorted(r2["perm"].tolist()) == list(range(inst.n)) and r2["makespan"] == makespan(inst.p, r2["perm"])


def test_small_instance_ig_reaches_optimum():
    inst = PfsInstance.random(6, 3, seed=9)
    opt = min(naive_makespan(inst.p, np.array(q)) for q in itertools.permutations(range(6)))
    r = pfsp_ig_run(inst, RandomSelector(seed=0, n_actions=2), [2, 3], max_iter=40, seed=0)
    assert r["makespan"] == opt
