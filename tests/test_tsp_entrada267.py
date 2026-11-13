"""Entrada 267: o ILS do TSP compilado (pmsp/tsp_numba.py): validade das operacoes, determinismo, uso das acoes e qualidade minima."""
import numpy as np

from pmsp import tsp_numba as TS
from pmsp.instance_tsp import build


def _valid(tour, pos):
    n = len(tour)
    assert sorted(tour.tolist()) == list(range(n))
    assert all(pos[tour[i]] == i for i in range(n))


def test_construct_gives_a_valid_local_optimum():
    D, NL = build("unif", 80, 1)
    tour, pos = TS.construct(D, NL)
    _valid(tour, pos)
    # nenhum 2-opt melhorador entre os candidatos das listas de vizinhos (criterio de ganho)
    n = len(tour)
    for i in range(n):
        c1 = tour[i]
        c2 = tour[(i + 1) % n]
        for c3 in NL[c1]:
            if D[c1, c3] >= D[c1, c2]:
                break
            c4 = tour[(pos[c3] + 1) % n]
            if c3 == c2 or c4 == c1:
                continue
            assert D[c1, c3] + D[c2, c4] - D[c1, c2] - D[c3, c4] >= -1e-6


def test_perturbation_then_local_search_keeps_a_permutation_and_does_not_worsen():
    D, NL = build("clust", 120, 2)
    tour0, pos0 = TS.construct(D, NL)
    n = len(tour0)
    rng_seed = 5
    for kind in range(3):
        for size in TS.SIZES[kind]:
            tour, pos = tour0.copy(), pos0.copy()
            stk = np.empty(n, dtype=np.int64)
            inq = np.zeros(n, dtype=np.int64)
            out = np.empty(n, dtype=np.int64)
            seg = np.empty(3, dtype=np.int64)
            buf = np.empty(n, dtype=np.int64)
            # semente do numba
            TS.tsp_agent_core(D, NL, tour0, np.array([0]), np.array([4]), 1, 0, 0, 0.1, 0.1, rng_seed, np.array([0.1, 0.9, 0.995, 0.05, 0.0]), 0.0)
            sp = TS._perturb(kind, int(size), tour, pos, stk, inq, 0, buf)
            _valid(tour, pos)
            perturbed = TS.tour_len(tour, D)
            TS.local_search(tour, pos, D, NL, stk, inq, sp, out, seg)
            _valid(tour, pos)
            assert TS.tour_len(tour, D) <= perturbed + 1e-6
            assert not inq.any()


def test_ils_is_deterministic_improves_and_uses_every_action_under_random_choice():
    D, NL = build("unif", 100, 3)
    tour0, _ = TS.construct(D, NL)
    kinds, sizes = TS.arm_table("T15")
    hp = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
    a = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 300, TS.AGENT_RANDOM, 0, 0.1, 0.1, 7, hp, 0.0)
    b = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 300, TS.AGENT_RANDOM, 0, 0.1, 0.1, 7, hp, 0.0)
    assert a[0] == b[0] and a[2] == b[2]
    assert a[0] <= TS.tour_len(tour0, D) + 1e-6
    assert a[3].sum() == 300 and (a[3] > 0).all()
    f = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 300, TS.AGENT_FIXED, 3, 0.1, 0.1, 7, hp, 0.0)
    assert f[3][3] == 300


def test_every_agent_runs_and_the_standard_q_repeats_the_first_action():
    D, NL = build("unif", 100, 4)
    tour0, _ = TS.construct(D, NL)
    kinds, sizes = TS.arm_table("T15")
    hp = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
    for ag, lr in ((TS.AGENT_Q, 0.1), (TS.AGENT_UCB, 0.1), (TS.AGENT_PG, 0.3), (TS.AGENT_ALNS, 0.1), (TS.AGENT_TS, 1.0)):
        r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 300, ag, 0, lr, 0.0, 11, hp, 0.0)
        assert r[3].sum() == 300 and r[0] > 0
    r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 300, TS.AGENT_QO, 0, 0.1, 0.0, 11, np.array([0.1, 0.9, 1.0, 0.05, 0.0]), 1.0)
    assert r[3].sum() == 300
    share0 = np.mean([TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 300, TS.AGENT_Q, 0, 0.1, 0.0, s, hp, 0.0)[3][0] / 300 for s in range(8)])
    assert share0 > 2.0 / 15            # the first-index lock-in of the standard Q-Learning shows up in this host too


def test_quality_is_close_to_the_random_uniform_constant():
    D, NL = build("unif", 200, 5)
    tour0, _ = TS.construct(D, NL)
    kinds, sizes = TS.arm_table("T5")
    r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 600, TS.AGENT_FIXED, 2, 0.1, 0.0, 3, np.array([0.1, 0.9, 0.995, 0.05, 0.0]), 0.0)
    ref = 0.7124 * np.sqrt(200 * 1e8) * (1 + 0.33 / 200 ** 0.5)       # Beardwood-Halton-Hammersley with a finite-size correction (rough)
    assert r[0] < 1.15 * ref
    assert r[0] < TS.tour_len(tour0, D)
