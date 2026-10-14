"""R||Cmax (Entrada 118): corretude da instancia, solucao, IG/ILS, LP e MILP contra forca bruta."""
import itertools

import numpy as np

from experiment import config as CFG
from pmsp.qlearning import QLearningAgent
from pmsp.selectors import NormalisedUCBSelector, RandomSelector
from pmsp.fixed_selector import FixedActionSelector
from pmsp.unrelated import (N_UR_DESTROY, N_UR_OPS, UnrelatedInstance, UrSol, UrStateFn, destroy_reconstruct, mct_greedy, ur_ig_run,
                            ur_ils_run, _best_move, _best_swap)
from pmsp.unrelated_exact import lp_bound, milp_solve


def _brute(inst):
    best = None
    for a in itertools.product(range(inst.m), repeat=inst.n):
        ms = UrSol(inst, np.array(a)).makespan
        best = ms if best is None or ms < best else best
    return best


def test_instance_bound_and_solution_consistency():
    inst = UnrelatedInstance.random(12, 4, 1, 100, seed=3)
    assert inst.lower_bound >= int(inst.p.min(axis=0).max())
    s = mct_greedy(inst)
    assert s.check() and s.makespan >= inst.lower_bound
    s2 = s.copy()
    s2.move(0, (s.a[0] + 1) % inst.m)
    assert s2.check() and s.check()


def test_destroy_reconstruct_keeps_solution_valid_for_all_strategies():
    inst = UnrelatedInstance.random(30, 5, 10, 100, seed=1)
    rng = np.random.default_rng(0)
    s = mct_greedy(inst)
    for st in range(N_UR_DESTROY):
        new = destroy_reconstruct(st, s, 9, rng)
        assert new.check() and set(new.a.tolist()) <= set(range(inst.m)) and s.check()


def test_local_search_operators_never_worsen_and_keep_consistency():
    inst = UnrelatedInstance.random(25, 5, 1, 100, seed=5)
    s = mct_greedy(inst)
    for op in (_best_move, _best_swap):
        t = s.copy()
        before = t.makespan
        gain = op(t)
        assert t.check() and t.makespan <= before and gain == before - t.makespan


def test_heuristics_bounded_by_brute_force_optimum_on_tiny_instances():
    for seed in range(5):
        inst = UnrelatedInstance.random(7, 3, 1, 30, seed=seed)
        opt = _brute(inst)
        assert opt >= inst.lower_bound
        g = mct_greedy(inst).makespan
        ig = ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), max_iter=150, seed=seed)["makespan"]
        il = ur_ils_run(inst, RandomSelector(seed=seed, n_actions=N_UR_OPS), max_iter=150, seed=seed)["makespan"]
        assert opt <= ig <= g and opt <= il <= g


def test_lp_and_milp_against_brute_force():
    for seed in range(4):
        inst = UnrelatedInstance.random(7, 3, 1, 30, seed=seed)
        opt = _brute(inst)
        assert lp_bound(inst) <= opt
        inc, dual, proven, t = milp_solve(inst, time_limit=20)
        assert inc == opt and dual <= opt and proven


def test_ig_default_is_deterministic_and_state_fn_works():
    inst = UnrelatedInstance.random(40, 6, 10, 100, seed=2)
    thr = {"f1": [0.3, 0.7], "f2": [0.3, 0.7]}
    mk = []
    for _ in range(2):
        ag = QLearningAgent(seed=4, n_actions=N_UR_DESTROY)
        mk.append(ur_ig_run(inst, ag, max_iter=120, seed=4, state_fn=UrStateFn(thr))["makespan"])
    assert mk[0] == mk[1]
    r = ur_ig_run(inst, FixedActionSelector(3), max_iter=80, seed=1)
    assert r["makespan"] <= mct_greedy(inst).makespan
    ucb = ur_ig_run(inst, NormalisedUCBSelector(C=1.4, seed=1, n_actions=N_UR_DESTROY), max_iter=80, seed=1)
    assert ucb["makespan"] <= mct_greedy(inst).makespan
