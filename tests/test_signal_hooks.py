"""Testes dos ganchos reward_fn/state_fn do IG, da recompensa densa e do seletor fixo (Entrada 104)."""
import numpy as np

from experiment import config as CFG
from experiment.runner import make_ils_selector
from pmsp.constructive import lpt
from pmsp.fixed_selector import FixedActionSelector
from pmsp.ig import ig_run, apply_destroy_reconstruct
from pmsp.instance import PMSPInstance
from pmsp.rewards import dense_balance_reward


def _inst(seed=1, n=40, m=6):
    return PMSPInstance.random(n=n, m=m, seed=seed, name="t")


def test_default_hooks_do_not_change_ig_ql():
    inst = _inst()
    a = ig_run(inst, make_ils_selector("ig_ql", 5), max_iter=150, seed=5, stagnation_window=CFG.STAGNATION_WINDOW)
    b = ig_run(inst, make_ils_selector("ig_ql", 5), max_iter=150, seed=5, stagnation_window=CFG.STAGNATION_WINDOW,
               reward_fn=None, state_fn=None)
    assert a["makespan"] == b["makespan"] and a["n_improvements"] == b["n_improvements"]


def test_dense_reward_is_nonzero_and_signed():
    inst = _inst(n=60, m=8)
    sol = lpt(inst)
    rng = np.random.default_rng(0)
    r = []
    for t in range(60):
        new, delta = apply_destroy_reconstruct(t % 3, sol, 18, rng)
        r.append(dense_balance_reward(sol, new, delta))
    r = np.array(r)
    assert (r != 0).mean() > 0.8 and r.min() < 0 < r.max()


def test_dense_reward_matches_balance_improvement():
    inst = PMSPInstance(p=[10, 10, 1, 1], m=2, name="x")
    from pmsp.solution import PMSPSolution
    bad = PMSPSolution(inst, np.array([0, 0, 1, 1]))     # cargas 20 / 2
    good = PMSPSolution(inst, np.array([0, 1, 0, 1]))    # cargas 11 / 11
    assert dense_balance_reward(bad, good, bad.makespan - good.makespan) > 0
    assert dense_balance_reward(good, bad, good.makespan - bad.makespan) < 0


def test_fixed_selector_always_returns_its_action_and_runs_in_ig():
    inst = _inst()
    s = FixedActionSelector(2)
    assert all(s.select(0, np.zeros(4)) == 2 for _ in range(10))
    r = ig_run(inst, s, max_iter=100, seed=1, stagnation_window=CFG.STAGNATION_WINDOW)
    assert r["makespan"] <= lpt(inst).makespan
