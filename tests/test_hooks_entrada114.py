"""Ganchos da Entrada 114: initial_solution no ILS e action_map no IG."""
import numpy as np

from experiment import config as CFG
from experiment.runner import make_ils_selector
from pmsp.constructive import lpt
from pmsp.ig import ig_run
from pmsp.ils import ils_run
from pmsp.instance import PMSPInstance
from pmsp.qlearning import QLearningAgent
from pmsp.seeding import best_constructive


def _inst(seed=2, n=40, m=6):
    return PMSPInstance.random(n=n, m=m, seed=seed, name="t")


def test_ils_default_unchanged_and_seeded_never_worse_than_seed():
    inst = _inst()
    a = ils_run(inst, make_ils_selector("ils_ql", 4), max_iter=120, seed=4, stagnation_window=CFG.STAGNATION_WINDOW)
    b = ils_run(inst, make_ils_selector("ils_ql", 4), max_iter=120, seed=4, stagnation_window=CFG.STAGNATION_WINDOW, initial_solution=None)
    assert a["makespan"] == b["makespan"]
    seed_sol = best_constructive(inst)
    c = ils_run(inst, make_ils_selector("ils_ql", 4), max_iter=120, seed=4, stagnation_window=CFG.STAGNATION_WINDOW, initial_solution=seed_sol)
    assert c["makespan"] <= seed_sol.makespan


def test_ig_action_map_changes_destruction_size_and_default_is_identical():
    inst = _inst(n=60, m=8)
    base = ig_run(inst, make_ils_selector("ig_random", 3), max_iter=100, seed=3, stagnation_window=CFG.STAGNATION_WINDOW)
    # action_map identica ao comportamento padrao -> mesmo resultado
    same = ig_run(inst, make_ils_selector("ig_random", 3), max_iter=100, seed=3, stagnation_window=CFG.STAGNATION_WINDOW,
                  action_map=lambda a, n: (a, max(2, round(0.3 * n))))
    assert same["makespan"] == base["makespan"]
    # 5 acoes = 5 tamanhos de destruicao
    fr = [0.05, 0.1, 0.2, 0.3, 0.4]
    ag = QLearningAgent(seed=1, n_actions=5)
    r = ig_run(inst, ag, max_iter=100, seed=1, stagnation_window=CFG.STAGNATION_WINDOW,
               action_map=lambda a, n: (0, max(2, round(fr[a] * n))))
    assert r["makespan"] <= lpt(inst).makespan


def test_ig_ls_fn_hook_default_identical_and_never_worse_than_start():
    from pmsp.operators import apply_best

    def ls(sol, rng):
        for _ in range(2):
            if apply_best(0, sol, rng) <= 0 and apply_best(1, sol, rng) <= 0:
                break

    inst = _inst(n=50, m=7)
    base = ig_run(inst, make_ils_selector("ig_random", 2), max_iter=100, seed=2, stagnation_window=CFG.STAGNATION_WINDOW)
    same = ig_run(inst, make_ils_selector("ig_random", 2), max_iter=100, seed=2, stagnation_window=CFG.STAGNATION_WINDOW, ls_fn=None)
    assert same["makespan"] == base["makespan"]
    r = ig_run(inst, make_ils_selector("ig_random", 2), max_iter=100, seed=2, stagnation_window=CFG.STAGNATION_WINDOW, ls_fn=ls)
    assert r["makespan"] <= lpt(inst).makespan
