"""Testes de pmsp/unrelated_srmilp.py (MILP restrito para polir solucoes de R||Cmax)."""
import numpy as np

from pmsp.unrelated import UnrelatedInstance, UrSol, mct_greedy
from pmsp.unrelated_srmilp import restricted_milp


def _inst(n=40, m=6, seed=2):
    return UnrelatedInstance.random(n, m, 1, 100, seed=seed)


def test_restricted_milp_never_worsens_and_returns_complete_assignment():
    for seed in range(4):
        inst = _inst(seed=seed)
        sol = mct_greedy(inst)
        new, mk, cpu = restricted_milp(inst, sol, k=3, time_limit=10.0)
        assert isinstance(new, UrSol)
        assert new.a.shape == (inst.n,) and new.a.min() >= 0 and new.a.max() < inst.m
        assert new.makespan <= sol.makespan
        assert new.makespan >= inst.lower_bound
        assert cpu >= 0.0


def test_restricted_model_contains_the_current_solution():
    # com k = 1 so' a maquina mais barata e a da solucao corrente sao permitidas: o problema continua viavel e o resultado nunca piora
    inst = _inst(n=30, m=5, seed=7)
    sol = mct_greedy(inst)
    new, mk, _ = restricted_milp(inst, sol, k=1, time_limit=10.0)
    assert new.makespan <= sol.makespan
    allowed = np.zeros((inst.m, inst.n), dtype=bool)
    allowed[np.argmin(inst.p, axis=0), np.arange(inst.n)] = True
    allowed[sol.a, np.arange(inst.n)] = True
    assert allowed[new.a, np.arange(inst.n)].all()
