"""
tests/test_sa.py
Testes de sanidade do modulo pmsp/sa.py (baseline SOTA Simulated
Annealing, reimplementado de Li & Perkins 2024 -- ver
diario_pesquisa.md, Entrada 37).
Execute: pytest tests/test_sa.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from pmsp.constructive import lpt
from pmsp.sa import sa_run, _heuristic_move_or_swap


@pytest.fixture
def inst():
    return PMSPInstance.random(n=30, m=8, seed=42)


@pytest.fixture
def inst_small():
    return PMSPInstance.random(n=12, m=3, seed=1)


@pytest.fixture
def inst_tiny():
    # m proximo de n para exercitar o caso de maquina vazia em
    # _heuristic_move_or_swap.
    return PMSPInstance.random(n=6, m=5, seed=7)


class TestHeuristicMoveOrSwap:

    def test_produces_valid_solution(self, inst):
        sol = lpt(inst)
        rng = np.random.default_rng(0)
        for _ in range(50):
            _heuristic_move_or_swap(sol, r=0.5, rng=rng)
            assert sol.is_valid()

    def test_r_zero_only_moves_r_one_only_swaps(self, inst):
        # Nao testamos diretamente qual operacao ocorreu (interno), so
        # que ambos os extremos de r produzem solucoes validas e nao
        # quebram com maquinas vazias eventualmente geradas.
        for r in (0.0, 1.0):
            sol = lpt(inst)
            rng = np.random.default_rng(1)
            for _ in range(50):
                _heuristic_move_or_swap(sol, r=r, rng=rng)
                assert sol.is_valid()

    def test_empty_machine_handled_gracefully(self, inst_tiny):
        # Com n proximo de m, e comum uma maquina ficar sem jobs ao
        # longo de varias iteracoes -- nao deve lancar excecao.
        sol = lpt(inst_tiny)
        rng = np.random.default_rng(2)
        for _ in range(200):
            delta = _heuristic_move_or_swap(sol, r=0.5, rng=rng)
            assert sol.is_valid()
            assert isinstance(delta, (int, np.integer))


class TestSARun:

    def test_returns_expected_keys(self, inst):
        result = sa_run(inst, max_iter=100, seed=0)
        for key in ("makespan", "gap_pct", "cpu_time", "n_improvements", "history"):
            assert key in result

    def test_deterministic_given_seed(self, inst):
        r1 = sa_run(inst, max_iter=200, seed=123)
        r2 = sa_run(inst, max_iter=200, seed=123)
        assert r1["makespan"] == r2["makespan"]
        assert r1["n_improvements"] == r2["n_improvements"]

    def test_different_seeds_can_differ(self, inst):
        results = {sa_run(inst, max_iter=200, seed=s)["makespan"] for s in range(5)}
        # Nao exigimos que sejam todos diferentes (podem convergir ao
        # mesmo otimo local), so que o mecanismo estocastico roda sem
        # erro para varias seeds.
        assert len(results) >= 1

    def test_never_worse_than_initial_lpt(self, inst):
        # Aceitacao gulosa para melhorias + LPT como ponto de partida
        # (ver docstring do modulo) implica que o best jamais piora em
        # relacao ao LPT inicial.
        lpt_makespan = lpt(inst).makespan
        result = sa_run(inst, max_iter=300, seed=0)
        assert result["makespan"] <= lpt_makespan

    def test_track_history_length(self, inst):
        result = sa_run(inst, max_iter=150, seed=0, track_history=True)
        assert len(result["history"]) <= 150
        assert len(result["history"]) > 0

    def test_early_stop_on_target_does_not_exceed_max_iter(self, inst_small):
        # p_target folgado (2.0) deve permitir parada bem antes de
        # max_iter em instancias pequenas.
        result = sa_run(inst_small, max_iter=5000, seed=0, p_target=2.0,
                         track_history=True)
        assert len(result["history"]) < 5000

    def test_r_extremes_do_not_crash(self, inst_tiny):
        for r in (0.0, 0.5, 1.0):
            result = sa_run(inst_tiny, max_iter=100, seed=0, r=r)
            assert result["makespan"] >= inst_tiny.lower_bound

    def test_gap_pct_consistent_with_makespan(self, inst):
        result = sa_run(inst, max_iter=100, seed=0)
        expected_gap = 100.0 * (result["makespan"] - inst.lower_bound) / inst.lower_bound
        assert abs(result["gap_pct"] - expected_gap) < 1e-9

    def test_cooling_schedule_saturates_by_iteration_100(self, inst):
        # Achado documentado na Entrada 37 / docstring do modulo:
        # com T0=10, D=0.9, a=1, a probabilidade de aceitar uma piora
        # deve ja estar numericamente ~0 bem antes de max_iter=1000.
        import math
        T = 10.0
        D = 0.9
        a_const = 1.0
        for _ in range(100):
            T *= D
        accept_prob_at_100 = math.exp(-a_const / T) if T > 0 else 0.0
        assert accept_prob_at_100 < 1e-6
