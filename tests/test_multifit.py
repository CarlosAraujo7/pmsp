"""
tests/test_multifit.py
Testes de sanidade do modulo pmsp/multifit.py (baseline SOTA MULTIFIT,
Coffman/Garey/Johnson 1978 -- ver diario_pesquisa.md, Entrada 37).
Execute: pytest tests/test_multifit.py -v
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
from pmsp.multifit import multifit, multifit_run, _ffd_fits


@pytest.fixture
def inst():
    return PMSPInstance.random(n=30, m=8, seed=42)


@pytest.fixture
def inst_small():
    return PMSPInstance.random(n=12, m=3, seed=1)


class TestFFDFits:

    def test_hand_computed_example(self):
        # p=[8,5,3,2], m=2 -- ver docstring de multifit() para o
        # calculo manual completo. C=9 deve ser infactivel; C=10 deve
        # ser factivel (jobs 8+2=10 numa maquina, 5+3=8 na outra).
        inst = PMSPInstance(p=[8, 5, 3, 2], m=2)
        order = np.argsort(inst.p)[::-1]
        feasible_9, _ = _ffd_fits(order, inst.p, 2, 9)
        feasible_10, assignment_10 = _ffd_fits(order, inst.p, 2, 10)
        assert feasible_9 is False
        assert feasible_10 is True
        loads = np.zeros(2, dtype=np.int64)
        for j, mach in enumerate(assignment_10):
            loads[mach] += inst.p[j]
        assert int(loads.max()) == 10

    def test_larger_capacity_never_less_feasible(self, inst):
        # Monotonicidade: se C e factivel, C+1 tambem deve ser.
        order = np.argsort(inst.p)[::-1]
        total = int(inst.p.sum())
        for C in range(int(inst.p.max()), total, max(1, total // 20)):
            feasible, _ = _ffd_fits(order, inst.p, inst.m, C)
            if feasible:
                feasible_plus, _ = _ffd_fits(order, inst.p, inst.m, C + 1)
                assert feasible_plus is True

    def test_capacity_below_max_job_always_infeasible(self, inst):
        order = np.argsort(inst.p)[::-1]
        feasible, _ = _ffd_fits(order, inst.p, inst.m, int(inst.p.max()) - 1)
        assert feasible is False


class TestMultifit:

    def test_produces_valid_solution(self, inst):
        sol = multifit(inst)
        assert sol.is_valid()

    def test_hand_computed_makespan(self):
        inst = PMSPInstance(p=[8, 5, 3, 2], m=2)
        sol = multifit(inst)
        assert sol.makespan == 10

    def test_makespan_at_least_lower_bound(self, inst):
        sol = multifit(inst)
        assert sol.makespan >= inst.lower_bound

    def test_deterministic(self, inst):
        sol_a = multifit(inst)
        sol_b = multifit(inst)
        assert sol_a.makespan == sol_b.makespan
        assert np.array_equal(sol_a.assignment, sol_b.assignment)

    def test_various_sizes_valid(self):
        for n, m, seed in [(5, 2, 0), (20, 4, 1), (20, 6, 2), (75, 15, 3), (100, 20, 4)]:
            inst = PMSPInstance.random(n=n, m=m, seed=seed)
            sol = multifit(inst)
            assert sol.is_valid(), f"invalido para n={n}, m={m}"

    def test_reasonable_quality_vs_theoretical_ratio(self, inst):
        # Fator de aproximacao provado: 13/11 ~= 1.1818 (Yue, 1990).
        # Testamos com folga (1.25) para nao sermos frageis a
        # arredondamento/instancias atipicas -- nao e um teste de
        # tightness, so uma rede de seguranca contra regressao grosseira.
        sol = multifit(inst)
        assert sol.makespan <= 1.25 * inst.lower_bound + 1


class TestMultifitRun:

    def test_returns_expected_keys(self, inst):
        result = multifit_run(inst)
        for key in ("makespan", "gap_pct", "cpu_time", "n_improvements", "history"):
            assert key in result

    def test_gap_pct_consistent_with_makespan(self, inst):
        result = multifit_run(inst)
        expected_gap = 100.0 * (result["makespan"] - inst.lower_bound) / inst.lower_bound
        assert abs(result["gap_pct"] - expected_gap) < 1e-9

    def test_n_improvements_zero_no_history(self, inst):
        result = multifit_run(inst)
        assert result["n_improvements"] == 0
        assert result["history"] == []
