"""
tests/test_phase1.py
Testes de sanidade da Fase 1.
Execute: pytest tests/test_phase1.py -v
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from pmsp.solution import PMSPSolution
from pmsp.bounds import (
    lower_bound, compute_state, decode_state, gap_percent,
    f1_relative_gap, f2_load_cv, f3_progress, f4_stagnation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_instance():
    # n=4, m=2, p=[3,5,2,8] -> LB = max(8, ceil(18/2)) = 9
    return PMSPInstance(p=[3, 5, 2, 8], m=2, name="test_small")

@pytest.fixture
def balanced_instance():
    # n=4, m=2, p=[5,5,5,5] -> LB = max(5, ceil(20/2)) = 10
    return PMSPInstance(p=[5, 5, 5, 5], m=2, name="test_balanced")

@pytest.fixture
def random_instance():
    return PMSPInstance.random(n=20, m=4, seed=42, name="rand_test")

# ---------------------------------------------------------------------------
# TestPMSPInstance
# ---------------------------------------------------------------------------

class TestPMSPInstance:

    def test_basic_attributes(self, small_instance):
        assert small_instance.n == 4
        assert small_instance.m == 2
        assert list(small_instance.p) == [3, 5, 2, 8]

    def test_lower_bound_small(self, small_instance):
        assert small_instance.lower_bound == 9

    def test_lower_bound_balanced(self, balanced_instance):
        assert balanced_instance.lower_bound == 10

    def test_lower_bound_dominated_by_longest_job(self):
        inst = PMSPInstance(p=[100, 1, 1], m=2)
        assert inst.lower_bound == 100

    def test_lower_bound_dominated_by_average(self):
        # sum=30, m=2, avg=15 > max=5 => LB=15
        inst = PMSPInstance(p=[5, 5, 5, 5, 5, 5], m=2)
        assert inst.lower_bound == 15

    def test_lb_cache_consistency(self, small_instance):
        assert small_instance.lower_bound == small_instance.lower_bound

    def test_random_reproducibility(self):
        a = PMSPInstance.random(n=10, m=3, seed=0)
        b = PMSPInstance.random(n=10, m=3, seed=0)
        assert np.array_equal(a.p, b.p)

    def test_random_different_seeds(self):
        a = PMSPInstance.random(n=10, m=3, seed=1)
        b = PMSPInstance.random(n=10, m=3, seed=2)
        assert not np.array_equal(a.p, b.p)

    def test_random_bounds_u1_100(self):
        inst = PMSPInstance.random(n=100, m=5, seed=99)
        assert inst.p.min() >= 1 and inst.p.max() <= 100

    def test_invalid_m_zero(self):
        with pytest.raises(ValueError):
            PMSPInstance(p=[1, 2, 3], m=0)

    def test_invalid_p_contains_zero(self):
        with pytest.raises(ValueError):
            PMSPInstance(p=[1, 0, 3], m=2)

    def test_serialization_roundtrip(self, small_instance, tmp_path):
        path = tmp_path / "inst.json"
        small_instance.save(path)
        loaded = PMSPInstance.load(path)
        assert loaded == small_instance
        assert loaded.name == small_instance.name

# ---------------------------------------------------------------------------
# TestPMSPSolution
# ---------------------------------------------------------------------------

class TestPMSPSolution:

    def test_makespan_manual(self, small_instance):
        # maq0: jobs[0(p=3),2(p=2)]=5; maq1: jobs[1(p=5),3(p=8)]=13
        sol = PMSPSolution(small_instance, assignment=[0, 1, 0, 1])
        assert sol.makespan == 13
        assert sol.loads[0] == 5
        assert sol.loads[1] == 13

    def test_makespan_ge_lower_bound(self, random_instance):
        n, m = random_instance.n, random_instance.m
        sol = PMSPSolution(random_instance, assignment=np.arange(n) % m)
        assert sol.makespan >= random_instance.lower_bound

    def test_validity_after_construction(self, small_instance):
        sol = PMSPSolution(small_instance, assignment=[0, 1, 0, 1])
        assert sol.is_valid()

    def test_validity_after_move_job(self, small_instance):
        sol = PMSPSolution(small_instance, assignment=[0, 1, 0, 1])
        # Move job3(p=8) de maq1 para maq0 => maq0=[3,2,8]=13, maq1=[5]=5
        sol.move_job(job=3, target_machine=0)
        assert sol.is_valid()
        assert sol.makespan == 13

    def test_validity_after_swap_inter(self, small_instance):
        # Swap job0(p=3,maq0) <-> job1(p=5,maq1)
        # => maq0=[job1(5),job2(2)]=7; maq1=[job0(3),job3(8)]=11
        sol = PMSPSolution(small_instance, assignment=[0, 1, 0, 1])
        sol.swap_jobs(job_a=0, job_b=1)
        assert sol.is_valid()
        assert sol.loads[0] == 7
        assert sol.loads[1] == 11

    def test_validity_after_swap_intra(self, small_instance):
        sol = PMSPSolution(small_instance, assignment=[0, 0, 1, 1])
        loads_before = sol.loads.copy()
        sol.swap_jobs(job_a=0, job_b=1)
        assert sol.is_valid()
        assert np.array_equal(sol.loads, loads_before)

    def test_move_job_noop(self, small_instance):
        sol = PMSPSolution(small_instance, assignment=[0, 1, 0, 1])
        loads_before = sol.loads.copy()
        sol.move_job(job=0, target_machine=0)
        assert np.array_equal(sol.loads, loads_before)
        assert sol.is_valid()

    def test_copy_independence(self, small_instance):
        # Inicial: maq0=[job0(3),job2(2)]=5; maq1=[job1(5),job3(8)]=13
        # Move job0(p=3) de maq0 para maq1 NA COPIA:
        #   copia.loads = [5-3, 13+3] = [2, 16] => makespan=16
        # Original permanece: loads=[5,13] => makespan=13
        sol = PMSPSolution(small_instance, assignment=[0, 1, 0, 1])
        copia = sol.copy()
        copia.move_job(job=0, target_machine=1)
        assert sol.makespan == 13,   f"Original alterado: makespan={sol.makespan}"
        assert copia.makespan == 16, f"Copia errada: makespan={copia.makespan}"
        assert sol.loads[0] == 5 and sol.loads[1] == 13
        assert sol.is_valid()
        assert copia.is_valid()

    def test_relative_gap_nonnegative(self, random_instance):
        n, m = random_instance.n, random_instance.m
        sol = PMSPSolution(random_instance, assignment=np.arange(n) % m)
        assert sol.relative_gap >= 0.0

    def test_machine_min_max(self, small_instance):
        # loads=[5,13] => min=maq0, max=maq1
        sol = PMSPSolution(small_instance, assignment=[0, 1, 0, 1])
        assert sol.machine_min == 0
        assert sol.machine_max == 1

    def test_invalid_assignment_out_of_range(self, small_instance):
        with pytest.raises(ValueError):
            PMSPSolution(small_instance, assignment=[0, 2, 0, 1])

# ---------------------------------------------------------------------------
# TestBounds
# ---------------------------------------------------------------------------

class TestBounds:

    def test_lower_bound_function(self, small_instance):
        assert lower_bound(small_instance) == 9

    def test_gap_percent_zero_at_lb(self, small_instance):
        lb = small_instance.lower_bound
        assert gap_percent(lb, lb) == pytest.approx(0.0)

    def test_gap_percent_positive(self, small_instance):
        lb = small_instance.lower_bound
        assert gap_percent(lb + 5, lb) > 0.0

    def test_f3_progress_boundaries(self):
        assert f3_progress(0, 100)   == pytest.approx(0.0)
        assert f3_progress(100, 100) == pytest.approx(1.0)
        assert f3_progress(200, 100) == pytest.approx(1.0)

    def test_f4_stagnation_saturation(self):
        assert f4_stagnation(0)   == pytest.approx(0.0)
        assert f4_stagnation(50)  == pytest.approx(1.0)
        assert f4_stagnation(999) == pytest.approx(1.0)

    def test_compute_state_range(self, random_instance):
        n, m = random_instance.n, random_instance.m
        sol = PMSPSolution(random_instance, assignment=np.arange(n) % m)
        for it in range(0, 110, 10):
            for no_imp in [0, 10, 50, 100]:
                s = compute_state(sol, it, 100, no_imp)
                assert 0 <= s <= 624, f"Estado fora do range: {s}"

    def test_decode_state_roundtrip(self):
        for s in range(81):
            b1, b2, b3, b4 = decode_state(s)
            assert b1 * 27 + b2 * 9 + b3 * 3 + b4 == s

    def test_decode_state_bins_in_range(self):
        for s in range(81):
            for b in decode_state(s):
                assert 0 <= b <= 2

# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_lb_le_makespan_for_many_random(self):
        for seed in range(50):
            inst = PMSPInstance.random(n=30, m=5, seed=seed)
            asgn = np.arange(inst.n) % inst.m
            sol = PMSPSolution(inst, assignment=asgn)
            assert sol.makespan >= inst.lower_bound, (
                f"Violacao de LB: seed={seed}, "
                f"makespan={sol.makespan}, LB={inst.lower_bound}"
            )

    def test_state_computation_is_valid(self, small_instance):
        sol = PMSPSolution(small_instance, assignment=[0, 1, 0, 1])
        s_before = compute_state(sol, 10, 100, 5)
        sol_copy = sol.copy()
        sol_copy.move_job(job=3, target_machine=0)
        s_after = compute_state(sol_copy, 10, 100, 0)
        assert 0 <= s_before <= 624
        assert 0 <= s_after <= 624
