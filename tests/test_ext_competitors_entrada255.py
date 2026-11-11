"""Testes dos concorrentes (irace e desenhos publicados) nas familias publicadas de R||Cmax (Entrada 255)."""
import numpy as np

from experiment import run_cycle30_ur_external_entrada254 as X
from experiment import run_cycle31_irace_ext_entrada255 as I
from experiment import run_cycle31_pubaos_ext_entrada255 as P
from experiment.run_agents_numba_entrada180 import FR, arm_table
from pmsp import ur_numba_agents as UA


def test_budget_equals_validated_tuning_cost():
    assert I.BASE["S5"] == X.N_TUNE * 2 * 5 and I.BASE["S15"] == X.N_TUNE * 2 * 15
    assert I.N_TUNE == X.N_TUNE and I.N_TEST == X.N_TEST and len(I.CLASSES) == 75


def test_static_cfg_equals_the_grid_arm():
    ci = 33
    f, n, m = I.CLASSES[ci]
    P_, a0, od, lb = I.instance(ci, 0, True)
    st, d, ls = arm_table("S5", n)
    for k in (0, 3):
        ref = UA.ur_agent_core(P_, a0, m, np.array([0]), np.array([d[k]]), np.array([0]), 1000, 0, 0, 0.1, 0.1, 77, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)[0]
        assert I.run_cfg("cfg5", ci, 0, True, (0, FR[k]), 77) == int(ref)
    assert I.cost("cfg15", ci, 0, 5, (1, 0.2)) > 0.0


def test_irace_q_with_default_hp_equals_the_compiled_q():
    ci = 12
    f, n, m = I.CLASSES[ci]
    P_, a0, od, lb = I.instance(ci, 0, True)
    st, d, ls = arm_table("S5", n)
    ref = UA.ur_agent_core(P_, a0, m, st, d, ls, 1000, 2, 0, 0.1, 0.1, 91, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)[0]
    assert I.run_cfg("q5", ci, 0, True, (0.1, 0.9, 0.995, 0.05), 91) == int(ref)


def test_seeds_unique_in_both_runners():
    seen = set()
    for kind in I.KINDS:
        for ci in (0, 74):
            for rep in (0, 19):
                for s in range(5):
                    v = I.seed_of(kind, ci, rep, s)
                    assert v not in seen
                    seen.add(v)
    seen = set()
    for sp in P.SPACES:
        for name in P.AGENT:
            for ci in (0, 74):
                for rep in (0, 19):
                    for s in range(5):
                        v = P.seed_of(sp, "test", ci, rep, s, name)
                        assert v not in seen
                        seen.add(v)


def test_published_designs_run_and_are_deterministic():
    ci = 20
    for name in P.AGENT:
        a = P.run1("S5", ci, 0, True, name, P.GRID[name][0], 5)
        assert a == P.run1("S5", ci, 0, True, name, P.GRID[name][0], 5)
        assert a[0] >= a[1] * 0.99
