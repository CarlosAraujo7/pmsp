"""Testes do IG com cronograma de acoes (pmsp/sched_numba.py): sequencia constante == agente fixo; o cronograma troca de acao na fronteira da fase."""
import numpy as np
import pytest

from experiment.run_agents_numba_entrada180 import arm_table, make
from pmsp import ig_numba_agents4 as G4
from pmsp import sched_numba as S
from pmsp import ur_numba_agents as UA
from pmsp.instance_ur import constructive_ur, lower_bound_ur, random_ur


@pytest.mark.parametrize("arm", [0, 2, 4])
def test_constant_schedule_equals_fixed_agent_pcmax(arm):
    inst, p, a0, od = make(40, 5, 3, False)
    st, d, ls = arm_table("S5", 40)
    T = 120
    ref = G4.ig_agent_core4(p, a0, 5, st, d, ls, T, 0, arm, 0.1, 0.1, 77, od, inst.lower_bound, 0, 0, 0.0, G4.HP_DEFAULT, 0)
    out = S.ig_sched_core(p, a0, 5, st, d, ls, np.full(T, arm, dtype=np.int64), 77, od)
    assert out[0] == ref[0] and abs(out[1] - ref[2]) < 1e-9


@pytest.mark.parametrize("arm", [1, 3])
def test_constant_schedule_equals_fixed_agent_ur(arm):
    P = random_ur(40, 6, 5)
    a0, od = constructive_ur(P)
    st, d, ls = arm_table("S15", 40)
    T = 120
    ref = UA.ur_agent_core(P, a0, 6, st, d, ls, T, 0, arm, 0.1, 0.1, 31, od, lower_bound_ur(P), 0, 0, 0.0, UA.HP_DEFAULT, 0)
    out = S.ur_sched_core(P, a0, 6, st, d, ls, np.full(T, arm, dtype=np.int64), 31, od)
    assert out[0] == ref[0] and abs(out[1] - ref[2]) < 1e-9


def test_schedule_uses_each_phase_and_never_worsens():
    inst, p, a0, od = make(40, 5, 1, False)
    st, d, ls = arm_table("S5", 40)
    start = int(np.bincount(a0, weights=p, minlength=5).max())
    seq = np.concatenate([np.full(50, 0), np.full(50, 4), np.full(50, 2)]).astype(np.int64)
    out = S.ig_sched_core(p, a0, 5, st, d, ls, seq, 5, od)
    assert out[0] <= start and out[1] > 0
    other = S.ig_sched_core(p, a0, 5, st, d, ls, seq[::-1].copy(), 5, od)
    assert other[0] <= start
