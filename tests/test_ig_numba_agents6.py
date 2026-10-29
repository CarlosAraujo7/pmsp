"""Testes dos agentes 11 (Q-Learning episodico de estado binario) e 12 (FRRMAB) de pmsp/ig_numba_agents6.py e pmsp/ur_numba_agents6.py; os agentes 0-10 reproduzem os de agents4/ur_numba_agents exatamente."""
import numpy as np
import pytest

from experiment.run_agents_numba_entrada180 import arm_table, make
from pmsp import ig_numba_agents4 as G4
from pmsp import ig_numba_agents6 as G6
from pmsp import ur_numba_agents as U4
from pmsp import ur_numba_agents6 as U6
from pmsp.instance_ur import constructive_ur, lower_bound_ur, random_ur

HP = G4.HP_DEFAULT


@pytest.mark.parametrize("agent", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_old_agents_unchanged_pcmax(agent):
    inst, p, a0, od = make(40, 5, 2, False)
    st, d, ls = arm_table("S5", 40)
    a = G4.ig_agent_core4(p, a0, 5, st, d, ls, 150, agent, 2, 0.1, 0.1, 13, od, inst.lower_bound, 0, 0, 0.0, HP, 0)
    b = G6.ig_agent_core6(p, a0, 5, st, d, ls, 150, agent, 2, 0.1, 0.1, 13, od, inst.lower_bound, 0, 0, 0.0, HP, 0)
    assert a[0] == b[0] and (a[3] == b[3]).all() and abs(a[2] - b[2]) < 1e-9


@pytest.mark.parametrize("agent", [1, 2, 3, 7])
def test_old_agents_unchanged_unrelated(agent):
    P = random_ur(40, 6, 9)
    a0, od = constructive_ur(P)
    st, d, ls = arm_table("S15", 40)
    lb = lower_bound_ur(P)
    a = U4.ur_agent_core(P, a0, 6, st, d, ls, 150, agent, 0, 0.1, 0.1, 21, od, lb, 0, 0, 0.0, HP, 0)
    b = U6.ur_agent_core6(P, a0, 6, st, d, ls, 150, agent, 0, 0.1, 0.1, 21, od, lb, 0, 0, 0.0, HP, 0)
    assert a[0] == b[0] and (a[3] == b[3]).all()


def run(agent, lr, lr2, hp, T=300, seed=5):
    inst, p, a0, od = make(40, 5, 1, False)
    st, d, ls = arm_table("S5", 40)
    start = int(np.bincount(a0, weights=p, minlength=5).max())
    out = G6.ig_agent_core6(p, a0, 5, st, d, ls, T, agent, 0, lr, lr2, seed, od, inst.lower_bound, 0, 0, 0.0, hp, 0)
    return out, start


def test_episodic_q_uses_blocks_and_is_deterministic():
    hp = np.array([0.1, 0.9, 0.9, 0.05, 0.5])
    (best, n_imp, ev, used, _), start = run(11, 0.1, 25.0, hp, T=300)
    assert best <= start and int(used.sum()) == 300
    assert all(int(u) % 25 == 0 for u in used)          # uma acao por episodio de 25 iteracoes
    again, _ = run(11, 0.1, 25.0, hp, T=300)
    assert again[0] == best and (again[3] == used).all()
    odd, _ = run(11, 0.1, 40.0, hp, T=300)               # 300 = 7 x 40 + 20: o ultimo episodio e' parcial
    assert int(odd[3].sum()) == 300


def test_frrmab_tries_every_action_first_and_is_deterministic():
    (best, n_imp, ev, used, _), start = run(12, 2.0, 0.5, HP, T=300)
    assert best <= start and int(used.sum()) == 300 and (used >= 1).all()
    again, _ = run(12, 2.0, 0.5, HP, T=300)
    assert again[0] == best and (again[3] == used).all()
    big, _ = run(12, 50.0, 1.0, HP, T=300)               # exploracao quase uniforme: todas as acoes bem usadas
    assert (big[3] > 15).all()
