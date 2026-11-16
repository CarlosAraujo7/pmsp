"""Entrada 280: os nucleos com desempate de maquinas identicas (pmsp/ig_ties.py) e do TSP (pmsp/tsp_ties.py) reproduzem os originais com tie_mode = 0 e desempatam ao acaso com tie_mode = 1."""
import numpy as np

from pmsp import ig_numba_agents5 as H5
from pmsp import ig_ties as IT
from pmsp import tsp_numba as TS
from pmsp import tsp_ties as TT
from pmsp.instance_tsp import build


def _lpt_assign(p, m):
    order = np.argsort(p)[::-1]
    loads = np.zeros(m, dtype=np.int64)
    a = np.empty(len(p), dtype=np.int64)
    for j in order:
        k = int(loads.argmin())
        a[j] = k
        loads[k] += p[j]
    return a


def _ig(seed=1, n=60, m=5, A=10):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 101, size=n).astype(np.int64)
    lb = max(int(np.ceil(p.sum() / m)), int(p.max()))
    st = np.array([k % 3 for k in range(A)], dtype=np.int64)
    d = np.array([max(2, round(f * n)) for f in np.linspace(0.05, 0.4, A)], dtype=np.int64)
    return p, _lpt_assign(p, m), m, st, d, np.zeros(A, dtype=np.int64), np.argsort(p)[::-1].astype(np.int64), lb


HP = np.array([0.1, 0.9, 0.995, 0.05, 0.0])


def test_ig_tie_mode_zero_reproduces_the_original_core():
    p, a0, m, st, d, ls, od, lb = _ig()
    A = len(st)
    for seed in (3, 4):
        a = H5.ig_agent_core5(p, a0, m, st, d, ls, 250, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, A)), 1.0)
        b = IT.ig_agent_core_ties(p, a0, m, st, d, ls, 250, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, A)), 1.0, 0)
        assert float(a[0]) == float(b[0]) and float(a[2]) == float(b[2]) and a[3].tolist() == b[3].tolist()


def test_ig_random_ties_spread_a_constant_table_and_first_maximum_repeats_arm_zero():
    p, a0, m, st, d, ls, od, lb = _ig(2)
    A = len(st)
    hp = np.array([0.0, 0.9, 1.0, 0.0, 0.0])
    z = np.zeros((81, A))
    first = IT.ig_agent_core_ties(p, a0, m, st, d, ls, 400, 2, 0, 0.1, 0.1, 5, od, lb, 0, 0, 0.0, hp, 0, z, 0.0, 0)[3]
    rand = IT.ig_agent_core_ties(p, a0, m, st, d, ls, 400, 2, 0, 0.1, 0.1, 5, od, lb, 0, 0, 0.0, hp, 0, z, 0.0, 1)[3]
    assert first[0] == 400 and rand.sum() == 400 and rand[0] < 100 and (rand > 0).sum() == A


def _tsp(seed=3, n=120):
    D, NL = build("unif", n, seed)
    tour0, _ = TS.construct(D, NL)
    return D, NL, tour0


def test_tsp_tie_mode_zero_reproduces_the_original_core_for_both_q_agents():
    D, NL, tour0 = _tsp()
    kinds, sizes = TS.arm_table("T15")
    for agent, hp, q0 in ((TS.AGENT_Q, HP, 0.0), (TS.AGENT_QO, np.array([0.1, 0.9, 1.0, 0.05, 0.0]), 0.02)):
        a = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 150, agent, 0, 0.1, 0.0, 17, hp, q0)
        b = TT.tsp_agent_core_ties(D, NL, tour0, kinds, sizes, 150, agent, 0, 0.1, 0.0, 17, hp, q0, 0)
        assert float(a[0]) == float(b[0]) and float(a[2]) == float(b[2]) and a[3].tolist() == b[3].tolist()


def test_tsp_random_ties_use_every_action_where_the_first_maximum_repeats_one():
    D, NL, tour0 = _tsp(5, 100)
    kinds, sizes = TS.arm_table("T15")
    hp = np.array([0.0, 0.9, 0.0, 0.0, 0.0])          # epsilon cai a 0 apos a primeira decisao e nao ha' aprendizagem: so' o desempate decide
    first = TT.tsp_agent_core_ties(D, NL, tour0, kinds, sizes, 300, TS.AGENT_Q, 0, 0.1, 0.0, 9, hp, 0.0, 0)[3]
    rand = TT.tsp_agent_core_ties(D, NL, tour0, kinds, sizes, 300, TS.AGENT_Q, 0, 0.1, 0.0, 9, hp, 0.0, 1)[3]
    assert first[0] >= 299 and rand.sum() == 300 and (rand > 0).sum() == 15 and rand[0] < 100
