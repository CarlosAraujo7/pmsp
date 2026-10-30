"""Testes do oraculo de malha fechada (Entrada 249): determinismo, invariantes, controle sem avaliacoes e ganho sobre o aleatorio."""
import numpy as np

from experiment import run_cycle27_closedloop_entrada249 as C
from experiment import run_cycle22_ur_entrada234 as UR
from experiment.run_agents_numba_entrada180 import arm_table
from pmsp import ig_numba as IGN
from pmsp import ur_numba as URN
from pmsp import ur_numba_agents as UA


def _ur(n=50, m=10):
    P, a0, od, lb = UR.make(n, m, 0, 5, True)
    return P, a0, od


def test_deterministic_and_invariants():
    P, a0, od = _ur()
    st, d, ls = arm_table("S5", 50)
    r1 = C.CL.ur_lookahead(P, a0, 10, st, d, ls, 60, 3, 11, od, 1)
    r2 = C.CL.ur_lookahead(P, a0, 10, st, d, ls, 60, 3, 11, od, 1)
    assert r1[0] == r2[0] and r1[1] == r2[1] and r1[2] == r2[2] and (r1[3] == r2[3]).all()
    init = int(URN.init_state(a0.astype(np.int64).copy(), P, 10)[0].max())
    assert r1[0] <= init
    assert int(r1[3].sum()) == 60
    assert r1[2] > 0.0                       # o oraculo gasta avaliacoes extras
    r0 = C.CL.ur_lookahead(P, a0, 10, st, d, ls, 60, 3, 11, od, 0)
    assert r0[2] == 0.0                      # o controle nao avalia
    assert int(r0[3].sum()) == 60


def test_pc_version_runs_and_improves():
    p, a0, od, lb = C.instance("pc", 50, 10, 0, 4, True)
    st, d, ls = arm_table("S15", 50)
    best, ev, evr, used = C.CL.pc_lookahead(p, a0, 10, st, d, ls, 80, 3, 5, od, 1)
    init = int(IGN.init_state(a0.astype(np.int64).copy(), p, 10)[0].max())
    assert best <= init and int(used.sum()) == 80 and evr > 0.0


def test_control_matches_random_agent_and_oracle_beats_it():
    """la0 reproduz o aleatorio em distribuicao; o oraculo de estado (S15) fica atras do aleatorio em makespan medio"""
    P, a0, od = _ur(60, 10)
    st, d, ls = arm_table("S15", 60)
    T, seeds = 150, range(120)
    rnd = np.array([UR.run("S15", P, a0, 10, od, 0.0, 60, UA.AGENT_RANDOM, 0, 0.1, 0.1, 1000 + s, T=T)[0] for s in seeds], float)
    la0 = np.array([C.CL.ur_lookahead(P, a0, 10, st, d, ls, T, 4, 2000 + s, od, 0)[0] for s in seeds], float)
    la1 = np.array([C.CL.ur_lookahead(P, a0, 10, st, d, ls, T, 4, 3000 + s, od, 1)[0] for s in seeds], float)
    se = np.sqrt(rnd.var(ddof=1) / len(rnd) + la0.var(ddof=1) / len(la0))
    assert abs(la0.mean() - rnd.mean()) <= 3.5 * se
    assert la1.mean() <= la0.mean()


def test_seeds_distinct():
    seen = set()
    for h in C.HOSTS:
        for sp in C.SPACES:
            for ci in (0, 11):
                for rep in (0, 99):
                    for o in (0, 1):
                        for s in range(C.N_SEED):
                            v = C.seed_of(h, sp, ci, rep, o, s)
                            assert v not in seen
                            seen.add(v)


def test_rollout_oracle_invariants_both_hosts():
    P, a0, od = _ur(40, 8)
    st, d, ls = arm_table("S5", 40)
    r1 = C.CL.ur_rollout(P, a0, 8, st, d, ls, 40, 2, 10, -1, 2, 7, od)
    r2 = C.CL.ur_rollout(P, a0, 8, st, d, ls, 40, 2, 10, -1, 2, 7, od)
    assert r1[0] == r2[0] and (r1[3] == r2[3]).all()
    init = int(URN.init_state(a0.astype(np.int64).copy(), P, 8)[0].max())
    assert r1[0] <= init and int(r1[3].sum()) == 40 and r1[2] > 0.0
    p, b0, odp, lb = C.instance("pc", 40, 8, 0, 3, True)
    best, ev, evr, used = C.CL.pc_rollout(p, b0, 8, st, d, ls, 40, 2, 10, 5, 1, 9, odp)
    assert best <= int(IGN.init_state(b0.astype(np.int64).copy(), p, 8)[0].max()) and int(used.sum()) == 40 and evr > 0.0


def test_rollout_with_single_candidate_base_is_the_fixed_agent_in_distribution():
    """com A = 1 a unica acao e' a base: o rollout executa sempre essa acao (a mesma trajetoria em distribuicao que o agente fixo)"""
    P, a0, od = _ur(40, 8)
    st, d, ls = arm_table("S5", 40)
    st1, d1, ls1 = st[2:3].copy(), d[2:3].copy(), ls[2:3].copy()
    T = 60
    fx = np.array([UR.run("S5", P, a0, 8, od, 0.0, 40, UA.AGENT_FIXED, 2, 0.1, 0.1, 500 + s, T=T)[0] for s in range(100)], float)
    ro = np.array([C.CL.ur_rollout(P, a0, 8, st1, d1, ls1, T, 1, 20, 0, 0, 900 + s, od)[0] for s in range(100)], float)
    se = np.sqrt(fx.var(ddof=1) / 100 + ro.var(ddof=1) / 100)
    assert abs(fx.mean() - ro.mean()) <= 3.5 * se + 1e-9
