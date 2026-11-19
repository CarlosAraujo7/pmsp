"""Entrada 282: o hospedeiro CVRP (pmsp/cvrp_alns.py) produz solucoes viaveis, os operadores reconstroem solucoes completas e a execucao e' reproduzivel."""
from pathlib import Path

import numpy as np
import pytest

from pmsp import cvrp_alns as C

XFILE = Path(__file__).resolve().parent.parent / "external_benchmarks" / "cvrp_x" / "raw" / "inst" / "X-n101-k25.vrp"


def _inst(n=40, pos="R", r=6, seed=3):
    return C.make_class_instance(n, pos, r, seed)


def test_canonical_form_collapses_zero_runs_and_keeps_empty_as_two_zeros():
    out = C.canon(np.array([0, 0, 3, 0, 0, 5, 6, 0, 0], dtype=np.int64), 9)
    assert out.tolist() == [0, 3, 0, 5, 6, 0]
    assert C.canon(np.array([0, 0, 0], dtype=np.int64), 3).tolist() == [0, 0]


def test_generator_is_deterministic_and_capacity_covers_the_largest_demand():
    a, b = _inst(), _inst()
    assert (a.D == b.D).all() and a.Q == b.Q and a.Q >= a.dem.max() and a.n == 40
    c = _inst(pos="C")
    assert c.D.shape == (41, 41) and (c.D == c.D.T).all()


def test_clarke_wright_start_is_feasible():
    inst = _inst(60)
    st = C.initial_state(inst)
    assert C.feasible(inst, st.seq) and st.objective() == C.cost_of(st.seq, inst.D)


@pytest.mark.parametrize("repair", [0, 1])
def test_destroy_and_repair_rebuild_a_feasible_complete_solution(repair):
    inst = _inst(50)
    rng = np.random.default_rng(1)
    st = C.initial_state(inst)
    for d in range(C.NUM_DESTROY):
        part = C.DESTROY_OPS[d](st, rng)
        assert len(part.unrouted) >= 2 and not set(part.unrouted.tolist()) & set(part.seq.tolist())
        full = C.REPAIR_OPS[repair](part, rng)
        assert C.feasible(inst, full.seq), (d, repair)


def test_every_one_of_the_sixteen_actions_keeps_feasibility_over_a_short_run():
    inst = _inst(40)
    init = C.initial_state(inst)
    for a in range(16):
        cost, dt, c0, best, info = C.run_alns(inst, "fixed", 7, 60, fixed=a, init=init)
        assert C.feasible(inst, best.seq) and cost <= c0 + 1e-9 and 0.0 <= info["share_d0"] <= 1.0


def test_library_selectors_run_and_the_run_is_reproducible_given_the_seed():
    inst = _inst(40)
    init = C.initial_state(inst)
    for name in ["random"] + C.LEARNERS:
        a = C.run_alns(inst, name, 11, 80, init=init)
        b = C.run_alns(inst, name, 11, 80, init=init)
        assert a[0] == b[0] and C.feasible(inst, a[3].seq), name


def test_permuted_operator_list_still_gives_feasible_solutions_and_fixed_select_is_constant():
    inst = _inst(40)
    init = C.initial_state(inst)
    cost, dt, c0, best, info = C.run_alns(inst, "aucb", 5, 80, init=init, perm=True)
    assert C.feasible(inst, best.seq)
    sel = C.FixedSelect(3, 1)
    assert sel(np.random.default_rng(0), None, None) == (3, 1)
    assert C.action_pair(7) == (3, 1)


@pytest.mark.skipif(not XFILE.exists(), reason="instancias X nao baixadas")
def test_published_instance_is_parsed_and_the_start_is_within_ten_percent_of_the_best_known():
    import vrplib
    root = XFILE.parent.parent
    inst = C.instance_from_vrplib(vrplib.read_instance(str(XFILE)), vrplib.read_solution(str(root / "bks" / "X-n101-k25.sol")))
    st = C.initial_state(inst)
    assert inst.n == 100 and inst.bks == 27591 and C.feasible(inst, st.seq) and st.objective() <= 1.10 * inst.bks
