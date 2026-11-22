"""Entrada 287: os nucleos com registro (pmsp/ig_logged.py, pmsp/ur_logged.py) devolvem, bit a bit, o mesmo que os originais (ig_ties, ur_ties) com a mesma semente, e o registro e' consistente."""
from __future__ import annotations

import numpy as np
import pytest

from pmsp import ig_logged as IL
from pmsp import ig_ties as IT
from pmsp import ur_logged as UL
from pmsp import ur_ties as UT
from pmsp.constructive import lpt
from pmsp.instance import PMSPInstance

HP = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
T = 150


def _ig_setup(n=30, m=5, seed=3):
    inst = PMSPInstance.random(n=n, m=m, seed=seed, name="t")
    p = np.asarray(inst.p, dtype=np.int64)
    a0 = lpt(inst).assignment.astype(np.int64)
    od = np.argsort(inst.p)[::-1].astype(np.int64)
    d = np.array([max(2, round(f * n)) for f in (0.05, 0.1, 0.2, 0.3, 0.4)], dtype=np.int64)
    return p, a0, m, np.zeros(5, dtype=np.int64), d, np.zeros(5, dtype=np.int64), od, float(inst.lower_bound)


def _ur_setup(n=30, m=4, seed=3):
    from experiment import run_cycle32_fanjul_entrada256 as F
    P = np.random.default_rng(seed).integers(1, 50, size=(n, m)).astype(np.int64)
    a0, od = F.constructive_ur(P)
    st, d, ls = F.arm_table_b("B5", n)
    return P, a0, m, st, d, ls, od, float(F.lower_bound_ur(P))


def _args(setup, agent, fixed, seed, tie):
    X, a0, m, st, d, ls, od, lb = setup
    return (X, a0, m, st, d, ls, T, agent, fixed, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, len(st))), 1.0, tie)


CASES = [(0, 2, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1)]          # (agente, braco fixo, tie_mode)


@pytest.mark.parametrize("kind", ["ig", "ur"])
@pytest.mark.parametrize("agent,fixed,tie", CASES)
def test_logged_core_equals_original(kind, agent, fixed, tie):
    setup = _ig_setup() if kind == "ig" else _ur_setup()
    orig_core, log_core = (IT.ig_agent_core_ties, IL.ig_agent_core_logged) if kind == "ig" else (UT.ur_agent_core_ties, UL.ur_agent_core_logged)
    for seed in (11, 12):
        o = orig_core(*_args(setup, agent, fixed, seed, tie))
        g = log_core(*_args(setup, agent, fixed, seed, tie))
        assert len(g) == len(o) + 5
        for x, y in zip(o, g[:len(o)]):
            assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("kind", ["ig", "ur"])
def test_log_is_consistent(kind):
    setup = _ig_setup() if kind == "ig" else _ur_setup()
    core = IL.ig_agent_core_logged if kind == "ig" else UL.ur_agent_core_logged
    best, n_imp, ev, used, hit2, Q, la, ls_, lc, lr_, lz = core(*_args(setup, 1, 0, 5, 0))
    assert la.shape == ls_.shape == lc.shape == lr_.shape == lz.shape == (T,)
    assert np.array_equal(np.bincount(la, minlength=len(used)), used)          # as contagens do registro sao as do nucleo
    assert ls_.min() >= 0 and ls_.max() <= 80 and set(np.unique(lz)) <= {0, 1}
    assert (lr_ >= 0).all()
    # o makespan so' diminui (aceita se nao piora): a soma das recompensas e' a melhora total, e o registro de 'corrente' e' nao crescente
    assert np.all(np.diff(lc) <= 1e-9)
    assert lc[0] - lr_.sum() == pytest.approx(best)
    assert (lz == 1).all()                                                       # o agente aleatorio nunca escreve em Q: a linha esta sempre zerada


@pytest.mark.parametrize("kind", ["ig", "ur"])
def test_zero_row_flag_tracks_the_q_table(kind):
    setup = _ig_setup(n=60, m=8) if kind == "ig" else _ur_setup(n=60, m=6)
    core = IL.ig_agent_core_logged if kind == "ig" else UL.ur_agent_core_logged
    out = core(*_args(setup, 2, 0, 9, 0))
    lz, la, lr_ = out[-1], out[6], out[9]
    assert lz[0] == 1                                                            # a primeira decisao ve a tabela zerada
    if lr_.sum() > 0:
        assert lz.min() == 0 or lz.mean() < 1.0                                  # depois de uma recompensa positiva ha' linhas nao zeradas
