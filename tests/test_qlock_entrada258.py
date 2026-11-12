"""Entrada 258 (mecanismo): o Q-Learning padrao (tabela zerada, recompensa nao negativa, argmax com desempate pelo primeiro indice) repete a acao listada primeiro, e o seu uso das acoes
segue a ORDEM da lista (nao a recompensa); permutar a lista em cada execucao (ql_perm, Entrada 260) tira essa dependencia da lista."""
import numpy as np

from experiment.run_agents_numba_entrada180 import arm_table
from pmsp import ig_numba_agents3 as H
from pmsp.constructive import lpt
from pmsp.instance import PMSPInstance

T = 400


def _setup(n=60, m=6, seed=0):
    p = np.random.default_rng(seed).integers(1, 100, size=n).astype(np.int64)
    inst = PMSPInstance(p=p, m=m, name="t")
    return p, m, lpt(inst).assignment.astype(np.int64), np.argsort(p)[::-1].astype(np.int64), inst.lower_bound


def _used(order, seed):
    p, m, a0, od, lb = _setup()
    st, d, ls = arm_table("S5", len(p))
    best, _, ev, used, _ = H.ig_agent_core3(p, a0, m, st[order].copy(), d[order].copy(), ls[order].copy(), T, H.AGENT_Q, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0)
    return np.asarray(used, dtype=float) / T


def test_q_repeats_the_first_listed_action():
    A = len(arm_table("S5", 60)[0])
    share0 = np.mean([_used(np.arange(A), s)[0] for s in range(6)])
    assert share0 > 2.0 / A          # well above the 1/A of random choice


def test_q_use_follows_the_order_of_the_list():
    A = len(arm_table("S5", 60)[0])
    fwd = np.mean([_used(np.arange(A), s)[0] for s in range(6)])
    rev = np.mean([_used(np.arange(A)[::-1], s)[0] for s in range(6)])       # the same action now stands last
    assert fwd > rev                                                          # the lock-in is on the index, not on the action


def test_random_permutation_averages_the_dependence_out():
    A = len(arm_table("S5", 60)[0])
    rng = np.random.default_rng(1)
    first = np.zeros(A)
    for s in range(30):
        perm = rng.permutation(A)
        u = _used(perm, s)
        first[perm[0]] += u[0]                                                # share spent on the action that happened to be listed first
    # with random lists no single action is systematically favoured: the shares by action are spread over the actions
    assert first.max() < 0.8 * first.sum()
