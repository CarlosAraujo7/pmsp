"""Testes do alvo do irace (Entrada 244): uma configuracao estatica continua reproduz a acao fixa da grade, o Q-Learning com os hiperparametros padrao reproduz o agente compilado, o soquete devolve o custo e o R ligado ao servidor fecha o ciclo (pulado sem R)."""
import json
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from experiment import run_cycle24_irace_entrada244 as I
from experiment.run_agents_numba_entrada180 import arm_table
from pmsp import ig_numba_agents as G
from pmsp import ur_numba_agents as UA

CI = 3   # 40 x 5


def test_static_cfg_equals_grid_arm_pcmax():
    n, m = I.CLASSES[CI]
    X, a0, od, lb = I.instance("pc", CI, 0, True)
    st, d, ls = arm_table("S15", n)
    for arm, (s_, f_) in {1: (0, 0.10), 8: (1, 0.20), 14: (2, 0.40)}.items():
        ref = G.ig_agent_core(X, a0, m, st, d, ls, 1000, G.AGENT_FIXED, arm, 0.1, 77, od, lb)[0]
        assert I.run_cfg("pc", "cfg15", CI, 0, True, (s_, f_), 77) == int(ref)


def test_default_q_equals_compiled_q_pcmax():
    n, m = I.CLASSES[CI]
    X, a0, od, lb = I.instance("pc", CI, 1, True)
    for sp, kind in (("S5", "q5"), ("S15", "q15")):
        st, d, ls = arm_table(sp, n)
        ref = G.ig_agent_core(X, a0, m, st, d, ls, 1000, G.AGENT_Q, 0, 0.1, 5, od, lb)[0]
        assert I.run_cfg("pc", kind, CI, 1, True, (0.1, 0.9, 0.995, 0.05), 5) == int(ref)


def test_static_and_default_q_unrelated():
    n, m = I.CLASSES[CI]
    P, a0, od, lb = I.instance("ur", CI, 0, True)
    st, d, ls = arm_table("S5", n)
    ref = UA.ur_agent_core(P, a0, m, st, d, ls, 1000, UA.AGENT_FIXED, 2, 0.1, 0.1, 9, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)[0]
    assert I.run_cfg("ur", "cfg5", CI, 0, True, (0, 0.20), 9) == int(ref)
    refq = UA.ur_agent_core(P, a0, m, st, d, ls, 1000, UA.AGENT_Q, 0, 0.1, 0.1, 9, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)[0]
    assert I.run_cfg("ur", "q5", CI, 0, True, (0.1, 0.9, 0.995, 0.05), 9) == int(refq)


def test_cost_is_the_gap_to_the_bound():
    lb = I.instance("pc", CI, 2, True)[3]
    mk = I.run_cfg("pc", "cfg5", CI, 2, True, (0, 0.2), 3)
    assert abs(I.cost("pc", "cfg5", CI, 2, 3, (0, 0.2)) - 100.0 * (mk - lb) / lb) < 1e-12
    assert mk >= lb


def test_budget_base_equals_validated_tuning_cost():
    # pc: 1 semente x 30 instancias x |A| acoes; ur: 2 sementes
    assert I.BASE == {("pc", "S5"): 30 * 5, ("pc", "S15"): 30 * 15, ("ur", "S5"): 30 * 5 * 2, ("ur", "S15"): 30 * 15 * 2}


@pytest.mark.skipif(not Path(I.RSCRIPT).exists(), reason="Rscript ausente")
def test_irace_loop_closes_with_the_server(tmp_path):
    # um job pequeno (pc, cfg5, classe 0, orcamento 150): o irace fala com o servidor e devolve configuracoes no intervalo
    r = I.tune_job(("pc", "cfg5", 0, 1, True))
    assert 0.05 <= r["best"]["dfrac"] <= 0.40
    assert r["n_exp"] <= 150 and r["n_exp"] > 100
    assert all(0.05 <= e["dfrac"] <= 0.40 for e in r["elites"])
