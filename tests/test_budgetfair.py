"""Propriedades verificaveis da biblioteca pmsp/budgetfair.py (protocolo de atribuicao justa ao orcamento)."""
import time

import numpy as np
import pytest

from pmsp import budgetfair as bf


def test_rel_diff_sign_and_scale():
    assert bf.rel_diff([90.0], [100.0])[0] == pytest.approx(-10.0)
    assert bf.rel_diff([110.0], [100.0])[0] == pytest.approx(10.0)


def test_paired_test_detects_a_real_shift_and_not_a_null():
    rng = np.random.default_rng(0)
    base = rng.integers(500, 1500, size=800).astype(float)
    shifted = base - rng.integers(0, 3, size=800)          # A melhora de 0 a 2 unidades
    t = bf.paired_test(shifted, base)
    assert t["mean_d_pct"] < 0 and t["p_raw"] < 1e-10 and t["better"] > t["worse"] == 0
    null = bf.paired_test(base, base)
    assert null["tie"] == 800 and np.isnan(null["p_raw"]) and null["equivalent"]


def test_equivalence_uses_the_margin():
    rng = np.random.default_rng(1)
    base = rng.integers(500, 1500, size=2000).astype(float)
    tiny = base - (rng.random(2000) < 0.05)                 # ganho esperado de ~0,004%
    t = bf.paired_test(tiny, base, margin=0.05)
    assert t["equivalent"] and -0.05 < t["mean_d_pct"] < 0
    assert not bf.paired_test(tiny, base, margin=0.001)["equivalent"]


def test_holm_matches_a_known_example_and_is_monotone():
    adj, rej = bf.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    # ordenados: 0,005 (x4) 0,01 (x3) 0,03 (x2) 0,04 (x1) -> 0,02; 0,03; 0,06; 0,06 (monotonizado)
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert list(rej) == [True, False, False, True]
    adj2, _ = bf.holm([0.2, np.nan, 0.001])
    assert np.isnan(adj2[1]) and adj2[2] == pytest.approx(0.002)


def test_reading_distinguishes_parity_from_a_real_gain():
    assert bf.reading(-0.02, 0.001, True) == "paridade"
    assert bf.reading(-0.20, 0.001, False) == "A melhor"
    assert bf.reading(0.20, 0.001, False) == "A pior"
    assert bf.reading(-0.20, 0.40, False) == "sem diferenca"


def test_budget_share_and_decision_cost_split():
    assert bf.budget_share(-0.010, -0.150) == pytest.approx(100 * (1 - 0.010 / 0.150))
    assert bf.budget_share(-0.150, -0.150) == 0.0
    s = bf.decision_cost_split(d_iter=0.001, d_cpu=-0.059)
    assert s["decision"] == pytest.approx(-0.001) and s["cost"] == pytest.approx(0.060) and s["total_cpu"] == pytest.approx(0.059)


def test_equal_cpu_iterations_rounds_and_floors():
    assert bf.equal_cpu_iterations(1000, 2.04) == 2040
    assert bf.equal_cpu_iterations(1000, 0.01) == 50


def test_cost_ratio_recovers_a_known_cost_ratio():
    def spin(sec):
        def run(inst, seed):
            t0 = time.process_time()
            while time.process_time() - t0 < sec:
                pass
        return run
    # time.process_time tem resolucao de ~15 ms no Windows: duracoes de dezenas de ms e janelas de ~1 s
    k = bf.cost_ratio(spin(0.12), spin(0.04), [0], min_seconds=0.8, repeats=1)
    assert 2.0 < k < 4.0


def test_mde_and_planted_power_are_coherent():
    rng = np.random.default_rng(2)
    base = rng.integers(500, 1500, size=1200).astype(float)
    a = base - (rng.random(1200) < 0.02)
    d = bf.rel_diff(a, base)
    unit = 100.0 / base
    m = bf.mde80(d)
    pw = bf.planted_power(d, unit, [0.0, 0.05, 0.3], alpha=0.05 / 12, n_boot=60)
    assert 0 < m < 0.2
    assert pw[0.3] >= 0.95 and pw[0.3] >= pw[0.05] >= pw[0.0]


def test_paired_test_matches_the_study_reports_exactly():
    """A biblioteca reproduz, para os mesmos dados, o teste usado nos relatorios do estudo (analysis/report_entrada180_agents.test)."""
    from analysis.report_entrada180_agents import test as study_test
    rng = np.random.default_rng(3)
    base = rng.integers(300, 900, size=400).astype(float)
    a = base - rng.integers(0, 2, size=400) + (rng.random(400) < 0.03)
    mine, ref = bf.paired_test(a, base), study_test(a, base)
    for k_mine, k_ref in [("mean_d_pct", "mean_d_pct"), ("ci90_lo", "ci90_lo"), ("ci90_hi", "ci90_hi"), ("p_raw", "p_raw"), ("better", "better"), ("worse", "worse"), ("tie", "tie")]:
        assert mine[k_mine] == pytest.approx(ref[k_ref])
