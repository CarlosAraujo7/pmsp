"""Testes do CPU igual no R||Cmax (Entrada 248): sementes, tempos de execucao e a reproducao dos resultados armazenados da Entrada 234."""
import csv
import json
from pathlib import Path

import pytest

from experiment import run_cycle22_ur_entrada234 as U
from experiment import run_cycle26_eqcpu_ur_entrada248 as E
from experiment.run_agents_numba_entrada180 import CLASSES
from pmsp import ur_numba_agents as UA

R = Path(__file__).resolve().parent.parent / "results"


def test_seeds_unique_and_disjoint_from_entrada234():
    seen = set()
    for sp in E.SPACES:
        for ci in (0, 5, 11):
            for rep in (0, 37, 99):
                for me in E.METHODS:
                    for s in range(5):
                        v = E.seed_of(sp, ci, rep, me, s)
                        assert v not in seen
                        seen.add(v)
                        assert v >= 100_000_000
    old = {U.seed_of(sp, ci, rep, me, s) for sp in E.SPACES for ci in (0, 5, 11) for rep in (0, 37, 99) for me in U.MID for s in range(5)}
    assert not (seen & old)


def test_run_lengths():
    K = {"ql": 0.55, "ucb": 1.04, "pg": 1.0, "alns": 0.98, "qo": 0.01, "val": 1.25}
    assert E.t_rand(K, "ql") == 550
    assert E.t_rand(K, "qo") == E.FLOOR
    assert E.t_val(K, "ql") == round(1000 * 0.55 / 1.25) == 440
    assert E.t_val(K, "ucb") == 832
    assert E.t_val({**K, "val": 100.0}, "ucb") == E.FLOOR


def test_learner_runner_matches_ur_run_and_is_deterministic():
    n, m = CLASSES[0]
    P, a0, od, lb = U.make(n, m, 0, 0, True)
    val, qo_cfg = E.load_choices("S5")
    a = E.run_learner("S5", "ql", P, a0, m, od, lb, n, 12345, qo_cfg)
    b = U.run("S5", P, a0, m, od, lb, n, UA.AGENT_Q, 0, 0.1, 0.1, 12345)
    assert a == b
    assert E.run_learner("S5", "qo", P, a0, m, od, lb, n, 99, qo_cfg) == E.run_learner("S5", "qo", P, a0, m, od, lb, n, 99, qo_cfg)
    cls = f"rand_{n}j_{m}m"
    f1 = U.run("S5", P, a0, m, od, lb, n, UA.AGENT_FIXED, val[cls], 0.1, 0.1, 7, T=400)
    f2 = U.run("S5", P, a0, m, od, lb, n, UA.AGENT_FIXED, val[cls], 0.1, 0.1, 7, T=400)
    assert f1 == f2


@pytest.mark.skipif(not list(R.glob("ur_test_S5_s*of3_entrada234.csv")), reason="resultados da Entrada 234 ausentes")
def test_stored_learner_results_reproduce():
    """os resultados T = 1000 dos aprendizes que o CPU igual reaproveita sao os que o codigo recalcula (mesma instancia e semente)"""
    val, qo_cfg = E.load_choices("S5")
    rows = []
    for f in sorted(R.glob("ur_test_S5_s*of3_entrada234.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["cls"] == f"rand_{CLASSES[0][0]}j_{CLASSES[0][1]}m" and r["rep"] == "0" and r["seed_i"] == "0" and r["method"] in ("ql", "ucb", "qo"):
                    rows.append(r)
    assert rows
    n, m = CLASSES[0]
    P, a0, od, lb = U.make(n, m, 0, 0, False)
    for r in rows:
        mk, ev = E.run_learner("S5", r["method"], P, a0, m, od, lb, n, U.seed_of("S5", 0, 0, r["method"], 0), qo_cfg)
        assert int(mk) == int(float(r["makespan"]))
