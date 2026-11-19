"""Entrada 281: o executor do regime de poucos dados reproduz as execucoes das Entradas 271 (aquecido) e 272 (congelado) com a mesma priori e a mesma semente, e a priori por subconjunto e' bem formada."""
import numpy as np

from experiment import run_cycle32_fanjul_entrada256 as F
from experiment import run_cycle43_warmqo_entrada271 as W271
from experiment import run_cycle44_frozenqo_entrada272 as Z272
from experiment import run_cycle49_scarce_entrada281 as S


def _first_tune_instance():
    man = F.manifest()
    r = man[man.role == "tune"].iloc[0]
    P, a0, od, lb = F.load(r)
    return P, a0, int(r.m), od, lb, int(r.n)


def test_score_is_in_unit_interval_and_peaks_at_the_best_action():
    r = np.array([1.05, 1.02, 1.10, 1.03])
    sc = S.score_of(r)
    assert sc.min() == 0.0 and sc.max() == 1.0 and int(sc.argmax()) == int(r.argmin())


def test_tuning_blocks_have_ten_instances_per_class_and_thirty_actions():
    T = S.tuning_blocks()
    assert len(T) == 28 and all(v.shape == (10, 30, 2) for v in T.values())


def test_runner_reproduces_the_warm_and_frozen_runs_of_entries_271_and_272():
    P, a0, m, od, lb, n = _first_tune_instance()
    F.warm()
    qo_cfg, _ = F.frozen("B30")
    sc = np.random.default_rng(3).random(30)
    sc = (sc - sc.min()) / (sc.max() - sc.min())
    for seed in (101, 102):
        assert S.run_one(P, a0, m, od, lb, n, seed, qo_cfg, sc, None) == W271.run_one(P, a0, m, od, lb, n, seed, qo_cfg, sc)
        assert S.run_one(P, a0, m, od, lb, n, seed, qo_cfg, sc, 0.0) == Z272.run_one(P, a0, m, od, lb, n, seed, qo_cfg, sc)


def test_variant_list_has_two_kinds_for_each_of_sixteen_subset_draws():
    assert len(S.VARIANTS) == 32 and len(S.KEYS) == 16 and {v[0] for v in S.VARIANTS} == {"w", "f"}
