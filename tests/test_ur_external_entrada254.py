"""Testes das familias publicadas de R||Cmax (Entrada 254): geradores, estrutura de correlacao, sementes e execucao."""
import numpy as np

from experiment import run_cycle30_ur_external_entrada254 as X
from pmsp.instance_ur_ext import FAMILIES, SIZES, gen


def test_sizes_and_classes():
    assert len(SIZES) == 25 and {m for _, m in SIZES} == {3, 5, 10, 15, 20} and {n for n, _ in SIZES} == {30, 50, 80, 100, 200}
    assert len(X.CLASSES) == 75 and len({X.cls_name(*c) for c in X.CLASSES}) == 75


def test_generators_follow_the_published_definitions():
    n, m = 200, 20
    P = gen("unc", n, m, 1)
    assert P.shape == (n, m) and P.dtype == np.int64 and P.min() >= 10 and P.max() <= 100
    P = gen("cj", n, m, 2)                       # p_ij = d_ij + b_j: the within-row spread is d (U{1,20}), the row level is b_j
    assert P.min() >= 2 and P.max() <= 120
    assert (P.max(axis=1) - P.min(axis=1)).max() <= 19
    assert np.corrcoef(P.mean(axis=1), P[:, 0])[0, 1] > 0.9
    P = gen("cm", n, m, 3)                       # p_ij = a_i + d_ij: the within-column spread is d, the column level is a_i
    assert P.min() >= 2 and P.max() <= 120
    assert (P.max(axis=0) - P.min(axis=0)).max() <= 19
    assert np.corrcoef(P.mean(axis=0), P[0, :])[0, 1] > -1.0
    for f in FAMILIES:
        assert (gen(f, 50, 5, 7) == gen(f, 50, 5, 7)).all() and not (gen(f, 50, 5, 7) == gen(f, 50, 5, 8)).all()


def test_instances_are_distinct_per_class_and_split():
    seen = set()
    for ci in (0, 30, 74):
        for rep in (0, 19):
            for tune in (False, True):
                P, a0, od, lb = X.make(ci, rep, tune)
                assert lb > 0 and len(a0) == P.shape[0]
                seen.add(P.tobytes())
    assert len(seen) == 12


def test_seeds_unique():
    seen = set()
    for sp in X.SPACES:
        for ci in (0, 74):
            for rep in (0, 19):
                for me in X.METHODS + [f"arm{a}" for a in (0, 14)]:
                    for s in range(5):
                        for tune in (False, True):
                            v = X.seed_of(sp, ci, rep, me, s, tune)
                            assert v not in seen
                            seen.add(v)


def test_methods_run_and_are_deterministic():
    P, a0, od, lb = X.make(7, 0, True)
    f, n, m = X.CLASSES[7]
    qo = (0.1, 0.3)
    for me in X.METHODS + ["arm3"]:
        r1 = X.run_method("S5", me, P, a0, m, od, lb, n, 11, 2, qo)
        r2 = X.run_method("S5", me, P, a0, m, od, lb, n, 11, 2, qo)
        assert r1 == r2 and r1[0] >= lb * 0.99
