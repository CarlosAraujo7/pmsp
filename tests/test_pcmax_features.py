import numpy as np
import pytest

from pmsp import pcmax_features as F
from pmsp.instance import PMSPInstance


def _inst(seed=0, n=40, m=5):
    p = np.random.default_rng(seed).integers(1, 100, size=n).astype(np.int64)
    inst = PMSPInstance(p=p, m=m, name="t")
    return p, m, inst.lower_bound


def test_shape_and_names():
    p, m, lb = _inst()
    f = F.features(p, m, lb)
    assert f.shape == (len(F.NAMES),)
    assert np.isfinite(f).all()


def test_scale_invariant_except_n_m():
    p, m, lb = _inst(1)
    a = F.features(p, m, lb)
    b = F.features(p * 1000, m, lb * 1000)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9)


def test_known_values():
    p = np.array([4, 4, 4, 4], dtype=np.int64)
    f = dict(zip(F.NAMES, F.features(p, 2, 8.0)))
    assert f["n"] == 4 and f["m"] == 2 and f["n_over_m"] == 2
    assert f["cv"] == 0 and f["max_over_mean"] == 1 and f["distinct_frac"] == 0.25
    assert f["lpt_gap_pct"] == 0.0 and f["lb_over_avg"] == 1.0 and f["m_pmax_over_sum"] == 0.5


def test_lpt_makespan_matches_reference():
    p = np.array([7, 5, 4, 3, 3, 2], dtype=np.int64)
    # LPT: 7 | 5 | 4 -> m2 (5+4=9)?; computed by hand with 2 machines: 7,5 -> loads (7,5); 4 -> 5+4=9; 3 -> 7+3=10; 3 -> 9+3=12; 2 -> 10+2=12
    assert F.lpt_makespan(p, 2) == 12


def test_constant_instance_no_nan():
    f = F.features(np.full(10, 7, dtype=np.int64), 3, 24.0)
    assert np.isfinite(f).all()
