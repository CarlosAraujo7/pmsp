import numpy as np

from pmsp import pfsp_features as PF
from pmsp.pfsp import PfsInstance


def _inst(seed=0, n=30, m=5):
    inst = PfsInstance.random(n, m, seed)
    return np.ascontiguousarray(inst.p), float(inst.lower_bound)


def test_shape_and_finite():
    p, lb = _inst()
    f = PF.features(p, lb)
    assert f.shape == (len(PF.NAMES),) and np.isfinite(f).all()


def test_scale_invariant_except_n_m_and_log():
    p, lb = _inst(1)
    a, b = PF.features(p, lb), PF.features(p * 7, lb * 7)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9)


def test_known_values_on_constant_matrix():
    p = np.full((4, 2), 5, dtype=np.int64)
    f = dict(zip(PF.NAMES, PF.features(p, 20.0)))
    assert f["n"] == 4 and f["m"] == 2 and f["n_over_m"] == 2 and f["cv"] == 0 and f["max_over_mean"] == 1
    assert f["load_max_over_mean"] == 1 and f["job_max_over_mean"] == 1 and f["lb_over_maxload"] == 1.0
