import numpy as np
import pytest

from pmsp.generators import DISTRIBUTIONS, make_instance, sample_times


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_generators_positive_integer_deterministic(dist):
    a = sample_times(dist, 80, 7)
    b = sample_times(dist, 80, 7)
    c = sample_times(dist, 80, 8)
    assert a.shape == (80,) and a.dtype.kind == "i" and a.min() >= 1
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_generator_ranges():
    assert sample_times("u50_100", 500, 1).min() >= 50
    assert sample_times("pareto", 2000, 1).max() <= 1000
    bm = sample_times("bimodal", 2000, 1)
    assert not ((bm > 25) & (bm < 75)).any()
    inst = make_instance("lognormal", 30, 5, 3)
    assert inst.n == 30 and inst.m == 5
