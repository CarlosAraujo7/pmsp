"""Testes do nucleo da politica neural (Entrada 259): determinismo por semente, uso total = T, vies pico = acao fixa, controle 'const' ignora as caracteristicas."""
import numpy as np

from pmsp import ur_numba_neural as NN
from pmsp.instance_ur import constructive_ur, lower_bound_ur

A, H, T = 5, 16, 60


def _setup():
    P = np.random.default_rng(3).integers(1, 60, size=(40, 5)).astype(np.int64)
    a0, od = constructive_ur(P)
    st = np.zeros(A, dtype=np.int64)
    d = np.array([2, 3, 4, 6, 8], dtype=np.int64)
    ls = np.zeros(A, dtype=np.int64)
    return P, a0, od, float(lower_bound_ur(P)), st, d, ls


def _theta(rng, peak=None):
    th = rng.normal(0, 0.3, NN.n_params(A, H))
    W1, b1, W2, b2 = NN.unpack(th, A, H)
    if peak is not None:
        W2[:] = 0.0
        b2[:] = -50.0
        b2[peak] = 50.0
    return W1, b1, W2, b2


def test_deterministic_and_usage_sums():
    P, a0, od, lb, st, d, ls = _setup()
    W1, b1, W2, b2 = _theta(np.random.default_rng(1))
    o1 = NN.ur_neural_core(P, a0, 5, st, d, ls, T, W1, b1, W2, b2, 1, 11, od, lb)
    o2 = NN.ur_neural_core(P, a0, 5, st, d, ls, T, W1, b1, W2, b2, 1, 11, od, lb)
    assert o1[0] == o2[0] and np.array_equal(o1[2], o2[2])
    assert int(o1[2].sum()) == T


def test_peaked_bias_is_a_fixed_arm():
    P, a0, od, lb, st, d, ls = _setup()
    W1, b1, W2, b2 = _theta(np.random.default_rng(2), peak=3)
    best, ev, used = NN.ur_neural_core(P, a0, 5, st, d, ls, T, W1, b1, W2, b2, 1, 5, od, lb)
    assert used[3] == T and ev > 0
    assert 0 < best <= 10 ** 9


def test_const_variant_ignores_features():
    P, a0, od, lb, st, d, ls = _setup()
    rng = np.random.default_rng(4)
    W1, b1, W2, b2 = _theta(rng)
    W1b = W1.copy()
    W1b[:, :NN.NF - 1] = rng.normal(0, 1.0, (H, NN.NF - 1))      # muda so' as colunas das caracteristicas (nao a do vies)
    o1 = NN.ur_neural_core(P, a0, 5, st, d, ls, T, W1, b1, W2, b2, 0, 9, od, lb)
    o2 = NN.ur_neural_core(P, a0, 5, st, d, ls, T, np.ascontiguousarray(W1b), b1, W2, b2, 0, 9, od, lb)
    assert o1[0] == o2[0] and np.array_equal(o1[2], o2[2])


def test_features_change_policy_when_used():
    P, a0, od, lb, st, d, ls = _setup()
    rng = np.random.default_rng(6)
    W1, b1, W2, b2 = _theta(rng)
    W1b = np.ascontiguousarray(W1 + rng.normal(0, 2.0, W1.shape))
    u1 = NN.ur_neural_core(P, a0, 5, st, d, ls, T, W1, b1, W2, b2, 1, 9, od, lb)[2]
    u2 = NN.ur_neural_core(P, a0, 5, st, d, ls, T, W1b, b1, W2, b2, 1, 9, od, lb)[2]
    assert not np.array_equal(u1, u2)
