"""
pmsp/instance_tsp.py (Entrada 267)
Geradores de instancias do TSP euclidiano para o segundo dominio do estudo (pmsp/tsp_numba.py): pontos uniformes e agrupados em [0, 10000]^2, matriz de distancias e listas de vizinhos.
"""
from __future__ import annotations

import numpy as np


def make_points(kind: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "unif":
        return rng.uniform(0.0, 10000.0, size=(n, 2))
    k = max(3, n // 40)
    centers = rng.uniform(1000.0, 9000.0, size=(k, 2))
    lab = rng.integers(0, k, size=n)
    pts = centers[lab] + rng.normal(0.0, 400.0, size=(n, 2))
    return np.clip(pts, 0.0, 10000.0)


def dist_matrix(pts: np.ndarray) -> np.ndarray:
    d = pts[:, None, :] - pts[None, :, :]
    return np.ascontiguousarray(np.sqrt((d * d).sum(axis=2)))


def neighbor_lists(D: np.ndarray, K: int = 10) -> np.ndarray:
    n = D.shape[0]
    K = min(K, n - 1)
    Dm = D + np.diag(np.full(n, np.inf))
    return np.ascontiguousarray(np.argsort(Dm, axis=1, kind="stable")[:, :K].astype(np.int64))


def build(kind: str, n: int, seed: int, K: int = 10):
    pts = make_points(kind, n, seed)
    D = dist_matrix(pts)
    return D, neighbor_lists(D, K)
