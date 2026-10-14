"""
pmsp/generators.py (Entrada 108)
Distribuicoes de tempos de processamento ALEM de U[1,100] para testar a generalidade das conclusoes (WP6).
Todas devolvem um vetor inteiro positivo de tamanho n, deterministico dado `seed`.

  u1_100     U{1..100}                        (baseline do artigo)
  u50_100    U{50..100}                       (faixa estreita)
  u1_10000   U{1..10000}                      (granularidade fina: quase sem repeticoes)
  lognormal  max(1, round(exp(N(3, 1))))      (cauda direita pesada)
  bimodal    50% U{5..25} + 50% U{75..100}    (mistura de tarefas pequenas e grandes)
  pareto     min(1000, ceil(10 (1-U)^(-1/1.5)))  (cauda pesada, alpha = 1,5)
"""
from __future__ import annotations

import numpy as np

from .instance import PMSPInstance

DISTRIBUTIONS = ["u1_100", "u50_100", "u1_10000", "lognormal", "bimodal", "pareto"]


def sample_times(dist: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist == "u1_100":
        p = rng.integers(1, 101, size=n)
    elif dist == "u50_100":
        p = rng.integers(50, 101, size=n)
    elif dist == "u1_10000":
        p = rng.integers(1, 10001, size=n)
    elif dist == "lognormal":
        p = np.maximum(1, np.rint(np.exp(rng.normal(3.0, 1.0, size=n)))).astype(np.int64)
    elif dist == "bimodal":
        small = rng.integers(5, 26, size=n)
        large = rng.integers(75, 101, size=n)
        p = np.where(rng.random(n) < 0.5, small, large)
    elif dist == "pareto":
        p = np.minimum(1000, np.ceil(10.0 * (1.0 - rng.random(n)) ** (-1.0 / 1.5))).astype(np.int64)
    else:
        raise ValueError(f"distribuicao desconhecida: {dist!r}")
    return np.asarray(p, dtype=np.int64)


def make_instance(dist: str, n: int, m: int, seed: int) -> PMSPInstance:
    return PMSPInstance(p=sample_times(dist, n, seed), m=m, name=f"{dist}_{n}j_{m}m_s{seed}")
