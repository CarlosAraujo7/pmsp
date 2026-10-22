"""
pmsp/pcmax_features.py (Entrada 200, Ciclo 18, frente B)
==========================================================
Atributos de instancia para o SELETOR OFFLINE condicionado a instancia (P||Cmax): so' quantidades invariantes a escala (razoes) e (n, m), calculadas dos tempos de processamento e do limite inferior classico. Nenhuma informacao de familia entra: o seletor so' ve' a instancia.
  n, m, log n, n/m, CV dos tempos, max/media, min/media, mediana/media, q10/media, q90/media, assimetria, curtose, fracao de valores distintos, m pmax / soma, LB / (soma/m), LPT / LB - 1 (o makespan da partida LPT, em %), n/m arredondado de fracao (n mod m)/m.
"""
from __future__ import annotations

import numpy as np

NAMES = ["n", "m", "log_n", "n_over_m", "cv", "max_over_mean", "min_over_mean", "median_over_mean", "q10_over_mean", "q90_over_mean", "skew", "kurt", "distinct_frac", "m_pmax_over_sum", "lb_over_avg", "lpt_gap_pct", "n_mod_m_frac"]


def lpt_makespan(p: np.ndarray, m: int) -> int:
    """Makespan do LPT (listas longas primeiro, cada tarefa na maquina menos carregada)."""
    import heapq
    loads = [0] * m
    heapq.heapify(loads)
    for x in sorted(p.tolist(), reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + x)
    return max(loads)


def features(p, m: int, lb: float) -> np.ndarray:
    """Vetor de atributos (ordem de NAMES). p: tempos (int64), m: maquinas, lb: limite inferior classico da instancia."""
    x = np.asarray(p, dtype=np.float64)
    n = len(x)
    mu = x.mean()
    sd = x.std()
    z = (x - mu) / sd if sd > 0 else np.zeros(n)
    s = np.sort(x)
    total = x.sum()
    lpt = lpt_makespan(np.asarray(p, dtype=np.int64), m)
    q10, q50, q90 = np.percentile(x, [10, 50, 90])
    return np.array([n, m, np.log(n), n / m, sd / mu, s[-1] / mu, s[0] / mu, q50 / mu, q10 / mu, q90 / mu, float((z ** 3).mean()), float((z ** 4).mean() - 3.0), len(np.unique(x)) / n,
                     m * s[-1] / total, lb / (total / m), 100.0 * (lpt - lb) / lb, (n % m) / m], dtype=np.float64)
