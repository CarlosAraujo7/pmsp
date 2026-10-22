"""
pmsp/pfsp_features.py (Entrada 205, Ciclo 18)
===============================================
Atributos de instancia do flowshop de permutacao para o SELETOR OFFLINE: so' razoes invariantes a escala e (n, m), calculadas da matriz de tempos p (n x m) e do limite inferior: n, m, log n, n/m, CV dos tempos, max/media, assimetria, curtose,
desequilibrio de carga das maquinas (max/media e CV das somas por maquina), dispersao das somas por tarefa (max/media e CV), LB / (maior carga de maquina).
"""
from __future__ import annotations

import numpy as np

NAMES = ["n", "m", "log_n", "n_over_m", "cv", "max_over_mean", "skew", "kurt", "load_max_over_mean", "load_cv", "job_max_over_mean", "job_cv", "lb_over_maxload"]


def features(p, lb: float) -> np.ndarray:
    x = np.asarray(p, dtype=np.float64)
    n, m = x.shape
    mu, sd = x.mean(), x.std()
    z = (x - mu) / sd if sd > 0 else np.zeros_like(x)
    loads, jobs = x.sum(axis=0), x.sum(axis=1)
    return np.array([n, m, np.log(n), n / m, sd / mu, x.max() / mu, float((z ** 3).mean()), float((z ** 4).mean() - 3.0), loads.max() / loads.mean(), loads.std() / loads.mean(), jobs.max() / jobs.mean(), jobs.std() / jobs.mean(), lb / loads.max()], dtype=np.float64)
