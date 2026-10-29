"""
pmsp/ur_features.py (Entrada 240, Ciclo 22)
=============================================
Atributos de instancia do SELETOR OFFLINE para MAQUINAS NAO RELACIONADAS (R||Cmax), invariantes a escala (razoes) e (n, m); calculados da matriz de tempos P (n x m), do limite inferior simples e do makespan da partida gulosa:
  n, m, log n, n/m, CV do tempo minimo por tarefa, media de min_k P / media_k P, CV da media por tarefa, CV da media por maquina (heterogeneidade de velocidade), amplitude (max - min)/media por tarefa, 2o menor / menor tempo,
  fracao de tarefas cuja melhor maquina e' a mais popular, LB / (soma dos minimos / m), folga da partida gulosa ao LB (%), (n mod m) / m.
"""
from __future__ import annotations

import numpy as np

NAMES = ["n", "m", "log_n", "n_over_m", "min_cv", "min_over_mean", "row_cv", "col_cv", "spread", "second_over_min", "popular_frac", "lb_over_avg", "greedy_gap_pct", "n_m_rem"]


def features(P, lb, greedy_makespan):
    P = np.asarray(P, dtype=np.float64)
    n, m = P.shape
    mn = P.min(axis=1)
    mean_row = P.mean(axis=1)
    col = P.mean(axis=0)
    srt = np.sort(P, axis=1)
    pop = np.bincount(P.argmin(axis=1), minlength=m).max() / n
    return np.array([n, m, np.log(n), n / m, mn.std() / mn.mean(), float((mn / mean_row).mean()), mean_row.std() / mean_row.mean(), col.std() / col.mean(), float(((P.max(axis=1) - mn) / mean_row).mean()),
                     float((srt[:, 1] / np.maximum(srt[:, 0], 1.0)).mean()), pop, lb / (mn.sum() / m), 100.0 * (greedy_makespan - lb) / lb, (n % m) / m], dtype=np.float64)
