"""
pmsp/rewards.py (Entrada 104)
Recompensas alternativas para o IG-QL (a original e' max(delta, 0), com delta = reducao do makespan, que e' nula
na quase totalidade das iteracoes -- ver Entrada 98).

dense_balance_reward: reducao relativa (com sinal) da soma dos quadrados das cargas das maquinas,
(SS_antes - SS_depois)/SS_antes. Como a carga total e' constante, SS cai se e somente se as cargas ficam mais
equilibradas; e' diferente de zero em praticamente toda iteracao (sinal denso), correlacionada com -- mas nao
igual a -- reducao do makespan. A escala da recompensa nao afeta a politica gulosa (Q inicializada em zero).
"""
from __future__ import annotations

import numpy as np


def dense_balance_reward(sol, sol_new, delta) -> float:
    a = np.asarray(sol.loads, dtype=np.float64)
    b = np.asarray(sol_new.loads, dtype=np.float64)
    ss_a = float(np.dot(a, a))
    if ss_a <= 0.0:
        return 0.0
    return (ss_a - float(np.dot(b, b))) / ss_a
