"""
pmsp/fixed_selector.py (Entrada 104)
Seletor que escolhe SEMPRE a mesma acao (estrategia de destruicao do IG). Usado para medir a "folga" (headroom)
disponivel para qualquer politica de selecao: se a melhor estrategia fixa nao supera a selecao uniforme
aleatoria, nao ha o que um seletor aprender.
"""
from __future__ import annotations

import numpy as np

from .selectors import BaseSelector


class FixedActionSelector(BaseSelector):
    def __init__(self, action: int) -> None:
        self._action = int(action)

    def select(self, state_id: int, features: np.ndarray) -> int:
        return self._action

    def update(self, state_id, features, action, reward, next_state_id) -> None:
        return None

    def reset(self) -> None:
        return None

    @property
    def name(self) -> str:
        return f"fixed{self._action}"
