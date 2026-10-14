"""
pmsp/window_bandits.py (Entrada 155)
=====================================
Seletores de acoes desenhados para a JANELA CURTA DE APRENDIZADO observada no estudo (>90% da recompensa de uma execucao chega nas primeiras ~10% das iteracoes; Entrada 147), ao contrario do
Q-Learning epsilon-guloso (que explora ao acaso exatamente nessa janela) e do UCB1 (bonus de exploracao grande frente a diferencas de recompensa minusculas). Sem estado; mesma interface BaseSelector.
  ThompsonBernoulliSelector   Thompson Sampling com sucesso = 'a iteracao melhorou o makespan' (recompensa > 0); prior Beta(a0, b0); fator de desconto opcional (nao estacionario)
  RacingSelector              eliminacao sucessiva (corrida): cada acao ativa e' puxada `n0` vezes por rodada; apos cada rodada elimina as acoes cuja taxa de sucesso e' menor que `keep` x a melhor; ao restar uma,
                              compromete-se com ela (empates entre sobreviventes: sorteio)
"""
from __future__ import annotations

import numpy as np

from .selectors import BaseSelector


class ThompsonBernoulliSelector(BaseSelector):
    def __init__(self, n_actions: int, seed: int | None = None, a0: float = 1.0, b0: float = 1.0, discount: float = 1.0) -> None:
        self._k = n_actions
        self._rng = np.random.default_rng(seed)
        self._a0, self._b0, self._gamma = a0, b0, discount
        self.reset()

    def reset(self) -> None:
        self._a = np.full(self._k, self._a0, dtype=np.float64)
        self._b = np.full(self._k, self._b0, dtype=np.float64)

    def select(self, state_id: int, features: np.ndarray) -> int:
        return int(np.argmax(self._rng.beta(self._a, self._b)))

    def update(self, state_id: int, features: np.ndarray, action: int, reward: float, next_state_id: int) -> None:
        if self._gamma < 1.0:
            self._a = self._a0 + self._gamma * (self._a - self._a0)
            self._b = self._b0 + self._gamma * (self._b - self._b0)
        if reward > 0:
            self._a[action] += 1.0
        else:
            self._b[action] += 1.0


class RacingSelector(BaseSelector):
    def __init__(self, n_actions: int, seed: int | None = None, n0: int = 10, keep: float = 0.5, min_survivors: int = 1) -> None:
        self._k = n_actions
        self._rng = np.random.default_rng(seed)
        self._n0, self._keep, self._min = n0, keep, min_survivors
        self.reset()

    def reset(self) -> None:
        self._active = list(range(self._k))
        self._succ = np.zeros(self._k)
        self._pulls = np.zeros(self._k)
        self._queue: list = []
        self._round_pulls = 0

    def _new_round(self) -> None:
        self._queue = [a for a in self._active for _ in range(self._n0)]
        self._rng.shuffle(self._queue)

    def select(self, state_id: int, features: np.ndarray) -> int:
        if len(self._active) == 1:
            return self._active[0]
        if not self._queue:
            self._new_round()
        return int(self._queue.pop())

    def update(self, state_id: int, features: np.ndarray, action: int, reward: float, next_state_id: int) -> None:
        self._pulls[action] += 1
        self._succ[action] += 1.0 if reward > 0 else 0.0
        if len(self._active) > 1 and not self._queue:               # fim de uma rodada: eliminar
            rate = self._succ[self._active] / np.maximum(self._pulls[self._active], 1)
            best = rate.max()
            survivors = [a for a, r in zip(self._active, rate) if r >= self._keep * best] if best > 0 else list(self._active)
            if len(survivors) < self._min:
                survivors = list(self._active)
            self._active = survivors
            if len(self._active) > 1 and all(self._pulls[a] >= 3 * self._n0 for a in self._active):   # rodadas esgotadas: sorteia entre os empatados
                self._active = [int(self._rng.choice(self._active))]
