"""
pmsp/linucb.py (Entrada 159, Ciclo 12)
=======================================
Bandit CONTEXTUAL linear (LinUCB disjunto; Li, Chu, Langford e Schapire, 2010) como seletor de acoes, para responder a critica 'o aprendiz e um espantalho' com um aprendiz de OUTRA classe que o Q-Learning tabular:
aproximacao de funcao linear sobre um contexto continuo, sem tabela de estados. Somente numpy.

Contexto x = [1, r1, r2, t/T, estagnacao], onde r1 e r2 sao as duas primeiras features do solver RELATIVAS ao valor no primeiro passo da execucao (mesma relativizacao do estado `rel` das Entradas 118/145;
limitadas a [0, 2]) e as duas ultimas ja vem em [0, 1]. Recompensa: 1 se a iteracao melhorou o makespan, 0 caso contrario (o sinal de sucesso das Entradas 155). Para cada acao a: A_a = lam I + soma x x', b_a = soma r x;
escolhe argmax_a theta_a x + alpha sqrt(x' A_a^-1 x), theta_a = A_a^-1 b_a; empates desfeitos por ruido de 1e-9 (semente do seletor).
Interface: BaseSelector (select(state_id, features), update(state_id, features, action, reward, next_state_id), reset()).
"""
from __future__ import annotations

import numpy as np

from pmsp.selectors import BaseSelector


class LinUCBSelector(BaseSelector):
    def __init__(self, n_actions: int, alpha: float = 1.0, lam: float = 1.0, seed: int | None = None, ctx_dim: int = 5) -> None:
        self.n_actions, self.alpha, self.lam, self.ctx_dim = n_actions, alpha, lam, ctx_dim
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.A = np.stack([self.lam * np.eye(self.ctx_dim) for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, self.ctx_dim))
        self._f0 = None

    def _context(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, dtype=float)
        if self._f0 is None:
            self._f0 = (f[0], f[1])
        r1 = f[0] / self._f0[0] if self._f0[0] > 0 else 0.0
        r2 = f[1] / self._f0[1] if self._f0[1] > 0 else 0.0
        return np.array([1.0, min(max(r1, 0.0), 2.0), min(max(r2, 0.0), 2.0), float(f[2]), float(f[3])])

    def select(self, state_id: int, features: np.ndarray) -> int:
        x = self._context(features)
        scores = np.empty(self.n_actions)
        for a in range(self.n_actions):
            ainv_x = np.linalg.solve(self.A[a], x)
            theta = np.linalg.solve(self.A[a], self.b[a])
            scores[a] = float(theta @ x) + self.alpha * float(np.sqrt(max(x @ ainv_x, 0.0)))
        scores += 1e-9 * self._rng.random(self.n_actions)
        return int(np.argmax(scores))

    def update(self, state_id: int, features: np.ndarray, action: int, reward: float, next_state_id: int) -> None:
        x = self._context(features)
        self.A[action] += np.outer(x, x)
        self.b[action] += (1.0 if reward > 0 else 0.0) * x

    def name(self) -> str:
        return f"LinUCB(alpha={self.alpha})"
