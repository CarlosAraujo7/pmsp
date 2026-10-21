"""
pmsp/neuralq.py (Entrada 162, Ciclo 13)
=========================================
Q-learning com REDE NEURAL pequena (um 'DQN minimo', somente numpy) como seletor de acoes: Q(x, a) = W2 tanh(W1 x + b1) + b2, com contexto continuo x = [1, r1, r2, progresso, estagnacao] (o mesmo do LinUCB;
r1 e r2 relativos ao valor no primeiro passo, em [0, 2]). Recompensa binaria de sucesso (1 se a iteracao melhorou o makespan). Aprendizado ONLINE, sem replay e sem rede-alvo:
no passo t o seletor guarda (x_t, a_t, r_t) e, quando o contexto x_{t+1} chega a `select`, faz um passo de gradiente descendente em (Q(x_t, a_t) - (r_t + gamma max_a' Q(x_{t+1}, a')))^2.
Exploracao epsilon-gulosa com os mesmos epsilon inicial/minimo/decaimento do agente tabular (1,0 -> 0,05, fator 0,995). Responde a critica 'o aprendiz nao e' neural'; o tamanho da rede e a taxa de aprendizado
(hidden, lr) sao os UNICOS parametros ajustados, em instancias de ajuste. Interface: BaseSelector.
"""
from __future__ import annotations

import numpy as np

from pmsp.selectors import BaseSelector


class NeuralQSelector(BaseSelector):
    def __init__(self, n_actions: int, hidden: int = 16, lr: float = 0.02, gamma: float = 0.9, epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995, seed: int | None = None, ctx_dim: int = 5) -> None:
        self.n_actions, self.hidden, self.lr, self.gamma = n_actions, hidden, lr, gamma
        self.eps0, self.eps_min, self.eps_decay, self.ctx_dim = epsilon_start, epsilon_min, epsilon_decay, ctx_dim
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        d, h, a = self.ctx_dim, self.hidden, self.n_actions
        self.W1 = self._rng.normal(0, 1.0 / np.sqrt(d), size=(h, d))
        self.b1 = np.zeros(h)
        self.W2 = self._rng.normal(0, 1.0 / np.sqrt(h), size=(a, h))
        self.b2 = np.zeros(a)
        self.eps = self.eps0
        self._pending = None
        self._f0 = None

    def _context(self, features):
        f = np.asarray(features, dtype=float)
        if self._f0 is None:
            self._f0 = (f[0], f[1])
        r1 = f[0] / self._f0[0] if self._f0[0] > 0 else 0.0
        r2 = f[1] / self._f0[1] if self._f0[1] > 0 else 0.0
        return np.array([1.0, min(max(r1, 0.0), 2.0), min(max(r2, 0.0), 2.0), float(f[2]), float(f[3])])

    def _q(self, x):
        h = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2, h

    def _sgd(self, x, a, target):
        q, h = self._q(x)
        err = float(np.clip(q[a] - target, -1.0, 1.0))        # gradiente da perda de Huber (recorte) para estabilidade
        dh = err * self.W2[a] * (1.0 - h * h)
        self.W2[a] -= self.lr * err * h
        self.b2[a] -= self.lr * err
        self.W1 -= self.lr * np.outer(dh, x)
        self.b1 -= self.lr * dh

    def select(self, state_id: int, features: np.ndarray) -> int:
        x = self._context(features)
        if self._pending is not None:                           # passo TD do instante anterior, agora que x_{t+1} e' conhecido
            px, pa, pr = self._pending
            self._sgd(px, pa, pr + self.gamma * float(self._q(x)[0].max()))
            self._pending = None
        if self._rng.random() < self.eps:
            a = int(self._rng.integers(0, self.n_actions))
        else:
            a = int(np.argmax(self._q(x)[0]))
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return a

    def update(self, state_id: int, features: np.ndarray, action: int, reward: float, next_state_id: int) -> None:
        self._pending = (self._context(features), int(action), 1.0 if reward > 0 else 0.0)

    def name(self) -> str:
        return f"NeuralQ(h={self.hidden}, lr={self.lr})"
