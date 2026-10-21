"""
pmsp/dqn.py (Entrada 173, Ciclo 14, frente I)
===============================================
DQN 'de tamanho respeitavel' (somente numpy) como seletor de acoes, para responder 'o aprendiz neural era pequeno demais' (pmsp/neuralq.py: uma camada de 16 unidades, sem replay, sem rede-alvo).
Rede Q(x, .) com DUAS camadas ocultas (64, 64, tanh), buffer de REPLAY (256 transicoes), REDE-ALVO (copiada a cada 50 passos), minibatch de 16, otimizador ADAM (beta1 0,9, beta2 0,999), erro TD recortado em [-1, 1] (Huber).
Contexto x (dimensao 5 + 2 |A|): [1, r1, r2, progresso, estagnacao] (os mesmos de NeuralQSelector) + a TAXA DE SUCESSO MOVEL de cada acao (media exponencial, fator 0,9, inicio 0,5) + a ultima acao (one-hot): o historico que um
aprendiz online precisa para distinguir acoes cuja utilidade depende do momento da busca. Recompensa binaria de sucesso (1 se a iteracao melhorou), gamma = 0,9, epsilon-gulosa 1,0 -> 0,05 por 0,995 (iguais ao agente tabular e ao NeuralQSelector).
Unico hiperparametro ajustado: a taxa de aprendizado (em instancias de ajuste). Interface: BaseSelector (select/update/reset/name).
"""
from __future__ import annotations

import numpy as np

from pmsp.selectors import BaseSelector


class DQNSelector(BaseSelector):
    def __init__(self, n_actions: int, hidden: tuple = (64, 64), lr: float = 0.003, gamma: float = 0.9, epsilon_start: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
                 buffer_size: int = 256, batch: int = 16, target_every: int = 50, success_decay: float = 0.9, seed: int | None = None) -> None:
        self.n_actions, self.hidden, self.lr, self.gamma = n_actions, tuple(hidden), lr, gamma
        self.eps0, self.eps_min, self.eps_decay = epsilon_start, epsilon_min, epsilon_decay
        self.buffer_size, self.batch, self.target_every, self.success_decay = buffer_size, batch, target_every, success_decay
        self.in_dim = 5 + 2 * n_actions
        self._seed = seed
        self.reset()

    # ------------------------------------------------------------------ rede
    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        sizes = [self.in_dim, *self.hidden, self.n_actions]
        self.W = [self._rng.normal(0.0, np.sqrt(1.0 / sizes[i]), size=(sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.W_t = [w.copy() for w in self.W]
        self.b_t = [b.copy() for b in self.b]
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._adam_t = 0
        self.eps = self.eps0
        self._f0 = None
        self._pending = None
        self._succ = np.full(self.n_actions, 0.5)
        self._last = -1
        self._buf_x = np.zeros((self.buffer_size, self.in_dim))
        self._buf_a = np.zeros(self.buffer_size, dtype=int)
        self._buf_r = np.zeros(self.buffer_size)
        self._buf_x2 = np.zeros((self.buffer_size, self.in_dim))
        self._buf_n = 0
        self._buf_i = 0
        self._steps = 0

    @staticmethod
    def _forward(W, b, X):
        """Retorna as ativacoes de cada camada (a ultima e' linear)."""
        acts = [X]
        h = X
        for i in range(len(W) - 1):
            h = np.tanh(h @ W[i] + b[i])
            acts.append(h)
        acts.append(h @ W[-1] + b[-1])
        return acts

    def _q(self, x, target=False):
        W, b = (self.W_t, self.b_t) if target else (self.W, self.b)
        return self._forward(W, b, x[None, :])[-1][0]

    def _context(self, features):
        f = np.asarray(features, dtype=float)
        if self._f0 is None:
            self._f0 = (f[0], f[1])
        r1 = f[0] / self._f0[0] if self._f0[0] > 0 else 0.0
        r2 = f[1] / self._f0[1] if self._f0[1] > 0 else 0.0
        last = np.zeros(self.n_actions)
        if self._last >= 0:
            last[self._last] = 1.0
        return np.concatenate([[1.0, min(max(r1, 0.0), 2.0), min(max(r2, 0.0), 2.0), float(f[2]), float(f[3])], self._succ, last])

    def _train_step(self):
        n = self._buf_n
        if n < self.batch:
            return
        idx = self._rng.integers(0, n, size=self.batch)
        X, A, Rw, X2 = self._buf_x[idx], self._buf_a[idx], self._buf_r[idx], self._buf_x2[idx]
        q_next = self._forward(self.W_t, self.b_t, X2)[-1].max(axis=1)
        target = Rw + self.gamma * q_next
        acts = self._forward(self.W, self.b, X)
        out = acts[-1]
        err = np.clip(out[np.arange(self.batch), A] - target, -1.0, 1.0) / self.batch
        delta = np.zeros_like(out)
        delta[np.arange(self.batch), A] = err
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)
        self._adam_t += 1
        t = self._adam_t
        c1, c2 = 1.0 - 0.9 ** t, 1.0 - 0.999 ** t
        for i in range(len(self.W)):
            self._mW[i] = 0.9 * self._mW[i] + 0.1 * gW[i]
            self._vW[i] = 0.999 * self._vW[i] + 0.001 * gW[i] ** 2
            self._mb[i] = 0.9 * self._mb[i] + 0.1 * gb[i]
            self._vb[i] = 0.999 * self._vb[i] + 0.001 * gb[i] ** 2
            self.W[i] -= self.lr * (self._mW[i] / c1) / (np.sqrt(self._vW[i] / c2) + 1e-8)
            self.b[i] -= self.lr * (self._mb[i] / c1) / (np.sqrt(self._vb[i] / c2) + 1e-8)

    # ------------------------------------------------------------ interface
    def select(self, state_id: int, features: np.ndarray) -> int:
        x = self._context(features)
        if self._pending is not None:                           # fecha a transicao anterior (x_{t+1} conhecido agora) e treina
            px, pa, pr = self._pending
            i = self._buf_i
            self._buf_x[i], self._buf_a[i], self._buf_r[i], self._buf_x2[i] = px, pa, pr, x
            self._buf_i = (i + 1) % self.buffer_size
            self._buf_n = min(self._buf_n + 1, self.buffer_size)
            self._train_step()
            self._steps += 1
            if self._steps % self.target_every == 0:
                self.W_t = [w.copy() for w in self.W]
                self.b_t = [b.copy() for b in self.b]
            self._pending = None
        if self._rng.random() < self.eps:
            a = int(self._rng.integers(0, self.n_actions))
        else:
            a = int(np.argmax(self._q(x)))
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        self._last_ctx = x
        return a

    def update(self, state_id: int, features: np.ndarray, action: int, reward: float, next_state_id: int) -> None:
        r = 1.0 if reward > 0 else 0.0
        self._pending = (self._last_ctx.copy(), int(action), r)
        self._succ[action] = self.success_decay * self._succ[action] + (1.0 - self.success_decay) * r
        self._last = int(action)

    def name(self) -> str:
        return f"DQN(h={self.hidden}, lr={self.lr})"
