"""
pmsp/qlearning.py
=================
Agente Q-Learning para selecao adaptativa de operadores de vizinhanca
no contexto da Learnheuristica ILS+RL para o PMSP.

Algoritmo de atualizacao (Q-Learning off-policy, Watkins 1989):

    Q(s, a) <- Q(s, a) + alpha * [ r  +  gamma * max_{a'} Q(s', a')  -  Q(s, a) ]

    onde:
        s         : estado atual em {0, ..., 624}
        a         : operador selecionado em {0, 1, 2, 3}
        r         : recompensa = melhora no makespan (delta >= 0)
        s'        : proximo estado
        alpha     : taxa de aprendizado (learning rate)
        gamma     : fator de desconto (discount factor)

Politica de selecao: epsilon-greedy com decaimento multiplicativo.
    - Com probabilidade epsilon: seleciona operador aleatoriamente (exploracao)
    - Com probabilidade 1-epsilon: seleciona argmax_a Q(s, a) (explotacao)
    - epsilon decai a cada passo: epsilon <- max(eps_min, epsilon * decay)

Espaco de estados: |S| = 3^4 = 81 (ver pmsp/bounds.py)
Espaco de acoes:   |A| = 4 operadores (Op1..Op4)

Compatibilidade: implementa a interface BaseSelector (pmsp/selectors.py)
para ser plugavel no mesmo motor experimental dos baselines.

Contribuicao XAI: a Q-table treinada revela a politica aprendida --
quais operadores sao preferidos em cada combinacao de (gap, desequilibrio,
progresso, estagnacao). Os metodos de analise (policy_matrix, q_heatmap_data)
sao usados diretamente na Fase 5 para gerar os heatmaps do artigo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .selectors import BaseSelector, N_OPERATORS

# Numero de estados no espaco discretizado: 3^4 = 81
N_STATES: int = 81


class QLearningAgent(BaseSelector):
    """
    Agente Q-Learning para AOS no ILS.

    Parametros
    ----------
    alpha : float
        Taxa de aprendizado em (0, 1]. Controla o quanto cada nova
        experiencia sobrescreve o valor Q anterior.
        Valor tipico: 0.1 (aprendizado moderado).
    gamma : float
        Fator de desconto em [0, 1). Pondera recompensas futuras.
        Para o PMSP (episodio finito), gamma=0.9 e comum.
    epsilon_start : float
        Epsilon inicial (1.0 = exploracao pura no inicio).
    epsilon_min : float
        Epsilon minimo apos decaimento completo.
    epsilon_decay : float
        Taxa de decaimento multiplicativo por passo.
        epsilon_start=1.0, decay=0.995, min=0.05:
            epsilon atinge 0.05 apos ~ ln(0.05/1.0)/ln(0.995) ~ 591 passos.
    seed : int | None
        Semente para reproducibilidade.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
        q_init = None,
        q_init_noise: float = 0.0,
        n_states: int = N_STATES,
        n_actions: int = N_OPERATORS,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)
        self._q_init = q_init  # optional pre-trained Q-table for warm-start
        # Ruido pequeno para cold-start (default 0.0 preserva comportamento
        # original exato). Adicionado em 30/08/2026 (diario, Entrada 14):
        # zeros exatos fazem estados nunca visitados "preferirem" a acao 0
        # (Op1) por desempate do np.argmax, nao por preferencia aprendida.
        self._q_init_noise = q_init_noise
        # Numero de estados (default N_STATES=81, desenho padrao). Configuravel
        # para o estudo de sensibilidade de granularidade (Entrada 16,
        # diario_pesquisa.md), usado com compute_state_cfg()/StateConfig em
        # pmsp/bounds.py. Default preserva comportamento original exato.
        self._n_states = n_states
        # Numero de acoes (default N_OPERATORS=4, os 4 operadores de
        # vizinhanca do ILS). Configuravel para reaproveitar esta MESMA
        # classe em frameworks de busca com outro espaco de acoes -- por
        # exemplo, o ig_ql (Iterated Greedy + Q-Learning) seleciona entre
        # 3 estrategias de destruicao, nao 4 operadores (ver pmsp/ig.py,
        # diario_pesquisa.md, Entrada 23). Default preserva comportamento
        # original exato.
        self._n_actions = n_actions
        self.reset(q_init=q_init)

    # ------------------------------------------------------------------
    # Interface BaseSelector
    # ------------------------------------------------------------------

    def select(self, state_id: int, features: np.ndarray) -> int:
        """
        Selecao epsilon-greedy.

        - Com prob. epsilon: acao aleatoria uniforme (exploracao).
        - Com prob. 1-epsilon: argmax_a Q[state_id, a] (explotacao).

        O epsilon e decrementado APOS a selecao (chamado a cada passo).
        """
        self._t += 1
        if self._rng.random() < self._epsilon:
            action = int(self._rng.integers(self._n_actions))
        else:
            action = int(np.argmax(self._Q[state_id]))
        # Decaimento multiplicativo do epsilon
        self._epsilon = max(self.epsilon_min,
                            self._epsilon * self.epsilon_decay)
        return action

    def update(
        self,
        state_id: int,
        features: np.ndarray,
        action: int,
        reward: float,
        next_state_id: int,
    ) -> None:
        """
        Atualiza a Q-table pelo erro TD (Temporal-Difference).

        TD-error = r + gamma * max_{a'} Q(s', a') - Q(s, a)
        Q(s, a) <- Q(s, a) + alpha * TD-error
        """
        best_next = float(self._Q[next_state_id].max())
        td_error = (reward
                    + self.gamma * best_next
                    - self._Q[state_id, action])
        self._Q[state_id, action] += self.alpha * td_error
        self._visits[state_id, action] += 1

    def reset(self, q_init=None) -> None:
        """
        Reinicia o agente. Se q_init fornecida (ou self._q_init definida
        no construtor), inicializa Q-table com valores pre-treinados
        (warm-start); caso contrario, zeros (cold-start) -- ou ruido
        uniforme pequeno em [-q_init_noise, +q_init_noise] se
        q_init_noise > 0, para evitar que estados nunca visitados
        "empatem" sempre para a acao 0 no argmax (default 0.0 preserva
        o comportamento original exato).

        Usa self._n_states (default N_STATES=81) em vez da constante do
        modulo diretamente, para suportar granularidades diferentes
        (Entrada 16, diario_pesquisa.md).
        """
        qi = q_init if q_init is not None else self._q_init
        if qi is not None:
            self._Q = qi.copy().astype(np.float64)
        elif self._q_init_noise > 0:
            self._Q = self._rng.uniform(
                -self._q_init_noise, self._q_init_noise,
                size=(self._n_states, self._n_actions))
        else:
            self._Q = np.zeros((self._n_states, self._n_actions), dtype=np.float64)
        self._visits = np.zeros((self._n_states, self._n_actions), dtype=np.int64)
        self._epsilon = self.epsilon_start
        self._t = 0

    def carry_forward(self) -> None:
        """
        Registra a Q-table atual como novo ponto de partida para a
        proxima chamada de reset() -- usado para ACUMULAR aprendizado
        entre multiplos episodios de treinamento (ver
        experiment/pilot_ql_traintest.py). Os metodos padrao
        (ils_ql/ils_ql_hi/ils_ql_ws) NAO usam isto -- cada execucao
        treina do zero ou de um warm-start fixo, sem acumulo.

        Adicionado em 30/08/2026 (diario_pesquisa.md, Entrada 14).
        """
        self._q_init = self._Q.copy()

    # ------------------------------------------------------------------
    # Acesso a Q-table (somente leitura para uso externo)
    # ------------------------------------------------------------------

    @property
    def Q(self) -> np.ndarray:
        """Q-table shape (N_STATES, N_OPERATORS). Retorna view (nao modifique)."""
        return self._Q

    @property
    def visits(self) -> np.ndarray:
        """Contador de visitas (state, action). Shape (N_STATES, N_OPERATORS)."""
        return self._visits

    @property
    def epsilon(self) -> float:
        """Valor atual de epsilon."""
        return self._epsilon

    @property
    def t(self) -> int:
        """Numero de passos executados desde o ultimo reset()."""
        return self._t

    # ------------------------------------------------------------------
    # Analise de politica (usada na Fase 5 -- XAI)
    # ------------------------------------------------------------------

    def best_action(self, state_id: int) -> int:
        """
        Retorna a acao greedy (argmax Q) para um estado, sem alterar epsilon.
        Usada para extrair a politica aprendida apos o treinamento.
        """
        return int(np.argmax(self._Q[state_id]))

    def policy_vector(self) -> np.ndarray:
        """
        Vetor de acoes greedy para todos os estados.
        shape: (N_STATES,), valores em {0, 1, 2, 3}.

        Usado na Fase 5 para plotar a politica aprendida em heatmaps.
        """
        return np.argmax(self._Q, axis=1).astype(np.int32)

    def q_values_for_action(self, action: int) -> np.ndarray:
        """
        Valores Q de um operador especifico para todos os estados.
        shape: (N_STATES,).

        Permite visualizar em heatmap a "preferencia" de cada operador
        ao longo do espaco de estados.
        """
        return self._Q[:, action].copy()

    def q_heatmap_data(self) -> dict:
        """
        Retorna dados estruturados para geracao de heatmaps (Fase 5).

        A politica e indexada por (f3_bin, f4_bin) para dois slices fixos:
            - inicio da busca: f1_bin=4 (alto gap), f2_bin=4 (alto desequilibrio)
            - fim da busca:    f1_bin=0 (gap baixo), f2_bin=0 (balanceado)

        Retorna
        -------
        dict com chaves:
            'policy_matrix' : ndarray (N_STATES,) -- acao greedy por estado
            'q_table'       : ndarray (N_STATES, N_OPERATORS) -- Q-table completa
            'visits'        : ndarray (N_STATES, N_OPERATORS) -- contagem de visitas
        """
        return {
            "policy_matrix": self.policy_vector(),
            "q_table": self._Q.copy(),
            "visits": self._visits.copy(),
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Salva a Q-table, contadores de visita e hiperparametros em formato NPZ.

        Arquivo gerado:
            <path>.npz com arrays: Q, visits
            <path>_meta.json com hiperparametros e estado do agente
        """
        path = Path(path)
        np.savez_compressed(
            path.with_suffix(".npz"),
            Q=self._Q,
            visits=self._visits,
        )
        meta = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_current": float(self._epsilon),
            "t": int(self._t),
            "n_states": N_STATES,
            "n_operators": N_OPERATORS,
        }
        meta_path = path.with_name(path.stem + "_meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        """
        Carrega um agente previamente salvo.

        Parametros
        ----------
        path : str | Path
            Caminho base (sem extensao). O metodo busca <path>.npz e
            <path>_meta.json.
        """
        path = Path(path)
        npz = np.load(path.with_suffix(".npz"))
        meta_path = path.with_name(path.stem + "_meta.json")
        meta = json.loads(meta_path.read_text())

        agent = cls(
            alpha=meta["alpha"],
            gamma=meta["gamma"],
            epsilon_start=meta["epsilon_start"],
            epsilon_min=meta["epsilon_min"],
            epsilon_decay=meta["epsilon_decay"],
        )
        agent._Q = npz["Q"]
        agent._visits = npz["visits"]
        agent._epsilon = meta["epsilon_current"]
        agent._t = meta["t"]
        return agent

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "Learnheuristica-QL"

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(alpha={self.alpha}, gamma={self.gamma}, "
            f"eps={self._epsilon:.4f}, t={self._t}, "
            f"Q_nonzero={int(np.count_nonzero(self._Q))})"
        )


class SarsaAgent(QLearningAgent):
    """
    Agente SARSA (on-policy TD control) para AOS no ILS.

    Adicionado em 31/08/2026 (PLANO_EXPERIMENTAL_EXTENSIVO.md, Secao 1;
    diario_pesquisa.md, Entrada 23) para contraste direto on-policy vs.
    off-policy com o QLearningAgent -- mesmo espaco de estados/acoes,
    mesma politica epsilon-greedy, mesma Q-table; a UNICA diferenca e a
    regra de atualizacao, que usa a acao REALMENTE executada a seguir
    (on-policy), nao o argmax (off-policy):

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]

    onde a' e a acao que o agente de fato vai executar no proximo
    passo -- diferente do Q-Learning, que usa max_a' Q(s', a')
    independente do que sera realmente escolhido a seguir.

    Detalhe de implementacao -- por que "espiar" a proxima acao:
    a interface BaseSelector.update() (pmsp/selectors.py) so recebe
    next_state_id, nao a proxima acao. Para um SARSA on-policy
    correto (nao "Expected SARSA" nem uma reamostragem inconsistente),
    a acao usada no alvo TD precisa ser EXATAMENTE a acao que sera de
    fato executada no proximo passo. A solucao: update() decide a'
    (via a mesma politica epsilon-greedy) e GUARDA essa decisao; a
    proxima chamada a select() para o MESMO estado reaproveita essa
    acao ja decidida em vez de sortear de novo. epsilon so decai
    dentro de select() (uma vez por passo real), nunca dentro do
    "espiar" de update() -- preserva o mesmo numero de decaimentos por
    episodio que o QLearningAgent.

    Referencia: Rummery, G. A., & Niranjan, M. (1994). "On-line
    Q-learning using connectionist systems." Technical Report
    CUED/F-INFENG/TR 166, Cambridge University. Ver tambem Sutton, R.
    S., & Barto, A. G. (2018). "Reinforcement Learning: An
    Introduction" (2a ed.), cap. 6, para o algoritmo SARSA classico.
    """

    def reset(self, q_init=None) -> None:
        super().reset(q_init=q_init)
        self._pending_state = None
        self._pending_action = None

    def _epsilon_greedy(self, state_id: int) -> int:
        """Mesma logica de decisao do QLearningAgent.select(), mas SEM
        avancar t/decair epsilon -- usada internamente por update()
        para decidir a' sem "gastar" um passo real (select() e quem
        de fato gasta um passo, ao usar essa acao)."""
        if self._rng.random() < self._epsilon:
            return int(self._rng.integers(self._n_actions))
        return int(np.argmax(self._Q[state_id]))

    def select(self, state_id: int, features: np.ndarray) -> int:
        self._t += 1
        if self._pending_state == state_id and self._pending_action is not None:
            action = self._pending_action
            self._pending_state = None
            self._pending_action = None
        else:
            # Primeiro passo do episodio: ainda nao ha decisao pendente
            # de um update() anterior (reset() zera o pendente).
            action = self._epsilon_greedy(state_id)
        self._epsilon = max(self.epsilon_min,
                            self._epsilon * self.epsilon_decay)
        return action

    def update(
        self,
        state_id: int,
        features: np.ndarray,
        action: int,
        reward: float,
        next_state_id: int,
    ) -> None:
        """
        Atualiza a Q-table pelo erro TD on-policy (SARSA).

        TD-error = r + gamma * Q(s', a') - Q(s, a)     [a' realmente executada]
        Q(s, a) <- Q(s, a) + alpha * TD-error
        """
        a_next = self._epsilon_greedy(next_state_id)
        td_error = (reward
                    + self.gamma * self._Q[next_state_id, a_next]
                    - self._Q[state_id, action])
        self._Q[state_id, action] += self.alpha * td_error
        self._visits[state_id, action] += 1
        self._pending_state = next_state_id
        self._pending_action = a_next

    def name(self) -> str:
        return "Learnheuristica-SARSA"

    def __repr__(self) -> str:
        return (
            f"SarsaAgent(alpha={self.alpha}, gamma={self.gamma}, "
            f"eps={self._epsilon:.4f}, t={self._t}, "
            f"Q_nonzero={int(np.count_nonzero(self._Q))})"
        )


# ===========================================================================
# ContextualUCBSelector -- terceira ablacao do ig_ql (Entrada 28)
# ===========================================================================

class ContextualUCBSelector(BaseSelector):
    """
    UCB1 condicionado por estado -- bandit contextual tabular.

    Desenhado para separar DUAS variaveis que o QLearningAgent combina:
    (a) condicionamento por estado (decidir com base em ONDE a busca esta
        -- gap relativo, CV de carga, progresso, estagnacao) e (b) aprendizado
    sequencial via TD (propagar valor entre estados com
    gamma * max_a' Q(s', a'), o "bootstrapping" do Q-Learning).

    Este seletor mantem uma tabela de contadores/recompensas por
    (estado, acao), no MESMO formato do Q-table do QLearningAgent -- ou
    seja, tem (a) -- mas a atualizacao e uma media simples por
    (estado, acao), sem gamma e sem propagar valor entre estados -- ou
    seja, NAO tem (b). A selecao usa UCB1 (mesma formula do UCBSelector),
    só que com as estatisticas do proprio estado, nao globais.

    Ver diario_pesquisa.md, Entrada 27 (ig_ucb, sem estado, ficou
    indistinguivel de ig_random) e Entrada 28 (este seletor): se
    ig_ucb_ctx chegar perto do ig_ql, a pequena vantagem do Q-Learning
    vem so do condicionamento por estado; se nao chegar, precisa mesmo
    do aprendizado sequencial via TD.
    """

    def __init__(self, C: float = math.sqrt(2), seed: int | None = None,
                 n_states: int = N_STATES, n_actions: int = N_OPERATORS) -> None:
        self.C = C
        self._rng = np.random.default_rng(seed)
        self._n_states = n_states
        self._n_actions = n_actions
        self.reset()

    def reset(self) -> None:
        self._counts = np.zeros((self._n_states, self._n_actions), dtype=np.int64)
        self._rewards = np.zeros((self._n_states, self._n_actions), dtype=np.float64)
        self._t = np.zeros(self._n_states, dtype=np.int64)

    def select(self, state_id: int, features: np.ndarray) -> int:
        self._t[state_id] += 1
        counts_s = self._counts[state_id]
        # Fase de inicializacao POR ESTADO: garante que cada acao seja
        # tentada ao menos uma vez antes de usar a formula UCB1 naquele
        # estado especifico (nao eh compartilhada entre estados).
        for k in range(self._n_actions):
            if counts_s[k] == 0:
                return k
        mu = self._rewards[state_id] / counts_s
        exploration = self.C * np.sqrt(np.log(self._t[state_id]) / counts_s)
        return int(np.argmax(mu + exploration))

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        # Sem TD-error, sem gamma, sem next_state_id -- media simples por
        # (estado, acao), de proposito, para isolar so o condicionamento
        # por estado.
        self._counts[state_id, action] += 1
        self._rewards[state_id, action] += reward

    def name(self) -> str:
        return "Contextual-UCB"

    def __repr__(self) -> str:
        visited = int(np.count_nonzero(self._counts.sum(axis=1)))
        return f"ContextualUCBSelector(C={self.C}, visited_states={visited}/{self._n_states})"


# ===========================================================================
# FlatQLearningSelector -- quarta celula do desenho fatorial 2x2 (Entrada 29)
# ===========================================================================

class FlatQLearningSelector(BaseSelector):
    """
    TD-learning (Q-Learning) SEM condicionamento por estado -- a celula
    que faltava no desenho fatorial 2x2 (estado x TD-learning) aberto nas
    Entradas 27-28:

        SEM estado  | ig_ucb        (sem TD)  | ig_ql_flat  (COM TD, esta classe)
        COM estado  | ig_ucb_ctx    (sem TD)  | ig_ql       (COM TD)

    Ignora completamente o `state_id`/`next_state_id` recebidos -- sempre
    delega a um QLearningAgent interno com n_states=1 (um unico estado
    agregado). Mantem TD-learning de verdade: bootstrap com
    gamma * max_a' Q(a'), epsilon-greedy com decaimento -- exatamente a
    mesma logica de aprendizado do ig_ql, so sem a dimensao "estado".

    Reaproveitar QLearningAgent internamente (em vez de reimplementar a
    logica TD) garante que a UNICA diferenca entre ig_ql_flat e ig_ql e'
    literalmente n_states=1 vs. n_states=N_STATES -- nao ha risco de uma
    reimplementacao divergir sutilmente da logica ja testada.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995, seed: int | None = None,
                 n_actions: int = N_OPERATORS) -> None:
        self._inner = QLearningAgent(
            alpha=alpha, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay, seed=seed,
            n_states=1, n_actions=n_actions,
        )

    def select(self, state_id: int, features: np.ndarray) -> int:
        return self._inner.select(0, features)

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        self._inner.update(0, features, action, reward, 0)

    def reset(self) -> None:
        self._inner.reset()

    def name(self) -> str:
        return "Flat-QLearning"

    def __repr__(self) -> str:
        return f"FlatQLearningSelector(inner={self._inner!r})"
