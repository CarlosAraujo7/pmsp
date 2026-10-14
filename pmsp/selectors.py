"""
pmsp/selectors.py
=================
Seletores de operadores (Adaptive Operator Selection - AOS).

Implementa a interface base e tres baselines para comparacao com
a Learnheuristica Q-Learning (Fase 3):

    RandomSelector  -- selecao uniforme aleatoria (baseline fraco)
    UCBSelector     -- Upper Confidence Bound (AOS-UCB, baseline forte)
    RFSelector      -- Random Forest online (baseline de ML supervisionado)

Interface
---------
Todos os seletores implementam:
    select(state_id, features) -> int   (escolhe operador 0..3)
    update(state_id, features, action, reward, next_state_id) -> None
    reset() -> None                     (reinicia para nova execucao)

Parametros comuns
-----------------
state_id  : int        indice do estado em [0, 624] (para Q-Learning)
features  : ndarray    vetor de 4 features continuas (para RF)
action    : int        operador escolhido em {0,1,2,3}
reward    : float      melhora no makespan (>= 0)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

N_OPERATORS = 4


# ===========================================================================
# Interface base
# ===========================================================================

class BaseSelector(ABC):
    """Interface que todos os seletores devem implementar."""

    @abstractmethod
    def select(self, state_id: int, features: np.ndarray) -> int:
        """Seleciona um operador dado o estado atual."""
        ...

    @abstractmethod
    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        """Atualiza o seletor com o feedback do passo anterior."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reinicia o estado interno (usado entre repeticoes)."""
        ...

    def name(self) -> str:
        return self.__class__.__name__


# ===========================================================================
# RandomSelector
# ===========================================================================

class RandomSelector(BaseSelector):
    """
    Selecao uniforme aleatoria de operadores.

    Baseline mais simples: ignora estado e historico, escolhe
    uniformemente entre os N_OPERATORS operadores disponíveis.
    Serve como limite inferior de desempenho do AOS.
    """

    def __init__(self, seed: int | None = None,
                 n_actions: int = N_OPERATORS) -> None:
        self._rng = np.random.default_rng(seed)
        # n_actions generalizado em 31/08/2026 (Entrada 26) no mesmo
        # espirito aditivo do n_actions de QLearningAgent (Entrada 23):
        # default preserva o comportamento original exato (N_OPERATORS=4);
        # necessario para o ig_random (3 estrategias de destruicao, ver
        # pmsp/ig.py), o controle de isolamento que testa se o ganho do
        # ig_ql vem do mecanismo de busca (destruir/reconstruir) em si ou
        # da selecao via Q-Learning entre estrategias.
        self._n_actions = n_actions

    def select(self, state_id: int, features: np.ndarray) -> int:
        return int(self._rng.integers(0, self._n_actions))

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        pass  # sem aprendizado

    def reset(self) -> None:
        pass  # sem estado a resetar

    def name(self) -> str:
        return "ILS-Random"


# ===========================================================================
# AlwaysOp1Selector
# ===========================================================================

class AlwaysOp1Selector(BaseSelector):
    """
    Seletor trivial: sempre escolhe o operador 0 (Op1, inter-machine swap).

    Adicionado em 29/08/2026 (Fase 1 do plano em ANALISE_E_PLANO_EJOR.md)
    para testar diretamente a hipotese levantada no rascunho do artigo: a
    politica aprendida pelo Q-Learning converge para Op1 em 87-96% dos
    estados em todas as classes de instancia -- este seletor testa se um
    ILS restrito so a Op1 (sem nenhuma selecao adaptativa) ja alcanca
    desempenho comparavel aos demais metodos, ou se os outros operadores
    contribuem de forma nao-obvia (por exemplo, via a fase de perturbacao,
    que sorteia entre os 4 operadores independente deste seletor).

    Nao aprende nada e nao tem estado interno alem do proprio nome --
    e deliberadamente o AOS mais simples possivel.
    """

    def select(self, state_id: int, features: np.ndarray) -> int:
        return 0

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        pass  # sem aprendizado

    def reset(self) -> None:
        pass  # sem estado a resetar

    def name(self) -> str:
        return "ILS-Op1Only"


# ===========================================================================
# AlwaysOp2Selector / AlwaysOp3Selector / AlwaysOp4Selector
# ===========================================================================

class AlwaysOp2Selector(BaseSelector):
    """
    Seletor trivial: sempre escolhe o operador 1 (Op2, reinsercao).

    Adicionado em 31/08/2026 (PLANO_EXPERIMENTAL_EXTENSIVO.md, Secao 1;
    diario_pesquisa.md, Entrada 23) para completar a bateria de
    isolamento -- ate agora so existia AlwaysOp1Selector.
    """

    def select(self, state_id: int, features: np.ndarray) -> int:
        return 1

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        pass  # sem aprendizado

    def reset(self) -> None:
        pass  # sem estado a resetar

    def name(self) -> str:
        return "ILS-Op2Only"


class AlwaysOp3Selector(BaseSelector):
    """
    Seletor trivial: sempre escolhe o operador 2 (Op3, troca intramaquina).

    Op3 NUNCA melhora o makespan por design (troca dentro da mesma
    maquina nao muda a soma de tempos de processamento dela) -- e o
    "controle negativo" da bateria de isolamento: espera-se que fique
    preso na solucao LPT inicial na fase de busca local, ja que
    apply_best nunca encontra uma melhoria por esse operador (a
    perturbacao do ILS, que sorteia entre os 4 operadores independente
    deste seletor, e o unico jeito de sol mudar de fato).

    Adicionado em 31/08/2026 -- ver diario_pesquisa.md, Entrada 23.
    """

    def select(self, state_id: int, features: np.ndarray) -> int:
        return 2

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        pass  # sem aprendizado

    def reset(self) -> None:
        pass  # sem estado a resetar

    def name(self) -> str:
        return "ILS-Op3Only"


class AlwaysOp4Selector(BaseSelector):
    """
    Seletor trivial: sempre escolhe o operador 3 (Op4, block-move).

    Adicionado em 31/08/2026 -- ver diario_pesquisa.md, Entrada 23.
    """

    def select(self, state_id: int, features: np.ndarray) -> int:
        return 3

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        pass  # sem aprendizado

    def reset(self) -> None:
        pass  # sem estado a resetar

    def name(self) -> str:
        return "ILS-Op4Only"


# ===========================================================================
# UCBSelector
# ===========================================================================

class UCBSelector(BaseSelector):
    """
    AOS baseado em Upper Confidence Bound (UCB1).

    Formula de selecao:
        k* = argmax_k [ mu_k  +  C * sqrt( ln(t) / n_k ) ]

    onde:
        mu_k  = recompensa media acumulada do operador k
        n_k   = numero de vezes que k foi selecionado
        t     = numero total de selecoes
        C     = constante de exploracao (default: sqrt(2))

    Os primeiros N_OPERATORS passos garantem que cada operador seja
    tentado pelo menos uma vez (fase de inicializacao).

    Referencia: Auer et al. (2002), "Finite-time Analysis of the
    Multiarmed Bandit Problem."
    """

    def __init__(self, C: float = math.sqrt(2),
                 seed: int | None = None,
                 n_actions: int = N_OPERATORS) -> None:
        """
        Parametros
        ----------
        C : float
            Constante de exploracao UCB (default: sqrt(2)).
        n_actions : int
            Tamanho do espaco de acoes (default: N_OPERATORS=4). Generalizado
            em 31/08/2026 (Entrada 27), mesmo padrao aditivo ja usado em
            QLearningAgent (Entrada 23) e RandomSelector (Entrada 26) --
            default preserva o comportamento original exato de ils_ucb;
            necessario para o ig_ucb (3 estrategias de destruicao do
            Iterated Greedy, ver pmsp/ig.py), que testa se um bandit simples
            (sem TD-learning) ja captura a pequena vantagem do ig_ql sobre
            o ig_random encontrada na Entrada 26.
        """
        self.C = C
        self._rng = np.random.default_rng(seed)
        self._n_actions = n_actions
        self.reset()

    def reset(self) -> None:
        self._counts = np.zeros(self._n_actions, dtype=np.int64)   # n_k
        self._rewards = np.zeros(self._n_actions, dtype=np.float64) # sum rewards
        self._t = 0

    def select(self, state_id: int, features: np.ndarray) -> int:
        self._t += 1
        # Fase de inicializacao: garante que cada operador seja testado
        for k in range(self._n_actions):
            if self._counts[k] == 0:
                return k
        # UCB1: argmax [ mu_k + C * sqrt(ln(t) / n_k) ]
        mu = self._rewards / self._counts
        exploration = self.C * np.sqrt(np.log(self._t) / self._counts)
        return int(np.argmax(mu + exploration))

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        self._counts[action] += 1
        self._rewards[action] += reward

    def name(self) -> str:
        return "ILS-UCB"

    @property
    def mean_rewards(self) -> np.ndarray:
        """Recompensa media por operador (para analise pos-execucao)."""
        with np.errstate(invalid="ignore"):
            return np.where(self._counts > 0,
                            self._rewards / self._counts, 0.0)


# ===========================================================================
# NormalisedUCBSelector (Entrada 98, 05/10/2026)
# ===========================================================================

class NormalisedUCBSelector(UCBSelector):
    """
    UCB1 com recompensa normalizada para [0, 1].

    O UCBSelector original usa a recompensa BRUTA (reducao de makespan em
    unidades de p_j), mas a analise de Auer et al. (2002) assume
    recompensas em [0, 1]: com recompensa bruta, a escala dos tempos de
    processamento decide se o bonus de exploracao domina (recompensas
    medias << 1: rodizio quase uniforme) ou some (recompensas >> 1:
    selecao gulosa) -- ver diario_pesquisa.md, Entrada 97.

    Aqui cada recompensa e' dividida pela MAIOR recompensa observada ate'
    o passo corrente (maxima corrente; recompensa 0 -> 0), o que a coloca
    em [0, 1] sem olhar a instancia. Nenhum parametro novo.
    UCBSelector permanece inalterado (todos os resultados ja publicados).
    """

    def reset(self) -> None:
        super().reset()
        self._rmax = 0.0

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        self._rmax = max(self._rmax, float(reward))
        r = float(reward) / self._rmax if self._rmax > 0 else 0.0
        super().update(state_id, features, action, r, next_state_id)

    def name(self) -> str:
        return "ILS-UCB-norm"


# ===========================================================================
# ProbabilityMatchingSelector (PM)
# ===========================================================================

class ProbabilityMatchingSelector(BaseSelector):
    """
    AOS classico: Probability Matching (Thierens, 2005; Fialho et al., 2010).

    Mantem uma estimativa de qualidade q_i por operador (media movel
    exponencial da recompensa), convertida em probabilidade de selecao
    proporcional, com piso p_min para garantir exploracao continua:

        q_i(t+1) = q_i(t) + alpha * (r(t) - q_i(t))      [se i foi usado em t]
        p_i = p_min + (1 - K*p_min) * q_i / sum_j(q_j)   [K = N_OPERATORS]

    Diferente do UCB (que usa um bonus de exploracao explicito baseado
    em contagem de visitas), o PM explora atraves de um piso p_min
    fixo -- toda acao mantem probabilidade >= p_min para sempre, mesmo
    com q_i baixo. Inicializacao com q_i iguais e positivos garante
    probabilidades iniciais uniformes (1/K), independente do valor
    absoluto escolhido.

    Parametros
    ----------
    alpha : float
        Taxa de adaptacao da media movel de qualidade (nao confundir
        com o alpha do Q-Learning). Valor tipico da literatura: 0.3.
    p_min : float
        Probabilidade minima por operador. Valor tipico: 0.1.
    seed : int | None

    Referencia: Fialho, A., Da Costa, L., Schoenauer, M., & Sebag, M.
    (2010). "Analyzing bandit-based adaptive operator selection
    strategies." Annals of Mathematics and Artificial Intelligence,
    60(1), 25-64.
    """

    def __init__(self, alpha: float = 0.3, p_min: float = 0.1,
                 seed: int | None = None) -> None:
        self.alpha = alpha
        self.p_min = p_min
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self._q = np.ones(N_OPERATORS, dtype=np.float64)

    def _probs(self) -> np.ndarray:
        total = self._q.sum()
        return self.p_min + (1.0 - N_OPERATORS * self.p_min) * (self._q / total)

    def select(self, state_id: int, features: np.ndarray) -> int:
        return int(self._rng.choice(N_OPERATORS, p=self._probs()))

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        self._q[action] += self.alpha * (reward - self._q[action])

    def name(self) -> str:
        return "ILS-PM"

    @property
    def probs(self) -> np.ndarray:
        """Probabilidades de selecao atuais (para analise pos-execucao)."""
        return self._probs()


# ===========================================================================
# AdaptivePursuitSelector (AP)
# ===========================================================================

class AdaptivePursuitSelector(BaseSelector):
    """
    AOS classico: Adaptive Pursuit (Thierens, 2005).

    Variante do Probability Matching que "persegue" o operador de
    maior qualidade estimada de forma mais agressiva: em vez de
    recalcular p_i do zero a cada passo (PM), a probabilidade do
    operador vencedor e empurrada diretamente em direcao a p_max, e as
    demais em direcao a p_min, a uma taxa beta:

        q_i(t+1)   = q_i(t) + alpha * (r(t) - q_i(t))        [mesma EMA do PM]
        i*         = argmax_i q_i(t+1)
        p_i*(t+1)  = p_i*(t) + beta * (p_max - p_i*(t))
        p_i(t+1)   = p_i(t)  + beta * (p_min - p_i(t))       para i != i*

    onde p_max = 1 - (K-1)*p_min garante soma 1 (K=N_OPERATORS).

    Implementacao vetorizada equivalente (usada abaixo): move-se PRIMEIRO
    toda a distribuicao em direcao a p_min, depois soma-se
    beta*(p_max - p_min) so no vencedor -- algebricamente identico as
    duas equacoes acima (verificado: para o vencedor,
    p(1-beta)+beta*p_min + beta*(p_max-p_min) = p(1-beta)+beta*p_max).
    Uma renormalizacao defensiva ao final de update() protege contra
    deriva de ponto flutuante em execucoes muito longas (>>1000 passos).

    Parametros
    ----------
    alpha : float
        Taxa de adaptacao da media movel de qualidade. Default 0.3.
    beta : float
        Taxa de "perseguicao" das probabilidades. Default 0.3.
    p_min : float
        Probabilidade minima por operador. Default 0.1.
    seed : int | None

    Referencia: Thierens, D. (2005). "An adaptive pursuit strategy for
    allocating operator probabilities." Proceedings of GECCO 2005. Ver
    tambem Fialho et al. (2010) para o uso conjunto com PM/UCB como
    trio classico de AOS.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3,
                 p_min: float = 0.1, seed: int | None = None) -> None:
        self.alpha = alpha
        self.beta = beta
        self.p_min = p_min
        self.p_max = 1.0 - (N_OPERATORS - 1) * p_min
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self._q = np.ones(N_OPERATORS, dtype=np.float64)
        self._p = np.full(N_OPERATORS, 1.0 / N_OPERATORS, dtype=np.float64)

    def select(self, state_id: int, features: np.ndarray) -> int:
        return int(self._rng.choice(N_OPERATORS, p=self._p))

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        self._q[action] += self.alpha * (reward - self._q[action])
        i_star = int(np.argmax(self._q))
        self._p += self.beta * (self.p_min - self._p)
        self._p[i_star] += self.beta * (self.p_max - self.p_min)
        self._p /= self._p.sum()  # seguranca numerica (soma deveria ja ser 1)

    def name(self) -> str:
        return "ILS-AP"

    @property
    def probs(self) -> np.ndarray:
        return self._p.copy()


# ===========================================================================
# RFSelector
# ===========================================================================

class RFSelector(BaseSelector):
    """
    Seletor baseado em Random Forest online com janela deslizante.

    Funciona como um classificador multi-classe que mapeia
    features de estado -> operador com maior recompensa esperada.

    Mecanismo de aprendizado online:
        1. Acumula experiencias (features, action, reward) num buffer.
        2. A cada `refit_interval` passos, retreina o RF com o buffer
           ponderado pela recompensa (amostra com reward maior recebe
           maior peso).
        3. Durante a fase pre-treino (buffer pequeno), usa exploracao
           epsilon-greedy pura.

    Parametros
    ----------
    n_estimators : int
        Numero de arvores no RF.
    refit_interval : int
        Frequencia de retreinamento (em numero de passos).
    window_size : int
        Tamanho maximo do buffer de experiencias.
    epsilon : float
        Taxa de exploracao epsilon-greedy (decai com o tempo).
    min_samples_fit : int
        Minimo de amostras por classe para iniciar o treino.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        refit_interval: int = 15,
        window_size: int = 300,
        epsilon: float = 0.20,
        min_samples_fit: int = 4,
        seed: int | None = None,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn e necessario para RFSelector. "
                "Instale com: pip install scikit-learn"
            )
        self._n_est = n_estimators
        self._refit_interval = refit_interval
        self._window_size = window_size
        self._epsilon = epsilon
        self._min_samples_fit = min_samples_fit
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self._model = RandomForestClassifier(
            n_estimators=self._n_est,
            random_state=0,
            n_jobs=1,
        )
        # Buffer: deque de (features: ndarray, action: int, reward: float)
        self._buffer: Deque[Tuple[np.ndarray, int, float]] = deque(
            maxlen=self._window_size
        )
        self._fitted = False
        self._t = 0

    def select(self, state_id: int, features: np.ndarray) -> int:
        self._t += 1
        # Exploracao epsilon-greedy
        if not self._fitted or self._rng.random() < self._epsilon:
            return int(self._rng.integers(0, N_OPERATORS))
        # Explotacao: usa o RF para prever o melhor operador
        feat = np.asarray(features, dtype=np.float64).reshape(1, -1)
        return int(self._model.predict(feat)[0])

    def update(self, state_id: int, features: np.ndarray,
               action: int, reward: float, next_state_id: int) -> None:
        self._buffer.append((np.asarray(features, dtype=np.float64),
                             action, float(reward)))
        # Retreina a cada refit_interval passos
        if self._t % self._refit_interval == 0:
            self._refit()

    def _refit(self) -> None:
        """Retreina o RF com o buffer atual ponderado pela recompensa."""
        if len(self._buffer) < self._min_samples_fit:
            return
        # Garante que cada classe apareca pelo menos uma vez
        actions_seen = {exp[1] for exp in self._buffer}
        if len(actions_seen) < N_OPERATORS:
            return  # ainda nao ha diversidade suficiente para treinar

        X = np.array([exp[0] for exp in self._buffer])
        y = np.array([exp[1] for exp in self._buffer], dtype=np.int32)
        # Pesos: reward + offset para evitar peso zero
        w = np.array([max(exp[2], 0.0) + 1e-3 for exp in self._buffer])
        w /= w.sum()   # normaliza

        self._model.fit(X, y, sample_weight=w)
        self._fitted = True

    def name(self) -> str:
        return "ILS-RF"

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ===========================================================================
# Factory
# ===========================================================================

def make_selector(name: str, seed: int | None = None) -> BaseSelector:
    """
    Fabrica de seletores pelo nome.

    Parametros
    ----------
    name : str
        Um de: "random", "ucb", "rf"
    seed : int | None

    Retorna
    -------
    BaseSelector
    """
    name = name.lower()
    if name == "random":
        return RandomSelector(seed=seed)
    if name == "ucb":
        return UCBSelector(seed=seed)
    if name == "rf":
        return RFSelector(seed=seed)
    if name == "pm":
        return ProbabilityMatchingSelector(seed=seed)
    if name == "ap":
        return AdaptivePursuitSelector(seed=seed)
    raise ValueError(
        f"Seletor desconhecido: {name!r}. "
        "Use 'random', 'ucb', 'rf', 'pm' ou 'ap'."
    )
