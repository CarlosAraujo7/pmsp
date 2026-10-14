"""
pmsp/bounds.py
==============
Funções puras de cálculo de bounds e das quatro features do espaço de
estados do agente Q-Learning.

Espaço de estados S (Fase 3)
-----------------------------
O agente observa um vetor de 4 features contínuas, cada uma discretizada
em N bins iguais, gerando |S| = N^4 estados possíveis. O desenho
padrão usado por todos os métodos já publicados é N=3, |S|=81
(ver _BINS_F1..F4 abaixo, limiares calibrados manualmente):

    f1 — Gap relativo:      (C_max - LB) / LB         ∈ [0, +∞)
    f2 — Coef. variação:    std(loads) / mean(loads)   ∈ [0, +∞)
    f3 — Progresso:         iter / max_iter            ∈ [0, 1]
    f4 — Estagnação:        iters_no_improve / 50      ∈ [0, +∞)

A discretização mapeia cada feature para um bin inteiro em {0,1,2,3,4}
usando limiares definidos por faixas calibradas empiricamente e descritas
no artigo.

Correção de documentação (30/08/2026, diario_pesquisa.md Entrada 16):
este parágrafo antes descrevia incorretamente N=5 bins / 625 estados,
defasado de uma versão anterior do desenho. O código (_BINS_F1..F4,
compute_state) sempre implementou N=3/81 nesta versão do projeto --
nenhum resultado publicado foi afetado, só o comentário estava errado.

Estudo de sensibilidade de granularidade (Entrada 16): compute_state_cfg()
generaliza a discretização para N configurável (testado N=2/16 estados,
N=3/81, N=4/256), usando limiares de quantis empíricos (ver
experiment/harvest_feature_distribution.py) em vez dos limiares manuais
_BINS_F1..F4. compute_state() original permanece inalterado e continua
sendo o caminho usado por padrão em ils_run() e por todos os métodos já
publicados (ils_ql, ils_ql_ws, ils_ql_hi, ils_rand_hi) -- compute_state_cfg()
só é usada quando ils_run() recebe o parâmetro state_cfg explicitamente.

Este módulo é intencionalmente sem estado (apenas funções puras) para
facilitar uso em ambientes paralelos (30 repetições simultâneas).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .instance import PMSPInstance
from .solution import PMSPSolution


# ===========================================================================
# Lower Bound
# ===========================================================================

def lower_bound(instance: PMSPInstance) -> int:
    """
    Retorna o lower bound clássico do PMSP:

        LB = max( max_j p_j ,  ceil( sum_j p_j / m ) )

    Este valor é cacheado em PMSPInstance.lower_bound, mas a função
    standalone é mantida aqui para uso em scripts independentes.
    """
    return instance.lower_bound


# ===========================================================================
# Features individuais (valores contínuos)
# ===========================================================================

def f1_relative_gap(solution: PMSPSolution) -> float:
    """
    f1: Gap relativo entre o makespan atual e o lower bound.

        f1 = (C_max - LB) / LB

    Interpreta: 0.0 = ótimo teórico atingido; 1.0 = 100% acima do LB.
    """
    return solution.relative_gap


def f2_load_cv(solution: PMSPSolution) -> float:
    """
    f2: Coeficiente de variação das cargas das máquinas.

        f2 = std(loads) / mean(loads)

    Interpreta: 0.0 = máquinas perfeitamente balanceadas;
                valores altos = desequilíbrio severo.
    """
    return solution.load_cv


def f3_progress(current_iter: int, max_iter: int) -> float:
    """
    f3: Progresso normalizado da busca.

        f3 = current_iter / max_iter  ∈ [0, 1]

    Permite ao agente ajustar a estratégia conforme o orçamento restante.
    """
    if max_iter <= 0:
        return 1.0
    return min(current_iter / max_iter, 1.0)


def f4_stagnation(iters_no_improve: int, window: int = 50) -> float:
    """
    f4: Normalização do número de iterações sem melhora.

        f4 = iters_no_improve / window

    Saturado em 1.0 para não crescer indefinidamente.
    Permite ao agente detectar estagnação e aumentar a diversificação.

    Parâmetros
    ----------
    iters_no_improve : int
        Iterações consecutivas sem melhora no makespan.
    window : int
        Janela de referência (padrão: 50 iterações, conforme definido
        no artigo).
    """
    return min(iters_no_improve / window, 1.0)


# ===========================================================================
# Discretização → estado inteiro
# ===========================================================================

# Limiares para cada feature (2 limites definem 3 bins: baixo/médio/alto).
# 3 bins por feature → 3^4 = 81 estados totais.
# Cobertura ~6× melhor que 5 bins (81 vs 625 estados) com o mesmo orçamento.
_BINS_F1 = [0.10, 0.35]              # gap relativo: baixo / médio / alto
_BINS_F2 = [0.15, 0.40]              # coef. variação carga: balanceado / médio / desbalanceado
_BINS_F3 = [0.33, 0.67]              # progresso: início / meio / fim
_BINS_F4 = [0.25, 0.60]              # estagnação: baixa / média / alta


def _digitize(value: float, thresholds: list[float]) -> int:
    """
    Mapeia um valor contínuo para um bin inteiro em {0, 1, 2, 3, 4}
    usando 4 limiares (np.digitize retorna índice em [0, len(bins)]).

    Exemplos com _BINS_F1 = [0.10, 0.35]:
        value=0.05 → bin 0   (< 0.10)   baixo gap
        value=0.20 → bin 1   (0.10 ≤ v < 0.35)  médio gap
        value=0.50 → bin 2   (≥ 0.35)   alto gap
    """
    return int(np.digitize(value, thresholds))   # resultado em {0..4}


def compute_state(
    solution: PMSPSolution,
    current_iter: int,
    max_iter: int,
    iters_no_improve: int,
    stagnation_window: int = 50,
) -> int:
    """
    Computa o estado inteiro s ∈ {0, …, 80} para o agente Q-Learning.

    Codificação posicional em base 3:
        s = b1 * 3^3  +  b2 * 3^2  +  b3 * 3^1  +  b4 * 3^0

    onde b_i ∈ {0,1,2} é o bin da feature f_i (3 bins: baixo/médio/alto).

    Parâmetros
    ----------
    solution : PMSPSolution
        Solução atual (de onde extraímos f1 e f2).
    current_iter : int
        Iteração atual do ILS (de onde extraímos f3).
    max_iter : int
        Orçamento total de iterações.
    iters_no_improve : int
        Contador de iterações sem melhora (para f4).
    stagnation_window : int
        Janela de normalização da estagnação (padrão 50).

    Retorna
    -------
    int
        Índice do estado em [0, 80].
    """
    b1 = _digitize(f1_relative_gap(solution),  _BINS_F1)
    b2 = _digitize(f2_load_cv(solution),        _BINS_F2)
    b3 = _digitize(f3_progress(current_iter, max_iter), _BINS_F3)
    b4 = _digitize(f4_stagnation(iters_no_improve, stagnation_window), _BINS_F4)

    return b1 * 27 + b2 * 9 + b3 * 3 + b4   # base-3 codificado (3^4=81 estados)


def decode_state(state_id: int) -> Tuple[int, int, int, int]:
    """
    Decodifica o índice de estado de volta para os 4 bins individuais.
    Útil para interpretação e geração de heatmaps (Fase 5).

    Retorna
    -------
    (b1, b2, b3, b4) : tupla de inteiros em {0..2}
    """
    b1 = state_id // 27
    rem = state_id % 27
    b2 = rem // 9
    rem = rem % 9
    b3 = rem // 3
    b4 = rem % 3
    return b1, b2, b3, b4


# ===========================================================================
# Métricas de avaliação de resultado (usadas no motor experimental, Fase 4)
# ===========================================================================

def gap_percent(makespan: int, lb: int) -> float:
    """
    GAP percentual entre o makespan obtido e o lower bound:

        GAP(%) = 100 * (C_max - LB) / LB

    Quanto menor, mais próxima do ótimo está a solução.
    """
    return 100.0 * (makespan - lb) / lb if lb > 0 else 0.0


# ===========================================================================
# Feature vector (companion to compute_state, used by RF selector and logs)
# ===========================================================================

def compute_features(
    solution,
    current_iter: int,
    max_iter: int,
    iters_no_improve: int,
    stagnation_window: int = 50,
) -> "np.ndarray":
    """
    Returns the 4 continuous features as a NumPy array.

    Shape: (4,)  --  [f1, f2, f3, f4]

    Companion to compute_state(): while compute_state() returns the
    discretized integer index for the Q-table, compute_features()
    returns the raw float values needed by the RF selector and for
    logging / visualization.

    Parameters
    ----------
    solution : PMSPSolution
    current_iter, max_iter, iters_no_improve, stagnation_window
        Same semantics as compute_state().
    """
    return np.array([
        f1_relative_gap(solution),
        f2_load_cv(solution),
        f3_progress(current_iter, max_iter),
        f4_stagnation(iters_no_improve, stagnation_window),
    ], dtype=np.float64)


# ===========================================================================
# Configuração de estado generalizada (Entrada 16, diario_pesquisa.md --
# estudo de sensibilidade de granularidade: 81 vs. 16 vs. 256 estados)
# ===========================================================================

from typing import NamedTuple as _NamedTuple


class StateConfig(_NamedTuple):
    """
    Agrupa os limiares de discretização das 4 features para uma dada
    granularidade. As 4 features usam o mesmo número de bins
    (len(thresholds) + 1 em cada lista -- assumido igual nas 4).

    Usada por compute_state_cfg() para generalizar compute_state() a um
    número de bins configurável. compute_state() (acima) permanece com
    os limiares fixos originais (_BINS_F1..F4, 3 bins) e NÃO usa esta
    estrutura -- é o caminho inalterado, usado por todos os métodos já
    publicados. Ver diario_pesquisa.md, Entrada 16.
    """
    bins_f1: list
    bins_f2: list
    bins_f3: list
    bins_f4: list

    @property
    def n_bins(self) -> int:
        return len(self.bins_f1) + 1

    @property
    def n_states(self) -> int:
        return self.n_bins ** 4


# Config padrão: reproduz exatamente compute_state() (3 bins, 81 estados,
# limiares manuais _BINS_F1..F4). Fornecida para uso explícito com
# compute_state_cfg() quando conveniente (ex.: testes de equivalência);
# NÃO é usada internamente por compute_state().
DEFAULT_STATE_CONFIG = StateConfig(
    bins_f1=list(_BINS_F1), bins_f2=list(_BINS_F2),
    bins_f3=list(_BINS_F3), bins_f4=list(_BINS_F4),
)


def compute_state_cfg(
    solution: PMSPSolution,
    current_iter: int,
    max_iter: int,
    iters_no_improve: int,
    stagnation_window: int,
    cfg: "StateConfig",
) -> int:
    """
    Generalização de compute_state() para uma StateConfig com número de
    bins configurável (igual nas 4 features).

    Codificação posicional em base n_bins:
        s = b1*n_bins^3 + b2*n_bins^2 + b3*n_bins + b4

    Para cfg=DEFAULT_STATE_CONFIG, produz exatamente os mesmos valores
    que compute_state() (verificado por execução real, Entrada 16).
    Ver compute_state() para a versão original (mais rápida, sem
    overhead de NamedTuple) -- usada por padrão em ils_run() quando
    state_cfg=None.
    """
    n = cfg.n_bins
    b1 = _digitize(f1_relative_gap(solution), cfg.bins_f1)
    b2 = _digitize(f2_load_cv(solution), cfg.bins_f2)
    b3 = _digitize(f3_progress(current_iter, max_iter), cfg.bins_f3)
    b4 = _digitize(f4_stagnation(iters_no_improve, stagnation_window), cfg.bins_f4)
    return b1 * n**3 + b2 * n**2 + b3 * n + b4


def decode_state_cfg(state_id: int, cfg: "StateConfig") -> Tuple[int, int, int, int]:
    """Generalização de decode_state() para uma StateConfig arbitrária."""
    n = cfg.n_bins
    b1 = state_id // n**3
    rem = state_id % n**3
    b2 = rem // n**2
    rem = rem % n**2
    b3 = rem // n
    b4 = rem % n
    return b1, b2, b3, b4
