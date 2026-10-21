"""
pmsp/ils.py
===========
Motor ILS (Iterated Local Search) com selecao adaptativa de operadores.

Estrutura do algoritmo
----------------------

    s <- LPT(instance)                     # solucao inicial
    s_best <- s.copy()
    iters_no_improve <- 0

    for t in 0..max_iter-1:
        # 1. Observa estado
        state <- compute_state(s, t, max_iter, iters_no_improve)
        feats <- compute_features(s, t, max_iter, iters_no_improve)

        # 2. Selecao do operador (Q-Learning / UCB / Random / RF)
        a <- selector.select(state, feats)

        # 3. Aplica operador best-improvement sobre copia
        s_new <- s.copy()
        delta <- apply_best(a, s_new, rng)
        reward <- delta                    # melhora no makespan

        # 4. Atualiza seletor (Q-table / contadores / buffer RF)
        state_new <- compute_state(s_new, t+1, max_iter, iters_no_improve)
        selector.update(state, feats, a, reward, state_new)

        # 5. Criterio de aceitacao (greedy: aceita se nao piora)
        if s_new.makespan <= s.makespan:
            s <- s_new
            if s.makespan < s_best.makespan:
                s_best <- s.copy()
                iters_no_improve <- 0
            else:
                iters_no_improve += 1
        else:
            iters_no_improve += 1

        # 6. Perturbacao ILS quando estagnado
        if iters_no_improve >= perturbation_threshold:
            s <- s_best.copy()
            for _ in range(n_perturb):
                apply_rand(rng.integers(4), s, rng)
            iters_no_improve <- 0

    return metricas de s_best
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from .instance import PMSPInstance
from .solution import PMSPSolution
from .constructive import lpt
from .operators import apply_best, apply_rand, N_OPERATORS
from .selectors import BaseSelector
from .bounds import compute_state, compute_features, gap_percent, compute_state_cfg


# ---------------------------------------------------------------------------
# Funcao principal: uma rodada do ILS
# ---------------------------------------------------------------------------

def ils_run(
    instance: PMSPInstance,
    selector: BaseSelector,
    max_iter: int = 500,
    perturbation_threshold: int = 50,
    n_perturb: int = 3,
    seed: int = 0,
    stagnation_window: int = 50,
    track_history: bool = False,
    state_cfg=None,
    initial_solution=None,
    best_fn=None,
) -> Dict[str, Any]:
    """
    Executa uma rodada do ILS com o seletor de operadores fornecido.

    Parametros
    ----------
    instance : PMSPInstance
        Instancia do PMSP a resolver.
    selector : BaseSelector
        Seletor de operadores (QLearningAgent, UCBSelector, etc.).
        O metodo reset() e chamado automaticamente no inicio.
    max_iter : int
        Numero maximo de iteracoes do loop principal.
    perturbation_threshold : int
        Numero de iteracoes sem melhora antes de perturbar.
    n_perturb : int
        Numero de movimentos aleatorios na perturbacao.
    seed : int
        Semente para o gerador de numeros aleatorios interno.
    stagnation_window : int
        Janela para normalizacao de f4 (estagnacao).
    track_history : bool
        Se True, registra o makespan do best a cada iteracao.
        Util para graficos de convergencia.
    state_cfg : pmsp.bounds.StateConfig | None
        Se None (padrao), usa compute_state() / limiares fixos originais
        (3 bins, 81 estados) -- comportamento identico ao de antes desta
        opcao existir (Entrada 16, diario_pesquisa.md). Se fornecido, usa
        compute_state_cfg() com os limiares desta config (estudo de
        sensibilidade de granularidade).

    Retorna
    -------
    dict com chaves:
        'makespan'      : int    -- melhor makespan encontrado
        'gap_pct'       : float  -- GAP% = 100*(makespan-LB)/LB
        'cpu_time'      : float  -- tempo de CPU em segundos
        'n_improvements': int    -- numero de melhorias encontradas
        'history'       : list   -- [makespan_t] se track_history=True, else []
    """
    rng = np.random.default_rng(seed)
    selector.reset()

    # --- Fase construtiva ---
    # Entrada 114: `initial_solution` opcional (None = LPT, comportamento original).
    sol = lpt(instance) if initial_solution is None else initial_solution.copy()
    s_best = sol.copy()
    best_makespan = sol.makespan
    iters_no_improve = 0
    n_improvements = 0

    history: List[int] = []
    t_start = time.perf_counter()

    # --- Loop principal ILS ---
    for t in range(max_iter):

        # 1. Observa estado atual
        if state_cfg is None:
            state_id = compute_state(
                sol, t, max_iter, iters_no_improve, stagnation_window)
        else:
            state_id = compute_state_cfg(
                sol, t, max_iter, iters_no_improve, stagnation_window, state_cfg)
        feats = compute_features(
            sol, t, max_iter, iters_no_improve, stagnation_window)

        # 2. Selecao do operador
        action = selector.select(state_id, feats)

        # 3. Aplica best-improvement sobre copia
        sol_new = sol.copy()
        delta = (apply_best if best_fn is None else best_fn)(action, sol_new, rng)   # Entrada 162: best_fn opcional (operadores vetorizados); None = comportamento anterior
        reward = float(delta)

        # 4. Proximo estado e atualizacao do seletor
        if state_cfg is None:
            next_state_id = compute_state(
                sol_new, t + 1, max_iter, iters_no_improve, stagnation_window)
        else:
            next_state_id = compute_state_cfg(
                sol_new, t + 1, max_iter, iters_no_improve, stagnation_window, state_cfg)
        selector.update(state_id, feats, action, reward, next_state_id)

        # 5. Criterio de aceitacao (greedy: nao piora)
        if sol_new.makespan <= sol.makespan:
            sol = sol_new
            if sol.makespan < best_makespan:
                best_makespan = sol.makespan
                s_best = sol.copy()
                n_improvements += 1
                iters_no_improve = 0
            else:
                iters_no_improve += 1
        else:
            iters_no_improve += 1

        # 6. Perturbacao ILS quando estagnado
        if iters_no_improve >= perturbation_threshold:
            sol = s_best.copy()
            for _ in range(n_perturb):
                rand_op = int(rng.integers(N_OPERATORS))
                apply_rand(rand_op, sol, rng)
            iters_no_improve = 0

        if track_history:
            history.append(best_makespan)

    cpu_time = time.perf_counter() - t_start
    lb = instance.lower_bound

    return {
        "makespan": best_makespan,
        "gap_pct": gap_percent(best_makespan, lb),
        "cpu_time": cpu_time,
        "n_improvements": n_improvements,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Baseline puro: LPT sem busca local
# ---------------------------------------------------------------------------

def lpt_run(instance: PMSPInstance) -> Dict[str, Any]:
    """
    Avalia o LPT como baseline puro (sem ILS).
    Usado como ponto de referencia e para calcular o speedup do ILS.
    """
    t_start = time.perf_counter()
    sol = lpt(instance)
    cpu_time = time.perf_counter() - t_start
    lb = instance.lower_bound
    return {
        "makespan": sol.makespan,
        "gap_pct": gap_percent(sol.makespan, lb),
        "cpu_time": cpu_time,
        "n_improvements": 0,
        "history": [],
    }
