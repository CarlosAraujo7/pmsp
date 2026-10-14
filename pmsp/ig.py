"""
pmsp/ig.py
==========
Iterated Greedy (IG) com selecao adaptativa de ESTRATEGIAS DE DESTRUICAO
via o mesmo agente de Q-Learning tabular da Learnheuristica ILS+RL --
metodo `ig_ql` (PLANO_EXPERIMENTAL_EXTENSIVO.md, Secao 1; diario_pesquisa.md,
Entrada 23).

Adaptado de Missaoui & O'Sullivan (2026), "Q-learning enhanced Iterated
Greedy" para escalonamento de maquinas paralelas. Diferenca em relacao
ao ILS ja existente (pmsp/ils.py): em vez de escolher entre 4
OPERADORES DE VIZINHANCA (Op1..Op4) e aplicar o melhor movimento local
a cada passo, o IG escolhe entre ESTRATEGIAS DE DESTRUICAO (remover um
subconjunto de d tarefas do cronograma) e reconstroi de forma gulosa
(estilo NEH: reinsere as tarefas removidas, uma de cada vez, em ordem
decrescente de tempo de processamento, cada uma na maquina de menor
carga no momento -- a mesma regra gulosa do LPT/Graham, aplicada so as
tarefas removidas). O passo de destruir+reconstruir substitui TANTO o
operador de vizinhanca QUANTO a perturbacao do ILS -- o IG classico nao
tem uma fase de perturbacao separada, o proprio ciclo destroi/reconstroi
a cada iteracao ja fornece busca e diversificacao.

Estrategias de destruicao (espaco de acoes do agente, |A|=3):
    0 -- remocao aleatoria uniforme de d tarefas
    1 -- remocao das d tarefas de maior tempo de processamento
         ("mais caras" -- mais dificeis de realocar bem, forcando o
         reconstrutor guloso a repensar as decisoes mais impactantes)
    2 -- remocao concentrada: escolhe 1 maquina aleatoria e remove ate
         d tarefas dela (completando de outras maquinas se ela tiver
         menos de d tarefas) -- redistribui a carga de uma maquina
         especifica, diferente das outras duas estrategias, que nao
         focam em nenhuma maquina em particular

Tamanho de destruicao d: por padrao PROPORCIONAL ao tamanho da
instancia (max(2, round(0.3*n))), nao um valor absoluto fixo -- decisao
deliberada a luz do achado da Entrada 20-22 (diario_pesquisa.md): um
tamanho de perturbacao/destruicao FIXO nao escala para instancias
grandes (150+ tarefas). Um usuario pode passar d explicitamente para
replicar um valor fixo se desejar comparar.

O valor 0.3 (30% das tarefas) e um default RAZOAVEL informado por um
diagnostico rapido (Entrada 23: testado d em {10%,30%,50%,70%} x 5
seeds, frac=0.3 deu a menor gap media e n_improvements > 0 de forma
consistente), nao um hiperparametro exaustivamente otimizado -- mesmo
espirito dos defaults de UCB_C=sqrt(2) etc. em experiment/config.py.
IMPORTANTE (achado de depuracao, Entrada 23): a reconstrucao guloso
precisa "zerar" a contribuicao das tarefas removidas ANTES de decidir
onde reinserir a primeira -- fazer isso incrementalmente (mover uma
tarefa de cada vez sem primeiro descontar as demais que ainda serao
movidas) produzia reconstrucoes sistematicamente PIORES que o LPT,
mesmo destruindo a instancia inteira (d=n deveria reproduzir o LPT
exatamente, e so passou a fazer isso apos o conserto -- ver
apply_destroy_reconstruct e o teste test_d_equals_n_reproduces_lpt em
tests/test_ig.py).

Convencao de recompensa: max(delta, 0.0) -- igual ao ILS (r = melhora
no makespan, r>=0), mesmo quando o passo bruto de destruir+reconstruir
PIORA a solucao (o que pode acontecer aqui, ao contrario do
apply_best() do ILS, que por construcao nunca piora). A decisao de
ACEITAR ou nao a solucao reconstruida usa o delta bruto (possivelmente
negativo); so o SINAL DE APRENDIZADO fica truncado em 0, preservando a
mesma convencao "recompensa = melhora, nunca negativa" usada em todo o
resto do projeto (ver docstring de QLearningAgent.update).

Criterio de aceitacao: igual ao ILS (aceita se nao piora: makespan_novo
<= makespan_atual) -- escolha deliberada para isolar a diferenca entre
frameworks (ILS vs. IG) exclusivamente no passo de busca em si
(operador de vizinhanca vs. destruir+reconstruir), sem confundir com
um criterio de aceitacao diferente.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .instance import PMSPInstance
from .solution import PMSPSolution
from .constructive import lpt
from .selectors import BaseSelector
from .bounds import compute_state, compute_features, gap_percent, compute_state_cfg

N_DESTROY_STRATEGIES = 3


# ---------------------------------------------------------------------------
# Estrategias de destruicao: cada uma retorna a LISTA de indices de
# tarefas escolhidas para remocao (a remocao em si acontece na
# reconstrucao, via move_job -- ver apply_destroy_reconstruct).
# ---------------------------------------------------------------------------

def _destroy_random(sol: PMSPSolution, d: int, rng: np.random.Generator) -> List[int]:
    n = sol.instance.n
    return list(rng.choice(n, size=min(d, n), replace=False))


def _destroy_most_expensive(sol: PMSPSolution, d: int, rng: np.random.Generator) -> List[int]:
    p = sol.instance.p
    order = np.argsort(p)[::-1]  # decrescente
    return list(order[:d])


def _destroy_machine_concentrated(sol: PMSPSolution, d: int, rng: np.random.Generator) -> List[int]:
    n, m = sol.instance.n, sol.instance.m
    machine = int(rng.integers(m))
    jobs_on_machine = np.where(sol.assignment == machine)[0]
    if len(jobs_on_machine) >= d:
        return list(rng.choice(jobs_on_machine, size=d, replace=False))
    other = np.setdiff1d(np.arange(n), jobs_on_machine)
    n_extra = min(d - len(jobs_on_machine), len(other))
    extra = rng.choice(other, size=n_extra, replace=False) if n_extra > 0 else np.array([], dtype=int)
    return list(jobs_on_machine) + list(extra)


_DESTROY_STRATEGIES = [_destroy_random, _destroy_most_expensive, _destroy_machine_concentrated]

DESTROY_STRATEGY_NAMES = [
    "Random",
    "MostExpensive",
    "MachineConcentrated",
]


def apply_destroy_reconstruct(
    strategy_id: int,
    sol: PMSPSolution,
    d: int,
    rng: np.random.Generator,
) -> Tuple[PMSPSolution, int]:
    """
    Um ciclo de Iterated Greedy: escolhe d tarefas para remover segundo
    `strategy_id` (0/1/2 -- ver _DESTROY_STRATEGIES), depois as
    reinsere de forma gulosa (estilo NEH/LPT): ordem decrescente de
    tempo de processamento, cada uma na maquina de MENOR carga no
    momento da insercao (recalculada apos cada insercao anterior).

    Opera sobre uma COPIA de `sol` (a original nao e alterada).

    Parametros
    ----------
    strategy_id : int
        Indice da estrategia de destruicao em {0, 1, 2}.
    sol : PMSPSolution
        Solucao atual (nao modificada).
    d : int
        Numero de tarefas a destruir/reconstruir.
    rng : np.random.Generator

    Retorna
    -------
    (sol_new, delta) : (PMSPSolution, int)
        sol_new  -- nova solucao candidata (copia, ja reconstruida).
        delta    -- makespan(sol) - makespan(sol_new); positivo se
                    melhorou, negativo se piorou (PODE ser negativo,
                    ao contrario do apply_best() do ILS -- ver
                    docstring do modulo).
    """
    if not (0 <= strategy_id < N_DESTROY_STRATEGIES):
        raise ValueError(f"strategy_id invalido para IG: {strategy_id!r}")

    sol_new = sol.copy()
    removed = _DESTROY_STRATEGIES[strategy_id](sol_new, d, rng)
    p = sol_new.instance.p

    # Remove de fato (em termos de carga) as tarefas escolhidas ANTES
    # de decidir onde reinserir a primeira delas -- senao o argmin() da
    # primeira decisao usaria cargas ainda "contaminadas" pelas
    # proprias tarefas removidas (que continuam fisicamente em seu
    # lugar em sol_new ate serem movidas), produzindo reconstrucoes
    # sistematicamente piores que o LPT mesmo destruindo a instancia
    # inteira -- bug encontrado e corrigido na Entrada 23,
    # diario_pesquisa.md. loads_tracker e um array PARALELO (nao a
    # carga real de sol_new) usado so para decidir o alvo de cada
    # reinsercao; sol_new.move_job() mantem a contabilidade real da
    # solucao, e as duas convergem para o mesmo resultado final.
    loads_tracker = sol_new.loads.copy()
    for j in removed:
        loads_tracker[sol_new.assignment[j]] -= p[j]

    order = sorted(removed, key=lambda j: -p[j])  # decrescente, estilo NEH
    for j in order:
        k = int(loads_tracker.argmin())
        loads_tracker[k] += p[j]
        sol_new.move_job(j, k)

    delta = sol.makespan - sol_new.makespan
    return sol_new, delta


# ---------------------------------------------------------------------------
# Funcao principal: uma rodada do IG com selecao adaptativa de destruicao
# ---------------------------------------------------------------------------

def ig_run(
    instance: PMSPInstance,
    selector: BaseSelector,
    max_iter: int = 500,
    d: Optional[int] = None,
    seed: int = 0,
    stagnation_window: int = 50,
    track_history: bool = False,
    state_cfg=None,
    initial_solution=None,
    reward_fn=None,
    state_fn=None,
    action_map=None,
    ls_fn=None,
) -> Dict[str, Any]:
    """
    Executa uma rodada do Iterated Greedy com o seletor de estrategias
    de destruicao fornecido (tipicamente um QLearningAgent com
    n_actions=3 -- ver make_ig_selector em experiment/runner.py).

    Espelha a assinatura e o dict de retorno de pmsp.ils.ils_run para
    ser plugavel no mesmo harness experimental (analysis/tables.py,
    analysis/plots.py) sem mudancas -- a UNICA diferenca estrutural e
    que nao ha perturbation_threshold/n_perturb (o proprio ciclo
    destruir/reconstruir, a cada iteracao, ja cumpre o papel de busca
    E diversificacao -- ver docstring do modulo).

    Parametros
    ----------
    instance : PMSPInstance
    selector : BaseSelector
        Seletor de estrategia de destruicao. O metodo reset() e chamado
        automaticamente no inicio.
    max_iter : int
        Numero maximo de iteracoes do loop principal.
    d : int | None
        Numero de tarefas destruidas/reconstruidas por iteracao. Se
        None (padrao), usa max(2, round(0.1 * instance.n)) -- proporcional
        ao tamanho da instancia (ver docstring do modulo).
    seed : int
        Semente para o gerador de numeros aleatorios interno.
    stagnation_window : int
        Janela para normalizacao de f4 (estagnacao) -- mesma semantica
        do ils_run.
    track_history : bool
        Se True, registra o makespan do best a cada iteracao.
    state_cfg : pmsp.bounds.StateConfig | None
        Mesma semantica do ils_run -- granularidade de estado
        configuravel (Entrada 16). None = 81 estados padrao.

    Retorna
    -------
    dict com as mesmas chaves de ils_run: 'makespan', 'gap_pct',
    'cpu_time', 'n_improvements', 'history'.
    """
    rng = np.random.default_rng(seed)
    selector.reset()

    if d is None:
        d = max(2, round(0.3 * instance.n))

    # Entrada 102: `initial_solution` opcional (default None = LPT, comportamento original).
    sol = lpt(instance) if initial_solution is None else initial_solution.copy()
    s_best = sol.copy()
    best_makespan = sol.makespan
    iters_no_improve = 0
    n_improvements = 0

    history: List[int] = []
    t_start = time.perf_counter()

    for t in range(max_iter):
        if state_fn is not None:      # Entrada 104: funcao de estado injetada (default None = comportamento original)
            state_id = state_fn(sol, t, max_iter, iters_no_improve, stagnation_window, state_cfg)
        elif state_cfg is None:
            state_id = compute_state(sol, t, max_iter, iters_no_improve, stagnation_window)
        else:
            state_id = compute_state_cfg(sol, t, max_iter, iters_no_improve, stagnation_window, state_cfg)
        feats = compute_features(sol, t, max_iter, iters_no_improve, stagnation_window)

        action = selector.select(state_id, feats)
        if action_map is None:
            sol_new, delta = apply_destroy_reconstruct(action, sol, d, rng)
        else:      # Entrada 114: a acao escolhe (estrategia, tamanho de destruicao)
            strat, d_t = action_map(action, instance.n)
            sol_new, delta = apply_destroy_reconstruct(strat, sol, d_t, rng)
        if ls_fn is not None:        # Entrada 124: busca local apos reconstruir (IG hibrido); default None = comportamento anterior
            ls_fn(sol_new, rng)
            delta = sol.makespan - sol_new.makespan
        reward = float(max(delta, 0.0)) if reward_fn is None else float(reward_fn(sol, sol_new, delta))

        if state_fn is not None:
            next_state_id = state_fn(sol_new, t + 1, max_iter, iters_no_improve, stagnation_window, state_cfg)
        elif state_cfg is None:
            next_state_id = compute_state(sol_new, t + 1, max_iter, iters_no_improve, stagnation_window)
        else:
            next_state_id = compute_state_cfg(sol_new, t + 1, max_iter, iters_no_improve, stagnation_window, state_cfg)
        selector.update(state_id, feats, action, reward, next_state_id)

        # Criterio de aceitacao: greedy, igual ao ILS (nao piora)
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
