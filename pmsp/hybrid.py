"""
pmsp/hybrid.py
==============
Prototipo do mecanismo ADAPTATIVO IG/ILS (Tarefa #68, diario_pesquisa.md
Entrada 59): em vez de escolher a PRIORI entre o framework ILS (motor de
busca local, pmsp/ils.py) e o framework IG (motor de destruir/reconstruir,
pmsp/ig.py) -- decisao que a Entrada 58/59 mostrou nao ter uma resposta
universal (ILS domina a familia externa B e boa parte de E; IG empata ou
domina em varias classes da familia C; familia F e mista) -- este modulo
UNIFICA os dois espacos de acao num unico agente de selecao, e deixa o
PROPRIO mecanismo de aprendizado (o mesmo QLearningAgent ja usado em
ils_ql/ig_ql, sem nenhuma mudanca de codigo) decidir, iteracao a
iteracao e a partir do MESMO vetor de estado/features de 4 dimensoes ja
usado pelos dois frameworks (compute_state/compute_features, pmsp/bounds.py
-- ver nota abaixo sobre por que isso e valido sem nenhuma mudanca), qual
dos 7 "movimentos" aplicar:

    Acoes 0-3 (heranca do ILS, pmsp/operators.py):
        0 -- Op1 Inter-machine swap    (best-improvement, exaustivo,
                                         focado na maquina gargalo)
        1 -- Op2 Single reinsertion    (idem)
        2 -- Op3 Intra-machine 2-opt   (neutro p/ makespan puro no PMSP)
        3 -- Op4 Block move            (idem Op1/Op2)
    Acoes 4-6 (heranca do IG, pmsp/ig.py):
        4 -- Destroy Random            (destroi d tarefas aleatorias e
                                         reconstroi guloso)
        5 -- Destroy MostExpensive     (destroi as d mais caras)
        6 -- Destroy MachineConcentrated (destroi ate d de 1 maquina)

Por que reusar compute_state/compute_features sem nenhuma mudanca e
valido: as 4 features (f1 gap relativo, f2 coef. variacao das cargas,
f3 progresso, f4 estagnacao -- ver pmsp/bounds.py) sao definidas
PURAMENTE em termos do ESTADO DA SOLUCAO ATUAL (loads, makespan, LB,
iteracao, iters_no_improve) -- nunca em termos de QUAL mecanismo a
produziu. Confirmado por leitura direta do modulo (nao e' suposicao):
bounds.py e "intencionalmente sem estado (apenas funcoes puras)". Isso
significa que o MESMO espaco de estados de 81 celulas ja calibrado e
usado por ils_ql e ig_ql permanece semanticamente identico aqui -- nao
e preciso recalibrar nada para comecar a testar a ideia.

Motivacao empirica (ver diario_pesquisa.md, Entrada 59, investigacao
da Tarefa #67): o motor ILS aplica, a cada iteracao, uma busca
best-improvement EXAUSTIVA restrita a maquina gargalo atual (Op1/2/4
iteram sobre TODOS os pares/alvos possiveis envolvendo
sol.machine_max) -- por construcao NUNCA piora (delta >= 0 sempre) e
tem alta probabilidade de achar alguma melhora quando ela existe
naquela vizinhanca. O motor IG destroi uma fracao fixa (30%) das
tarefas e reconstroi de forma gulosa/estocastica, SEM direcionamento
para a maquina gargalo -- pode piorar (delta < 0, rejeitado) e, medido
diretamente (diagnostico ad-hoc, Entrada 59), acerta uma melhora em
apenas ~0-3% das iteracoes em instancias de alta variancia (familia B
externa), contra bottleneck-machine-search do ILS. Por outro lado, em
instancias com MUITAS maquinas e POUCAS tarefas por maquina (ex.
familia C, classe C5: n=144, m=48, ~3 tarefas/maquina), o diagnostico
ad-hoc mostrou o oposto: a reconstrucao ampla do IG (que redistribui
tarefas por TODAS as m maquinas de uma vez) superou claramente a busca
estritamente local do ILS (focada em 1 maquina por vez) -- gap 2.58%
(IG/MachineConcentrated) vs. 7.74% (ILS) apos 300 iteracoes a partir
do mesmo LPT. Ou seja: nenhum dos dois mecanismos domina
universalmente, e a razao aparenta ser estrutural (perfil da
instancia), nao um artefato de sintonia fina de nenhum dos dois --
exatamente o cenario em que um agente que aprende a ESCOLHER entre
"vizinhancas pequenas e exaustivas" e "saltos grandes e estocasticos",
condicionado ao estado observado, pode superar QUALQUER framework
fixo isoladamente.

Decisao de design deliberada (v1, prototipo -- ver diario, Entrada 59):
SEM o passo de perturbacao por estagnacao do ILS (perturbation_threshold/
n_perturb). Motivo: as 3 acoes de destruicao (4-6) ja fornecem, em tese,
o papel de "salto grande" que a perturbacao do ILS fornecia
artificialmente (kick de 3 movimentos aleatorios a cada 50 iteracoes
estagnadas) -- deixar o PROPRIO agente decidir quando usar um salto
grande (em vez de um gatilho fixo, cego ao estado) e mais consistente
com a filosofia de "selecao adaptativa" do projeto inteiro, e evita
introduzir 2 hiperparametros novos (threshold, n_perturb) sem
justificativa empirica ainda. Isso e uma simplificacao deliberada do
prototipo -- comparar COM vs. SEM um kick residual de estagnacao e uma
ablacao natural para depois que a ideia se mostrar promissora (nao
feita ainda).

Criterio de aceitacao e convencao de recompensa: identicos aos dois
frameworks originais (aceita se nao piora; recompensa = max(delta,0)) --
ver docstrings de pmsp/ils.py e pmsp/ig.py. Isso NAO e coincidencia: os
dois frameworks ja compartilhavam esses dois pontos, o que e o que
torna a unificacao do espaco de acao segura (nao ha conflito de
convencao a resolver).

Custo computacional por acao NAO e uniforme (achado da Entrada 59): as
acoes 0/1/3 (ILS) custam O(n^2)/O(n*m) por chamada (busca exaustiva);
as acoes 4-6 (IG) custam O(n) (uma unica reconstrucao gulosa, sem
comparar alternativas). O agente aprende a politica em termos de
NUMERO DE ITERACOES (recompensa = melhora de makespan), nao em termos
de tempo de CPU -- ou seja, o hybrid_run herda a MESMA limitacao (nao
normalizada por custo) que ja existe na comparacao atual ils_ql vs.
ig_ql (ver Entrada 59: cpu_time de ils_ql chega a ser 13.5x maior que
ig_ql na familia B em alta variancia). Nao corrigido aqui -- fica
registrado como limitacao conhecida do prototipo v1.

Este modulo NAO foi adicionado a METHODS/METHODS_NEW em
experiment/config.py nem a make_ils_selector/run_single em
experiment/runner.py -- e um prototipo isolado, testado via piloto
manual (ver experiment/pilot_hybrid.py), nao parte da bateria
principal. Decisao de integra-lo formalmente (novo metodo "hybrid_ql"
na grade completa) fica para depois que o piloto mostrar sinal
suficiente e Carlos aprovar o custo computacional adicional.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .instance import PMSPInstance
from .solution import PMSPSolution
from .constructive import lpt
from .operators import apply_best, N_OPERATORS
from .ig import apply_destroy_reconstruct, N_DESTROY_STRATEGIES, DESTROY_STRATEGY_NAMES
from .selectors import BaseSelector
from .bounds import compute_state, compute_features, gap_percent, compute_state_cfg

#: Espaco de acao unificado: 0..3 = operadores ILS, 4..6 = estrategias IG
N_HYBRID_ACTIONS = N_OPERATORS + N_DESTROY_STRATEGIES  # 4 + 3 = 7

HYBRID_ACTION_NAMES = [
    "Op1-InterSwap", "Op2-Reinsertion", "Op3-IntraSwap", "Op4-BlockMove",
    "Destroy-Random", "Destroy-MostExpensive", "Destroy-MachineConcentrated",
]


def apply_hybrid_action(
    action_id: int,
    sol: PMSPSolution,
    d: int,
    rng: np.random.Generator,
) -> Tuple[PMSPSolution, int]:
    """
    Despacha uma acao do espaco unificado de 7 acoes.

    Acoes 0..N_OPERATORS-1 (ILS): sol NAO e modificada, apply_best roda
    sobre uma COPIA (mesma convencao de ils_run). Acoes seguintes (IG):
    apply_destroy_reconstruct ja copia `sol` internamente.

    Retorna (sol_new, delta), delta = makespan(sol) - makespan(sol_new)
    (positivo = melhorou; pode ser negativo so nas acoes IG, nunca nas
    acoes ILS -- ver docstrings de pmsp/operators.py e pmsp/ig.py).
    """
    if 0 <= action_id < N_OPERATORS:
        sol_new = sol.copy()
        delta = apply_best(action_id, sol_new, rng)
        return sol_new, delta
    strat_id = action_id - N_OPERATORS
    if not (0 <= strat_id < N_DESTROY_STRATEGIES):
        raise ValueError(f"action_id invalido para hybrid: {action_id!r}")
    return apply_destroy_reconstruct(strat_id, sol, d, rng)


def hybrid_run(
    instance: PMSPInstance,
    selector: BaseSelector,
    max_iter: int = 500,
    d: Optional[int] = None,
    seed: int = 0,
    stagnation_window: int = 50,
    track_history: bool = False,
    track_action_history: bool = False,
    state_cfg=None,
) -> Dict[str, Any]:
    """
    Executa uma rodada do mecanismo hibrido IG/ILS com selecao
    adaptativa sobre o espaco unificado de 7 acoes (ver docstring do
    modulo). Espelha a assinatura/dict de retorno de ils_run/ig_run
    para ser plugavel no mesmo harness experimental.

    Parametros identicos a ig_run/ils_run, mais:
    track_action_history : bool
        Se True, registra o id da acao escolhida a cada iteracao (para
        inspecionar se a politica aprendida realmente se especializa
        por tipo de instancia -- ver analysis/ da Entrada 59+).

    Retorna
    -------
    dict com as mesmas chaves de ig_run/ils_run, mais 'action_history'
    (lista vazia se track_action_history=False).
    """
    rng = np.random.default_rng(seed)
    selector.reset()

    if d is None:
        d = max(2, round(0.3 * instance.n))

    sol = lpt(instance)
    s_best = sol.copy()
    best_makespan = sol.makespan
    iters_no_improve = 0
    n_improvements = 0

    history: List[int] = []
    action_history: List[int] = []
    t_start = time.perf_counter()

    for t in range(max_iter):
        if state_cfg is None:
            state_id = compute_state(sol, t, max_iter, iters_no_improve, stagnation_window)
        else:
            state_id = compute_state_cfg(sol, t, max_iter, iters_no_improve, stagnation_window, state_cfg)
        feats = compute_features(sol, t, max_iter, iters_no_improve, stagnation_window)

        action = selector.select(state_id, feats)
        sol_new, delta = apply_hybrid_action(action, sol, d, rng)
        reward = float(max(delta, 0.0))

        if state_cfg is None:
            next_state_id = compute_state(sol_new, t + 1, max_iter, iters_no_improve, stagnation_window)
        else:
            next_state_id = compute_state_cfg(sol_new, t + 1, max_iter, iters_no_improve, stagnation_window, state_cfg)
        selector.update(state_id, feats, action, reward, next_state_id)

        if track_action_history:
            action_history.append(action)

        # Criterio de aceitacao: greedy, identico aos dois frameworks originais
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
        "action_history": action_history,
    }


# ---------------------------------------------------------------------------
# Pre-treinamento (warm-start) da Q-table hibrida -- Tarefa #69
# ---------------------------------------------------------------------------
# Generalizacao de experiment/runner.py:pretrain_qtable() (usada para dar
# warm-start ao ils_ql_ws) para o agente hibrido de 7 acoes definido acima.
# Mantida NESTE modulo -- e nao em experiment/runner.py -- para preservar
# pmsp/hybrid.py como prototipo isolado que nao mexe na bateria principal
# (ver diario_pesquisa.md, Entrada 59) e para nao acoplar pmsp/ (algoritmos)
# a experiment/config.py (hiperparametros de orquestracao): os hiperparametros
# permanecem argumentos explicitos, no mesmo padrao ja usado por hybrid_run()
# acima e por ils_run()/ig_run() em pmsp/ils.py e pmsp/ig.py. O import de
# QLearningAgent/N_STATES fica aqui (em vez de subir para o bloco de imports
# no topo do arquivo) para manter esta adicao estritamente append-only e
# auditavel por diff, seguindo a convencao do projeto.
#
# Diferenca crucial em relacao ao pretrain_qtable() original: aquela funcao
# chama PMSPInstance.random(n=n, m=m, seed=seed) SEM passar low/high, entao
# so' amostra do range de variancia padrao [1,100] (baixa variancia) --
# nunca do regime de alta variancia. A Tarefa #70 (diario_pesquisa.md,
# Entrada 60) mostrou, com um fatorial controlado 2x3 (variancia x n/m),
# que a variancia dos tempos de processamento -- e NAO o n/m -- e' o fator
# causal dominante que decide se ILS ou IG (e, por extensao, quais das 7
# acoes do hibrido) sao favorecidas: em variancia alta, ILS vence em TODOS
# os niveis de n/m testados; em variancia baixa, IG empata ou vence em
# todos os niveis. Por isso, esta funcao alterna deliberadamente entre
# baixa e alta variancia a cada instancia de pre-treino (variance_ranges),
# para que a Q-table resultante carregue sinal aprendido sobre AMBOS os
# regimes antes mesmo da primeira iteracao da rodada real -- ao contrario
# do pretrain_qtable() original, que nunca "via" alta variancia.

from .qlearning import QLearningAgent, N_STATES  # import local, ver nota acima


def pretrain_hybrid_qtable(
    source_sizes,
    n_pretrain,
    max_iter,
    variance_ranges=((1, 100), (1, 10 ** 8)),
    base_seed=0,
    alpha=0.1,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    stagnation_window=50,
    verbose=True,
):
    """
    Pre-treina um QLearningAgent de N_HYBRID_ACTIONS=7 acoes sobre uma
    MISTURA deliberada de instancias sinteticas de baixa e alta variancia,
    e retorna a Q-table media acumulada sobre todas as rodadas.

    Ver nota de modulo acima para o racional completo (Tarefa #69/#70,
    diario_pesquisa.md, Entrada 60).

    Parameters
    ----------
    source_sizes : list[tuple[int, int]]
        Lista de (n, m) das instancias sinteticas de pre-treino.
    n_pretrain : int
        Numero de rodadas de pre-treino por tamanho (n, m).
    max_iter : int
        Orcamento de iteracoes de hybrid_run() por rodada de pre-treino.
    variance_ranges : sequence[tuple[int, int]]
        Pares (low, high) para PMSPInstance.random(), alternados
        ciclicamente por inst_id (inst_id % len(variance_ranges)).
        Default replica os dois extremos do fatorial da Tarefa #70:
        (1, 100) = "baixa variancia", (1, 10**8) = "alta variancia".
    base_seed, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay,
    stagnation_window
        Ver QLearningAgent (pmsp/qlearning.py) e hybrid_run() acima.

    Returns
    -------
    np.ndarray shape (N_STATES, N_HYBRID_ACTIONS)
    """
    Q_accum = np.zeros((N_STATES, N_HYBRID_ACTIONS), dtype=np.float64)
    count = 0
    n_var = len(variance_ranges)

    if verbose:
        print(f"  [pretrain_hybrid] Iniciando pre-treinamento da Q-table "
              f"hibrida (mix de {n_var} regime(s) de variancia)...")

    for n, m in source_sizes:
        for inst_id in range(n_pretrain):
            seed = base_seed + inst_id
            low, high = variance_ranges[inst_id % n_var]
            inst = PMSPInstance.random(n=n, m=m, low=low, high=high, seed=seed)
            agent = QLearningAgent(
                alpha=alpha,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                seed=seed,
                n_actions=N_HYBRID_ACTIONS,
            )
            hybrid_run(
                instance=inst,
                selector=agent,
                max_iter=max_iter,
                seed=seed,
                stagnation_window=stagnation_window,
            )
            Q_accum += agent._Q
            count += 1

    Q_avg = Q_accum / count if count > 0 else Q_accum
    if verbose:
        dominant = int(np.bincount(np.argmax(Q_avg, axis=1)).argmax())
        print(f"  [pretrain_hybrid] {count} rodadas concluidas. "
              f"Acao dominante na politica media: {HYBRID_ACTION_NAMES[dominant]}")
    return Q_avg
