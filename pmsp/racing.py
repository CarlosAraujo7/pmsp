"""
pmsp/racing.py
===============
Prototipo do mecanismo de RACING/PORTFOLIO -- segunda "ideia
estruturalmente diferente" do hybrid_ql pedida por Carlos apos a
Entrada 60. Diferente do classificador a priori (pmsp/apriori_select.py,
que decide de uma estatistica ESTATICA da instancia, sem rodar nada) e
do hybrid_ql (que decide iteracao a iteracao, aprendendo uma Q-table),
este mecanismo decide EMPIRICAMENTE e SEM NENHUM APRENDIZADO: roda um
"probe" curto de CADA framework (ILS e IG) a partir da MESMA solucao
LPT inicial, ve' qual dos dois chegou mais longe nesse orcamento
pequeno, e compromete o RESTANTE do orcamento inteiro com o vencedor --
continuando a busca a partir de onde o probe vencedor parou (nao
reiniciando do LPT).

Vantagem conceitual sobre o classificador a priori: nao depende de uma
feature-proxy (dup_ratio) que pode errar (Entrada 61 mostrou 72,95% de
acuracia global, com uma familia -- E -- onde a regra fica pior que
aleatorio). Racing mede DIRETAMENTE o desempenho na instancia real,
entao deveria ser robusto a instancias "atipicas" onde uma regra fixa
erraria. Custo: gasta 2x probe_iters de orcamento so' em reconhecimento
(uma fracao do probe e' necessariamente "desperdicada" no framework
perdedor) -- um trade-off deliberado e documentado, nao escondido.

Design:
-------
- probe_iters = max(2, round(probe_frac * max_iter)) iteracoes de CADA
  framework, comecando do MESMO lpt(instance) (mesma seed base) --
  selecao de acao aleatoria simples dentro do probe (nao usa
  QLearningAgent aqui: o probe e' so' reconhecimento rapido, nao vale
  a pena aprender uma politica pra um orcamento tao curto).
- Vencedor = o probe que atingiu o menor makespan (gap_pct implicito).
- Fase de compromisso: continua a partir da solucao do probe vencedor,
  usando um QLearningAgent proprio (mesmo padrao de ils_ql/ig_ql) pelo
  orcamento restante (max_iter - 2*probe_iters).
- Orcamento TOTAL gasto = max_iter, igual aos outros metodos (2 probes
  curtos + 1 fase de compromisso longa), garantindo comparacao justa.

Reusa apply_best/apply_destroy_reconstruct de baixo nivel (mesmas
funcoes que pmsp/hybrid.py ja usa) -- nenhuma mudanca em pmsp/ils.py,
pmsp/ig.py, pmsp/operators.py ou pmsp/hybrid.py.

HISTORICO DE VERSAO (honesto, nao escondido):
-----------------------------------------------
v1 (Entrada 61, primeira versao pilotada): fase de compromisso do ramo
"ils" era um loop simplificado -- so' selecao de operador + aceitacao
greedy, SEM a perturbacao-por-estagnacao que pmsp/ils.py::ils_run tem
(ver secao 6 do algoritmo, docstring de ils_run). Pilotada nas mesmas 5
instancias reais de Entradas 59/60: o probe identificou o framework
vencedor corretamente 5/5 vezes (concordando com apriori_select em
todas), mas a fase de compromisso do ramo "ils" ficou visivelmente
pior que ils_ql nas 4 instancias onde ILS venceu (ex.: B/13 gap_pct
racing=0,0352 vs ils_ql=0,0087; n_improvements medio 3,33 vs 34,00 --
sintoma claro de estagnacao sem escape). No unico caso onde IG venceu
(C5), o desempenho ficou proximo/equivalente as referencias baseadas
em IG -- esperado, ja que o IG classico NAO tem fase de perturbacao
separada (o proprio ciclo destruir/reconstruir ja diversifica a cada
iteracao -- ver docstring de pmsp/ig.py), entao o ramo "ig" do racing
nunca teve essa lacuna.
v2 (esta versao): adiciona ao ramo "ils" da fase de compromisso a
MESMA perturbacao-por-estagnacao de ils_run (identica: reinicia de
s_best, aplica n_perturb movimentos ALEATORIOS via apply_rand quando
iters_no_improve >= perturbation_threshold). Parametros novos
`perturbation_threshold` (default 50) e `n_perturb` (default 3) --
mesmos defaults de ils_run/CFG.PERTURBATION_THRESHOLD/CFG.N_PERTURB.
Efeito nos resultados: ver diario_pesquisa.md, Entrada 61 (repiloto
v1 vs v2 documentado la' com numeros exatos, honesto sobre o quanto
fechou ou nao a lacuna).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .constructive import lpt
from .operators import apply_best, apply_rand, N_OPERATORS
from .ig import apply_destroy_reconstruct, N_DESTROY_STRATEGIES
from .qlearning import QLearningAgent
from .bounds import compute_state, compute_features, gap_percent

DEFAULT_PROBE_FRAC = 0.05


def _probe(instance, sol0, kind: str, probe_iters: int, d: int, rng) -> Dict[str, Any]:
    """
    Roda probe_iters iteracoes de um unico framework ('ils' ou 'ig'),
    partindo de sol0 (nao modifica sol0), selecao de acao ALEATORIA
    simples (sem aprendizado -- o probe e' so' reconhecimento rapido).
    Retorna a melhor solucao encontrada e seu makespan.
    """
    sol = sol0.copy()
    best_makespan = sol.makespan
    s_best = sol.copy()
    n_actions = N_OPERATORS if kind == "ils" else N_DESTROY_STRATEGIES

    for _ in range(probe_iters):
        action = int(rng.integers(n_actions))
        if kind == "ils":
            sol_new = sol.copy()
            apply_best(action, sol_new, rng)
        else:
            sol_new, _delta = apply_destroy_reconstruct(action, sol, d, rng)
        if sol_new.makespan <= sol.makespan:
            sol = sol_new
            if sol.makespan < best_makespan:
                best_makespan = sol.makespan
                s_best = sol.copy()

    return {"sol": sol, "s_best": s_best, "best_makespan": best_makespan}


def racing_run(
    instance,
    max_iter: int = 1000,
    probe_frac: float = DEFAULT_PROBE_FRAC,
    d: Optional[int] = None,
    seed: int = 0,
    stagnation_window: int = 50,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    perturbation_threshold: int = 50,
    n_perturb: int = 3,
    track_history: bool = False,
) -> Dict[str, Any]:
    """
    Executa o mecanismo de racing/portfolio completo (probe dos dois
    frameworks + compromisso com o vencedor). Ver docstring do modulo
    para o design completo e o historico de versao (v1 -> v2).

    Retorna um dict no mesmo formato de ils_run/ig_run/hybrid_run
    (makespan, gap_pct, cpu_time, n_improvements, history), mais
    'framework_vencedor', 'probe_ils_makespan', 'probe_ig_makespan' e
    'probe_iters' para auditoria/transparencia.
    """
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    if d is None:
        d = max(2, round(0.3 * instance.n))
    probe_iters = max(2, round(probe_frac * max_iter))

    sol0 = lpt(instance)

    probe_ils = _probe(instance, sol0, "ils", probe_iters, d, rng)
    probe_ig = _probe(instance, sol0, "ig", probe_iters, d, rng)

    if probe_ils["best_makespan"] <= probe_ig["best_makespan"]:
        winner = "ils"
        sol = probe_ils["sol"]
        s_best = probe_ils["s_best"]
        best_makespan = probe_ils["best_makespan"]
    else:
        winner = "ig"
        sol = probe_ig["sol"]
        s_best = probe_ig["s_best"]
        best_makespan = probe_ig["best_makespan"]

    n_improvements = 0  # conta so' as melhoras da fase de compromisso (probe e' reconhecimento)
    history = [best_makespan] if track_history else []

    remaining = max_iter - 2 * probe_iters
    if remaining > 0:
        n_actions = N_OPERATORS if winner == "ils" else N_DESTROY_STRATEGIES
        agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon_start=epsilon_start,
                                epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                                seed=seed + 1, n_actions=n_actions)
        iters_no_improve = 0
        for t in range(remaining):
            state_id = compute_state(sol, t, remaining, iters_no_improve, stagnation_window)
            feats = compute_features(sol, t, remaining, iters_no_improve, stagnation_window)
            action = agent.select(state_id, feats)

            if winner == "ils":
                sol_new = sol.copy()
                delta = apply_best(action, sol_new, rng)
            else:
                sol_new, delta = apply_destroy_reconstruct(action, sol, d, rng)
            reward = float(max(delta, 0.0))

            next_state_id = compute_state(sol_new, t + 1, remaining, iters_no_improve, stagnation_window)
            agent.update(state_id, feats, action, reward, next_state_id)

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

            # Perturbacao-por-estagnacao (v2): SO' no ramo "ils", replicando
            # exatamente pmsp/ils.py::ils_run secao 6 -- o ramo "ig" nao
            # precisa disso (destruir/reconstruir ja diversifica a cada
            # iteracao, ver docstring do modulo e de pmsp/ig.py).
            if winner == "ils" and iters_no_improve >= perturbation_threshold:
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
        "framework_vencedor": winner,
        "probe_ils_makespan": probe_ils["best_makespan"],
        "probe_ig_makespan": probe_ig["best_makespan"],
        "probe_iters": probe_iters,
    }
