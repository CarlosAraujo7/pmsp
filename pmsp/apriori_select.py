"""
pmsp/apriori_select.py
=======================
Prototipo do mecanismo de SELECAO A PRIORI -- uma das "ideias
estruturalmente diferentes" do hybrid_ql que Carlos pediu para explorar
apos a Entrada 60 (deixando de lado, por enquanto, a unificacao de
espaco de acao ja prototipada em pmsp/hybrid.py).

Em vez de aprender, iteracao a iteracao, qual mecanismo usar (hybrid_ql,
Entradas 59/60), este prototipo decide UMA UNICA VEZ, ANTES de rodar
qualquer iteracao, se a instancia sera' resolvida inteiramente por ILS
ou inteiramente por IG -- usando so' uma estatistica barata computada
diretamente do vetor de tempos de processamento da instancia (nenhuma
busca, nenhum aprendizado, custo ~O(n log n) por causa do np.unique).

Motivacao / achado que fundamenta a escolha da feature (Entrada 61):
---------------------------------------------------------------------
A primeira tentativa (CV = desvio padrao / media de p_j) FALHOU: para
p_j ~ U[1,H], CV(p_j) converge pra ~0,53-0,58 independente de H (e'
uma propriedade da FORMA uniforme, nao da escala) -- confirmado
varrendo H de 100 a 10**8 (n=150,m=50 fixos, 3 seeds cada): CV ficou
entre 0,532 e 0,539 em TODOS os 8 niveis de H testados, apesar do
vencedor (ILS vs IG) trocar completamente entre H=100 (IG vence) e
H=100000+ (ILS vence). CV nao serve como preditor a priori.

A feature que efetivamente separa os dois regimes: dup_ratio = fracao
de jobs cujo tempo de processamento e' IGUAL ao de pelo menos outro job
da mesma instancia (1 - n_valores_unicos/n). Intuicao: com poucos
valores possiveis (H pequeno relativo a n), o principio da casa dos
pombos forca repeticoes -- ha muitas solucoes de balanceamento
equivalentes, um problema "facil" onde a reconstrucao estocastica do
IG basta. Com H grande (ou em instancias reais com tempos de
processamento essencialmente unicos), o balanceamento fino vira um
quebra-cabeca combinatorio "dificil" (proximo da transicao de fase
classica do problema de particionamento de numeros -- literatura de
Karmarkar-Karp/Mertens), onde a busca local exaustiva do ILS tem
vantagem real.

Validacao honesta (Entrada 61): threshold dup_ratio=0,1 (dup_ratio >
0,1 -> prediz IG, senao -> prediz ILS), testado contra ils_ql/ig_ql JA
CALCULADOS na bateria externa (1.612 instancias reais B/C/E/F).
Acuracia GLOBAL: 72,95%. Por familia: B=82,0% (dominada por ILS, poucos
duplicados), C=73,3% (dominada por IG, muitos duplicados), F=64,0%,
E=48,75% (PIOR que aleatorio -- limitacao real e documentada, dup_ratio
nao captura o que decide o vencedor especificamente na familia E). Nao
e' um oraculo -- e' um sinal barato, simples e parcialmente
informativo, com uma limitacao clara.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from .ils import ils_run
from .ig import ig_run, N_DESTROY_STRATEGIES
from .qlearning import QLearningAgent
from .selectors import N_OPERATORS

DEFAULT_DUP_RATIO_THRESHOLD = 0.1


def dup_ratio(instance) -> float:
    """
    Fracao de jobs cujo tempo de processamento e' duplicado dentro da
    mesma instancia (1 menos a razao entre o numero de valores UNICOS
    de p_j e n). Ver docstring do modulo para o racional e a validacao.
    """
    p = np.asarray(instance.p)
    n_unique = len(np.unique(p))
    return 1.0 - n_unique / len(p)


def apriori_select(instance, threshold: float = DEFAULT_DUP_RATIO_THRESHOLD) -> str:
    """
    Decide 'ils' ou 'ig' a partir so' de dup_ratio(instance), sem rodar
    nenhuma busca. Retorna a string do framework escolhido.
    """
    return "ig" if dup_ratio(instance) > threshold else "ils"


def apriori_run(
    instance,
    max_iter: int = 500,
    threshold: float = DEFAULT_DUP_RATIO_THRESHOLD,
    seed: int = 0,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    stagnation_window: int = 50,
    perturbation_threshold: int = 50,
    n_perturb: int = 3,
) -> Dict[str, Any]:
    """
    Roda o framework (ILS ou IG) escolhido a priori por apriori_select(),
    usando o MESMO seletor QLearningAgent ja' validado (ils_ql/ig_ql)
    para a decisao de baixo nivel DENTRO do framework escolhido -- ou
    seja, este prototipo so' decide o FRAMEWORK (uma vez, no inicio),
    nao a operacao dentro dele (isso continua aprendido normalmente).

    Retorna o mesmo dict de ils_run/ig_run (makespan, gap_pct,
    cpu_time, n_improvements, ...), mais 'framework_escolhido' e
    'dup_ratio' para auditoria/transparencia.
    """
    t_start = time.perf_counter()
    dr = dup_ratio(instance)
    framework = "ig" if dr > threshold else "ils"

    if framework == "ils":
        agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon_start=epsilon_start,
                                epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                                seed=seed, n_actions=N_OPERATORS)
        result = ils_run(instance=instance, selector=agent, max_iter=max_iter,
                          perturbation_threshold=perturbation_threshold, n_perturb=n_perturb,
                          seed=seed, stagnation_window=stagnation_window)
    else:
        agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon_start=epsilon_start,
                                epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                                seed=seed, n_actions=N_DESTROY_STRATEGIES)
        result = ig_run(instance=instance, selector=agent, max_iter=max_iter,
                         seed=seed, stagnation_window=stagnation_window)

    result = dict(result)
    result["framework_escolhido"] = framework
    result["dup_ratio"] = dr
    # cpu_time recalculado aqui pra incluir o custo (desprezivel) da
    # decisao a priori em cima do que ils_run/ig_run ja mediram.
    result["cpu_time"] = time.perf_counter() - t_start
    return result
