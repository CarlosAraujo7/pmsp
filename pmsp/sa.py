"""
pmsp/sa.py
==========
Simulated Annealing (SA) para o PMSP, reimplementado a partir de Li &
Perkins (2024), "A Simulated Annealing Approach to Identical Parallel
Machine Scheduling" (arXiv:2410.11880, preprint -- nenhuma publicacao
revisada por pares localizada ate a Entrada 37) -- baseline SOTA de
comparacao empirica (ver diario_pesquisa.md, Entradas 34, 35, 37;
ANALISE_E_PLANO_EJOR.md, Atualizacoes 19 e 21).

Fidelidade ao artigo original (texto completo extraido diretamente do
PDF do arXiv na Entrada 37 -- nao resumo de IA, texto bruto via
pdftotext):

    - Vizinhanca (Algoritmo 1 do paper): escolhe 2 maquinas aleatorias
      m1 != m2; com probabilidade r faz SWAP de um job aleatorio de m1
      com um job aleatorio de m2; com probabilidade (1-r) faz MOVE de
      um job aleatorio de m1 para m2.
    - Aceitacao (Algoritmo 2 do paper): se o vizinho e pior, aceita com
      probabilidade exp(-a/T). ATENCAO (achado verificado, Entrada 37):
      essa formula NAO depende da magnitude da piora (delta) -- so da
      temperatura -- diferente do SA classico (que usa exp(-delta/T)).
      Isso e uma caracteristica confirmada do artigo original (nao erro
      de transcricao nossa), replicada fielmente aqui.
    - T decai geometricamente a cada iteracao: T <- T * D.
    - Parametros usados no artigo (Secao 4): T=10, D=0.9, a=1, p=1.01.
    - Estimativa do otimo k = max(maior p_j, soma_p/n) -- EXATAMENTE o
      lower_bound classico ja usado neste projeto
      (PMSPInstance.lower_bound). Parada antecipada se makespan <= k*p.

Tres diferencas DELIBERADAS em relacao ao artigo original, decididas
com Carlos e registradas na Entrada 37:

    1. Solucao inicial: o artigo original usa alocacao ALEATORIA (Secao
       3); os proprios autores reconhecem na Conclusao que isso e uma
       limitacao ("instead of randomly distributing the jobs, a simple
       heuristic method such as list scheduling could be implemented").
       Aqui partimos do LPT, igual a TODOS os outros metodos deste
       projeto (ver docstring de pmsp/constructive.py) -- isolando a
       comparacao no MECANISMO de busca, nao no ponto de partida.

    2. Orcamento/parada: o artigo original usa uma regra de parada
       propria por TEMPO DECORRIDO (M = n milissegundos -- regra
       incomum, nao normalizada por hardware/linguagem, e o proprio
       paper nao reporta quantas iteracoes isso corresponde). A pedido
       explicito de Carlos ("mesmo orcamento dos nossos metodos",
       Entrada 37), substituimos por max_iter no mesmo sentido de
       ils_run/ig_run (numero fixo de iteracoes, tipicamente
       MAX_ITER=1000 -- o mesmo orcamento do ig_ql).

       Nota tecnica (verificada por calculo direto e depois em runtime,
       Entrada 37): com T0=10 e D=0.9, o schedule satura (T -> ~0,
       aceitacao de pioras -> ~0) por volta da iteracao ~60-100,
       INDEPENDENTE do orcamento total -- o SA passa a maior parte de
       um orcamento de 1000 iteracoes em regime essencialmente guloso.
       Isso NAO e uma distorcao introduzida pela equalizacao de
       orcamento: e uma caracteristica do proprio desenho de parametros
       do artigo original (os autores reconhecem que o tuning nao foi
       exaustivo). Mantemos T0/D/a EXATAMENTE como publicados -- nao
       "consertamos" o cooling schedule deles.

    3. Parametro r (prob. de swap vs. move, Algoritmo 1): o artigo
       declara `r` como parametro de entrada (0<r<1) mas NUNCA
       especifica o valor usado nos experimentos (Secao 4 lista T, D,
       a, p mas nao r) -- confirmado por leitura do texto completo, nao
       e omissao nossa. Default neutro r=0.5, documentado e
       configuravel.

IMPORTANTE -- colisao de nomes no pseudocodigo original: o Algoritmo 1
usa `a` para um sorteio aleatorio POR ITERACAO (decide mover vs.
trocar), enquanto o Algoritmo 2 usa `a` para a CONSTANTE fixa da
formula de aceitacao (a=1) -- duas coisas diferentes com o mesmo nome
no paper original. Aqui usamos `coin` para o sorteio da Alg. 1 e
`a_const` para a constante da Alg. 2, evitando a colisao.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List

import numpy as np

from .instance import PMSPInstance
from .solution import PMSPSolution
from .constructive import lpt
from .bounds import gap_percent


def _heuristic_move_or_swap(
    sol: PMSPSolution,
    r: float,
    rng: np.random.Generator,
) -> int:
    """
    Um passo da heuristica de vizinhanca (Algoritmo 1 de Li & Perkins,
    2024): escolhe 2 maquinas distintas aleatorias e, com probabilidade
    r, troca (swap) um job aleatorio de cada uma; com probabilidade
    (1-r), move um job aleatorio de uma para a outra.

    Modifica `sol` IN-PLACE -- quem chama decide se opera sobre uma
    copia (ver sa_run, que sempre passa uma copia). Retorna o delta de
    makespan (positivo=melhorou, pode ser negativo=piorou -- mesma
    convencao de apply_destroy_reconstruct em pmsp/ig.py).

    Caso extremo nao tratado no paper original (que assume implicitamente
    que toda maquina tem >=1 job): se a maquina de origem sorteada
    estiver vazia, a jogada e tratada como nula (delta=0) -- robustez
    defensiva documentada aqui por transparencia.
    """
    m = sol.instance.m
    makespan_before = sol.makespan

    m1 = int(rng.integers(m))
    m2 = int(rng.integers(m - 1))
    if m2 >= m1:
        m2 += 1  # garante m2 != m1, uniforme sobre as m-1 restantes

    jobs_m1 = sol.machine_jobs[m1]
    jobs_m2 = sol.machine_jobs[m2]

    coin = rng.random()
    if coin < r:
        if not jobs_m1 or not jobs_m2:
            return 0
        j1 = int(rng.choice(jobs_m1))
        j2 = int(rng.choice(jobs_m2))
        sol.swap_jobs(j1, j2)
    else:
        if not jobs_m1:
            return 0
        j1 = int(rng.choice(jobs_m1))
        sol.move_job(j1, m2)

    return makespan_before - sol.makespan


def sa_run(
    instance: PMSPInstance,
    max_iter: int = 500,
    seed: int = 0,
    T0: float = 10.0,
    D: float = 0.9,
    a_const: float = 1.0,
    p_target: float = 1.01,
    r: float = 0.5,
    track_history: bool = False,
) -> Dict[str, Any]:
    """
    Executa uma rodada do Simulated Annealing de Li & Perkins (2024),
    reimplementado como baseline de comparacao. Ver docstring do modulo
    para a fidelidade ao artigo original e as diferencas deliberadas
    (solucao inicial, orcamento, parametro r).

    Espelha a assinatura e o dict de retorno de pmsp.ils.ils_run /
    pmsp.ig.ig_run para plugar no mesmo harness experimental sem
    mudancas.

    Parametros
    ----------
    instance : PMSPInstance
    max_iter : int
        Numero maximo de iteracoes (substitui a regra M=n ms do artigo
        original, a pedido de Carlos -- Entrada 37).
    seed : int
    T0, D, a_const : float
        Parametros do cooling schedule, exatamente como no artigo
        original (T=10, D=0.9, a=1 na Secao 4 deles).
    p_target : float
        Fator de aproximacao alvo para parada antecipada (Cmax<=k*p),
        1.01 no artigo original.
    r : float
        Probabilidade de escolher SWAP em vez de MOVE a cada iteracao.
        Nao especificado no artigo original -- default neutro 0.5.
    track_history : bool

    Retorna
    -------
    dict com as mesmas chaves de ils_run/ig_run: 'makespan', 'gap_pct',
    'cpu_time', 'n_improvements', 'history'.
    """
    rng = np.random.default_rng(seed)

    sol = lpt(instance)  # diferenca deliberada #1 -- ver docstring do modulo
    best_makespan = sol.makespan
    n_improvements = 0

    lb = instance.lower_bound
    target_makespan = lb * p_target

    T = T0
    history: List[int] = []
    t_start = time.perf_counter()

    for _ in range(max_iter):
        if best_makespan <= target_makespan:
            break

        sol_new = sol.copy()
        _heuristic_move_or_swap(sol_new, r, rng)

        if sol_new.makespan > sol.makespan:
            # Vizinho pior: aceita com prob. exp(-a_const/T) (Alg. 2 do
            # paper -- NAO depende do delta, ver docstring do modulo)
            accept_prob = math.exp(-a_const / T) if T > 0 else 0.0
            if rng.random() < accept_prob:
                sol = sol_new
        else:
            sol = sol_new

        if sol.makespan < best_makespan:
            best_makespan = sol.makespan
            n_improvements += 1

        T *= D
        if track_history:
            history.append(best_makespan)

    cpu_time = time.perf_counter() - t_start

    return {
        "makespan": best_makespan,
        "gap_pct": gap_percent(best_makespan, lb),
        "cpu_time": cpu_time,
        "n_improvements": n_improvements,
        "history": history,
    }
