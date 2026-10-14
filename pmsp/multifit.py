"""
pmsp/multifit.py
=================
MULTIFIT: heuristica construtiva classica para o PMSP baseada em busca
binaria + First-Fit-Decreasing (FFD) de bin-packing, usada como
baseline SOTA de comparacao (ver diario_pesquisa.md, Entradas 33 e 37;
ANALISE_E_PLANO_EJOR.md, Atualizacoes 19 e 21).

Referencia original:
    Coffman, E. G., Garey, M. R., & Johnson, D. S. (1978). "An
    Application of Bin-Packing to Multiprocessor Scheduling." SIAM
    Journal on Computing, 7(1), 1-17.

Fator de aproximacao (historico, verificado via multiplas fontes
independentes -- Wikipedia + OSTI + Springer, Entrada 37):
    original (CGJ, 1978): 1.22
    Friesen (1984): 6/5 = 1.2
    Yue (1990): 13/11 ~= 1.1818 (bound apertado, provado justo)

Algoritmo (busca binaria sobre a capacidade C de cada maquina):
    1. L <- max(max_j p_j, ceil(sum_p / m))       [== instance.lower_bound]
    2. U <- max(ceil(2*sum_p/m), max_j p_j)       [limite superior classico,
       garantidamente factivel para FFD em m bins -- resultado classico
       de bin-packing]
    3. Enquanto L < U:
           C <- (L + U) // 2
           Roda FFD com capacidade C nas m maquinas
           Se coube tudo: U <- C   Senao: L <- C + 1
    4. Retorna a alocacao associada ao ultimo U factivel encontrado.

FFD (First-Fit-Decreasing) com capacidade fixa C e m maquinas: ordena
jobs em ordem DECRESCENTE de p_j (mesma convencao do LPT deste
projeto). Para cada job, aloca a PRIMEIRA maquina (por indice 0..m-1,
NAO por carga -- isso seria "Best Fit") cuja carga atual + p_j <= C.
Se nenhuma das m maquinas tem espaco, C e infactivel.

Nota de fidelidade (Entrada 37): o algoritmo classico usa um numero
fixo k de iteracoes de busca binaria como parametro de precisao. Aqui
usamos convergencia exata (L < U, ambos inteiros, pois p_j e portanto
capacidades sao inteiras) -- mesmo resultado final que rodar k grande
o suficiente, so que sem iteracoes desnecessarias apos L==U. Otimizacao
de implementacao, nao mudanca no algoritmo.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .instance import PMSPInstance
from .solution import PMSPSolution
from .constructive import lpt
from .bounds import gap_percent


def _ffd_fits(
    order: np.ndarray,
    p: np.ndarray,
    m: int,
    capacity: int,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Tenta empacotar os jobs em `order` (ja em ordem decrescente de p_j)
    nas m maquinas via First-Fit-Decreasing com capacidade fixa
    `capacity`. First-Fit: cada job vai para a PRIMEIRA maquina (por
    indice, 0..m-1) que tem espaco.

    Retorna (True, assignment) se coube tudo; (False, None) se algum
    job nao coube em nenhuma das m maquinas.
    """
    n = len(order)
    loads = np.zeros(m, dtype=p.dtype)  # herda de p -- ver Entrada 53
    assignment = np.full(n, -1, dtype=np.int32)

    for j in order:
        pj = int(p[j])
        placed = False
        for k in range(m):
            if loads[k] + pj <= capacity:
                loads[k] += pj
                assignment[j] = k
                placed = True
                break
        if not placed:
            return False, None

    return True, assignment


def multifit(instance: PMSPInstance) -> PMSPSolution:
    """
    Constroi uma solucao pelo algoritmo MULTIFIT (Coffman, Garey &
    Johnson, 1978). Ver docstring do modulo para detalhes e referencia.

    Exemplo
    -------
    >>> inst = PMSPInstance(p=[8, 5, 3, 2], m=2)
    >>> sol = multifit(inst)
    >>> sol.makespan
    10
    """
    n, m = instance.n, instance.m
    p = instance.p
    order = np.argsort(p)[::-1]  # decrescente, mesma convencao do LPT

    total = int(p.sum())
    # ceil() exato em inteiros (equivalente a math.ceil(total/m), mas
    # sem passar por float -- este perdia precisao para somas extremas,
    # ver Entrada 53 do diario_pesquisa.md)
    L = max(int(p.max()), -(-total // m))
    U = max(-(-(2 * total) // m), int(p.max()))

    # Sanity check defensivo: U classico e garantidamente factivel, mas
    # confirmamos em vez de presumir. Se nao for (nao deveria
    # acontecer), caimos para o LPT (sempre factivel) como rede de
    # seguranca.
    feasible_U, assignment_U = _ffd_fits(order, p, m, U)
    if not feasible_U:
        return lpt(instance)

    best_assignment = assignment_U

    while L < U:
        C = (L + U) // 2
        feasible, assignment = _ffd_fits(order, p, m, C)
        if feasible:
            U = C
            best_assignment = assignment
        else:
            L = C + 1

    return PMSPSolution(instance, best_assignment)


def multifit_run(instance: PMSPInstance) -> Dict[str, Any]:
    """
    Avalia o MULTIFIT como baseline de comparacao (heuristica
    construtiva classica revisada por pares, sem busca local) --
    espelha lpt_run() em pmsp/ils.py para plugar no mesmo harness
    experimental sem mudancas.
    """
    t_start = time.perf_counter()
    sol = multifit(instance)
    cpu_time = time.perf_counter() - t_start
    lb = instance.lower_bound
    return {
        "makespan": sol.makespan,
        "gap_pct": gap_percent(sol.makespan, lb),
        "cpu_time": cpu_time,
        "n_improvements": 0,
        "history": [],
    }
