"""
pmsp/constructive.py
====================
Heuristica construtiva LPT (Longest Processing Time) para o PMSP.

Algoritmo LPT (Graham, 1969):
    1. Ordene os jobs em ordem decrescente de p_j.
    2. Para cada job na ordem, aloque-o a maquina com menor carga atual.

Propriedade teorica: LPT produz C_max <= (4/3 - 1/(3m)) * OPT.
O LPT e usado como solucao inicial para o ILS em todos os baselines,
garantindo um ponto de partida de qualidade consistente.

Complexidade: O(n log n) para ordenacao + O(n log m) com heap.
Implementacao atual: O(n * m) com argmin NumPy (suficiente para n<=200).
"""

from __future__ import annotations

import numpy as np

from .instance import PMSPInstance
from .solution import PMSPSolution


def lpt(instance: PMSPInstance) -> PMSPSolution:
    """
    Constroi uma solucao pelo algoritmo LPT.

    Parametros
    ----------
    instance : PMSPInstance
        Instancia do PMSP a ser resolvida.

    Retorna
    -------
    PMSPSolution
        Solucao construida pelo LPT. Garante validez (is_valid() == True).

    Exemplo
    -------
    >>> inst = PMSPInstance(p=[8, 5, 3, 2], m=2)
    >>> sol = lpt(inst)
    >>> sol.makespan  # jobs 8->maq0; 5->maq1; 3->maq1; 2->maq0 => [10, 8] => 10
    10
    """
    n, m = instance.n, instance.m
    p = instance.p

    # Ordena os indices dos jobs em ordem DECRESCENTE de p_j
    # O job mais longo eh alocado primeiro para reduzir o desequilibrio
    order = np.argsort(p)[::-1]  # shape (n,), indices em ordem decrescente

    assignment = np.empty(n, dtype=np.int32)
    loads = np.zeros(m, dtype=p.dtype)  # herda de p -- ver Entrada 53 do diario

    for j in order:
        # Aloca job j a maquina com menor carga atual: O(m) por job
        k = int(loads.argmin())
        assignment[j] = k
        loads[k] += p[j]

    return PMSPSolution(instance, assignment)


def random_construction(
    instance: PMSPInstance,
    rng: np.random.Generator,
) -> PMSPSolution:
    """
    Constroi uma solucao aleatoria (round-robin com permutacao aleatoria).
    Usada para gerar solucoes diversas na fase de perturbacao do ILS.

    Parametros
    ----------
    instance : PMSPInstance
    rng : np.random.Generator
        Gerador de numeros aleatorios (para reproducibilidade).

    Retorna
    -------
    PMSPSolution
    """
    n, m = instance.n, instance.m
    order = rng.permutation(n)
    assignment = np.empty(n, dtype=np.int32)
    loads = np.zeros(m, dtype=instance.p.dtype)  # herda de instance.p -- ver Entrada 53

    for j in order:
        k = int(loads.argmin())
        assignment[j] = k
        loads[k] += instance.p[j]

    return PMSPSolution(instance, assignment)
