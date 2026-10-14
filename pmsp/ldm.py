"""
pmsp/ldm.py
===========
Largest Differencing Method (LDM) -- generalizacao m-way do metodo de diferenciacao de
Karmarkar & Karp (1982) para P||Cmax; analise de desempenho em Michiels, Korst, Aarts &
van Leeuwen (2007), J. Comb. Optim. 13(1):19-32 (DOI 10.1007/s10878-006-9010-z).
Heuristica construtiva, deterministica, sem orcamento de iteracoes (como MULTIFIT).

Algoritmo: cada tarefa j e' uma "tupla parcial" de m somas ordenadas de forma decrescente,
(p_j, 0, ..., 0). Enquanto houver mais de uma tupla, retira-se as duas com maior diferenca
(maior soma - menor soma) e as combina somando a maior soma de uma a menor da outra, a
segunda maior a segunda menor, etc.; a tupla resultante e' reordenada e reinserida. A tupla
final contem as cargas das m maquinas. Cada soma carrega a lista de tarefas que a compoe,
de modo que a atribuicao e' reconstruida exatamente.
"""
from __future__ import annotations

import heapq
import time
from typing import Any, Dict

import numpy as np

from .bounds import gap_percent
from .instance import PMSPInstance
from .solution import PMSPSolution


def ldm(instance: PMSPInstance) -> PMSPSolution:
    n, m = instance.n, instance.m
    p = [int(x) for x in instance.p]
    heap = []
    for j in range(n):
        tup = [(p[j], [j])] + [(0, []) for _ in range(m - 1)]
        heapq.heappush(heap, (-(p[j]), j, tup))
    counter = n
    while len(heap) > 1:
        _, _, a = heapq.heappop(heap)
        _, _, b = heapq.heappop(heap)
        merged = [(a[i][0] + b[m - 1 - i][0], a[i][1] + b[m - 1 - i][1]) for i in range(m)]
        merged.sort(key=lambda t: -t[0])
        diff = merged[0][0] - merged[-1][0]
        counter += 1
        heapq.heappush(heap, (-diff, counter, merged))
    _, _, final = heap[0]
    assignment = np.zeros(n, dtype=np.int64)
    for machine, (_, jobs) in enumerate(final):
        for j in jobs:
            assignment[j] = machine
    return PMSPSolution(instance, assignment)


def ldm_run(instance: PMSPInstance) -> Dict[str, Any]:
    """Interface do harness (mesmo formato de multifit_run / lpt_run)."""
    t0 = time.perf_counter()
    sol = ldm(instance)
    cpu = time.perf_counter() - t0
    lb = instance.lower_bound
    return {"makespan": sol.makespan, "gap_pct": gap_percent(sol.makespan, lb), "cpu_time": cpu,
            "n_improvements": 0, "history": []}
