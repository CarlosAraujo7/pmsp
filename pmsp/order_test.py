"""
pmsp/order_test.py (Entrada 288, Ciclo 54)
===========================================
TESTE DE INVARIANCIA A ORDEM das acoes: um diagnostico reutilizavel para qualquer seletor de operadores (proposta E da revisao cega #2 do CEC; Revisor 2, M3). Pergunta: o resultado do seletor depende da ORDEM em que as acoes
foram listadas? Um seletor invariante a ordem tem resultado que nao muda quando a lista e' permutada; um Q-Learning com tabela zerada, recompensa nao negativa e desempate pelo primeiro maximo NAO e' invariante (a acao listada
primeiro e' o padrao quando a linha da tabela esta zerada).

Delineamento: R ordens (permutacoes) da lista de acoes, executadas nas MESMAS instancias (bloco) com algumas sementes. y[i, r] = resultado (por exemplo o makespan relativo) da instancia i sob a ordem r (media das sementes).
Estatistica: SQ_ordem = soma sobre as ordens de n_blocos (media da ordem r - media geral)^2, depois de centrar cada instancia (efeito de bloco removido). Hipotese nula: as ordens sao permutaveis DENTRO de cada instancia; o p-valor
vem de embaralhar os rotulos de ordem dentro de cada instancia (n_perm vezes). Devolve a estatistica, o p-valor e o DP entre as ordens do resultado medio (na unidade de y), que mede o TAMANHO do efeito de ordem.
"""
from __future__ import annotations

import numpy as np


def order_invariance_test(y, n_perm: int = 2000, seed: int = 0) -> dict:
    """y: matriz (instancias x ordens). Retorna dict(stat, p, sd_between, n_blocks, n_orders)."""
    y = np.asarray(y, dtype=float)
    n, r = y.shape
    yc = y - y.mean(axis=1, keepdims=True)                  # remove o efeito de instancia
    col = yc.mean(axis=0)
    stat = float(n * np.sum(col ** 2))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        sh = np.take_along_axis(yc, np.argsort(rng.random((n, r)), axis=1), axis=1)     # embaralha as colunas dentro de cada linha
        null[b] = n * np.sum(sh.mean(axis=0) ** 2)
    p = float((1 + np.sum(null >= stat)) / (1 + n_perm))
    return dict(stat=stat, p=p, sd_between=float(col.std(ddof=1)), n_blocks=n, n_orders=r)
