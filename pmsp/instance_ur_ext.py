"""
pmsp/instance_ur_ext.py (Entrada 254, Ciclo 30)
================================================
Familias PUBLICADAS de instancias de R||Cmax, para testar o protocolo FORA do gerador usado em todo o artigo (tempos independentes U[1, 100]). As tres familias e os tamanhos seguem a descricao de Aablad, Stromberg e Spensieri (Open Journal of Mathematical Optimization, 2021, secao de experimentos; configuracoes quase identicas as de Fanjul-Peyro e Ruiz):
  unc  nao correlacionada: p_ij ~ U{10, 100};
  cj   correlacionada por tarefa: p_ij = d_ij + b_j, b_j ~ U{1, 100}, d_ij ~ U{1, 20};
  cm   correlacionada por maquina: p_ij = a_i + d_ij, a_i ~ U{1, 100}, d_ij ~ U{1, 20}.
Tamanhos: m em {3, 5, 10, 15, 20} e n em {30, 50, 80, 100, 200} (25 por familia). A funcao gen(familia, n, m, semente) devolve P (n x m, int64); partida gulosa e limite inferior sao os de pmsp/instance_ur.py.
"""
from __future__ import annotations

import numpy as np

FAMILIES = ("unc", "cj", "cm")
SIZES = [(n, m) for m in (3, 5, 10, 15, 20) for n in (30, 50, 80, 100, 200)]


def gen(family, n, m, seed):
    rng = np.random.default_rng(seed)
    if family == "unc":
        P = rng.integers(10, 101, size=(n, m))
    elif family == "cj":
        b = rng.integers(1, 101, size=(n, 1))
        d = rng.integers(1, 21, size=(n, m))
        P = b + d
    elif family == "cm":
        a = rng.integers(1, 101, size=(1, m))
        d = rng.integers(1, 21, size=(n, m))
        P = a + d
    else:
        raise ValueError(family)
    return P.astype(np.int64)
