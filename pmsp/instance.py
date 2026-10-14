"""
pmsp/instance.py
================
Define a classe PMSPInstance, que encapsula todos os dados estáticos
de uma instância do Parallel Machine Scheduling Problem (PMSP).

Problema:
    Dado um conjunto de n jobs com tempos de processamento p_j > 0
    e m máquinas idênticas, alocar cada job a exatamente uma máquina
    minimizando o makespan C_max = max_i { sum_{j: a_j=i} p_j }.

Esta classe é o ponto de entrada de todo o pipeline:
    geração aleatória → leitura de arquivo → objeto PMSPInstance
                                                      ↓
                                           PMSPSolution / bounds / operadores
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

# Teto de seguranca para a soma dos tempos de processamento -- ver
# Entrada 53 do diario_pesquisa.md. Instancias cuja soma total
# ultrapassa este valor usam dtype=object (inteiros Python de precisao
# arbitraria) para o vetor `p` e para todos os acumuladores de carga
# derivados dele, em vez de int64 -- imune a overflow silencioso.
# 10**15 fica confortavelmente abaixo tanto do limite de precisao
# exata do float64 (2**53 ~= 9.007e15, relevante para bco.py) quanto
# do teto do int64 (2**63-1 ~= 9.223e18).
SAFE_INT64_SUM_THRESHOLD = 10 ** 15


class PMSPInstance:
    """
    Representa uma instância do PMSP de forma imutável após a criação.

    Atributos
    ----------
    name : str
        Identificador textual da instância (ex.: "tai_20_5" ou "rand_50_3_seed42").
    n : int
        Número de jobs.
    m : int
        Número de máquinas idênticas.
    p : np.ndarray, shape (n,), dtype=int64
        Vetor de tempos de processamento p_j, indexado de 0 a n-1.
    """

    # ------------------------------------------------------------------
    # Construtores
    # ------------------------------------------------------------------

    def __init__(self, p: Sequence[int], m: int, name: str = "unnamed") -> None:
        """
        Parâmetros
        ----------
        p : sequência de inteiros positivos
            Tempos de processamento dos n jobs.
        m : int
            Número de máquinas (≥ 1).
        name : str
            Nome/identificador da instância.
        """
        if m < 1:
            raise ValueError(f"O número de máquinas deve ser ≥ 1, recebido: {m}")
        p_arr_probe = np.asarray(p)
        if p_arr_probe.ndim != 1 or len(p_arr_probe) == 0:
            raise ValueError("O vetor de tempos de processamento deve ser 1-D e não vazio.")
        # Decide o dtype por soma em Python puro (precisao arbitraria,
        # imune a overflow) ANTES de fixar o array numpy -- ver Entrada
        # 53 do diario_pesquisa.md (bug de overflow int64 descoberto na
        # familia B, classe 18, do benchmark externo Symmetry 2025:
        # soma dos tempos de processamento ~2.78e19, muito acima do
        # teto do int64). Instancias "normais" ficam muito abaixo do
        # limiar e continuam em int64 nativo, sem qualquer mudanca de
        # desempenho.
        p_list = [int(x) for x in p_arr_probe]
        total_check = sum(p_list)
        p_dtype = np.int64 if total_check <= SAFE_INT64_SUM_THRESHOLD else object
        p_arr = np.asarray(p_list, dtype=p_dtype)
        if np.any(p_arr <= 0):
            raise ValueError("Todos os tempos de processamento devem ser > 0.")

        self._p = p_arr
        self._m = int(m)
        self._name = name

        # Cache de lower bound (calculado sob demanda)
        self._lb: int | None = None

    # ------------------------------------------------------------------
    # Propriedades (somente leitura após construção)
    # ------------------------------------------------------------------

    @property
    def p(self) -> np.ndarray:
        """Vetor de tempos de processamento (somente leitura)."""
        return self._p.view()          # view sem cópia, mas protegido contra escrita

    @property
    def n(self) -> int:
        """Número de jobs."""
        return len(self._p)

    @property
    def m(self) -> int:
        """Número de máquinas."""
        return self._m

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Lower Bound
    # ------------------------------------------------------------------

    @property
    def lower_bound(self) -> int:
        """
        Lower bound clássico para o PMSP com makespan:

            LB = max( max_j p_j ,  ceil( sum_j p_j / m ) )

        - O primeiro termo garante que nenhum job pode terminar antes
          do job mais longo.
        - O segundo termo é a carga média arredondada para cima (bound
          de capacidade).

        O resultado é cacheado para evitar recálculos em laços internos.
        """
        if self._lb is None:
            total = int(self._p.sum())
            self._lb = max(
                int(self._p.max()),        # job mais longo
                -(-total // self._m),      # carga media arredondada para cima --
                                            # ceil() EXATO em inteiros (equivalente a
                                            # math.ceil(total/m), mas sem passar por
                                            # float -- este perdia precisao para somas
                                            # extremas, ver Entrada 53 do diario)
            )
        return self._lb

    # ------------------------------------------------------------------
    # Persistência (JSON simples)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serializa a instância em dicionário (para JSON / pickle)."""
        return {
            "name": self._name,
            "n": self.n,
            "m": self._m,
            "p": self._p.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PMSPInstance":
        """Reconstrói uma instância a partir de um dicionário."""
        return cls(p=d["p"], m=d["m"], name=d.get("name", "unnamed"))

    def save(self, path: str | Path) -> None:
        """Salva a instância em formato JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PMSPInstance":
        """Carrega uma instância a partir de um arquivo JSON."""
        d = json.loads(Path(path).read_text())
        return cls.from_dict(d)

    # ------------------------------------------------------------------
    # Geração aleatória
    # ------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        n: int,
        m: int,
        low: int = 1,
        high: int = 100,
        seed: int | None = None,
        name: str | None = None,
    ) -> "PMSPInstance":
        """
        Gera uma instância aleatória com p_j ~ U[low, high].

        Parâmetros
        ----------
        n : int
            Número de jobs.
        m : int
            Número de máquinas.
        low, high : int
            Intervalo fechado para os tempos de processamento.
        seed : int | None
            Semente para reprodutibilidade.
        name : str | None
            Se None, gera automaticamente como "rand_{n}_{m}_s{seed}".
        """
        rng = np.random.default_rng(seed)
        p = rng.integers(low=low, high=high + 1, size=n)
        auto_name = name or f"rand_{n}j_{m}m_s{seed}"
        return cls(p=p, m=m, name=auto_name)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PMSPInstance(name={self._name!r}, n={self.n}, m={self._m}, "
            f"sum_p={int(self._p.sum())}, LB={self.lower_bound})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PMSPInstance):
            return NotImplemented
        return (
            self._m == other._m
            and np.array_equal(self._p, other._p)
        )
