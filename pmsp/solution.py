"""
pmsp/solution.py
================
Define a classe PMSPSolution, que representa uma solução completa para
uma instância do PMSP e mantém estruturas de dados redundantes para
que os operadores de vizinhança (Fase 2) operem em O(1) ou O(log n).

Representação da solução
------------------------
Mantemos TRÊS visões sincronizadas, todas atualizadas juntas a cada
movimento, para evitar recálculos caros em laços internos:

    1. assignment[j]  → índice da máquina alocada ao job j   (shape: n)
    2. loads[i]       → carga total (soma dos p_j) da máquina i  (shape: m)
    3. machine_jobs[i]→ lista ordenada dos jobs na máquina i  (lista de listas)

Makespan:
    C_max = max(loads)   ← acesso O(1) via NumPy

Invariante fundamental:
    Para todo i: loads[i] == sum(instance.p[j] for j in machine_jobs[i])
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np

from .instance import PMSPInstance


class PMSPSolution:
    """
    Solução para o PMSP com manutenção eficiente de carga por máquina.

    Parâmetros
    ----------
    instance : PMSPInstance
        Instância do problema à qual esta solução pertence.
    assignment : array-like de inteiros, shape (n,), opcional
        assignment[j] = índice da máquina do job j (0-indexado).
        Se None, cria solução vazia (todos os jobs sem alocação).
    """

    def __init__(
        self,
        instance: PMSPInstance,
        assignment: np.ndarray | None = None,
    ) -> None:
        self._inst = instance
        n, m = instance.n, instance.m

        if assignment is not None:
            asgn = np.asarray(assignment, dtype=np.int32)
            if asgn.shape != (n,):
                raise ValueError(
                    f"assignment deve ter shape ({n},), recebido {asgn.shape}"
                )
            if asgn.min() < 0 or asgn.max() >= m:
                raise ValueError(
                    f"Índices de máquina devem estar em [0, {m-1}]."
                )
            self._assignment = asgn.copy()
        else:
            # Solução vazia: assignment inválido, usado apenas como rascunho
            self._assignment = np.full(n, -1, dtype=np.int32)

        # Reconstrói cargas e listas de jobs a partir de assignment
        self._loads = np.zeros(m, dtype=self._inst.p.dtype)  # ver Entrada 53
        self._machine_jobs: List[List[int]] = [[] for _ in range(m)]
        self._rebuild_from_assignment()

    # ------------------------------------------------------------------
    # Reconstrução interna
    # ------------------------------------------------------------------

    def _rebuild_from_assignment(self) -> None:
        """
        Reconstrói loads[] e machine_jobs[] a partir de self._assignment.
        Chamado no construtor e após operações de carga em lote.
        Complexidade: O(n).
        """
        m = self._inst.m
        self._loads[:] = 0
        self._machine_jobs = [[] for _ in range(m)]
        for j, mach in enumerate(self._assignment):
            if mach >= 0:
                self._loads[mach] += self._inst.p[j]
                self._machine_jobs[mach].append(j)

    # ------------------------------------------------------------------
    # Propriedades de leitura
    # ------------------------------------------------------------------

    @property
    def instance(self) -> PMSPInstance:
        return self._inst

    @property
    def assignment(self) -> np.ndarray:
        """Vetor de alocação job→máquina (cópia somente leitura)."""
        return self._assignment.copy()

    @property
    def loads(self) -> np.ndarray:
        """Cargas atuais de cada máquina (view, não modificar diretamente)."""
        return self._loads

    @property
    def machine_jobs(self) -> List[List[int]]:
        """Listas de jobs por máquina (referência interna — não modificar)."""
        return self._machine_jobs

    # ------------------------------------------------------------------
    # Makespan e métricas derivadas
    # ------------------------------------------------------------------

    @property
    def makespan(self) -> int:
        """
        C_max = max_i { loads[i] }
        Acesso O(1) pois loads[] é mantido incrementalmente.
        """
        return int(self._loads.max())

    @property
    def machine_min(self) -> int:
        """Índice da máquina com menor carga (útil para Op2)."""
        return int(self._loads.argmin())

    @property
    def machine_max(self) -> int:
        """Índice da máquina com maior carga (= máquina gargalo)."""
        return int(self._loads.argmax())

    @property
    def load_std(self) -> float:
        """Desvio padrão das cargas (usado em f2 do espaço de estados)."""
        return float(self._loads.std())

    @property
    def load_mean(self) -> float:
        """Média das cargas."""
        return float(self._loads.mean())

    @property
    def relative_gap(self) -> float:
        """
        f1 do espaço de estados Q-Learning:
            gap = (C_max - LB) / LB
        Mede o quão longe estamos do ótimo teórico.
        """
        lb = self._inst.lower_bound
        return (self.makespan - lb) / lb if lb > 0 else 0.0

    @property
    def load_cv(self) -> float:
        """
        f2 do espaço de estados Q-Learning:
            CV = std(loads) / mean(loads)
        Coeficiente de variação — desequilíbrio relativo de carga.
        """
        mu = self.load_mean
        return self.load_std / mu if mu > 0 else 0.0

    # ------------------------------------------------------------------
    # Primitivas de movimento (chamadas pelos operadores de vizinhança)
    # ------------------------------------------------------------------

    def move_job(self, job: int, target_machine: int) -> None:
        """
        Move o job `job` da sua máquina atual para `target_machine`.
        Atualiza assignment, loads e machine_jobs em O(n_i) onde n_i
        é o número de jobs na máquina de origem.

        Parâmetros
        ----------
        job : int
            Índice do job a mover (0-indexado).
        target_machine : int
            Índice da máquina de destino.
        """
        src = int(self._assignment[job])
        if src == target_machine:
            return  # movimento nulo — sem efeito

        pj = int(self._inst.p[job])

        # Atualiza cargas
        self._loads[src] -= pj
        self._loads[target_machine] += pj

        # Atualiza listas de jobs
        self._machine_jobs[src].remove(job)
        self._machine_jobs[target_machine].append(job)

        # Atualiza vetor de alocação
        self._assignment[job] = target_machine

    def swap_jobs(self, job_a: int, job_b: int) -> None:
        """
        Troca as alocações de dois jobs (possivelmente em máquinas distintas).
        Atualiza loads em O(1) e machine_jobs em O(n_i).

        Parâmetros
        ----------
        job_a, job_b : int
            Índices dos dois jobs a trocar.
        """
        mach_a = int(self._assignment[job_a])
        mach_b = int(self._assignment[job_b])

        if mach_a == mach_b:
            # Swap intra-máquina: cargas não mudam, apenas ordem na lista
            ia = self._machine_jobs[mach_a].index(job_a)
            ib = self._machine_jobs[mach_a].index(job_b)
            self._machine_jobs[mach_a][ia] = job_b
            self._machine_jobs[mach_a][ib] = job_a
            self._assignment[job_a] = mach_b   # mesmo, mas mantém consistência
            self._assignment[job_b] = mach_a
        else:
            # Swap inter-máquinas: diferença de carga = pa - pb
            pa = int(self._inst.p[job_a])
            pb = int(self._inst.p[job_b])
            delta = pa - pb   # mach_a perde pa e ganha pb => -delta

            self._loads[mach_a] += pb - pa   # equivalente: -= delta
            self._loads[mach_b] += pa - pb   # equivalente: += delta

            # Atualiza listas
            self._machine_jobs[mach_a].remove(job_a)
            self._machine_jobs[mach_a].append(job_b)
            self._machine_jobs[mach_b].remove(job_b)
            self._machine_jobs[mach_b].append(job_a)

            # Atualiza assignment
            self._assignment[job_a] = mach_b
            self._assignment[job_b] = mach_a

    # ------------------------------------------------------------------
    # Cópia profunda eficiente
    # ------------------------------------------------------------------

    def copy(self) -> "PMSPSolution":
        """
        Retorna uma cópia profunda desta solução.
        Usar em vez de copy.deepcopy() para maior eficiência.
        """
        new_sol = PMSPSolution.__new__(PMSPSolution)
        new_sol._inst = self._inst          # instância é imutável: referência ok
        new_sol._assignment = self._assignment.copy()
        new_sol._loads = self._loads.copy()
        new_sol._machine_jobs = [lst.copy() for lst in self._machine_jobs]
        return new_sol

    # ------------------------------------------------------------------
    # Validação de invariante (usada nos testes e em modo debug)
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """
        Verifica se as três representações estão sincronizadas.
        Deve retornar True após qualquer operação válida.
        """
        n, m = self._inst.n, self._inst.m
        p = self._inst.p

        # 1. Todo job está alocado a uma máquina válida
        if np.any(self._assignment < 0) or np.any(self._assignment >= m):
            return False

        # 2. machine_jobs deve cobrir exatamente os n jobs (sem duplicatas)
        all_jobs = [j for lst in self._machine_jobs for j in lst]
        if sorted(all_jobs) != list(range(n)):
            return False

        # 3. machine_jobs deve ser consistente com assignment
        for i, jobs in enumerate(self._machine_jobs):
            for j in jobs:
                if self._assignment[j] != i:
                    return False

        # 4. Cargas devem corresponder à soma dos tempos de processamento
        expected_loads = np.array(
            [sum(p[j] for j in self._machine_jobs[i]) for i in range(m)],
            dtype=self._inst.p.dtype,  # ver Entrada 53
        )
        if not np.array_equal(self._loads, expected_loads):
            return False

        return True

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PMSPSolution(makespan={self.makespan}, "
            f"LB={self._inst.lower_bound}, "
            f"gap={self.relative_gap:.4f}, "
            f"loads={self._loads.tolist()})"
        )
