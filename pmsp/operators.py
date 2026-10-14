"""
pmsp/operators.py
=================
Quatro operadores de vizinhanca para o PMSP.

Cada operador expoe DUAS funcoes:
    opX_best(sol, rng)   -> int  (best-improvement: aplica o melhor movimento)
    opX_rand(sol, rng)   -> int  (random-move: aplica um movimento aleatorio)

Retorno: melhora no makespan (delta = old - new >= 0).
         0 indica que nenhuma melhora foi encontrada/aplicada.

A solucao e modificada IN-PLACE. O codigo chamador deve copiar antes:
    s_new = s.copy(); delta = op1_best(s_new, rng)

Operadores implementados
------------------------
Op1 - Inter-machine swap  : troca dois jobs entre maquinas distintas.
Op2 - Single reinsertion  : move um job para a maquina menos carregada.
Op3 - Intra-machine 2-opt : inverte a ordem de dois jobs na mesma maquina
                            (neutro para makespan puro; preparatorio para setup).
Op4 - Block move          : move um bloco consecutivo de jobs para outra maquina.

Interface unificada
-------------------
OPERATORS_BEST = [op1_best, op2_best, op3_best, op4_best]
OPERATORS_RAND = [op1_rand, op2_rand, op3_rand, op4_rand]
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .solution import PMSPSolution


# ===========================================================================
# Utilitarios internos
# ===========================================================================

def _new_makespan_after_loads(loads: np.ndarray, i: int, li: int,
                               j: int, lj: int) -> int:
    """
    Calcula o novo makespan apos alterar as cargas das maquinas i e j
    para li e lj, respectivamente, sem modificar o array original.

    Complexidade: O(m).
    """
    cmax = 0
    for k in range(len(loads)):
        if k == i:
            v = li
        elif k == j:
            v = lj
        else:
            v = loads[k]
        if v > cmax:
            cmax = v
    return cmax


# ===========================================================================
# Op1 -- Inter-machine swap
# ===========================================================================

def op1_best(sol: PMSPSolution, rng: np.random.Generator) -> int:
    """
    Melhor troca de dois jobs entre maquinas distintas.

    Estrategia: itera sobre todos os jobs da maquina gargalo (max load)
    e tenta trocar com cada job de cada outra maquina. Aplica a troca
    com maior reducao de makespan.

    Complexidade: O(n_bottleneck * n_others * m) ~ O(n^2 / m * m) = O(n^2).
    """
    loads = sol.loads           # referencia (nao copia)
    p = sol.instance.p
    mach_jobs = sol.machine_jobs

    best_delta = 0
    best_ja = best_jb = -1

    mach_a = sol.machine_max    # maquina gargalo
    old_makespan = int(loads.max())

    for ja in mach_jobs[mach_a]:
        pa = int(p[ja])
        for mach_b, jobs_b in enumerate(mach_jobs):
            if mach_b == mach_a or len(jobs_b) == 0:
                continue
            la = int(loads[mach_a])
            lb = int(loads[mach_b])
            for jb in jobs_b:
                pb = int(p[jb])
                # Cargas apos troca
                new_la = la - pa + pb
                new_lb = lb - pb + pa
                new_cmax = _new_makespan_after_loads(
                    loads, mach_a, new_la, mach_b, new_lb)
                delta = old_makespan - new_cmax
                if delta > best_delta:
                    best_delta = delta
                    best_ja, best_jb = ja, jb

    if best_ja >= 0:
        sol.swap_jobs(best_ja, best_jb)
    return best_delta


def op1_rand(sol: PMSPSolution, rng: np.random.Generator) -> int:
    """
    Troca aleatoria de dois jobs de maquinas distintas.
    Usada como perturbacao no ILS.
    """
    m = sol.instance.m
    if m < 2:
        return 0
    mach_jobs = sol.machine_jobs
    # Seleciona duas maquinas nao-vazias distintas
    non_empty = [i for i in range(m) if len(mach_jobs[i]) > 0]
    if len(non_empty) < 2:
        return 0
    mach_a, mach_b = rng.choice(non_empty, size=2, replace=False)
    ja = int(rng.choice(mach_jobs[mach_a]))
    jb = int(rng.choice(mach_jobs[mach_b]))
    old = sol.makespan
    sol.swap_jobs(ja, jb)
    return max(0, old - sol.makespan)


# ===========================================================================
# Op2 -- Single reinsertion
# ===========================================================================

def op2_best(sol: PMSPSolution, rng: np.random.Generator) -> int:
    """
    Move o melhor job da maquina gargalo para a maquina de menor carga,
    ou para qualquer maquina que reduza o makespan.

    Estrategia: para cada job na maquina gargalo, avalia mover para
    cada outra maquina e registra o melhor delta.

    Complexidade: O(n_bottleneck * m^2) ~ O(n/m * m^2) = O(n*m).
    """
    loads = sol.loads
    p = sol.instance.p
    mach_jobs = sol.machine_jobs
    mach_a = sol.machine_max
    old_makespan = int(loads.max())

    best_delta = 0
    best_j = -1
    best_target = -1

    la = int(loads[mach_a])
    for j in mach_jobs[mach_a]:
        pj = int(p[j])
        new_la = la - pj
        for mach_b in range(sol.instance.m):
            if mach_b == mach_a:
                continue
            lb = int(loads[mach_b])
            new_lb = lb + pj
            new_cmax = _new_makespan_after_loads(
                loads, mach_a, new_la, mach_b, new_lb)
            delta = old_makespan - new_cmax
            if delta > best_delta:
                best_delta = delta
                best_j, best_target = j, mach_b

    if best_j >= 0:
        sol.move_job(best_j, best_target)
    return best_delta


def op2_rand(sol: PMSPSolution, rng: np.random.Generator) -> int:
    """
    Move um job aleatorio da maquina gargalo para a maquina menos carregada.
    """
    mach_a = sol.machine_max
    jobs_a = sol.machine_jobs[mach_a]
    if len(jobs_a) == 0:
        return 0
    j = int(rng.choice(jobs_a))
    target = sol.machine_min
    if target == mach_a:
        # Tenta outra maquina
        candidates = [i for i in range(sol.instance.m) if i != mach_a]
        if not candidates:
            return 0
        target = int(rng.choice(candidates))
    old = sol.makespan
    sol.move_job(j, target)
    return max(0, old - sol.makespan)


# ===========================================================================
# Op3 -- Intra-machine 2-opt
# ===========================================================================

def op3_best(sol: PMSPSolution, rng: np.random.Generator) -> int:
    """
    Inverte a ordem de dois jobs dentro da mesma maquina.

    Para o PMSP puro (sem setup dependente de sequencia), esta operacao
    nao altera o makespan (delta = 0 sempre). E incluida como preparatorio
    para extensoes com tempos de setup e como operador de diversificacao.

    Aplica a inversao na maquina com mais jobs (maior potencial de exploracao
    de sequencias em versoes futuras do modelo).
    """
    mach_jobs = sol.machine_jobs
    # Seleciona a maquina com mais jobs
    best_mach = max(range(sol.instance.m),
                    key=lambda i: len(mach_jobs[i]))
    jobs = mach_jobs[best_mach]
    if len(jobs) < 2:
        return 0

    # Tenta todos os pares e aplica o primeiro (semanticamente neutro)
    ia = 0
    ib = 1
    ja, jb = jobs[ia], jobs[ib]
    old = sol.makespan
    sol.swap_jobs(ja, jb)
    return max(0, old - sol.makespan)  # sempre 0 no PMSP puro


def op3_rand(sol: PMSPSolution, rng: np.random.Generator) -> int:
    """
    Troca aleatoria de dois jobs dentro de uma maquina aleatoria.
    """
    m = sol.instance.m
    mach_jobs = sol.machine_jobs
    candidates = [i for i in range(m) if len(mach_jobs[i]) >= 2]
    if not candidates:
        return 0
    mach = int(rng.choice(candidates))
    jobs = mach_jobs[mach]
    ia, ib = rng.choice(len(jobs), size=2, replace=False)
    ja, jb = jobs[ia], jobs[ib]
    old = sol.makespan
    sol.swap_jobs(ja, jb)
    return max(0, old - sol.makespan)


# ===========================================================================
# Op4 -- Block move
# ===========================================================================

def op4_best(sol: PMSPSolution, rng: np.random.Generator,
             max_block: int = 3) -> int:
    """
    Move o melhor bloco consecutivo de jobs de uma maquina para outra.

    Um bloco e um subconjunto contiguo de jobs em machine_jobs[i].
    Tamanho do bloco: 1 ate min(max_block, len(machine_jobs[i])-1).

    Estrategia: itera sobre a maquina gargalo, todos os blocos de
    tamanho 1..max_block, e todos os destinos possiveis.

    Complexidade: O(max_block * n/m * m) = O(max_block * n).
    """
    loads = sol.loads
    p = sol.instance.p
    mach_jobs = sol.machine_jobs
    mach_a = sol.machine_max
    old_makespan = int(loads.max())
    jobs_a = mach_jobs[mach_a]

    best_delta = 0
    best_start = best_size = best_target = -1

    la = int(loads[mach_a])
    n_a = len(jobs_a)

    for size in range(1, min(max_block + 1, n_a)):
        for start in range(n_a - size + 1):
            block = jobs_a[start:start + size]
            block_load = int(sum(p[j] for j in block))
            new_la = la - block_load

            for mach_b in range(sol.instance.m):
                if mach_b == mach_a:
                    continue
                lb = int(loads[mach_b])
                new_lb = lb + block_load
                new_cmax = _new_makespan_after_loads(
                    loads, mach_a, new_la, mach_b, new_lb)
                delta = old_makespan - new_cmax
                if delta > best_delta:
                    best_delta = delta
                    best_start, best_size, best_target = start, size, mach_b

    if best_start >= 0:
        block = list(mach_jobs[mach_a][best_start:best_start + best_size])
        for j in block:
            sol.move_job(j, best_target)
    return best_delta


def op4_rand(sol: PMSPSolution, rng: np.random.Generator,
             max_block: int = 3) -> int:
    """
    Move um bloco aleatorio de jobs da maquina gargalo para outra maquina.
    """
    mach_a = sol.machine_max
    jobs_a = sol.machine_jobs[mach_a]
    n_a = len(jobs_a)
    if n_a < 2:
        return 0
    size = int(rng.integers(1, min(max_block, n_a - 1) + 1))
    start = int(rng.integers(0, n_a - size + 1))
    block = list(jobs_a[start:start + size])

    candidates = [i for i in range(sol.instance.m) if i != mach_a]
    if not candidates:
        return 0
    target = int(rng.choice(candidates))
    old = sol.makespan
    for j in block:
        sol.move_job(j, target)
    return max(0, old - sol.makespan)


# ===========================================================================
# Registro central de operadores
# ===========================================================================

#: Lista de funcoes best-improvement (indexada por ID do operador 0..3)
OPERATORS_BEST = [op1_best, op2_best, op3_best, op4_best]

#: Lista de funcoes random-move para perturbacao (indexada por ID 0..3)
OPERATORS_RAND = [op1_rand, op2_rand, op3_rand, op4_rand]

#: Nomes legíveis dos operadores (para logs e visualizacoes)
OPERATOR_NAMES = [
    "Op1-InterSwap",
    "Op2-Reinsertion",
    "Op3-IntraSwap",
    "Op4-BlockMove",
]

N_OPERATORS = len(OPERATORS_BEST)


def apply_best(op_id: int, sol: PMSPSolution,
               rng: np.random.Generator) -> int:
    """Despacha para o operador best-improvement pelo seu indice."""
    return OPERATORS_BEST[op_id](sol, rng)


def apply_rand(op_id: int, sol: PMSPSolution,
               rng: np.random.Generator) -> int:
    """Despacha para o operador random-move pelo seu indice."""
    return OPERATORS_RAND[op_id](sol, rng)
