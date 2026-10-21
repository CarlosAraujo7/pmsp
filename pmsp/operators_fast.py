"""
pmsp/operators_fast.py (Entrada 162, Ciclo 13)
================================================
Versoes VETORIZADAS (numpy) dos dois operadores de melhor melhora usados na busca local do IG hibrido de P||Cmax: troca entre maquinas (op1_best) e reinsercao (op2_best) de pmsp/operators.py.
Motivacao: as versoes originais sao lacos Python triplos com uma funcao O(m) no corpo; a razao de CPU entre o IG com busca local e sem ela (2,1 a 82,9) media, em parte, a ineficiencia da implementacao, e as conclusoes
'a CPU igual' dependem dessa razao (critica de revisor: o custo de CPU e' propriedade da implementacao). As versoes rapidas sao EXATAMENTE equivalentes: mesma ordem de varredura (tarefa da maquina gargalo, maquina, tarefa),
mesma regra de desempate (primeiro maximo estrito de delta, delta > 0) e mesmos retornos; a equivalencia e' testada em tests/test_operators_fast.py. Para tempos de processamento fora de int64 (familia B, dtype object) cai nas versoes originais.
Ideia: para uma troca/movimento entre a maquina gargalo a e a maquina b, o novo makespan e' max(nova carga de a, nova carga de b, maior carga entre as demais maquinas); o maior das demais se obtem dos dois maiores valores
de carga entre as maquinas != a (se b e' a maquina do maximo, usa-se o segundo maior).
"""
from __future__ import annotations

import numpy as np

from . import operators as _ops


def _other_max(loads: np.ndarray, a: int) -> np.ndarray:
    """other[b] = maior carga entre as maquinas k != a, k != b (0 se nao houver), para b != a (o valor em b == a nao e' usado)."""
    m = len(loads)
    rest_idx = np.delete(np.arange(m), a)
    rest = loads[rest_idx].astype(np.int64)
    out = np.zeros(m, dtype=np.int64)
    if len(rest) == 0:
        return out
    i1 = int(rest.argmax())
    m1 = int(rest[i1])
    m2 = int(np.delete(rest, i1).max()) if len(rest) > 1 else 0
    out[:] = m1
    out[rest_idx[i1]] = m2
    return out


def _exact_dtype(sol) -> bool:
    return sol.instance.p.dtype.kind in "iu" and sol.loads.dtype.kind in "iu"


def op1_best_fast(sol, rng) -> int:
    """Equivalente a operators.op1_best (melhor troca entre a maquina gargalo e as demais)."""
    if not _exact_dtype(sol):
        return _ops.op1_best(sol, rng)
    loads, p, mj = sol.loads, sol.instance.p, sol.machine_jobs
    a = sol.machine_max
    old = int(loads.max())
    if len(mj[a]) == 0:
        return 0
    others = [(b, jobs) for b, jobs in enumerate(mj) if b != a and len(jobs) > 0]
    if not others:
        return 0
    A = np.asarray(mj[a], dtype=np.int64)
    jb = np.concatenate([np.asarray(jobs, dtype=np.int64) for _, jobs in others])
    mb = np.concatenate([np.full(len(jobs), b, dtype=np.int64) for b, jobs in others])
    pa = p[A].astype(np.int64)[:, None]
    pb = p[jb].astype(np.int64)[None, :]
    la = int(loads[a])
    lb = loads[mb].astype(np.int64)[None, :]
    new_cmax = np.maximum(np.maximum(la - pa + pb, lb - pb + pa), _other_max(loads, a)[mb][None, :])
    delta = (old - new_cmax).ravel()
    k = int(delta.argmax())
    best = int(delta[k])
    if best <= 0:
        return 0
    ia, ib = divmod(k, len(jb))
    sol.swap_jobs(int(A[ia]), int(jb[ib]))
    return best


def op2_best_fast(sol, rng) -> int:
    """Equivalente a operators.op2_best (melhor reinsercao de uma tarefa da maquina gargalo)."""
    if not _exact_dtype(sol):
        return _ops.op2_best(sol, rng)
    loads, p, mj = sol.loads, sol.instance.p, sol.machine_jobs
    a = sol.machine_max
    m = len(loads)
    old = int(loads.max())
    if len(mj[a]) == 0 or m < 2:
        return 0
    A = np.asarray(mj[a], dtype=np.int64)
    bs = np.delete(np.arange(m), a)
    pj = p[A].astype(np.int64)[:, None]
    la = int(loads[a])
    lb = loads[bs].astype(np.int64)[None, :]
    new_cmax = np.maximum(np.maximum(la - pj, lb + pj), _other_max(loads, a)[bs][None, :])
    delta = (old - new_cmax).ravel()
    k = int(delta.argmax())
    best = int(delta[k])
    if best <= 0:
        return 0
    ia, ib = divmod(k, len(bs))
    sol.move_job(int(A[ia]), int(bs[ib]))
    return best


def op4_best_fast(sol, rng, max_block: int = 3) -> int:
    """Equivalente a operators.op4_best (melhor bloco de 1 a max_block jobs consecutivos da maquina gargalo movido para outra maquina)."""
    if not _exact_dtype(sol):
        return _ops.op4_best(sol, rng, max_block)
    loads, p, mj = sol.loads, sol.instance.p, sol.machine_jobs
    a = sol.machine_max
    m = len(loads)
    old = int(loads.max())
    jobs_a = mj[a]
    n_a = len(jobs_a)
    if m < 2 or n_a < 2:
        return 0
    pa = p[np.asarray(jobs_a, dtype=np.int64)].astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(pa)])
    loads_b, starts, sizes = [], [], []
    for size in range(1, min(max_block + 1, n_a)):                 # mesma ordem dos lacos: tamanho, inicio
        s = np.arange(n_a - size + 1)
        loads_b.append(cs[s + size] - cs[s])
        starts.append(s)
        sizes.append(np.full(len(s), size))
    bl = np.concatenate(loads_b)[:, None]
    st = np.concatenate(starts)
    sz = np.concatenate(sizes)
    bs = np.delete(np.arange(m), a)
    la = int(loads[a])
    lb = loads[bs].astype(np.int64)[None, :]
    new_cmax = np.maximum(np.maximum(la - bl, lb + bl), _other_max(loads, a)[bs][None, :])
    delta = (old - new_cmax).ravel()
    k = int(delta.argmax())
    best = int(delta[k])
    if best <= 0:
        return 0
    ib_, ibm = divmod(k, len(bs))
    start, size, target = int(st[ib_]), int(sz[ib_]), int(bs[ibm])
    for j in list(jobs_a[start:start + size]):
        sol.move_job(j, target)
    return best


def apply_best_fast(op_id: int, sol, rng) -> int:
    """Despacha como operators.apply_best, com os tres operadores caros vetorizados (op3 e' barato e fica como esta)."""
    if op_id == 0:
        return op1_best_fast(sol, rng)
    if op_id == 1:
        return op2_best_fast(sol, rng)
    if op_id == 3:
        return op4_best_fast(sol, rng)
    return _ops.apply_best(op_id, sol, rng)


def make_ls_fast(passes: int = 2):
    """Mesma busca local de run_igls_p_entrada124.make_ls (inter-troca e reinsercao, ate' 2 passes), com os operadores vetorizados."""
    def ls(sol, rng):
        for _ in range(passes):
            if op1_best_fast(sol, rng) <= 0 and op2_best_fast(sol, rng) <= 0:
                break
    return ls
