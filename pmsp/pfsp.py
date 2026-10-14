"""
pmsp/pfsp.py (Entrada 151)
===========================
Flowshop de permutacao (PFSP, Fm|prmu|Cmax) para o TERCEIRO problema do estudo de atribuicao 'mecanismo x politica': o IG de Ruiz e Stutzle (2007) e o ILS com a mesma busca local de insercao, ambos
com o gancho de selecao de acoes (tamanho de destruicao d) usado nos outros problemas. Somente numpy (sem numba): as cabecas e caudas usam o prefixo de maximos
  C[k, i] = S_i[k] + max_{l <= k} (C[l, i-1] - S_i[l-1]),   S_i = somas acumuladas de p[., i] na ordem da sequencia,
e a aceleracao de Taillard avalia todas as posicoes de insercao de uma tarefa em O(m) operacoes vetorizadas.
  PfsInstance.random(n, m, seed)                instancia 'estilo Taillard' (p ~ U{1..99}); lower_bound = max_i (min cabeca + soma + min cauda)
  makespan(p, perm), heads(p, perm), tails(p, perm)
  best_insertion(p, perm, job, rng)             melhor posicao de insercao de `job` em `perm` (sem ele) -> (posicao, makespan)
  neh(p)                                        NEH com aceleracao de Taillard
  local_search(p, perm, rng)                    descida de insercao (ordem aleatoria; ate nao melhorar)
  pfsp_ig_run(...), pfsp_ils_run(...)           ver docstrings
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np


class PfsInstance:
    def __init__(self, p: np.ndarray, name: str = "pfsp") -> None:
        p = np.asarray(p, dtype=np.int64)
        assert p.ndim == 2 and p.min() >= 1
        self.p = p                       # (n, m): tempo da tarefa j na maquina i
        self.n, self.m = p.shape
        self.name = name
        # limite inferior classico (Taillard 1993): max_i ( min_j soma das maquinas antes de i + soma_j p_ji + min_j soma das maquinas depois de i )
        pre = np.cumsum(p, axis=1) - p
        post = np.cumsum(p[:, ::-1], axis=1)[:, ::-1] - p
        self.lower_bound = int(max(pre[:, i].min() + p[:, i].sum() + post[:, i].min() for i in range(self.m)))

    @staticmethod
    def random(n: int, m: int, seed: int, name: Optional[str] = None) -> "PfsInstance":
        rng = np.random.default_rng(seed)
        return PfsInstance(rng.integers(1, 100, size=(n, m)), name or f"pfsp_{n}x{m}_s{seed}")


def heads(p: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """e[k, i]: conclusao da k-esima tarefa da sequencia (k = 1..K) na maquina i; e[0, :] = 0. Forma (K + 1, m)."""
    q = p[perm]
    K, m = q.shape
    e = np.zeros((K + 1, m), dtype=np.int64)
    prev = np.zeros(K, dtype=np.int64)
    for i in range(m):
        s = np.cumsum(q[:, i])
        e[1:, i] = s + np.maximum.accumulate(prev - (s - q[:, i]))
        prev = e[1:, i]
    return e


def tails(p: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """t[k, i]: caminho mais longo a partir da k-esima tarefa (k = 0..K-1), inclusive, ate' o fim; t[K, :] = 0. Forma (K + 1, m)."""
    K = len(perm)
    r = heads(p[:, ::-1], perm[::-1])                  # problema invertido (tarefas e maquinas): cabecas = caudas
    t = np.zeros((K + 1, p.shape[1]), dtype=np.int64)
    t[:K] = r[1:][::-1][:, ::-1]
    return t


def makespan(p: np.ndarray, perm: np.ndarray) -> int:
    return int(heads(p, perm)[-1, -1])


def best_insertion(p: np.ndarray, perm: np.ndarray, job: int, rng=None, e=None, t=None):
    """Melhor posicao (0..K) para inserir `job` em `perm` (K tarefas); empates: primeira posicao (rng=None) ou sorteio. Devolve (posicao, makespan)."""
    K = len(perm)
    if e is None:
        e = heads(p, perm)
    if t is None:
        t = tails(p, perm)
    m = p.shape[1]
    pj = p[job]
    f = np.empty((K + 1, m), dtype=np.int64)
    f[:, 0] = e[:, 0] + pj[0]
    for i in range(1, m):
        f[:, i] = np.maximum(f[:, i - 1], e[:, i]) + pj[i]
    ms = (f + t).max(axis=1)
    best = ms.min()
    cand = np.flatnonzero(ms == best)
    pos = int(cand[0]) if rng is None or len(cand) == 1 else int(cand[rng.integers(len(cand))])
    return pos, int(best)


def neh(p: np.ndarray) -> np.ndarray:
    n = p.shape[0]
    order = np.argsort(-p.sum(axis=1), kind="stable")
    perm = np.array([order[0]], dtype=np.int64)
    for job in order[1:]:
        pos, _ = best_insertion(p, perm, int(job))
        perm = np.insert(perm, pos, job)
    return perm


def local_search(p: np.ndarray, perm: np.ndarray, rng, cur: Optional[int] = None) -> tuple:
    """Descida de insercao (Ruiz e Stutzle): ate' nenhuma tarefa, em ordem aleatoria, melhorar o makespan. Devolve (perm, makespan)."""
    perm = perm.copy()
    cur = makespan(p, perm) if cur is None else cur
    improved = True
    while improved:
        improved = False
        for job in rng.permutation(len(perm)):
            idx = int(np.flatnonzero(perm == job)[0])
            rest = np.delete(perm, idx)
            pos, ms = best_insertion(p, rest, int(job), rng=None)
            if ms < cur:
                perm = np.insert(rest, pos, job)
                cur = ms
                improved = True
    return perm, cur


def _accept(new: int, cur: int, temp: float, rng) -> bool:
    if new <= cur:
        return True
    return bool(rng.random() < np.exp(-(new - cur) / temp))


def pfsp_ig_run(inst: PfsInstance, selector, d_choices, max_iter: int = 1000, seed: int = 0, state_fn=None, use_ls: bool = True, tp: float = 0.4) -> Dict[str, Any]:
    """IG de Ruiz e Stutzle: NEH; a cada iteracao o `selector` escolhe uma acao a -> tamanho de destruicao d = d_choices[a]; remove d tarefas aleatorias, reinsere pela melhor posicao (aceleracao de
    Taillard), busca local de insercao (se use_ls), aceitacao com temperatura constante T = tp * soma(p) / (n * m * 10). Recompensa = max(melhora do makespan corrente, 0)."""
    rng = np.random.default_rng(seed)
    p, n = inst.p, inst.n
    selector.reset()
    temp = tp * p.sum() / (n * inst.m * 10.0)
    perm = neh(p)
    cur = makespan(p, perm)
    init = cur
    best, best_perm = cur, perm.copy()
    no_imp = 0
    t0 = time.perf_counter()
    for t in range(max_iter):
        s_id = state_fn(cur, best, init, t, max_iter, no_imp) if state_fn is not None else 0
        feats = np.array([0.0, 0.0, t / max_iter, min(no_imp / 50.0, 1.0)])
        a = selector.select(s_id, feats)
        d = int(d_choices[a])
        removed = rng.choice(n, size=d, replace=False)
        newp = perm[~np.isin(perm, removed)]
        for job in rng.permutation(removed):
            pos, _ = best_insertion(p, newp, int(job), rng=None)
            newp = np.insert(newp, pos, job)
        new_ms = makespan(p, newp)
        if use_ls:
            newp, new_ms = local_search(p, newp, rng, new_ms)
        reward = float(max(cur - new_ms, 0))
        ns_id = state_fn(new_ms if new_ms < cur else cur, best, init, t + 1, max_iter, no_imp) if state_fn is not None else 0
        selector.update(s_id, feats, a, reward, ns_id)
        if _accept(new_ms, cur, temp, rng):
            perm, cur = newp, new_ms
        if cur < best:
            best, best_perm = cur, perm.copy()
            no_imp = 0
        else:
            no_imp += 1
    cpu = time.perf_counter() - t0
    return {"makespan": best, "cpu_time": cpu, "perm": best_perm, "gap_pct": 100.0 * (best - inst.lower_bound) / inst.lower_bound}


def pfsp_ils_run(inst: PfsInstance, max_iter: int = 1000, seed: int = 0, n_perturb: int = 2, tp: float = 0.4) -> Dict[str, Any]:
    """ILS com a mesma busca local de insercao e perturbacao por `n_perturb` insercoes aleatorias; mesma aceitacao do IG. Sem selecao de acao."""
    rng = np.random.default_rng(seed)
    p, n = inst.p, inst.n
    temp = tp * p.sum() / (n * inst.m * 10.0)
    perm = neh(p)
    perm, cur = local_search(p, perm, rng)
    best, best_perm = cur, perm.copy()
    t0 = time.perf_counter()
    for t in range(max_iter):
        newp = perm.copy()
        for _ in range(n_perturb):
            i, j = int(rng.integers(n)), int(rng.integers(n))
            job = newp[i]
            newp = np.insert(np.delete(newp, i), j, job)
        newp, new_ms = local_search(p, newp, rng)
        if _accept(new_ms, cur, temp, rng):
            perm, cur = newp, new_ms
        if cur < best:
            best, best_perm = cur, perm.copy()
    cpu = time.perf_counter() - t0
    return {"makespan": best, "cpu_time": cpu, "perm": best_perm, "gap_pct": 100.0 * (best - inst.lower_bound) / inst.lower_bound}
