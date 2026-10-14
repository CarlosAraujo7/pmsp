"""
pmsp/unrelated.py (Entrada 118)
================================
Extensao do protocolo "mecanismo vs politica" para maquinas NAO RELACIONADAS (R||Cmax): n tarefas, m maquinas, tempo p[i, j] da tarefa j
na maquina i; minimizar o makespan. E' a variante em que os metodos exatos nao acompanham o tamanho (MILP com milhares de binarias),
sobre a qual existe literatura de IG (Fanjul-Peyro & Ruiz, EJOR 2010).

Conteudo (tudo deterministico dado o seed):
  UnrelatedInstance             p inteiro (m, n); lower_bound classico max(max_j min_i p_ij, ceil(sum_j min_i p_ij / m))
  mct_greedy                    construtiva: tarefas por min_i p_ij decrescente, cada uma na maquina de menor conclusao
  ur_ig_run                     Iterated Greedy (destruir d tarefas / reconstruir guloso) com 4 estrategias de destruicao
                                (0 aleatoria, 1 mais cara na maquina atual, 2 concentrada numa maquina sorteada, 3 maquina critica),
                                selecao por qualquer BaseSelector (QL, aleatoria, UCB, fixa), aceitacao gulosa, recompensa max(delta, 0)
  ur_ils_run                    ILS (busca local de melhor melhoria em 3 operadores a partir da maquina critica + perturbacao
                                aleatoria de 3 tarefas apos `perturbation_threshold` iteracoes sem melhora)
  UrStateFn                     estado de 81 bins (f1 gap e f2 CV das cargas RELATIVOS ao valor inicial, com limiares de calibracao;
                                f3 progresso; f4 estagnacao), como o `rel` da Entrada 98
Os mesmos QLearningAgent / RandomSelector / NormalisedUCBSelector / FixedActionSelector do P||Cmax sao reutilizados.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

N_UR_DESTROY = 4
N_UR_OPS = 3


class UnrelatedInstance:
    def __init__(self, p: np.ndarray, name: str = "ur") -> None:
        p = np.asarray(p, dtype=np.int64)
        assert p.ndim == 2 and p.min() >= 1
        self.p = p
        self.name = name
        self.m, self.n = p.shape
        pm = p.min(axis=0)
        self.lower_bound = int(max(pm.max(), -(-int(pm.sum()) // self.m)))

    @staticmethod
    def random(n: int, m: int, low: int, high: int, seed: int, name: Optional[str] = None) -> "UnrelatedInstance":
        rng = np.random.default_rng(seed)
        return UnrelatedInstance(rng.integers(low, high + 1, size=(m, n)), name or f"ur_{n}j_{m}m_U{low}_{high}_s{seed}")


class UrSol:
    def __init__(self, inst: UnrelatedInstance, assignment: np.ndarray) -> None:
        self.inst = inst
        self.a = np.asarray(assignment, dtype=np.int64).copy()
        self.loads = np.zeros(inst.m, dtype=np.int64)
        np.add.at(self.loads, self.a, inst.p[self.a, np.arange(inst.n)])

    @property
    def makespan(self) -> int:
        return int(self.loads.max())

    def copy(self) -> "UrSol":
        s = UrSol.__new__(UrSol)
        s.inst, s.a, s.loads = self.inst, self.a.copy(), self.loads.copy()
        return s

    def move(self, j: int, k: int) -> None:
        i = self.a[j]
        if i == k:
            return
        self.loads[i] -= self.inst.p[i, j]
        self.loads[k] += self.inst.p[k, j]
        self.a[j] = k

    def check(self) -> bool:
        ref = np.zeros(self.inst.m, dtype=np.int64)
        np.add.at(ref, self.a, self.inst.p[self.a, np.arange(self.inst.n)])
        return bool((ref == self.loads).all())


def mct_greedy(inst: UnrelatedInstance) -> UrSol:
    order = np.argsort(-inst.p.min(axis=0), kind="stable")
    loads = np.zeros(inst.m, dtype=np.int64)
    a = np.zeros(inst.n, dtype=np.int64)
    for j in order:
        k = int(np.argmin(loads + inst.p[:, j]))
        a[j] = k
        loads[k] += inst.p[k, j]
    return UrSol(inst, a)


# ------------------------------------------------------------------ destruicao / reconstrucao
def _d_random(sol, d, rng):
    return list(rng.choice(sol.inst.n, size=min(d, sol.inst.n), replace=False))


def _d_expensive(sol, d, rng):
    cost = sol.inst.p[sol.a, np.arange(sol.inst.n)]
    return list(np.argsort(-cost, kind="stable")[:d])


def _from_machine(sol, d, rng, machine):
    n = sol.inst.n
    on = np.where(sol.a == machine)[0]
    if len(on) >= d:
        return list(rng.choice(on, size=d, replace=False))
    other = np.setdiff1d(np.arange(n), on)
    extra = rng.choice(other, size=min(d - len(on), len(other)), replace=False) if d > len(on) and len(other) else np.array([], dtype=int)
    return list(on) + list(extra)


def _d_machine(sol, d, rng):
    return _from_machine(sol, d, rng, int(rng.integers(sol.inst.m)))


def _d_critical(sol, d, rng):
    return _from_machine(sol, d, rng, int(np.argmax(sol.loads)))


_DESTROY = [_d_random, _d_expensive, _d_machine, _d_critical]
DESTROY_NAMES = ["Random", "MostExpensive", "MachineConcentrated", "CriticalMachine"]


def destroy_reconstruct(strategy: int, sol: UrSol, d: int, rng) -> UrSol:
    new = sol.copy()
    removed = [int(j) for j in _DESTROY[strategy](new, d, rng)]
    p = new.inst.p
    tracker = new.loads.copy()
    for j in removed:
        tracker[new.a[j]] -= p[new.a[j], j]
    order = sorted(removed, key=lambda j: (-int(p[:, j].min()), j))
    for j in order:
        k = int(np.argmin(tracker + p[:, j]))
        tracker[k] += p[k, j]
        new.move(j, k)
    return new


# ------------------------------------------------------------------ estado
class UrStateFn:
    """Estado de 81 bins; f1 e f2 RELATIVOS ao valor no primeiro passo da execucao (limiares thr = {'f1': [t1, t2], 'f2': [t1, t2]})."""

    def __init__(self, thr: Dict[str, List[float]], window: int = 50) -> None:
        self.thr, self.window = thr, window
        self.f10 = self.f20 = None

    @staticmethod
    def features(sol: UrSol):
        lb = sol.inst.lower_bound
        f1 = (sol.makespan - lb) / lb
        mean = sol.loads.mean()
        f2 = float(sol.loads.std() / mean) if mean > 0 else 0.0
        return f1, f2

    def __call__(self, sol: UrSol, t: int, max_iter: int, no_improve: int) -> int:
        f1, f2 = self.features(sol)
        if self.f10 is None:
            self.f10, self.f20 = f1, f2
        r1 = f1 / self.f10 if self.f10 > 0 else 0.0
        r2 = f2 / self.f20 if self.f20 > 0 else 0.0
        b1 = int(np.digitize(r1, self.thr["f1"]))
        b2 = int(np.digitize(r2, self.thr["f2"]))
        b3 = int(np.digitize(min(t / max_iter, 1.0), [0.33, 0.67]))
        b4 = int(np.digitize(min(no_improve / self.window, 1.0), [0.25, 0.60]))
        return b1 * 27 + b2 * 9 + b3 * 3 + b4


def default_state_fn(sol, t, max_iter, no_improve, window=50):
    """Estado SEM relativizar (limiares absolutos do P||Cmax) -- nao usado nas analises; so' para testes."""
    f1, f2 = UrStateFn.features(sol)
    return (int(np.digitize(f1, [0.10, 0.35])) * 27 + int(np.digitize(f2, [0.15, 0.40])) * 9
            + int(np.digitize(min(t / max_iter, 1.0), [0.33, 0.67])) * 3 + int(np.digitize(min(no_improve / window, 1.0), [0.25, 0.60])))


# ------------------------------------------------------------------ IG
def ur_ig_run(inst: UnrelatedInstance, selector, max_iter: int = 1000, d: Optional[int] = None, seed: int = 0, state_fn=None,
              initial_solution: Optional[UrSol] = None, stagnation_window: int = 50, action_map=None, ls_steps: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    selector.reset()
    if d is None:
        d = max(2, round(0.3 * inst.n))
    sol = mct_greedy(inst) if initial_solution is None else initial_solution.copy()
    best = sol.makespan
    best_sol = sol.copy()
    no_imp = 0
    n_imp = 0
    sf = state_fn if state_fn is not None else (lambda s, t, T, k: default_state_fn(s, t, T, k, stagnation_window))
    t0 = time.perf_counter()
    for t in range(max_iter):
        s_id = sf(sol, t, max_iter, no_imp)
        feats = np.array(UrStateFn.features(sol) + (t / max_iter, min(no_imp / stagnation_window, 1.0)))
        action = selector.select(s_id, feats)
        strat, d_t = (action, d) if action_map is None else action_map(action, inst.n)
        new = destroy_reconstruct(strat, sol, d_t, rng)
        if ls_steps > 0:      # IG com busca local (descida mover/trocar a partir da maquina critica, ate ls_steps melhorias)
            _descent(new, ls_steps)
        delta = sol.makespan - new.makespan
        reward = float(max(delta, 0))
        ns_id = sf(new, t + 1, max_iter, no_imp)
        selector.update(s_id, feats, action, reward, ns_id)
        if new.makespan <= sol.makespan:
            sol = new
            if sol.makespan < best:
                best = sol.makespan
                best_sol = sol.copy()
                n_imp += 1
                no_imp = 0
            else:
                no_imp += 1
        else:
            no_imp += 1
    cpu = time.perf_counter() - t0
    return {"makespan": best, "gap_pct": 100.0 * (best - inst.lower_bound) / inst.lower_bound, "cpu_time": cpu, "n_improvements": n_imp, "solution": best_sol}


# ------------------------------------------------------------------ ILS
def _best_move(sol: UrSol) -> int:
    """Melhor realocacao de uma tarefa da maquina critica; devolve a reducao do makespan (0 se nada melhora)."""
    p, loads = sol.inst.p, sol.loads
    c = int(np.argmax(loads))
    cur = int(loads[c])
    jobs = np.where(sol.a == c)[0]
    if len(jobs) <= 1:
        return 0
    newc = cur - p[c, jobs]                                  # (|J|,)
    cand = loads[None, :] + p[:, jobs].T                      # (|J|, m): carga de k apos receber j
    cand[:, c] = np.iinfo(np.int64).max
    worst = np.maximum(newc[:, None], cand)                   # max do par (c, k)
    idx = np.unravel_index(int(np.argmin(worst)), worst.shape)
    if worst[idx] >= cur:
        return 0
    j, k = int(jobs[idx[0]]), int(idx[1])
    old_ms = cur
    sol.move(j, k)
    return old_ms - sol.makespan


def _best_swap(sol: UrSol) -> int:
    p, loads, a = sol.inst.p, sol.loads, sol.a
    c = int(np.argmax(loads))
    cur = int(loads[c])
    jobs_c = np.where(a == c)[0]
    best = (cur, None)
    for k in range(sol.inst.m):
        if k == c:
            continue
        jobs_k = np.where(a == k)[0]
        if len(jobs_k) == 0 or len(jobs_c) == 0:
            continue
        newc = cur - p[c, jobs_c][:, None] + p[c, jobs_k][None, :]
        newk = loads[k] - p[k, jobs_k][None, :] + p[k, jobs_c][:, None]
        worst = np.maximum(newc, newk)
        idx = np.unravel_index(int(np.argmin(worst)), worst.shape)
        if worst[idx] < best[0]:
            best = (int(worst[idx]), (int(jobs_c[idx[0]]), int(jobs_k[idx[1]]), k))
    if best[1] is None:
        return 0
    j, l, k = best[1]
    old_ms = sol.makespan
    sol.move(j, k)
    sol.move(l, c)
    return old_ms - sol.makespan


def _double_move(sol: UrSol) -> int:
    old = sol.makespan
    _best_move(sol)
    _best_move(sol)
    return old - sol.makespan


_OPS = [_best_move, _best_swap, _double_move]


def _descent(sol: UrSol, max_steps: int) -> None:
    for _ in range(max_steps):
        if _best_move(sol) == 0 and _best_swap(sol) == 0:
            break


def ur_ils_run(inst: UnrelatedInstance, selector, max_iter: int = 1000, perturbation_threshold: int = 50, n_perturb: int = 3, seed: int = 0,
               state_fn=None, initial_solution: Optional[UrSol] = None, stagnation_window: int = 50) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    selector.reset()
    sol = mct_greedy(inst) if initial_solution is None else initial_solution.copy()
    best = sol.makespan
    best_sol = sol.copy()
    no_imp = 0
    n_imp = 0
    sf = state_fn if state_fn is not None else (lambda s, t, T, k: default_state_fn(s, t, T, k, stagnation_window))
    t0 = time.perf_counter()
    for t in range(max_iter):
        s_id = sf(sol, t, max_iter, no_imp)
        feats = np.array(UrStateFn.features(sol) + (t / max_iter, min(no_imp / stagnation_window, 1.0)))
        action = selector.select(s_id, feats)
        new = sol.copy()
        delta = _OPS[action](new)
        ns_id = sf(new, t + 1, max_iter, no_imp)
        selector.update(s_id, feats, action, float(max(delta, 0)), ns_id)
        if new.makespan <= sol.makespan:
            sol = new
            if sol.makespan < best:
                best = sol.makespan
                best_sol = sol.copy()
                n_imp += 1
                no_imp = 0
            else:
                no_imp += 1
        else:
            no_imp += 1
        if no_imp >= perturbation_threshold:
            sol = best_sol.copy()
            for _ in range(n_perturb):
                sol.move(int(rng.integers(inst.n)), int(rng.integers(inst.m)))
            no_imp = 0
    cpu = time.perf_counter() - t0
    return {"makespan": best, "gap_pct": 100.0 * (best - inst.lower_bound) / inst.lower_bound, "cpu_time": cpu, "n_improvements": n_imp, "solution": best_sol}
