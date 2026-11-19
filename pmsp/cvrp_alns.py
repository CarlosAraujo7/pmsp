"""
pmsp/cvrp_alns.py (Entrada 282, Ciclo 50)
==========================================
O CVRP como hospedeiro do protocolo, com a biblioteca ALNS de terceiros (Wouda e Lan, JOSS 2023; licenca MIT) fornecendo o laco, o criterio de aceitacao e TODOS os seletores de operadores (RandomSelect, RouletteWheel, SegmentedRouletteWheel, AlphaUCB, MABSelector):
nenhum aprendiz e' implementado por nos. Escrevemos so' o problema e os operadores (nucleos em numba; solucao = vetor com os clientes 1..n e o deposito 0 como separador de rotas, distancias euclidianas arredondadas, como nas instancias X de Uchoa et al. 2017):
  destruir (8 operadores = 4 tipos x 2 fracoes de n, 10% e 30%): aleatoria, dos piores (maior economia de remocao, com aleatorizacao), relacionada (Shaw, por distancia, com aleatorizacao) e de rotas inteiras;
  reparar (2 operadores): insercao gulosa (clientes em ordem aleatoria) e insercao por arrependimento-2. Uma ACAO = um par (destruir, reparar): 16 acoes.
Solucao inicial: economias de Clarke e Wright (paralelo). Aceitacao: RecordToRecordTravel.autofit (como no exemplo de CVRP da biblioteca). Estatica = um par fixo (FixedSelect). Orcamento = iteracoes (T).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from alns import ALNS, State
from alns.accept import RecordToRecordTravel
from alns.select import AlphaUCB, MABSelector, OperatorSelectionScheme, RandomSelect, RouletteWheel, SegmentedRouletteWheel
from alns.stop import MaxIterations

BIG = 10 ** 9
FRACS = (0.10, 0.30)
DESTROY_KINDS = ("random", "worst", "related", "route")
NUM_DESTROY = len(DESTROY_KINDS) * len(FRACS)      # 8
NUM_REPAIR = 2                                     # 0 guloso, 1 arrependimento-2
SCORES = [3, 2, 1, 0.5]                            # os do exemplo de CVRP da biblioteca (melhor, melhor que a corrente, aceita, rejeitada)


@dataclass
class Instance:
    name: str
    D: np.ndarray        # (n + 1) x (n + 1), int64; indice 0 = deposito
    dem: np.ndarray      # n + 1, dem[0] = 0
    Q: int
    n: int
    bks: float = float("nan")


# --------------------------------------------------------------------------------------- nucleos numba
@njit(cache=True)
def cost_of(seq, D):
    c = 0
    for i in range(len(seq) - 1):
        c += D[seq[i], seq[i + 1]]
    return c


@njit(cache=True)
def canon(buf, L):
    """Forma canonica: comeca e termina em 0, sem zeros consecutivos; solucao vazia = [0, 0]."""
    out = np.empty(L + 2, dtype=np.int64)
    k = 0
    out[k] = 0
    k += 1
    for i in range(L):
        v = buf[i]
        if v != 0:
            out[k] = v
            k += 1
        elif out[k - 1] != 0:
            out[k] = 0
            k += 1
    if out[k - 1] != 0:
        out[k] = 0
        k += 1
    if k == 1:
        out[1] = 0
        k = 2
    return out[:k].copy()


@njit(cache=True)
def remove_customers(seq, mask):
    buf = np.empty(len(seq), dtype=np.int64)
    k = 0
    for i in range(len(seq)):
        v = seq[i]
        if v == 0 or not mask[v]:
            buf[k] = v
            k += 1
    return canon(buf, k)


@njit(cache=True)
def removal_gains(seq, D, n):
    g = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, len(seq) - 1):
        v = seq[i]
        if v != 0:
            g[v] = D[seq[i - 1], v] + D[v, seq[i + 1]] - D[seq[i - 1], seq[i + 1]]
    return g


@njit(cache=True)
def route_ids(seq, n):
    rid = np.full(n + 1, -1, dtype=np.int64)
    r = -1
    for i in range(len(seq)):
        if seq[i] == 0:
            r += 1
        else:
            rid[seq[i]] = r
    return rid


@njit(cache=True)
def _edge_loads(w, L, dem, load, rid):
    """Para cada aresta p (entre w[p] e w[p + 1]): a carga e o indice da rota que a contem."""
    j = 0
    r = 0
    while j < L - 1:
        k = j + 1
        s = 0
        while k < L and w[k] != 0:
            s += dem[w[k]]
            k += 1
        for p in range(j, k):
            load[p] = s
            rid[p] = r
        r += 1
        j = k


@njit(cache=True)
def _insert(w, L, p, c):
    for t in range(L, p + 1, -1):
        w[t] = w[t - 1]
    w[p + 1] = c
    L += 1
    if not (w[L - 1] == 0 and w[L - 2] == 0):
        w[L] = 0
        L += 1
    return L


@njit(cache=True)
def greedy_repair(seq, order, D, dem, Q):
    q = len(order)
    cap = len(seq) + 2 * q + 6
    w = np.empty(cap, dtype=np.int64)
    L = len(seq)
    for i in range(L):
        w[i] = seq[i]
    w[L] = 0
    L += 1
    load = np.zeros(cap, dtype=np.int64)
    rid = np.zeros(cap, dtype=np.int64)
    for t in range(q):
        c = order[t]
        _edge_loads(w, L, dem, load, rid)
        bp = -1
        bd = BIG
        for p in range(L - 1):
            if load[p] + dem[c] <= Q:
                d = D[w[p], c] + D[c, w[p + 1]] - D[w[p], w[p + 1]]
                if d < bd:
                    bd = d
                    bp = p
        L = _insert(w, L, bp, c)
    return canon(w, L)


@njit(cache=True)
def regret_repair(seq, rem, D, dem, Q):
    q = len(rem)
    cap = len(seq) + 2 * q + 6
    w = np.empty(cap, dtype=np.int64)
    L = len(seq)
    for i in range(L):
        w[i] = seq[i]
    w[L] = 0
    L += 1
    load = np.zeros(cap, dtype=np.int64)
    rid = np.zeros(cap, dtype=np.int64)
    done = np.zeros(q, dtype=np.bool_)
    for _ in range(q):
        _edge_loads(w, L, dem, load, rid)
        pick = -1
        pick_reg = -1
        pick_b1 = BIG
        pick_p = -1
        for t in range(q):
            if done[t]:
                continue
            c = rem[t]
            b1 = BIG
            b2 = BIG
            p1 = -1
            cur = -1
            cur_best = BIG
            cur_p = -1
            for p in range(L - 1):
                r = rid[p]
                if r != cur:
                    if cur >= 0 and cur_best < BIG:
                        if cur_best < b1:
                            b2 = b1
                            b1 = cur_best
                            p1 = cur_p
                        elif cur_best < b2:
                            b2 = cur_best
                    cur = r
                    cur_best = BIG
                    cur_p = -1
                if load[p] + dem[c] <= Q:
                    d = D[w[p], c] + D[c, w[p + 1]] - D[w[p], w[p + 1]]
                    if d < cur_best:
                        cur_best = d
                        cur_p = p
            if cur >= 0 and cur_best < BIG:
                if cur_best < b1:
                    b2 = b1
                    b1 = cur_best
                    p1 = cur_p
                elif cur_best < b2:
                    b2 = cur_best
            reg = (b2 - b1) if b2 < BIG else BIG // 2
            if reg > pick_reg or (reg == pick_reg and b1 < pick_b1):
                pick = t
                pick_reg = reg
                pick_b1 = b1
                pick_p = p1
        done[pick] = True
        L = _insert(w, L, pick_p, rem[pick])
    return canon(w, L)


# --------------------------------------------------------------------------------------- estado e operadores
class CVRPState(State):
    def __init__(self, inst: Instance, seq: np.ndarray, unrouted: np.ndarray | None = None):
        self.inst = inst
        self.seq = seq
        self.unrouted = unrouted if unrouted is not None else np.empty(0, dtype=np.int64)
        self._obj = None

    def objective(self) -> float:
        if self._obj is None:
            self._obj = float(cost_of(self.seq, self.inst.D))
        return self._obj

    def n_routes(self) -> int:
        return int((self.seq == 0).sum()) - 1


def _pick_by_rank(rng, order, q, p):
    """Escolhe q elementos de `order` (melhor primeiro) com a aleatorizacao y ** p de Ropke e Pisinger."""
    avail = list(order)
    out = []
    for _ in range(min(q, len(avail))):
        y = rng.random()
        out.append(avail.pop(int(len(avail) * y ** p)))
    return out


def make_destroy(kind: str, frac: float):
    def destroy(state: CVRPState, rng) -> CVRPState:
        inst = state.inst
        n = inst.n
        q = max(2, int(round(frac * n)))
        seq = state.seq
        mask = np.zeros(n + 1, dtype=np.bool_)
        if kind == "random":
            chosen = rng.choice(np.arange(1, n + 1), size=q, replace=False)
        elif kind == "worst":
            g = removal_gains(seq, inst.D, n)
            order = np.argsort(-g[1:], kind="stable") + 1
            chosen = _pick_by_rank(rng, order, q, 3.0)
        elif kind == "related":
            s = int(rng.integers(1, n + 1))
            order = np.argsort(inst.D[s, 1:], kind="stable") + 1
            chosen = _pick_by_rank(rng, order, q, 6.0)
        else:  # route
            rid = route_ids(seq, n)
            routes = rng.permutation(np.unique(rid[1:]))
            chosen = []
            for r in routes:
                chosen += [c for c in range(1, n + 1) if rid[c] == r]
                if len(chosen) >= q:
                    break
        chosen = np.asarray(chosen, dtype=np.int64)
        mask[chosen] = True
        return CVRPState(inst, remove_customers(seq, mask), chosen)

    destroy.__name__ = f"destroy_{kind}_{int(round(frac * 100))}"
    return destroy


def greedy_repair_op(state: CVRPState, rng) -> CVRPState:
    order = rng.permutation(state.unrouted).astype(np.int64)
    return CVRPState(state.inst, greedy_repair(state.seq, order, state.inst.D, state.inst.dem, state.inst.Q))


def regret_repair_op(state: CVRPState, rng) -> CVRPState:
    return CVRPState(state.inst, regret_repair(state.seq, state.unrouted.astype(np.int64), state.inst.D, state.inst.dem, state.inst.Q))


DESTROY_OPS = [make_destroy(k, f) for k in DESTROY_KINDS for f in FRACS]
REPAIR_OPS = [greedy_repair_op, regret_repair_op]


def action_pair(a: int):
    """Acao a (0..15) -> (indice do operador de destruicao, indice do de reparo)."""
    return a // NUM_REPAIR, a % NUM_REPAIR


class FixedSelect(OperatorSelectionScheme):
    """A acao estatica: sempre o mesmo par (destruir, reparar)."""

    def __init__(self, d_idx: int, r_idx: int):
        super().__init__(NUM_DESTROY, NUM_REPAIR)
        self._pair = (d_idx, r_idx)

    def __call__(self, rng, best, curr):
        return self._pair

    def update(self, candidate, d_idx, r_idx, outcome):
        pass


# --------------------------------------------------------------------------------------- solucao inicial e instancias
def clarke_wright(inst: Instance) -> np.ndarray:
    n, D, dem, Q = inst.n, inst.D, inst.dem, inst.Q
    routes = {c: [c] for c in range(1, n + 1)}
    where = {c: c for c in range(1, n + 1)}
    load = {c: int(dem[c]) for c in range(1, n + 1)}
    ii, jj = np.triu_indices(n, k=1)
    ii, jj = ii + 1, jj + 1
    sav = D[0, ii] + D[0, jj] - D[ii, jj]
    for t in np.argsort(-sav, kind="stable"):
        i, j = int(ii[t]), int(jj[t])
        ri, rj = where[i], where[j]
        if ri == rj or load[ri] + load[rj] > Q:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] == i and b[0] == j:
            new = a + b
        elif a[0] == i and b[-1] == j:
            new = b + a
        elif a[0] == i and b[0] == j:
            new = a[::-1] + b
        elif a[-1] == i and b[-1] == j:
            new = a + b[::-1]
        else:
            continue
        routes[ri] = new
        load[ri] += load[rj]
        del routes[rj], load[rj]
        for c in new:
            where[c] = ri
    seq = [0]
    for r in routes.values():
        seq += r + [0]
    return np.asarray(seq, dtype=np.int64)


def make_class_instance(n: int, pos: str, rsize: int, seed: int) -> Instance:
    """Instancia sintetica no estilo de Uchoa et al. (2017): grade 1000 x 1000, deposito central, clientes aleatorios (R) ou agrupados (C), demandas U[1, 100], capacidade para `rsize` clientes por rota em media."""
    rng = np.random.default_rng(seed)
    G = 1000
    if pos == "R":
        xy = rng.integers(0, G + 1, size=(n, 2))
    else:
        S = int(rng.integers(3, 9))
        pts = [rng.integers(0, G + 1, size=2) for _ in range(S)]
        while len(pts) < n:
            for _ in range(10000):
                cand = rng.integers(0, G + 1, size=2)
                p = sum(np.exp(-np.hypot(*(cand - q)) / 40.0) for q in pts)
                if rng.random() < min(1.0, p):
                    pts.append(cand)
                    break
            else:
                pts.append(rng.integers(0, G + 1, size=2))
        xy = np.array(pts[:n])
    coords = np.vstack([[G // 2, G // 2], xy]).astype(float)
    D = np.rint(np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))).astype(np.int64)
    dem = np.concatenate([[0], rng.integers(1, 101, size=n)]).astype(np.int64)
    Q = int(max(np.ceil(rsize * dem.sum() / n), dem.max()))
    return Instance(name=f"{pos}{rsize}_n{n}_s{seed}", D=D, dem=dem, Q=Q, n=n)


def instance_from_vrplib(inst: dict, sol: dict | None = None) -> Instance:
    xy = np.asarray(inst["node_coord"], dtype=float)
    D = np.rint(np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))).astype(np.int64)
    dem = np.asarray(inst["demand"], dtype=np.int64)
    return Instance(name=str(inst["name"]), D=D, dem=dem, Q=int(inst["capacity"]), n=int(inst["dimension"]) - 1, bks=float(sol["cost"]) if sol else float("nan"))


def feasible(inst: Instance, seq: np.ndarray) -> bool:
    cust = seq[seq != 0]
    if sorted(cust.tolist()) != list(range(1, inst.n + 1)):
        return False
    load, ok = 0, True
    for v in seq:
        if v == 0:
            ok = ok and load <= inst.Q
            load = 0
        else:
            load += int(inst.dem[v])
    return ok


# --------------------------------------------------------------------------------------- execucao
def make_selector(name: str, seed: int, fixed: int | None = None):
    nd, nr = NUM_DESTROY, NUM_REPAIR
    if name == "fixed":
        return FixedSelect(*action_pair(fixed))
    if name == "random":
        return RandomSelect(nd, nr)
    if name == "rw":
        return RouletteWheel(SCORES, 0.8, nd, nr)
    if name == "srw":
        return SegmentedRouletteWheel(SCORES, 0.8, 100, nd, nr)
    if name == "aucb":
        return AlphaUCB(SCORES, 0.05, nd, nr)
    from mabwiser.mab import LearningPolicy
    if name == "mab_ucb1":
        return MABSelector(SCORES, nd, nr, learning_policy=LearningPolicy.UCB1(alpha=1.0), seed=seed)
    if name == "mab_eps":
        return MABSelector(SCORES, nd, nr, learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15), seed=seed)
    raise ValueError(name)


LEARNERS = ["rw", "srw", "aucb", "mab_ucb1", "mab_eps"]


def initial_state(inst: Instance) -> CVRPState:
    seq = clarke_wright(inst)
    return CVRPState(inst, canon(seq, len(seq)))


def run_alns(inst: Instance, selector: str, seed: int, T: int = 1000, fixed: int | None = None, init: CVRPState | None = None, perm: bool = False):
    """Uma execucao: devolve (melhor custo, segundos, custo inicial, melhor solucao, info). Com perm = True os operadores sao adicionados em ordem aleatoria (a lista que o seletor ve e' permutada; diagnostico de ordem).
    info: parcela das iteracoes nos operadores de destruicao e de reparo LISTADOS PRIMEIRO."""
    rng = np.random.default_rng(seed)
    alns = ALNS(rng)
    dorder = list(range(NUM_DESTROY))
    rorder = list(range(NUM_REPAIR))
    if perm:
        prng = np.random.default_rng(seed + 12345)
        dorder, rorder = [int(x) for x in prng.permutation(NUM_DESTROY)], [int(x) for x in prng.permutation(NUM_REPAIR)]
    for k in dorder:
        alns.add_destroy_operator(DESTROY_OPS[k])
    for k in rorder:
        alns.add_repair_operator(REPAIR_OPS[k])
    init = init if init is not None else initial_state(inst)
    sel = make_selector(selector, seed, fixed)
    accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0.0, T)
    t0 = time.perf_counter()
    res = alns.iterate(init, sel, accept, MaxIterations(T))
    dt = time.perf_counter() - t0
    dc = res.statistics.destroy_operator_counts
    rc = res.statistics.repair_operator_counts
    info = dict(share_d0=float(sum(dc.get(DESTROY_OPS[dorder[0]].__name__, [0]))) / T, share_r0=float(sum(rc.get(REPAIR_OPS[rorder[0]].__name__, [0]))) / T)
    best = res.best_state
    return best.objective(), dt, init.objective(), best, info
