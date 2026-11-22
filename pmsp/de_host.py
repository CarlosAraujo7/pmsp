"""
pmsp/de_host.py (Entrada 289, Ciclo 55)
=========================================
HOSPEDEIRO DE EVOLUCAO DIFERENCIAL COM SELECAO DE ESTRATEGIA, o teste de um regime em que o sinal de aprendizado e' DENSO (resposta a C da revisao cega #2 do CEC: um hospedeiro padrao de computacao evolutiva em que o aprendizado
online deveria ter chance). Funcoes: BBOB (24 funcoes sem ruido, 15 instancias cada) pelas implementacoes de referencia do pacote `cma` (cma.bbobbenchmarks), D = 10, dominio [-5, 5]^D.

DE geracional com NP = 100 (10 D). A cada GERACAO o seletor escolhe UMA estrategia para os NP vetores-alvo (T = 1000 decisoes por execucao, como nos hospedeiros de escalonamento; orcamento NP (T + 1) = 100.100 avaliacoes, 10^4 D):
  acao 0  DE/rand/1/bin          v = x_r1 + F (x_r2 - x_r3)
  acao 1  DE/best/1/bin          v = x_best + F (x_r1 - x_r2)
  acao 2  DE/current-to-best/1/bin   v = x_i + F (x_best - x_i) + F (x_r1 - x_r2)
  acao 3  DE/rand/2/bin          v = x_r1 + F (x_r2 - x_r3) + F (x_r4 - x_r5)
  acao 4  DE/current-to-rand/1   u = x_i + K (x_r1 - x_i) + F (x_r2 - x_r3), K ~ U(0,1), sem cruzamento
F = 0,5 e CR = 0,9 para todas (so' a ESTRATEGIA varia, como no pool do SaDE). Cruzamento binomial com um indice forcado; componentes fora do dominio sao sorteados de novo em [-5, 5]; o alvo e' substituido se o filho nao for pior.
Recompensa da decisao = taxa de SUCESSO da geracao (fracao dos NP filhos estritamente melhores que o alvo), em [0, 1]: credito padrao do SaDE/CoDE, sem escala.
Estado (81, como no IG): progresso (0,33/0,67), estagnacao (geracoes sem melhora do melhor / 50; 0,25/0,60), diversidade (media dos desvios por dimensao / 2,887; 0,02/0,20), sucesso da geracao anterior (0,05/0,20).
Seletores: fixo, aleatorio, Q-Learning (alfa 0,1; gama 0,9; epsilon 1 -> 0,05 por 0,995; tabela inicial zero; desempate pelo primeiro maximo OU aleatorio), UCB1 (recompensa normalizada pelo maximo corrente; C = raiz de 2),
gradiente de politica (por estado; lr 0,1), probabilidade proporcional (PM), perseguicao adaptativa (AP) e explorar-depois-comprometer (ETC). Tudo com numpy e sementes fixas (reprodutivel).
Desempenho = erro final do melhor, f_best - f_opt (BBOB conhece f_opt); em escala log10 com piso 1e-8.
"""
from __future__ import annotations

import numpy as np

LOW, HIGH = -5.0, 5.0
NSTRAT = 5
F_DE, CR_DE = 0.5, 0.9
STD0 = (HIGH - LOW) / np.sqrt(12.0)
FLOOR = 1e-8
_CACHE: dict = {}


def bbob(ifun: int, iinst: int):
    if (ifun, iinst) not in _CACHE:
        from cma import bbobbenchmarks as bb
        _CACHE[(ifun, iinst)] = bb.instantiate(ifun, iinstance=iinst)
    return _CACHE[(ifun, iinst)]


# ------------------------------------------------------------------------------------------------ geracao
def _donors(rng, NP, n):
    r = rng.random((NP, NP))
    r[np.arange(NP), np.arange(NP)] = 2.0                      # o alvo nunca e' doador
    return np.argsort(r, axis=1)[:, :n]


def generation(pop, fit, a, rng, f, F=F_DE, CR=CR_DE):
    """Uma geracao com a estrategia a; atualiza pop e fit no lugar; devolve a taxa de sucesso."""
    NP, D = pop.shape
    ib = int(np.argmin(fit))
    idx = _donors(rng, NP, 5)
    x = lambda k: pop[idx[:, k]]
    if a == 0:
        v = x(0) + F * (x(1) - x(2))
    elif a == 1:
        v = pop[ib] + F * (x(0) - x(1))
    elif a == 2:
        v = pop + F * (pop[ib] - pop) + F * (x(0) - x(1))
    elif a == 3:
        v = x(0) + F * (x(1) - x(2)) + F * (x(3) - x(4))
    else:
        v = pop + rng.random((NP, 1)) * (x(0) - pop) + F * (x(1) - x(2))
    if a != 4:
        mask = rng.random((NP, D)) < CR
        mask[np.arange(NP), rng.integers(0, D, NP)] = True
        v = np.where(mask, v, pop)
    bad = (v < LOW) | (v > HIGH)
    if bad.any():
        v[bad] = rng.uniform(LOW, HIGH, size=int(bad.sum()))
    fv = f(v)
    succ = fv < fit
    rep = fv <= fit
    pop[rep] = v[rep]
    fit[rep] = fv[rep]
    return float(succ.mean())


def state_id(t, T, stag, div, succ):
    b1 = 0 if t / T < 0.33 else (1 if t / T < 0.67 else 2)
    f2 = min(stag / 50.0, 1.0)
    b2 = 0 if f2 < 0.25 else (1 if f2 < 0.60 else 2)
    b3 = 0 if div < 0.02 else (1 if div < 0.20 else 2)
    b4 = 0 if succ < 0.05 else (1 if succ < 0.20 else 2)
    return b1 * 27 + b2 * 9 + b3 * 3 + b4


# ------------------------------------------------------------------------------------------------ seletores
class Fixed:
    def __init__(self, a): self.a = a
    def select(self, s, t): return self.a
    def update(self, a, r, s, s2): pass


class Rand:
    def __init__(self, K, seed): self.K, self.rng = K, np.random.default_rng(seed)
    def select(self, s, t): return int(self.rng.integers(0, self.K))
    def update(self, a, r, s, s2): pass


class QL:
    def __init__(self, K, seed, tie, alpha=0.1, gamma=0.9):
        self.K, self.rng, self.tie, self.alpha, self.gamma = K, np.random.default_rng(seed), tie, alpha, gamma
        self.Q, self.eps = np.zeros((81, K)), 1.0

    def select(self, s, t):
        if self.rng.random() < self.eps:
            a = int(self.rng.integers(0, self.K))
        else:
            q = self.Q[s]
            a = int(np.argmax(q)) if self.tie == 0 else int(self.rng.choice(np.flatnonzero(q == q.max())))
        self.eps = max(0.05, self.eps * 0.995)
        return a

    def update(self, a, r, s, s2):
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2].max() - self.Q[s, a])


class UCB1:
    def __init__(self, K, seed): self.K, self.n, self.rs, self.rmax, self.t = K, np.zeros(K), np.zeros(K), 0.0, 0
    def select(self, s, t):
        self.t += 1
        z = np.flatnonzero(self.n == 0)
        if len(z):
            return int(z[0])
        return int(np.argmax(self.rs / self.n + np.sqrt(2.0) * np.sqrt(np.log(self.t) / self.n)))
    def update(self, a, r, s, s2):
        self.rmax = max(self.rmax, r)
        self.n[a] += 1
        self.rs[a] += (r / self.rmax if self.rmax > 0 else 0.0)


class PG:
    def __init__(self, K, seed, lr=0.1): self.K, self.rng, self.lr, self.H, self.rbar, self.n, self.rmax, self.pi = K, np.random.default_rng(seed), lr, np.zeros((81, K)), 0.0, 0, 0.0, None
    def select(self, s, t):
        h = self.H[s]
        p = np.exp(h - h.max())
        self.pi = p / p.sum()
        return int(self.rng.choice(self.K, p=self.pi))
    def update(self, a, r, s, s2):
        self.rmax = max(self.rmax, r)
        rn = r / self.rmax if self.rmax > 0 else 0.0
        ind = np.zeros(self.K)
        ind[a] = 1.0
        self.H[s] += self.lr * (rn - self.rbar) * (ind - self.pi)
        self.n += 1
        self.rbar += (rn - self.rbar) / self.n


class PM:
    def __init__(self, K, seed, alpha=0.3, pmin=0.05): self.K, self.rng, self.alpha, self.pmin, self.q = K, np.random.default_rng(seed), alpha, pmin, np.ones(K)
    def select(self, s, t):
        p = self.pmin + (1.0 - self.K * self.pmin) * self.q / self.q.sum()
        return int(self.rng.choice(self.K, p=p))
    def update(self, a, r, s, s2): self.q[a] = (1 - self.alpha) * self.q[a] + self.alpha * r


class AP:
    def __init__(self, K, seed, alpha=0.3, beta=0.1, pmin=0.05):
        self.K, self.rng, self.alpha, self.beta, self.pmin, self.q = K, np.random.default_rng(seed), alpha, beta, pmin, np.ones(K)
        self.pmax, self.p = 1.0 - (K - 1) * pmin, np.full(K, 1.0 / K)
    def select(self, s, t): return int(self.rng.choice(self.K, p=self.p / self.p.sum()))
    def update(self, a, r, s, s2):
        self.q[a] = (1 - self.alpha) * self.q[a] + self.alpha * r
        ab = int(np.argmax(self.q))
        for k in range(self.K):
            tgt = self.pmax if k == ab else self.pmin
            self.p[k] += self.beta * (tgt - self.p[k])


class ETC:
    def __init__(self, K, seed, m=20): self.K, self.m, self.n, self.rs, self.commit = K, m, np.zeros(K), np.zeros(K), None
    def select(self, s, t):
        if self.commit is not None:
            return self.commit
        k = int(np.argmin(self.n))
        return k
    def update(self, a, r, s, s2):
        self.n[a] += 1
        self.rs[a] += r
        if self.commit is None and (self.n >= self.m).all():
            self.commit = int(np.argmax(self.rs / self.n))


LEARNERS = {"ql_std": lambda K, seed: QL(K, seed, 0), "ql_rt": lambda K, seed: QL(K, seed, 1), "ucb": UCB1, "pg": PG, "pm": PM, "ap": AP, "etc": ETC}


def make_selector(method, K, seed):
    if method == "rand":
        return Rand(K, seed)
    if method.startswith("arm"):
        return Fixed(int(method[3:]))
    return LEARNERS[method](K, seed)


# ------------------------------------------------------------------------------------------------ execucao
def run_de(ifun, iinst, method, seed, T=1000, NP=100, D=10, log=False):
    f, fopt = bbob(ifun, iinst)
    rng = np.random.default_rng(seed)
    sel = make_selector(method, NSTRAT, seed + 7_919)
    pop = rng.uniform(LOW, HIGH, size=(NP, D))
    fit = np.asarray(f(pop), dtype=float)
    best, stag, succ = float(fit.min()), 0, 1.0
    div = float(pop.std(axis=0).mean() / STD0)
    s = state_id(0, T, stag, div, succ)
    A = np.zeros(T, dtype=np.int8) if log else None
    Rw = np.zeros(T, dtype=np.float32) if log else None
    S = np.zeros(T, dtype=np.int8) if log else None
    for t in range(T):
        a = sel.select(s, t)
        succ = generation(pop, fit, a, rng, f)
        nb = float(fit.min())
        if nb < best:
            best, stag = nb, 0
        else:
            stag += 1
        div = float(pop.std(axis=0).mean() / STD0)
        s2 = state_id(t + 1, T, stag, div, succ)
        sel.update(a, succ, s, s2)
        if log:
            A[t], Rw[t], S[t] = a, succ, s
        s = s2
    err = best - fopt
    out = dict(err=float(err), lg=float(np.log10(max(err, FLOOR))))
    if log:
        out.update(A=A, Rw=Rw, S=S)
    return out
