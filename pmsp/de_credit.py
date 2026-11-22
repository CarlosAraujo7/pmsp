"""
pmsp/de_credit.py (Entrada 291, Ciclo 56)
===========================================
O hospedeiro de evolucao diferencial de pmsp/de_host.py (congelado) com ATRIBUICAO DE CREDITO ALTERNATIVA, para testar o dilema informacao-alinhamento da Entrada 289c: o credito por taxa de sucesso (denso) esta' desalinhado do erro final
(a acao de maior sucesso e' a validada em so' 5 de 24 funcoes); um credito ALINHADO com o objetivo tambem faz o aprendiz ganhar do aleatorio? Tres creditos, todos medidos por geracao:
  success         a taxa de sucesso da geracao (fracao dos NP filhos estritamente melhores que o alvo): EXATAMENTE a de de_host.run_de (a mesma sequencia de numeros aleatorios; tests/test_de_credit_entrada291.py);
  log_incumbent   a reducao, em decadas, do erro do INCUMBENTE: max(0, log10(erro antes) - log10(erro depois)), erro = f_best - f_opt com piso 1e-8. Alinhado com a metrica de desempenho POR CONSTRUCAO (a soma dos creditos e'
                  a reducao total de log10 do erro) e USA f_opt, que um otimizador real nao conhece: e' o LIMITE SUPERIOR de um credito alinhado, nao um metodo praticavel;
  rel_incumbent   a melhora do incumbente relativa a' dispersao da populacao, (melhor antes - melhor depois) / (media antes - melhor antes), em [0, 1]: praticavel (nao usa f_opt) e sem a queda de ordens de grandeza do credito linear.
Os dois ultimos sao normalizados pelo MAXIMO corrente antes de chegar ao seletor (como o UCB1 e o PG ja' fazem por dentro; assim todos os seletores recebem recompensas em [0, 1]).
"""
from __future__ import annotations

import numpy as np

from . import de_host as DH

CREDITS = ("success", "log_incumbent", "rel_incumbent")


def credit_value(credit, succ, best_before, best_after, mean_before, fopt):
    if credit == "success":
        return float(succ)
    if credit == "log_incumbent":
        return max(0.0, float(np.log10(max(best_before - fopt, DH.FLOOR)) - np.log10(max(best_after - fopt, DH.FLOOR))))
    spread = mean_before - best_before
    if spread <= 1e-300:
        return 0.0
    return float(min(max((best_before - best_after) / spread, 0.0), 1.0))


def run_de_credit(ifun, iinst, method, seed, credit, T=1000, NP=100, D=10, log=False):
    f, fopt = DH.bbob(ifun, iinst)
    rng = np.random.default_rng(seed)
    sel = DH.make_selector(method, DH.NSTRAT, seed + 7_919)
    pop = rng.uniform(DH.LOW, DH.HIGH, size=(NP, D))
    fit = np.asarray(f(pop), dtype=float)
    best, stag, succ = float(fit.min()), 0, 1.0
    div = float(pop.std(axis=0).mean() / DH.STD0)
    s = DH.state_id(0, T, stag, div, succ)
    rmax, csum = 0.0, 0.0
    A = np.zeros(T, dtype=np.int8) if log else None
    Rc = np.zeros(T, dtype=np.float32) if log else None
    S = np.zeros(T, dtype=np.int8) if log else None
    for t in range(T):
        a = sel.select(s, t)
        best_before, mean_before = float(fit.min()), float(fit.mean())
        succ = DH.generation(pop, fit, a, rng, f)
        nb = float(fit.min())
        r = credit_value(credit, succ, best_before, nb, mean_before, fopt)
        if credit == "success":
            r_pass = r
        else:
            rmax = max(rmax, r)
            r_pass = r / rmax if rmax > 0 else 0.0
        if nb < best:
            best, stag = nb, 0
        else:
            stag += 1
        div = float(pop.std(axis=0).mean() / DH.STD0)
        s2 = DH.state_id(t + 1, T, stag, div, succ)
        sel.update(a, r_pass, s, s2)
        csum += r
        if log:
            A[t], Rc[t], S[t] = a, r, s
        s = s2
    err = best - fopt
    out = dict(err=float(err), lg=float(np.log10(max(err, DH.FLOOR))), crew=float(csum))
    if log:
        out.update(A=A, Rc=Rc, S=S)
    return out
