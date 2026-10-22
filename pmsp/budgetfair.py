"""
pmsp/budgetfair.py (Entrada 186, Ciclo 16)
============================================
O PROTOCOLO de atribuicao justa ao orcamento como uma biblioteca pequena e independente do problema (so numpy e scipy), extraida dos scripts do estudo: para auditar o ganho de qualquer componente adaptativo de uma busca (um seletor de operadores, uma busca local, um parametro)
contra um controle e contra um concorrente estatico, com o mesmo orcamento.
  rel_diff, paired_test      diferenca relativa pareada d = 100 (A - B) / B (negativo: A melhor); Wilcoxon bilateral (empates descartados), IC90 bootstrap da media, equivalencia a 0 (IC90 dentro de +-margem)
  holm, reading              correcao de Holm sobre uma familia e a leitura 'paridade' / 'melhor' / 'pior' / 'sem diferenca' (licao II.24: significativo mas dentro da margem e' paridade)
  cost_ratio, equal_cpu_iterations   razao de CPU K = tempo(A)/tempo(B) medida em SEQUENCIA com time.process_time, e o numero de iteracoes que o braco barato recebe para a CPU igual
  budget_share, decision_cost_split  fracao do ganho que e' orcamento, e a separacao do ganho a CPU igual de um aprendiz em efeito de DECISAO (a iteracoes iguais) e efeito de CUSTO (a diferenca)
  mde80, planted_power       efeito minimo detectavel analitico e poder empirico por efeito plantado (o teste de que o banco detectaria um ganho se ele existisse)
Nenhum acoplamento com pmsp.instance ou com os agentes: as execucoes entram como funcoes `run(instance, seed) -> valor`.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
from scipy import stats


def rel_diff(a, b) -> np.ndarray:
    """d = 100 (a - b) / b, elemento a elemento (negativo: a e' melhor, para minimizacao)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 100.0 * (a - b) / b


def paired_test(a, b, margin: float = 0.05, n_boot: int = 2000, seed: int = 0) -> dict:
    """Teste pareado de a contra b (minimizacao). Retorna n, melhor/pior/empate, media de d, IC90 bootstrap, p do Wilcoxon bilateral (empates descartados; nan se < 6 pares nao nulos) e se o IC90 cabe em +-margem."""
    d = rel_diff(a, b)
    nz = d[d != 0]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi), p_raw=p,
                equivalent=bool(lo > -margin and hi < margin))


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Correcao de Holm-Bonferroni: (p ajustados, rejeita). p nan nao rejeita e nao entra na ordenacao."""
    p = np.asarray(pvalues, float)
    ok = ~np.isnan(p)
    adj = np.full(len(p), np.nan)
    rej = np.zeros(len(p), dtype=bool)
    order = np.argsort(p[ok])
    idx = np.flatnonzero(ok)[order]
    m = len(idx)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
        rej[i] = run < alpha
    return adj, rej


def reading(mean_d: float, p_holm: float, equivalent: bool, alpha: float = 0.05) -> str:
    """'sem diferenca' (nao significativo), 'paridade' (significativo mas dentro da margem), 'A melhor' ou 'A pior'."""
    if not (p_holm < alpha):
        return "sem diferenca"
    if equivalent:
        return "paridade"
    return "A melhor" if mean_d < 0 else "A pior"


def cost_ratio(run_a: Callable, run_b: Callable, instances: Sequence, min_seconds: float = 1.0, repeats: int = 2) -> float:
    """K = tempo de CPU de a / tempo de CPU de b, medido em sequencia (nunca em paralelo) com time.process_time, na mediana de `repeats` instancias. run_x(instancia, semente) executa uma vez."""
    ratios = []
    for inst in instances[:repeats]:
        def cpu(run):
            reps, t0 = 0, time.process_time()
            while time.process_time() - t0 < min_seconds:
                run(inst, reps + 7)
                reps += 1
            return (time.process_time() - t0) / reps
        tb = cpu(run_b)
        ratios.append(cpu(run_a) / tb)
    return float(np.median(ratios))


def equal_cpu_iterations(T: int, K: float, floor: int = 50) -> int:
    """Iteracoes que o braco barato recebe para ter a mesma CPU do caro: round(T K), com piso."""
    return max(int(floor), int(round(T * K)))


def budget_share(d_budget: float, d_iter: float) -> float:
    """Porcentagem do ganho a iteracoes iguais que e' so' orcamento: 100 (1 - |d_orcamento| / |d_iteracoes|); 100 = todo o ganho some quando o orcamento e' igualado."""
    return 100.0 * (1.0 - abs(d_budget) / abs(d_iter))


def decision_cost_split(d_iter: float, d_cpu: float) -> dict:
    """Ganhos (positivos: o aprendiz ajuda) de um aprendiz contra o aleatorio: efeito de DECISAO = -d a iteracoes iguais; efeito de CUSTO = -(d_cpu - d_iter); o ganho a CPU igual e' a soma."""
    dec = -d_iter
    return dict(decision=dec, cost=-(d_cpu - d_iter), total_cpu=-d_cpu)


def mde80(d, alpha: float = 0.05) -> float:
    """Efeito minimo detectavel analitico a 80% de poder para a media das diferencas pareadas d (aproximacao normal; otimista para testes de postos com muitos empates)."""
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)) * se)


def planted_power(d, unit, deltas: Sequence[float], alpha: float = 0.05, n_boot: int = 500, seed: int = 0) -> dict:
    """Poder empirico por efeito plantado: reamostra os pares e, em cada um, com probabilidade delta / media(unit) o aprendiz melhora de UMA unidade de objetivo (unit_i = 100 / objetivo_i, em %), de modo que o ganho esperado e' delta;
    detecta quando o Wilcoxon bilateral (empates descartados) rejeita em `alpha` e a media do d plantado e' negativa. Retorna {delta: poder}."""
    d, unit = np.asarray(d, float), np.asarray(unit, float)
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for delta in deltas:
        pi = min(1.0, delta / float(unit.mean()))
        hit = 0
        for b in range(n_boot):
            j = idx[b]
            dp = d[j] - np.where(rng.random(n) < pi, unit[j], 0.0)
            nz = dp[dp != 0]
            hit += int(len(nz) >= 6 and stats.wilcoxon(nz).pvalue < alpha and dp.mean() < 0)
        out[float(delta)] = hit / n_boot
    return out
