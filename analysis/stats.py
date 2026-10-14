"""
analysis/stats.py
==================
Correcao para comparacoes multiplas (Holm-Bonferroni), comparacao
Wilcoxon pareada por familia de hipoteses, effect sizes (correlacao
rank-biserial pareada + IC 95% bootstrap da diferenca media) e teste
formal de igualdade de variancia (Levene/Fligner-Killeen).

Historico:
- Holm-Bonferroni + pairwise_wilcoxon_family: criados na Entrada 37
  (diario_pesquisa.md) a pedido explicito de Carlos: nenhuma correcao
  para multiplas comparacoes tinha sido aplicada em nenhum lugar do
  projeto ate aqui, apesar de sinalizada como pendencia 3 vezes
  (ANALISE_E_PLANO_EJOR.md Secao 8 Fase 4; PLANO_EXPERIMENTAL_EXTENSIVO.md;
  uma entrada antiga do diario). Decisao de Carlos: aplicar
  retroativamente sobre TODAS as comparacoes ja existentes.
- rank_biserial_paired, bootstrap_ci_mean_diff, variance_equality_family:
  adicionados na Entrada 53 (Task #45) -- effect sizes/IC 95% planejados
  desde PLANO_EXPERIMENTAL_EXTENSIVO.md, e o teste formal de variancia
  motivado pela hipotese "Q-Learning reduz variancia, nao media"
  levantada independentemente por 2 consultas ao Gemini (Entrada 47).
  pairwise_wilcoxon_family GANHOU 4 colunas novas (mean_diff,
  rank_biserial, ci_low, ci_high) sem quebrar nenhuma chamada
  existente -- todos os parametros novos tem default e todas as
  colunas antigas permanecem com o mesmo nome/significado.

Definicao de "familia" (decisao de projeto, documentada aqui por
transparencia): tratamos cada FAMILIA CIENTIFICA DE HIPOTESES -- um
conjunto de comparacoes que juntas sustentam UMA alegacao especifica do
artigo -- como uma familia separada para fins de correcao, em vez de
uma unica correcao global sobre TODOS os p-valores ja calculados no
projeto inteiro (o que seria estatisticamente valido mas
desnecessariamente conservador e misturaria perguntas cientificas nao
relacionadas). Isso e a pratica padrao em artigos empiricos de
OR/ML: a familia e o conjunto de testes que sustentam uma alegacao,
nao "todo p-valor que o projeto ja calculou". Ver diario_pesquisa.md,
Entrada 37, para a lista completa de familias aplicadas e a
justificativa desta escolha.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Correcao de Holm-Bonferroni (step-down) para comparacoes multiplas.
    Controla o Family-Wise Error Rate (FWER) em `alpha`; uniformemente
    mais poderosa que Bonferroni simples (rejeita pelo menos tantas
    hipoteses quanto Bonferroni, tipicamente mais), sem exigir
    suposicoes de independencia entre os testes.

    Procedimento (Holm, 1979): ordena os m p-valores em ordem
    crescente p_(1) <= ... <= p_(m). p-valor ajustado da posicao i:
        p_adj_(i) = max_{j<=i} [ (m - j + 1) * p_(j) ]   (truncado em 1.0)
    Rejeita H_(i) se p_adj_(i) <= alpha.

    Parametros
    ----------
    pvalues : sequencia de float
        P-valores brutos de UMA familia de testes relacionados (ver
        docstring do modulo para a definicao de familia usada aqui).
    alpha : float

    Retorna
    -------
    dict com 'p_adjusted' (mesma ordem de entrada), 'reject' (bool),
    e 'm' (tamanho da familia).
    """
    pvalues = list(pvalues)
    m = len(pvalues)
    if m == 0:
        return {"p_adjusted": [], "reject": [], "m": 0}

    order = sorted(range(m), key=lambda i: pvalues[i])
    p_adj_sorted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(factor * pvalues[idx], 1.0)
        running_max = max(running_max, val)
        p_adj_sorted[rank] = running_max

    p_adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        p_adjusted[idx] = p_adj_sorted[rank]

    reject = [p <= alpha for p in p_adjusted]
    return {"p_adjusted": p_adjusted, "reject": reject, "m": m}


# ===========================================================================
# Effect sizes (Entrada 53 / Task #45)
# ===========================================================================

def rank_biserial_paired(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Correlacao rank-biserial PAREADA (matched-pairs rank-biserial
    correlation -- Kerby, 2014, "The simple difference formula",
    Innovative Teaching 3(1)). Effect size companion natural do teste
    de Wilcoxon signed-rank -- mesma quantidade que pacotes como
    pingouin.wilcoxon() reportam na coluna 'RBC'.

    Formula: sejam d_i = x_i - y_i as diferencas pareadas NAO NULAS
    (i=1..n), e rank(|d_i|) o rank (medio em caso de empate) do valor
    absoluto de cada diferenca. Com W+ = soma dos ranks onde d_i > 0 e
    W- = soma dos ranks onde d_i < 0 (W+ + W- = n(n+1)/2):

        r_rb = (W+ - W-) / (W+ + W-)

    r_rb in [-1, 1]. r_rb=0 -> nenhuma tendencia sistematica; |r_rb|=1
    -> toda diferenca favorece o mesmo lado.

    CONVENCAO DE SINAL (documentar no texto do artigo): d_i = x_i - y_i,
    entao r_rb > 0 significa que x tende a ser MAIOR que y. Usado em
    pairwise_wilcoxon_family com x=gap do metodo de referencia e
    y=gap do metodo comparado -- r_rb > 0 ai significa que o metodo de
    referencia tem gap MAIOR (PIOR desempenho), mesma convencao de
    sinal ja usada na coluna 'direction' (mean_ref > mean_other ->
    "ref_worse").

    Retorna NaN se nao houver diferencas nao-nulas (empate total em
    todos os pares) -- mesmo caso em que scipy.stats.wilcoxon nao
    consegue calcular p-valor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    return (w_pos - w_neg) / (w_pos + w_neg)


def bootstrap_ci_mean_diff(
    x: Sequence[float],
    y: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 9999,
    method: str = "BCa",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Intervalo de confianca bootstrap para a diferenca media PAREADA
    mean(x - y), via scipy.stats.bootstrap.

    method='BCa' (bias-corrected and accelerated, Efron & Tibshirani
    1993) por padrao -- mais preciso que o metodo percentil simples
    para distribuicoes assimetricas, o caso tipico do gap% (limitado
    em 0, cauda longa a direita). Cai automaticamente para
    method='percentile' se o BCa falhar (caso degenerado comum: todas
    as diferencas identicas -> variancia zero -> aceleracao
    indefinida); se AINDA assim falhar, retorna NaN nos limites em vez
    de interromper a bateria inteira por causa de uma unica comparacao
    degenerada.

    Parametros
    ----------
    x, y : sequencias pareadas do MESMO tamanho e MESMA ordem (mesmo
        'rep' na posicao i de x e y).
    confidence : nivel de confianca (padrao 0.95 -> IC 95%).
    n_resamples : numero de reamostragens bootstrap (padrao 9999).
    seed : semente para reprodutibilidade (recomendado sempre passar
        um valor explicito em uso em lote -- ver bootstrap_seed em
        pairwise_wilcoxon_family).

    Retorna
    -------
    dict com 'mean_diff' (estimativa pontual, nao-bootstrap), 'ci_low',
    'ci_high'.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    n = len(d)
    mean_diff = float(d.mean()) if n > 0 else float("nan")

    if n < 2:
        return {"mean_diff": mean_diff, "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    for m in (method, "percentile"):
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                res = stats.bootstrap(
                    (d,), np.mean, confidence_level=confidence,
                    n_resamples=n_resamples, method=m, random_state=rng,
                )
            ci_low = float(res.confidence_interval.low)
            ci_high = float(res.confidence_interval.high)
            return {"mean_diff": mean_diff, "ci_low": ci_low, "ci_high": ci_high}
        except Exception:
            continue
    return {"mean_diff": mean_diff, "ci_low": float("nan"), "ci_high": float("nan")}


def pairwise_wilcoxon_family(
    df: "pd.DataFrame",
    reference_method: str,
    other_methods: Sequence[str],
    sizes: Optional[Sequence[Tuple[int, int]]] = None,
    alpha: float = 0.05,
    family_name: str = "",
    n_resamples: int = 9999,
    bootstrap_seed: int = 0,
) -> "pd.DataFrame":
    """
    Teste de Wilcoxon pareado entre `reference_method` e cada metodo em
    `other_methods`, para cada (n,m) em `sizes` (todas as presentes em
    df, por padrao) -- o conjunto INTEIRO resultante e tratado como UMA
    familia de hipoteses para a correcao de Holm-Bonferroni (ver
    docstring do modulo).

    Desde a Entrada 53 (Task #45), TAMBEM calcula para cada comparacao:
    a diferenca media pareada (mean_diff = mean(ref) - mean(other)),
    seu IC 95% bootstrap (bootstrap_ci_mean_diff, BCa) e a correlacao
    rank-biserial pareada (rank_biserial_paired) -- effect size do
    proprio teste de Wilcoxon, nao so o p-valor. Isso distingue
    "nao significativo" (p_holm >= alpha) de "comprovadamente nulo"
    (IC 95% estreito em torno de zero) -- ver LEI_GERAL_PESQUISA.md,
    precedente EDP, e ANALISE_E_PLANO_EJOR.md Secao 8.1.

    O pareamento usa a coluna 'rep' (mesma instancia/seed para os dois
    metodos) via merge -- pares com N diferente entre metodos (comum
    neste projeto, onde alguns metodos tem N=30 e outros N=100 numa
    mesma classe) usam automaticamente so os reps em comum (min
    disponivel), sem vies.

    Parametros novos (Entrada 53)
    ------------------------------
    n_resamples : numero de reamostragens bootstrap por comparacao
        (repassado a bootstrap_ci_mean_diff).
    bootstrap_seed : semente base; a comparacao de indice i na familia
        usa seed=bootstrap_seed+i (reprodutivel, mas cada comparacao
        usa um padrao de reamostragem distinto).

    Retorna DataFrame longo com colunas: family, reference, other, n, m,
    n_pairs, p_raw, p_holm, reject_holm, mean_ref, mean_other,
    mean_diff, ci_low, ci_high, rank_biserial, direction.
    """
    if sizes is None:
        sizes = (
            df[["n", "m"]]
            .drop_duplicates()
            .sort_values(["n", "m"])
            .itertuples(index=False, name=None)
        )
        sizes = list(sizes)

    rows: List[Dict[str, Any]] = []
    for other in other_methods:
        for n_val, m_val in sizes:
            sub_ref = df[(df["method"] == reference_method) & (df["n"] == n_val) & (df["m"] == m_val)]
            sub_other = df[(df["method"] == other) & (df["n"] == n_val) & (df["m"] == m_val)]
            merged = pd.merge(
                sub_ref[["rep", "gap_pct"]].rename(columns={"gap_pct": "ref"}),
                sub_other[["rep", "gap_pct"]].rename(columns={"gap_pct": "other"}),
                on="rep",
            )
            n_pairs = len(merged)
            mean_ref = float(merged["ref"].mean()) if n_pairs else float("nan")
            mean_other = float(merged["other"].mean()) if n_pairs else float("nan")

            p_raw = float("nan")
            rank_biserial = float("nan")
            mean_diff = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
            if n_pairs >= 3:
                diffs = merged["ref"] - merged["other"]
                if not (diffs == 0).all():
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")
                        _, p_raw = stats.wilcoxon(
                            merged["ref"], merged["other"],
                            alternative="two-sided", zero_method="wilcox",
                        )
                    rank_biserial = rank_biserial_paired(merged["ref"], merged["other"])
                boot = bootstrap_ci_mean_diff(
                    merged["ref"], merged["other"], n_resamples=n_resamples,
                    seed=bootstrap_seed + len(rows),
                )
                mean_diff, ci_low, ci_high = boot["mean_diff"], boot["ci_low"], boot["ci_high"]

            rows.append(dict(
                family=family_name, reference=reference_method, other=other,
                n=n_val, m=m_val, n_pairs=n_pairs, p_raw=p_raw,
                mean_ref=mean_ref, mean_other=mean_other,
                mean_diff=mean_diff, ci_low=ci_low, ci_high=ci_high,
                rank_biserial=rank_biserial,
            ))

    result = pd.DataFrame(rows)
    valid = result["p_raw"].notna()
    holm = holm_bonferroni(result.loc[valid, "p_raw"].tolist(), alpha=alpha)
    # Constroi arrays de tamanho completo antes de atribuir -- evita o
    # FutureWarning do pandas sobre dtype incompativel ao usar .loc[]
    # com uma lista (possivelmente vazia) via mascara booleana.
    p_holm_full = np.full(len(result), np.nan)
    reject_full = np.zeros(len(result), dtype=bool)
    p_holm_full[valid.to_numpy()] = holm["p_adjusted"]
    reject_full[valid.to_numpy()] = holm["reject"]
    result["p_holm"] = p_holm_full
    result["reject_holm"] = reject_full

    def _direction(r):
        if pd.isna(r["mean_ref"]) or pd.isna(r["mean_other"]):
            return "n/a"
        if r["mean_ref"] < r["mean_other"]:
            return "ref_better"
        if r["mean_ref"] > r["mean_other"]:
            return "ref_worse"
        return "tie"

    result["direction"] = result.apply(_direction, axis=1)
    return result


# ===========================================================================
# Igualdade de variancia (Entrada 53 / Task #45, motivado pela Entrada 47)
# ===========================================================================

def variance_equality_family(
    df: "pd.DataFrame",
    reference_method: str,
    other_methods: Sequence[str],
    sizes: Optional[Sequence[Tuple[int, int]]] = None,
    alpha: float = 0.05,
    family_name: str = "",
) -> "pd.DataFrame":
    """
    Testa IGUALDADE DE VARIANCIA do gap% entre `reference_method` e
    cada metodo em `other_methods`, para cada (n,m) -- usa Levene
    (centrado na mediana -- variante Brown-Forsythe, mais robusta a
    nao-normalidade que o Levene classico centrado na media) E
    Fligner-Killeen (nao-parametrico, ainda mais robusto a
    nao-normalidade -- gap% e limitado em 0 e tipicamente assimetrico
    a direita, violando a normalidade que o Levene classico assume)
    LADO A LADO -- quando os dois concordam na decisao, e evidencia
    mais forte do que qualquer um isolado.

    Motivacao (diario_pesquisa.md, Entrada 47): duas consultas
    INDEPENDENTES ao Gemini (flash-lite e 3-flash-preview, sem uma
    saber da outra) levantaram a mesma hipotese -- Q-Learning reduz a
    VARIANCIA do gap sem necessariamente reduzir a MEDIA. Uma checagem
    descritiva rapida (nao formal) confirmou o padrao (std do gap% de
    ig_ql <= ig_random) em TODAS as 6 classes originais. Esta funcao
    formaliza esse teste.

    Diferente de pairwise_wilcoxon_family, NAO faz pareamento por
    'rep' -- Levene/Fligner comparam a DISPERSAO de duas amostras
    (podem ate ter N diferente), nao diferencas pareadas observacao a
    observacao.

    Os 2 testes (Levene e Fligner) sao corrigidos por Holm-Bonferroni
    SEPARADAMENTE (2 colunas p_holm distintas) -- sao 2 estatisticas
    diferentes da MESMA hipotese substantiva; misturar as duas listas
    de p-valores numa unica correcao infla artificialmente o tamanho
    nominal da familia (dobraria o numero de testes sem dobrar o
    numero de PERGUNTAS cientificas feitas).

    Retorna DataFrame longo com colunas: family, reference, other, n,
    m, n_ref, n_other, std_ref, std_other, ratio_std (std_ref/std_other),
    p_levene, p_holm_levene, reject_holm_levene, p_fligner,
    p_holm_fligner, reject_holm_fligner, agreement (True se os 2 testes
    concordam na decisao rejeitar/nao-rejeitar).
    """
    if sizes is None:
        sizes = (
            df[["n", "m"]]
            .drop_duplicates()
            .sort_values(["n", "m"])
            .itertuples(index=False, name=None)
        )
        sizes = list(sizes)

    rows: List[Dict[str, Any]] = []
    for other in other_methods:
        for n_val, m_val in sizes:
            sub_ref = df[(df["method"] == reference_method) & (df["n"] == n_val) & (df["m"] == m_val)]["gap_pct"]
            sub_other = df[(df["method"] == other) & (df["n"] == n_val) & (df["m"] == m_val)]["gap_pct"]
            n_ref, n_other = len(sub_ref), len(sub_other)
            std_ref = float(sub_ref.std(ddof=1)) if n_ref >= 2 else float("nan")
            std_other = float(sub_other.std(ddof=1)) if n_other >= 2 else float("nan")
            ratio_std = (std_ref / std_other) if (std_other not in (0.0, float("nan")) and not pd.isna(std_other)) else float("nan")

            p_levene = float("nan")
            p_fligner = float("nan")
            if n_ref >= 2 and n_other >= 2:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    try:
                        _, p_levene = stats.levene(sub_ref, sub_other, center="median")
                    except Exception:
                        pass
                    try:
                        _, p_fligner = stats.fligner(sub_ref, sub_other)
                    except Exception:
                        pass

            rows.append(dict(
                family=family_name, reference=reference_method, other=other,
                n=n_val, m=m_val, n_ref=n_ref, n_other=n_other,
                std_ref=std_ref, std_other=std_other, ratio_std=ratio_std,
                p_levene=p_levene, p_fligner=p_fligner,
            ))

    result = pd.DataFrame(rows)

    for test_col, holm_col, reject_col in (
        ("p_levene", "p_holm_levene", "reject_holm_levene"),
        ("p_fligner", "p_holm_fligner", "reject_holm_fligner"),
    ):
        valid = result[test_col].notna()
        holm = holm_bonferroni(result.loc[valid, test_col].tolist(), alpha=alpha)
        p_holm_full = np.full(len(result), np.nan)
        reject_full = np.zeros(len(result), dtype=bool)
        p_holm_full[valid.to_numpy()] = holm["p_adjusted"]
        reject_full[valid.to_numpy()] = holm["reject"]
        result[holm_col] = p_holm_full
        result[reject_col] = reject_full

    result["agreement"] = result["reject_holm_levene"] == result["reject_holm_fligner"]
    return result
