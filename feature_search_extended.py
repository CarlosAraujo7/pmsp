import sys
import time

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy import stats as spstats
from pmsp.constructive import lpt
from pmsp.instance import PMSPInstance
from external_benchmarks.pcmax_symmetry2025.loader import load_instance_array

# ---------------------------------------------------------------------------
# Diagnostico dirigido a` familia E (Task #80, a pedido de Carlos: "vamos
# pela primeira opcao e continuar investigando e testando variacoes de
# novos metodos pra propor na literatura"). apriori_select ja' foi aceito
# como o candidato mais solido (Entrada 63); tres tentativas de melhorar a
# REGRA de despacho nao renderam ganho robusto (Entradas 64-65). Em vez de
# testar mais variacoes as cegas na REGRA, este script busca uma FEATURE
# melhor que dup_ratio -- focado na familia E, que e' onde dup_ratio mais
# falha (Entradas 61-62).
#
# Calcula um conjunto de features candidatas, todas O(n log n) ou mais
# baratas, pra TODAS as 1.612 instancias (nao so' E, pra comparacao):
#   dup_ratio       -- baseline ja' conhecido (1 - n_unicos/n)
#   cv              -- coef. de variacao (std/media) -- ja' refutado em
#                       instancias SINTETICAS (Task #72); aqui testado
#                       direto contra os dados REAIS, pela primeira vez
#   skewness        -- assimetria da distribuicao de p_j
#   kurtosis        -- curtose
#   entropy_hist    -- entropia de Shannon do histograma (20 bins) de p_j
#                       -- mede quao "espalhados"/uniformes os valores sao
#   n_unique_over_m -- fracao de valores unicos relativa ao numero de
#                       MAQUINAS (nao ao numero de jobs, como dup_ratio)
#   range_over_mean -- (max-min)/media
#   iqr_over_mean   -- (p75-p25)/media
#   gini            -- coeficiente de Gini de p_j (desigualdade)
#   lpt_load_cv     -- CV das cargas das MAQUINAS apos LPT (nao dos p_j
#                       brutos) -- unica feature que exige rodar algo
#                       (lpt(), O(n log n), nao e' busca/aprendizado)
#
# Nenhuma simulacao de busca (ils_run/ig_run/etc.) e' rodada aqui -- so'
# leitura de instancia + estatisticas fechadas + 1 chamada de lpt() por
# instancia. actual_winner/ils_ql/ig_ql reaproveitados do CSV ja' calculado.
# ---------------------------------------------------------------------------

REF_CSV = "_scratch_dup_ratio_validation_full.csv"
OUT_CSV = "_scratch_extended_features.csv"


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1])


def entropy_hist(x, bins=20):
    counts, _ = np.histogram(x, bins=bins)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def main():
    ref = pd.read_csv(REF_CSV)
    t0 = time.time()
    rows = []
    for i, row in enumerate(ref.itertuples()):
        m, n, p = load_instance_array(row.family, row.member)
        p = np.asarray(p, dtype=np.float64)

        inst = PMSPInstance(p=p, m=m, name=row.member)
        sol = lpt(inst)
        lpt_load_cv = float(sol.load_cv)  # ja' calculado pela propria classe (pmsp/solution.py)

        n_unique = len(np.unique(p))
        p25, p75 = np.percentile(p, [25, 75])
        rows.append(dict(
            member=row.member, family=row.family,
            cv=float(np.std(p) / np.mean(p)) if np.mean(p) > 0 else np.nan,
            skewness=float(spstats.skew(p)),
            kurtosis=float(spstats.kurtosis(p)),
            entropy_hist=entropy_hist(p),
            n_unique_over_m=n_unique / m,
            range_over_mean=float((p.max() - p.min()) / np.mean(p)) if np.mean(p) > 0 else np.nan,
            iqr_over_mean=float((p75 - p25) / np.mean(p)) if np.mean(p) > 0 else np.nan,
            gini=gini(p),
            lpt_load_cv=lpt_load_cv,
        ))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ref)} ({time.time()-t0:.0f}s)", flush=True)

    feats = pd.DataFrame(rows)
    merged = ref.merge(feats, on=["member", "family"], how="inner")
    merged.to_csv(OUT_CSV, index=False)
    print(f"\ntotal: {len(merged)} instancias, {time.time()-t0:.0f}s. salvo em {OUT_CSV}")
    print("DONE")


if __name__ == "__main__":
    main()
