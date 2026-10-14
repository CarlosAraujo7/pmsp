import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from analysis.stats import pairwise_wilcoxon_family

pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 200)

# ---------------------------------------------------------------------------
# A pedido de Carlos: antes de escalar a execucao (ampliar amostra externa +
# completar metodos nas classes grandes + rodar as 2 classes zeradas),
# checar com rigor se os metodos ja rodados (ig_ql, ablacoes, baselines
# SOTA) estao trazendo contribuicoes UTEIS -- nao reciclar conclusoes do
# diario, recalcular direto de results/results.csv com o proprio toolkit
# estatistico do projeto (Wilcoxon pareado + Holm-Bonferroni + rank-biserial
# + IC 95% bootstrap, ja implementado em analysis/stats.py).
# ---------------------------------------------------------------------------

df = pd.read_csv("results/results.csv")
print(f"total de linhas em results.csv: {len(df)}")
print(f"classes (n,m) presentes: {sorted(df[['n','m']].drop_duplicates().apply(tuple, axis=1).tolist())}")
print()

ORIGINAIS = [(20, 4), (20, 6), (30, 8), (50, 10), (75, 15), (100, 20)]
GRANDES = [(150, 15), (150, 30), (250, 25), (300, 20)]

REF = "ig_ql"
ABLACOES = ["ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql_flat"]
SOTA = ["bco", "multifit", "sa_li_perkins"]
ILS_FORTES_ORIGINAIS = ["ils_ql", "ils_ql_hi", "ils_ql_ws", "ils_rand_hi", "ils_rf", "ils_ucb"]
ILS_FORTES_GRANDES = ["ils_ql", "ils_rf", "ils_ucb"]  # so esses existem nas classes grandes


def resumo(res, titulo):
    print(f"\n=== {titulo} ===")
    cols = ["other", "n", "m", "n_pairs", "mean_ref", "mean_other", "mean_diff",
            "ci_low", "ci_high", "rank_biserial", "p_holm", "reject_holm", "direction"]
    print(res[cols].round(4).to_string(index=False))
    sig = res[res.reject_holm]
    print(f"\n-> {len(sig)}/{len(res)} comparacoes significativas (Holm-Bonferroni, alpha=0,05)")
    if len(sig):
        favor_ref = (sig.direction == "ref_better").sum()
        print(f"   das quais {favor_ref} favoraveis a {REF}, {len(sig)-favor_ref} desfavoraveis")


print("############################################################")
print("# 1. ig_ql vs SOTA (bco, multifit, sa_li_perkins) -- 6 classes originais")
print("############################################################")
res1 = pairwise_wilcoxon_family(df, REF, SOTA, sizes=ORIGINAIS, family_name="originais_vs_SOTA")
resumo(res1, "ig_ql vs SOTA, classes originais (20/4 a 100/20)")

print("\n\n############################################################")
print("# 2. ig_ql vs SOTA -- 4 classes GRANDES (150-300 jobs)")
print("############################################################")
res2 = pairwise_wilcoxon_family(df, REF, SOTA, sizes=GRANDES, family_name="grandes_vs_SOTA")
resumo(res2, "ig_ql vs SOTA, classes grandes (150/15 a 300/20)")

print("\n\n############################################################")
print("# 3. ig_ql vs suas proprias ablacoes -- 6 classes originais (isola mecanismo)")
print("############################################################")
res3 = pairwise_wilcoxon_family(df, REF, ABLACOES, sizes=ORIGINAIS, family_name="originais_vs_ablacoes")
resumo(res3, "ig_ql vs ablacoes, classes originais")

print("\n\n############################################################")
print("# 4. ig_ql vs suas proprias ablacoes -- 4 classes GRANDES")
print("############################################################")
res4 = pairwise_wilcoxon_family(df, REF, ABLACOES, sizes=GRANDES, family_name="grandes_vs_ablacoes")
resumo(res4, "ig_ql vs ablacoes, classes grandes")

print("\n\n############################################################")
print("# 5. ig_ql vs melhores concorrentes ILS -- 6 classes originais")
print("############################################################")
res5 = pairwise_wilcoxon_family(df, REF, ILS_FORTES_ORIGINAIS, sizes=ORIGINAIS, family_name="originais_vs_ILS_fortes")
resumo(res5, "ig_ql vs ILS fortes, classes originais")

print("\n\n############################################################")
print("# 6. ig_ql vs melhores concorrentes ILS -- 4 classes GRANDES")
print("############################################################")
res6 = pairwise_wilcoxon_family(df, REF, ILS_FORTES_GRANDES, sizes=GRANDES, family_name="grandes_vs_ILS_fortes")
resumo(res6, "ig_ql vs ILS fortes, classes grandes")

print("\n\nDONE")
