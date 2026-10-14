import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

pd.set_option("display.width", 220)

# ---------------------------------------------------------------------------
# Continuacao imediata de gray_zone_analysis.py: a zona cinzenta (baixa
# acuracia perto do limiar) so' apareceu de forma clara na familia C
# (dentro=59,66% vs fora=77,94% a w=0,05); em B nao faz diferenca (ja'
# 100% dentro), em E o padrao e' INVERTIDO (dentro e' MELHOR que fora),
# em F e' achatado. Antes de escrever isso como "vale a pena so' pra C",
# verifica com o MESMO rigor de CV da Entrada 64 (nao confiar em UM
# corte) se um gray_zone especifico pra familia C generaliza -- ou se e'
# outro caso de overfitting num subconjunto pequeno, como os limiares
# por familia da Entrada 64 ja mostraram ser um risco real.
# ---------------------------------------------------------------------------

THRESHOLD = 0.1
df = pd.read_csv("_scratch_dup_ratio_validation_full.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)
dfc = df[df.family == "C"].copy()
dfc["pred_thr"] = np.where(dfc.dup_ratio > THRESHOLD, "ig", "ils")
dfc["gap_thr"] = np.where(dfc.pred_thr == "ig", dfc.ig_ql, dfc.ils_ql)
dfc["dist_to_thr"] = (dfc.dup_ratio - THRESHOLD).abs()
print(f"familia C: {len(dfc)} instancias")
print(f"gap_pct medio (regra atual, threshold=0,1), so' familia C: {dfc.gap_thr.mean():.4f}")
print(f"oracle_best medio, so' familia C: {dfc.oracle_best.mean():.4f}")


def gap_with_gray_zone(sub, w):
    """gap_pct medio se, dentro da faixa [thr-w, thr+w], usarmos
    oracle_best (melhor caso possivel de um probe perfeito); fora,
    a regra de threshold normal."""
    in_zone = sub.dist_to_thr <= w
    return np.where(in_zone, sub.oracle_best, sub.gap_thr).mean()


# ---------------------------------------------------------------------
# 5-fold CV: pra cada fold, escolhe w no TREINO (minimizando o teto
# teorico -- OTIMISTA de proposito, pra dar a melhor chance a` ideia),
# avalia no TESTE
# ---------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
ws_tested = np.arange(0.0, 0.31, 0.01)
fold_gaps_gated, fold_gaps_baseline, fold_ws = [], [], []

for fold_id, (train_idx, test_idx) in enumerate(skf.split(dfc, dfc.actual_winner)):
    train, test = dfc.iloc[train_idx], dfc.iloc[test_idx]
    best_w, best_gap_train = None, np.inf
    for w in ws_tested:
        g = gap_with_gray_zone(train, w)
        if g < best_gap_train:
            best_gap_train, best_w = g, w
    gap_test_gated = gap_with_gray_zone(test, best_w)
    gap_test_baseline = test.gap_thr.mean()
    fold_gaps_gated.append(gap_test_gated)
    fold_gaps_baseline.append(gap_test_baseline)
    fold_ws.append(best_w)
    print(f"fold {fold_id}: n_test={len(test)}  w_escolhido(treino)={best_w:.2f}  "
          f"gap_baseline(teste)={gap_test_baseline:.4f}  gap_gated_TEORICO(teste)={gap_test_gated:.4f}")

print(f"\n5-fold CV, so' familia C (teto teorico OTIMISTA -- assume probe perfeito na zona):")
print(f"  baseline (threshold puro):        {np.mean(fold_gaps_baseline):.4f} +/- {np.std(fold_gaps_baseline):.4f}")
print(f"  gated (teto teorico, probe perfeito): {np.mean(fold_gaps_gated):.4f} +/- {np.std(fold_gaps_gated):.4f}")
print(f"  larguras escolhidas por fold: {[f'{w:.2f}' for w in fold_ws]}")
print(f"\n  MELHORA (mesmo no MELHOR caso teorico, CV-robusta): {np.mean(fold_gaps_baseline) - np.mean(fold_gaps_gated):.4f}")

print("\nDONE")
