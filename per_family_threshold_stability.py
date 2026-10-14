import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

pd.set_option("display.width", 220)

# ---------------------------------------------------------------------------
# Entrada 64 testou limiares por familia com UM split 70/30 (escolhidos de
# subconjuntos de treino pequenos, por familia) e concluiu que pioravam no
# teste -- mas aquele teste combinava as 4 familias de uma vez, escondendo
# se alguma familia ESPECIFICA (como E, acabou de mostrar esta entrada)
# tem um limiar genuinamente estavel. Este script repete, com o MESMO
# rigor de 5-fold CV usado agora pra E, pra TODAS as 4 familias
# separadamente -- decide, por familia, se o limiar retunado e' estavel
# (converge pro mesmo valor entre folds) e se generaliza (CV-gap melhor
# que o threshold=0,1 fixo). So' aceita como achado real o que passar nos
# dois criterios -- estabilidade E generalizacao -- nao so' um dos dois.
# ---------------------------------------------------------------------------

SEED = 2026
df = pd.read_csv("_scratch_extended_features.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)


def gap_of_mask(sub, choose_ig_mask):
    return np.where(choose_ig_mask, sub.ig_ql, sub.ils_ql).mean()


def best_threshold_gap(train, feat="dup_ratio"):
    best = (None, None, np.inf)
    vals = train[feat].dropna()
    qs = np.quantile(vals, np.linspace(0.02, 0.98, 49))
    for direction in ["gt", "lt"]:
        for thr in qs:
            mask = train[feat] > thr if direction == "gt" else train[feat] < thr
            g = gap_of_mask(train, mask)
            if g < best[2]:
                best = (direction, thr, g)
    return best


print(f"{'familia':8s} {'n':>5s} {'gap_thr=0.1':>12s} {'gap_CV_retunado':>16s} {'limiares_por_fold':>50s} {'estavel?':>9s}")
results = []
for fam, dfam in df.groupby("family"):
    dfam = dfam.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baseline = gap_of_mask(dfam, dfam.dup_ratio > 0.1)
    fold_gaps, fold_thrs = [], []
    for train_idx, test_idx in skf.split(dfam, dfam.actual_winner):
        train, test = dfam.iloc[train_idx], dfam.iloc[test_idx]
        direction, thr, _ = best_threshold_gap(train)
        mask_test = test.dup_ratio > thr if direction == "gt" else test.dup_ratio < thr
        fold_gaps.append(gap_of_mask(test, mask_test))
        fold_thrs.append((direction, round(thr, 4)))
    cv_gap = np.mean(fold_gaps)
    unique_thrs = set(fold_thrs)
    estavel = len(unique_thrs) == 1
    results.append((fam, len(dfam), baseline, cv_gap, fold_thrs, estavel))
    print(f"{fam:8s} {len(dfam):5d} {baseline:12.4f} {cv_gap:16.4f} {str(fold_thrs):>50s} {str(estavel):>9s}")

print("\n=== resumo: melhora (baseline - cv_gap), so' aceitando como achado real se ESTAVEL ===")
for fam, n, baseline, cv_gap, fold_thrs, estavel in results:
    melhora = baseline - cv_gap
    veredito = "REAL (estavel + melhora)" if (estavel and melhora > 0) else (
        "instavel -- nao confiar" if not estavel else "estavel mas sem melhora real")
    print(f"  {fam}: melhora={melhora:+.4f}  -> {veredito}")

print("\nDONE")
