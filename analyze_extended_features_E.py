import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# Analise das features estendidas (feature_search_extended.py), focada na
# familia E -- ja aplicando de saida o rigor de CV que a Entrada 64 mostrou
# ser necessario (nao repetir o erro de confiar num split unico).
# ---------------------------------------------------------------------------

SEED = 2026
FEATURES = ["dup_ratio", "cv", "skewness", "kurtosis", "entropy_hist",
            "n_unique_over_m", "range_over_mean", "iqr_over_mean", "gini", "lpt_load_cv"]

df = pd.read_csv("_scratch_extended_features.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)
dfe = df[df.family == "E"].copy().reset_index(drop=True)
print(f"familia E: {len(dfe)} instancias")
print(f"gap_pct medio: ils_ql={dfe.ils_ql.mean():.4f}  ig_ql={dfe.ig_ql.mean():.4f}  "
      f"oracle_best={dfe.oracle_best.mean():.4f}")
print(f"distribuicao actual_winner: \n{dfe.actual_winner.value_counts()}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def gap_of_mask(sub, choose_ig_mask):
    return np.where(choose_ig_mask, sub.ig_ql, sub.ils_ql).mean()


def best_threshold_gap(train, feat):
    """Sweep de limiar (as 2 direcoes) minimizando gap_pct no treino.
    Retorna (direcao, limiar, gap_treino)."""
    best = (None, None, np.inf)
    vals = train[feat].dropna()
    qs = np.quantile(vals, np.linspace(0.02, 0.98, 49))
    for direction in ["gt", "lt"]:
        for thr in qs:
            if direction == "gt":
                mask = train[feat] > thr
            else:
                mask = train[feat] < thr
            g = gap_of_mask(train, mask)
            if g < best[2]:
                best = (direction, thr, g)
    return best


# ---------------------------------------------------------------------
# 1. Cada feature isolada, sweep de limiar, 5-fold CV (so' familia E)
# ---------------------------------------------------------------------
print("\n=== cada feature, sweep de limiar (2 direcoes), 5-fold CV, so' familia E ===")
feature_cv_results = {}
for feat in FEATURES:
    fold_gaps = []
    for train_idx, test_idx in skf.split(dfe, dfe.actual_winner):
        train, test = dfe.iloc[train_idx], dfe.iloc[test_idx]
        direction, thr, _ = best_threshold_gap(train, feat)
        mask_test = test[feat] > thr if direction == "gt" else test[feat] < thr
        fold_gaps.append(gap_of_mask(test, mask_test))
    feature_cv_results[feat] = (np.mean(fold_gaps), np.std(fold_gaps))

baseline_thr01 = gap_of_mask(dfe, dfe.dup_ratio > 0.1)
print(f"\nlinha de base atual (dup_ratio, threshold=0,1 fixo, SEM retunar): {baseline_thr01:.4f}")
print(f"oracle_best (teto teorico absoluto): {dfe.oracle_best.mean():.4f}")
print(f"sempre ils_ql: {dfe.ils_ql.mean():.4f}   sempre ig_ql: {dfe.ig_ql.mean():.4f}")
print("\nresultado por feature (CV, limiar retunado por fold):")
res = pd.DataFrame([(f, m, s) for f, (m, s) in feature_cv_results.items()],
                    columns=["feature", "gap_pct_CV_mean", "gap_pct_CV_std"]).sort_values("gap_pct_CV_mean")
print(res.round(4).to_string(index=False))

# ---------------------------------------------------------------------
# 2. Combinacao de todas as features (arvore rasa + logistica), 5-fold CV
# ---------------------------------------------------------------------
print("\n=== combinacao de todas as 10 features (arvore depth=3 e logistica), 5-fold CV ===")
X = dfe[FEATURES].fillna(dfe[FEATURES].median())
y = dfe.actual_winner
fold_gaps_tree, fold_gaps_logit = [], []
for train_idx, test_idx in skf.split(dfe, dfe.actual_winner):
    train, test = dfe.iloc[train_idx], dfe.iloc[test_idx]
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr = y.iloc[train_idx]

    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED, min_samples_leaf=10)
    tree.fit(Xtr, ytr)
    pred_tree = tree.predict(Xte)
    fold_gaps_tree.append(gap_of_mask(test, pred_tree == "ig"))

    scaler = StandardScaler().fit(Xtr)
    logit = LogisticRegression(random_state=SEED, max_iter=2000)
    logit.fit(scaler.transform(Xtr), ytr)
    pred_logit = logit.predict(scaler.transform(Xte))
    fold_gaps_logit.append(gap_of_mask(test, pred_logit == "ig"))

print(f"arvore rasa (todas features): {np.mean(fold_gaps_tree):.4f} +/- {np.std(fold_gaps_tree):.4f}")
print(f"regressao logistica (todas features): {np.mean(fold_gaps_logit):.4f} +/- {np.std(fold_gaps_logit):.4f}")

# ---------------------------------------------------------------------
# 3. Resumo final
# ---------------------------------------------------------------------
print("\n=== RESUMO FINAL, so' familia E ===")
summary = [
    ("oracle_best (teto)", dfe.oracle_best.mean()),
    ("sempre ils_ql", dfe.ils_ql.mean()),
    ("sempre ig_ql", dfe.ig_ql.mean()),
    ("dup_ratio threshold=0,1 (atual, sem retunar)", baseline_thr01),
    (f"melhor feature isolada ({res.iloc[0].feature}, retunada por fold)", res.iloc[0].gap_pct_CV_mean),
    ("arvore rasa (todas as 10 features)", np.mean(fold_gaps_tree)),
    ("regressao logistica (todas as 10 features)", np.mean(fold_gaps_logit)),
]
print(pd.DataFrame(summary, columns=["regra", "gap_pct (CV ou fixo)"]).round(4).to_string(index=False))

print("\nDONE")
