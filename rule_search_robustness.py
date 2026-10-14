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
# Continuacao de rule_search_apriori.py: aquele script usou UM split
# treino/teste (70/30, seed=2026) e achou que a regressao logistica
# (gap_pct=1,0606) bate o limiar atual (1,0677) no teste -- uma melhora de
# 0,0071 no gap_pct medio (~15,7% do gap ate o oraculo fechado). Antes de
# tratar isso como um achado real (e nao sorte de UM split especifico),
# este script roda 5-fold CV ESTRATIFICADA por familia -- cada instancia
# vira teste exatamente 1 vez, reporta media +/- desvio padrao do
# gap_pct por fold, pra cada regra. Tambem faz a analise de DISCORDANCIA
# par-a-par entre o limiar atual e a logistica: quando discordam, quem
# acerta mais?
# ---------------------------------------------------------------------------

SEED = 2026
df = pd.read_csv("_scratch_dup_ratio_validation_full.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)
df["nm_ratio"] = df.n / df.m
df["log_n"] = np.log(df.n)
FEATS = ["dup_ratio", "n", "m", "nm_ratio", "log_n"]


def gap_of_mask(sub, choose_ig_mask):
    return np.where(choose_ig_mask, sub.ig_ql, sub.ils_ql).mean()


# ---------------------------------------------------------------------
# 1. 5-fold CV estratificada por familia -- cada instancia e' teste 1x
# ---------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_results = {"threshold_0.10": [], "threshold_0.11": [], "logistic": [], "tree_depth3": []}

for fold_id, (train_idx, test_idx) in enumerate(skf.split(df, df.family)):
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    # limiar atual (0.10), sem re-tunar -- fixo, e' o que o codigo usa hoje
    mask_thr10 = test.dup_ratio.values > 0.10
    fold_results["threshold_0.10"].append(gap_of_mask(test, mask_thr10))

    # limiar re-tunado no treino deste fold
    best_thr, best_gap = None, np.inf
    for thr in np.arange(0.0, 0.61, 0.01):
        g = gap_of_mask(train, train.dup_ratio.values > thr)
        if g < best_gap:
            best_gap, best_thr = g, thr
    mask_thr_tuned = test.dup_ratio.values > best_thr
    fold_results["threshold_0.11"].append(gap_of_mask(test, mask_thr_tuned))

    # logistica
    scaler = StandardScaler().fit(train[FEATS])
    logit = LogisticRegression(random_state=SEED, max_iter=1000)
    logit.fit(scaler.transform(train[FEATS]), train.actual_winner)
    pred = logit.predict(scaler.transform(test[FEATS]))
    fold_results["logistic"].append(gap_of_mask(test, pred == "ig"))

    # arvore rasa
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED, min_samples_leaf=20)
    tree.fit(train[FEATS], train.actual_winner)
    pred_t = tree.predict(test[FEATS])
    fold_results["tree_depth3"].append(gap_of_mask(test, pred_t == "ig"))

    print(f"fold {fold_id}: n_test={len(test)}  thr0.10={fold_results['threshold_0.10'][-1]:.4f}  "
          f"thr_tuned({best_thr:.2f})={fold_results['threshold_0.11'][-1]:.4f}  "
          f"logistic={fold_results['logistic'][-1]:.4f}  tree={fold_results['tree_depth3'][-1]:.4f}", flush=True)

print("\n=== 5-fold CV: media +/- desvio padrao do gap_pct (cada instancia foi teste 1x) ===")
for name, vals in fold_results.items():
    vals = np.array(vals)
    print(f"  {name:20s}: {vals.mean():.4f} +/- {vals.std():.4f}   (folds: {np.round(vals,4)})")

oracle_all = df.oracle_best.mean()
print(f"\n  oracle_best (referencia, nao e' um fold-CV, e' so a media global): {oracle_all:.4f}")

# ---------------------------------------------------------------------
# 2. Analise de discordancia: threshold=0.10 (regra atual) vs logistica
#    treinada em TODOS os dados exceto o proprio ponto (leave-one-out
#    seria caro; aqui usamos o modelo treinado no split 70/30 original
#    pra looking at onde os dois discordam no teste)
# ---------------------------------------------------------------------
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df.family)
scaler = StandardScaler().fit(train[FEATS])
logit = LogisticRegression(random_state=SEED, max_iter=1000)
logit.fit(scaler.transform(train[FEATS]), train.actual_winner)
print("\nclasses_ da logistica (ordem alfabetica, define o sinal dos coeficientes):", logit.classes_)

test = test.copy()
test["pred_thr10"] = np.where(test.dup_ratio.values > 0.10, "ig", "ils")
test["pred_logit"] = logit.predict(scaler.transform(test[FEATS]))
disagree = test[test.pred_thr10 != test.pred_logit].copy()
print(f"\n=== discordancia entre threshold=0.10 e logistica, no TESTE: {len(disagree)}/{len(test)} instancias ===")
disagree["gap_thr10"] = np.where(disagree.pred_thr10 == "ig", disagree.ig_ql, disagree.ils_ql)
disagree["gap_logit"] = np.where(disagree.pred_logit == "ig", disagree.ig_ql, disagree.ils_ql)
logit_better = (disagree.gap_logit < disagree.gap_thr10 - 1e-9).sum()
thr10_better = (disagree.gap_thr10 < disagree.gap_logit - 1e-9).sum()
tie = len(disagree) - logit_better - thr10_better
print(f"  quando discordam: logistica melhor em {logit_better}, threshold=0.10 melhor em {thr10_better}, empate em {tie}")
print(f"  soma do gap 'a mais' pago pelo threshold=0.10 nessas discordancias: {(disagree.gap_thr10 - disagree.gap_logit).sum():.4f}")
print("\n  amostra das discordancias (10 primeiras):")
print(disagree[["member","family","n","m","dup_ratio","actual_winner","pred_thr10","pred_logit","gap_thr10","gap_logit"]].head(10).to_string(index=False))

print("\nDONE")
