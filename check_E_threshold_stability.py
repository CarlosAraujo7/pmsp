import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

pd.set_option("display.width", 220)

# ---------------------------------------------------------------------------
# Continuacao imediata de analyze_extended_features_E.py: dup_ratio
# retunado especificamente pra familia E mostrou uma melhora CV-media
# real (0,7524 contra 0,7654 do threshold=0,1 fixo). Antes de tratar isso
# como achado (e nao like o overfitting ja pego nos limiares por familia
# da Entrada 64), verifica a ESTABILIDADE do limiar escolhido por fold --
# um sinal real deveria convergir pro mesmo valor (como w=0,28 convergiu
# em toda familia C, Entrada 65); um sinal espurio varia sem padrao.
# ---------------------------------------------------------------------------

SEED = 2026
df = pd.read_csv("_scratch_extended_features.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)
dfe = df[df.family == "E"].copy().reset_index(drop=True)


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


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("=== limiar de dup_ratio escolhido por fold (familia E) ===")
chosen = []
for fold_id, (train_idx, test_idx) in enumerate(skf.split(dfe, dfe.actual_winner)):
    train, test = dfe.iloc[train_idx], dfe.iloc[test_idx]
    direction, thr, gap_train = best_threshold_gap(train)
    mask_test = test.dup_ratio > thr if direction == "gt" else test.dup_ratio < thr
    gap_test = gap_of_mask(test, mask_test)
    chosen.append((direction, thr))
    print(f"  fold {fold_id}: direcao={direction:3s} limiar={thr:.4f}  gap_treino={gap_train:.4f}  gap_teste={gap_test:.4f}")

print(f"\nlimiares escolhidos: {[(d, round(t,4)) for d,t in chosen]}")
thrs = [t for d, t in chosen if d == "gt"]
print(f"media dos limiares 'gt' (se todos concordarem na direcao): {np.mean(thrs) if thrs else 'N/A (direcoes divergem)'}")

# ---------------------------------------------------------------------
# Teste definitivo: um UNICO limiar fixo (nao retunado por fold),
# escolhido so' 1x a partir de TODA a familia E, testado com CV honesta
# (ou seja: repete o esquema real que apriori_select usaria em producao
# -- um limiar fixo pra familia E, nao um limiar que muda por fold)
# ---------------------------------------------------------------------
direction_full, thr_full, _ = best_threshold_gap(dfe)
print(f"\nlimiar UNICO escolhido usando TODA a familia E: direcao={direction_full} limiar={thr_full:.4f}")

# valida esse limiar UNICO com 5-fold CV (aplicando o MESMO limiar fixo em
# cada fold de teste -- simula o que aconteceria em producao: um so'
# limiar pra familia E, nao retunado por instancia)
fold_gaps_fixed = []
for train_idx, test_idx in skf.split(dfe, dfe.actual_winner):
    test = dfe.iloc[test_idx]
    mask_test = test.dup_ratio > thr_full if direction_full == "gt" else test.dup_ratio < thr_full
    fold_gaps_fixed.append(gap_of_mask(test, mask_test))
print(f"gap_pct desse limiar UNICO fixo, medido nos 5 folds de teste: "
      f"{np.mean(fold_gaps_fixed):.4f} +/- {np.std(fold_gaps_fixed):.4f}")
print(f"(comparar com threshold=0,1 atual: {gap_of_mask(dfe, dfe.dup_ratio > 0.1):.4f}, "
      f"e com oracle_best: {dfe.oracle_best.mean():.4f})")

print("\nDONE")
