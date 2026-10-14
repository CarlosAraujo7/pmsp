import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

pd.set_option("display.width", 220)

# ---------------------------------------------------------------------------
# verify_discretization_artifact.py mostrou que a "melhora" de +0,0130 em E
# (Entrada 66) e' dominada por 2 instancias de E1 (n=6,m=3, dup_ratio caindo
# EXATAMENTE em 1/6=0,1667 -- artefato de granularidade discreta, nao sinal
# de familia) -- excluindo E1 inteiramente, o efeito NA MEDIA JA INVERTE pra
# -0,0087 (piora). Este script fecha a duvida decisiva: repete a MESMA
# validacao de estabilidade por 5-fold CV da Entrada 66 (mesmo protocolo:
# escolhe o limiar so' no treino, mede no teste), agora restrita a E2+E3
# (360 das 400 instancias de E, excluindo o subconjunto degenerado E1) --
# pra ver se sobra ALGUM sinal real depois de remover o outlier, ou se a
# familia E, na sua maioria representativa, simplesmente NAO se beneficia
# de um limiar diferente de 0,1 (igual as familias B/C/F).
# ---------------------------------------------------------------------------

SEED = 2026
df = pd.read_csv("_scratch_extended_features.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)
dfe_full = df[df.family == "E"].copy().reset_index(drop=True)
dfe_no_e1 = df[(df.family == "E") & (df.subfolder != "E1")].copy().reset_index(drop=True)


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


for label, dfx in [("familia E COMPLETA (com E1, 400 inst.) -- repete Entrada 66", dfe_full),
                    ("familia E SEM E1 (so E2+E3, 360 inst.)", dfe_no_e1)]:
    print(f"\n=== {label} ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baseline = gap_of_mask(dfx, dfx.dup_ratio > 0.1)
    fold_gaps, fold_thrs = [], []
    for train_idx, test_idx in skf.split(dfx, dfx.actual_winner):
        train, test = dfx.iloc[train_idx], dfx.iloc[test_idx]
        direction, thr, _ = best_threshold_gap(train)
        mask_test = test.dup_ratio > thr if direction == "gt" else test.dup_ratio < thr
        fold_gaps.append(gap_of_mask(test, mask_test))
        fold_thrs.append((direction, round(thr, 4)))
    cv_gap = np.mean(fold_gaps)
    print(f"  baseline (threshold=0,1): {baseline:.4f}")
    print(f"  CV-retunado: {cv_gap:.4f}  (melhora: {baseline-cv_gap:+.4f})")
    print(f"  limiares por fold: {fold_thrs}")
    print(f"  estavel? {len(set(fold_thrs)) == 1}")

print("\n=== checagem adicional: e' so' tirar QUALQUER 2 instancias, ou e' especifico dessas 2 de E1? ===")
print("(reamostragem: tirar 2 instancias aleatorias de E, 200 repeticoes, ver quantas vezes a melhora reportada")
print(" (+0,0130) sobrevive -- se for fragil a QUALQUER remocao de 2 instancias, reforca que 'sinal' e' artefato)")
rng = np.random.default_rng(SEED)
melhoras = []
gap01_full = gap_of_mask(dfe_full, dfe_full.dup_ratio > 0.1)
for _ in range(200):
    drop_idx = rng.choice(dfe_full.index, size=2, replace=False)
    sub = dfe_full.drop(drop_idx)
    g01 = gap_of_mask(sub, sub.dup_ratio > 0.1)
    g01667 = gap_of_mask(sub, sub.dup_ratio > 0.1667)
    melhoras.append(g01 - g01667)
melhoras = np.array(melhoras)
print(f"  melhora original (com as 400): +0,0130")
print(f"  removendo 2 instancias aleatorias (200 repeticoes): media={melhoras.mean():.4f}  "
      f"mediana={np.median(melhoras):.4f}  min={melhoras.min():.4f}  max={melhoras.max():.4f}")
print(f"  fracao das 200 repeticoes onde a melhora vira NEGATIVA so' de tirar 2 instancias aleatorias: "
      f"{(melhoras < 0).mean()*100:.1f}%")

print("\nDONE")
