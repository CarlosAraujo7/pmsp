import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# Hipotese levantada por investigate_E1_outlier.py: o limiar "otimo" de
# 0,1667 pra familia E pode ser um artefato de GRANULARIDADE, nao um sinal
# genuino. Para n=6 (todas as 40 instancias de E1), dup_ratio = 1 - k/6 so'
# pode assumir os valores {0; 0,1667; 0,3333; 0,5; 0,6667; 0,8333; 1,0} --
# 0,1667 = 1/6 EXATAMENTE. B (n=4,m=2 fixo) e F (n=10,m=5 fixo) tambem tem
# dup_ratio discreto, mas nenhum de seus niveis cai perto de 0,1667 (B: 0;
# 0,25; 0,5; 0,75 -- F: multiplos de 0,1), por isso threshold=0,1 e
# threshold=0,1667 dao EXATAMENTE o mesmo resultado nesses dois (diff=0,000
# ja confirmado). Esse script verifica diretamente: (1) confirma a
# aritmetica, (2) identifica EXATAMENTE quais instancias mudam de
# classificacao entre os dois limiares, (3) olha o efeito POR INSTANCIA
# (nao so' a media) pra ver se a "melhora" vem de poucos casos extremos.
# ---------------------------------------------------------------------------

df = pd.read_csv("_scratch_extended_features.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)

print("=== aritmetica: valores possiveis de dup_ratio pra n pequeno ===")
for n in [4, 6, 10, 12]:
    vals = sorted(set(round(1 - k / n, 4) for k in range(1, n + 1)))
    marker = " <- 1/6 = 0,1667 exato aqui" if any(abs(v - 0.16666666) < 1e-4 for v in vals) else ""
    print(f"  n={n:3d}: valores possiveis = {vals}{marker}")

print("\n=== quais instancias de E mudam de classificacao entre threshold=0,1 e threshold=0,1667? ===")
dfe = df[df.family == "E"].copy()
pred_01 = dfe.dup_ratio > 0.1
pred_01667 = dfe.dup_ratio > 0.1667
mudou = pred_01 != pred_01667
print(f"total em E: {len(dfe)}, mudaram de classificacao: {mudou.sum()} ({mudou.sum()/len(dfe)*100:.1f}%)")
print("\ndistribuicao de dup_ratio DAS QUE MUDARAM, por subfolder:")
print(dfe[mudou].groupby("subfolder").agg(
    n_instancias=("dup_ratio", "count"),
    dup_ratio_valores=("dup_ratio", lambda x: sorted(x.unique())),
))
print("\nn (tamanho da instancia) das que mudaram, por subfolder:")
print(dfe[mudou].groupby("subfolder").n.describe()[["count", "min", "max"]])

print("\n=== efeito por instancia (nao so' a media): gap_pct sob threshold=0,1 MENOS gap_pct sob threshold=0,1667 ===")
print("(positivo = threshold novo ajuda NESSA instancia; soma/400 = a melhora media reportada)")
dfe["gap_01"] = np.where(pred_01, dfe.ig_ql, dfe.ils_ql)
dfe["gap_01667"] = np.where(pred_01667, dfe.ig_ql, dfe.ils_ql)
dfe["efeito"] = dfe.gap_01 - dfe.gap_01667
print(f"\nmedia do efeito (= melhora reportada): {dfe.efeito.mean():.4f}")
print(f"mediana do efeito: {dfe.efeito.median():.4f}")
print(f"instancias com efeito != 0: {(dfe.efeito != 0).sum()} de {len(dfe)}")
print(f"soma do efeito so' nas instancias que mudaram: {dfe.loc[mudou, 'efeito'].sum():.4f}  "
      f"(dividido por 400 = {dfe.loc[mudou, 'efeito'].sum()/400:.4f}, deve bater com a media acima)")
print("\ntop 10 instancias por |efeito| (as que mais pesam na media):")
top10 = dfe.reindex(dfe.efeito.abs().sort_values(ascending=False).index).head(10)
print(top10[["member", "subfolder", "n", "m", "dup_ratio", "ils_ql", "ig_ql", "efeito"]].to_string(index=False))

print("\n=== o mesmo efeito, EXCLUINDO E1 inteiramente (so' E2+E3, 360 das 400 instancias) ===")
dfe_no_e1 = dfe[dfe.subfolder != "E1"]
g01_no_e1 = np.where(dfe_no_e1.dup_ratio > 0.1, dfe_no_e1.ig_ql, dfe_no_e1.ils_ql).mean()
g01667_no_e1 = np.where(dfe_no_e1.dup_ratio > 0.1667, dfe_no_e1.ig_ql, dfe_no_e1.ils_ql).mean()
print(f"gap(thr=0,1) sem E1: {g01_no_e1:.4f}   gap(thr=0,1667) sem E1: {g01667_no_e1:.4f}   "
      f"diff: {g01_no_e1-g01667_no_e1:+.4f}")

print("\nDONE")
