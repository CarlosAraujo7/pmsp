import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

# ---------------------------------------------------------------------------
# Fecha o diagnostico da familia E: quantifica o impacto de uma regra MISTA
# (usar o limiar retunado 0,1667 so' pra familia E, manter 0,1 nas outras 3)
# sobre as 1.612 instancias da bateria externa inteira -- tanto em gap_pct
# (a metrica que realmente importa pro artigo) quanto em acuracia de
# classificacao (pra deixar explicito o paradoxo: acuracia PIORA em E mesmo
# com gap_pct melhorando -- os dois nao sao a mesma coisa, ja discutido na
# Entrada 64, mas aqui a divergencia e' mais forte e merece registro).
# ---------------------------------------------------------------------------

df = pd.read_csv("_scratch_extended_features.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)

# --- regra atual: threshold=0,1 uniforme pra todas as familias ---
pred_atual = np.where(df.dup_ratio > 0.1, "ig", "ils")
gap_atual = np.where(pred_atual == "ig", df.ig_ql, df.ils_ql).mean()
acc_atual = (pred_atual == df.actual_winner).mean()

# --- regra mista: E usa 0,1667 (achado desta entrada), resto usa 0,1 ---
thr_por_familia = df.family.map({"B": 0.1, "C": 0.1, "E": 0.1667, "F": 0.1})
pred_misto = np.where(df.dup_ratio > thr_por_familia, "ig", "ils")
gap_misto = np.where(pred_misto == "ig", df.ig_ql, df.ils_ql).mean()
acc_misto = (pred_misto == df.actual_winner).mean()

oracle_global = df.oracle_best.mean()

print("=== impacto GLOBAL (1.612 instancias) da regra mista E=0,1667/resto=0,1 ===")
print(f"gap_pct  atual (threshold=0,1 uniforme):      {gap_atual:.4f}")
print(f"gap_pct  misto (E=0,1667, resto=0,1):         {gap_misto:.4f}")
print(f"melhora absoluta:                             {gap_atual - gap_misto:.4f}")
print(f"melhora relativa:                             {(gap_atual - gap_misto) / gap_atual * 100:.2f}%")
print(f"oracle_best global (teto absoluto):            {oracle_global:.4f}")
print(f"gap total ate' o oraculo (regra atual):        {gap_atual - oracle_global:.4f}")
print(f"fracao do gap total fechada pela regra mista:  {(gap_atual - gap_misto) / (gap_atual - oracle_global) * 100:.2f}%")
print()
print(f"acuracia GLOBAL atual (threshold=0,1):         {acc_atual:.4f}")
print(f"acuracia GLOBAL mista (E=0,1667):              {acc_misto:.4f}")
print(f"delta acuracia GLOBAL:                         {acc_misto - acc_atual:+.4f}")
print()

dfe = df[df.family == "E"]
pred_e_atual = np.where(dfe.dup_ratio > 0.1, "ig", "ils")
pred_e_misto = np.where(dfe.dup_ratio > 0.1667, "ig", "ils")
acc_e_atual = (pred_e_atual == dfe.actual_winner).mean()
acc_e_misto = (pred_e_misto == dfe.actual_winner).mean()
majority_e = dfe.actual_winner.value_counts(normalize=True).max()
print("=== so' familia E: acuracia vs gap_pct (o paradoxo) ===")
print(f"acuracia E, threshold=0,1 (atual):             {acc_e_atual:.4f}")
print(f"acuracia E, threshold=0,1667 (retunado):       {acc_e_misto:.4f}")
print(f"baseline de maioria em E (sempre prever '{dfe.actual_winner.value_counts().idxmax()}'): {majority_e:.4f}")
print(f"gap_pct E, threshold=0,1 (atual):              {np.where(pred_e_atual=='ig', dfe.ig_ql, dfe.ils_ql).mean():.4f}")
print(f"gap_pct E, threshold=0,1667 (retunado):        {np.where(pred_e_misto=='ig', dfe.ig_ql, dfe.ils_ql).mean():.4f}")

print("\nDONE")
