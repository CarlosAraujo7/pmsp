import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# Carlos pediu pra investir mais tempo procurando um proxy CALCULAVEL pra
# "comporta-se como familia E" antes de decidir os 3 caminhos da Entrada 66.
# Este e' o primeiro passo, o mais barato possivel: verificar se features
# ja triviais (n, m, n/m -- sempre disponiveis, nenhum calculo extra) ou a
# propria subestrutura de E (subfolders E1/E2/E3, multiplos geradores
# U[baixo,alto] diferentes) ja separam E do resto, ou se a heterogeneidade
# de E vem so' de uma parte dela. Puramente descritivo -- nenhum modelo
# treinado ainda, so' groupby/describe/overlap.
# ---------------------------------------------------------------------------

df = pd.read_csv("_scratch_extended_features.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)
df["nm_ratio"] = df.n / df.m

print("=== n, m, n/m por familia (describe) ===")
for feat in ["n", "m", "nm_ratio"]:
    print(f"\n--- {feat} ---")
    print(df.groupby("family")[feat].describe()[["count", "min", "25%", "50%", "75%", "max"]].round(3))

print("\n=== overlap de n/m: fracao das instancias de E cujo n/m cai dentro do range [min,max] de outra familia ===")
e_nm = df[df.family == "E"].nm_ratio
for fam in ["B", "C", "F"]:
    other = df[df.family == fam].nm_ratio
    lo, hi = other.min(), other.max()
    frac_overlap = ((e_nm >= lo) & (e_nm <= hi)).mean()
    print(f"  E dentro do range de {fam} [{lo:.3f}, {hi:.3f}]: {frac_overlap*100:.1f}% das instancias de E")

print("\n=== subfolders dentro de cada familia (contagem) ===")
print(df.groupby(["family", "subfolder"]).size())

print("\n=== familia E: dup_ratio, gap_pct e limiar otimo POR SUBFOLDER (E1/E2/E3 ou equivalente) ===")
dfe = df[df.family == "E"].copy()


def gap_of_mask(sub, choose_ig_mask):
    return np.where(choose_ig_mask, sub.ig_ql, sub.ils_ql).mean()


for sf, dsub in dfe.groupby("subfolder"):
    gap_thr01 = gap_of_mask(dsub, dsub.dup_ratio > 0.1)
    gap_thr01667 = gap_of_mask(dsub, dsub.dup_ratio > 0.1667)
    print(f"  {sf}: n={len(dsub):4d}  dup_ratio[min={dsub.dup_ratio.min():.4f} "
          f"med={dsub.dup_ratio.median():.4f} max={dsub.dup_ratio.max():.4f}]  "
          f"gap(thr=0,1)={gap_thr01:.4f}  gap(thr=0,1667)={gap_thr01667:.4f}  "
          f"diff={gap_thr01-gap_thr01667:+.4f}  n_e_range=[{dsub.n.min()}-{dsub.n.max()}] m_range=[{dsub.m.min()}-{dsub.m.max()}]")

print("\n=== dup_ratio: distribuicao por familia (pra contextualizar por que o limiar de E e' diferente) ===")
print(df.groupby("family").dup_ratio.describe()[["count", "min", "25%", "50%", "75%", "max"]].round(4))

print("\nDONE")
