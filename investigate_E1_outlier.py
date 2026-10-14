import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# family_separability_check.py revelou algo que muda a leitura da Entrada 66:
# o ganho agregado de familia E (+0,0130) NAO e' uniforme dentro de E --
# decompoe em E1 (+0,2083, so' 40/400=10% de E, MAS gap_pct em escala >5,
# muito maior que o resto) empurrando sozinho o resultado, E2 quase neutro
# (+0,0030), E3 (50% de E, a MAIOR fatia) na verdade PIORA (-0,0180) com o
# limiar novo. E1 tem TODAS as 40 instancias com n=6, m=3 fixos -- uma
# configuracao minuscula e degenerada, repetida.
#
# Pergunta central: isso e' um artefato de ESCALA (instancias minusculas
# tem gap_pct naturalmente extremo e sensivel, INDEPENDENTE de familia),
# ou e' algo especifico de "familia E"? Se for artefato de escala, "n
# pequeno" (feature trivial, sempre calculavel) pode ser um proxy melhor
# e MAIS GERAL que "familia E" -- generalizaria pra qualquer instancia
# nova, nao so' pra esse benchmark especifico.
# ---------------------------------------------------------------------------

df = pd.read_csv("_scratch_extended_features.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)


def gap_of_mask(sub, choose_ig_mask):
    return np.where(choose_ig_mask, sub.ig_ql, sub.ils_ql).mean()


print("=== instancias com n<=10 em CADA familia (nao so' E) -- existem? quantas? ===")
for fam, dfam in df.groupby("family"):
    small = dfam[dfam.n <= 10]
    print(f"  {fam}: {len(small)}/{len(dfam)} instancias com n<=10"
          + (f"  (n,m combos: {sorted(small[['n','m']].drop_duplicates().apply(tuple, axis=1).tolist())})" if len(small) else ""))

print("\n=== gap_pct medio (ils_ql, ig_ql, oracle_best) POR FAIXA DE n, todas as familias juntas ===")
bins = [0, 10, 20, 50, 100, 300, 1000, 10000]
labels = ["<=10", "11-20", "21-50", "51-100", "101-300", "301-1000", ">1000"]
df["n_bin"] = pd.cut(df.n, bins=bins, labels=labels)
print(df.groupby("n_bin", observed=True).agg(
    n_instancias=("n", "count"),
    ils_ql=("ils_ql", "mean"),
    ig_ql=("ig_ql", "mean"),
    oracle_best=("oracle_best", "mean"),
    dup_ratio_medio=("dup_ratio", "mean"),
).round(4))

print("\n=== so' pra confirmar: gap_pct medio POR FAIXA DE n, DENTRO DE CADA FAMILIA (pra isolar efeito de n do efeito de familia) ===")
for fam, dfam in df.groupby("family"):
    print(f"\n--- familia {fam} ---")
    g = dfam.groupby("n_bin", observed=True).agg(
        n_instancias=("n", "count"),
        oracle_best=("oracle_best", "mean"),
    ).round(4)
    print(g[g.n_instancias > 0])

print("\n=== teste direto: threshold=0,1 vs 0,1667, restrito a n<=10, EM CADA FAMILIA separadamente ===")
for fam, dfam in df.groupby("family"):
    small = dfam[dfam.n <= 10]
    if len(small) == 0:
        print(f"  {fam}: nenhuma instancia com n<=10")
        continue
    g01 = gap_of_mask(small, small.dup_ratio > 0.1)
    g01667 = gap_of_mask(small, small.dup_ratio > 0.1667)
    print(f"  {fam}: n_instancias(n<=10)={len(small)}  gap(thr=0,1)={g01:.4f}  gap(thr=0,1667)={g01667:.4f}  diff={g01-g01667:+.4f}")

print("\n=== teste direto: threshold=0,1 vs 0,1667, restrito a n<=10, TODAS AS FAMILIAS JUNTAS (proxy = 'n pequeno', sem olhar familia) ===")
small_all = df[df.n <= 10]
g01 = gap_of_mask(small_all, small_all.dup_ratio > 0.1)
g01667 = gap_of_mask(small_all, small_all.dup_ratio > 0.1667)
print(f"  n_instancias(n<=10, todas familias)={len(small_all)}  gap(thr=0,1)={g01:.4f}  gap(thr=0,1667)={g01667:.4f}  diff={g01-g01667:+.4f}")
print(f"  composicao por familia desse subconjunto: {small_all.family.value_counts().to_dict()}")

print("\n=== E1 especificamente: essas 40 instancias sao TODAS n=6,m=3? Confirmando e olhando ils_ql/ig_ql brutos ===")
dfe1 = df[(df.family == "E") & (df.subfolder == "E1")]
print(dfe1[["member", "n", "m", "dup_ratio", "ils_ql", "ig_ql", "actual_winner"]].describe(include="all"))
print(dfe1[["member", "n", "m", "dup_ratio", "ils_ql", "ig_ql", "actual_winner"]].head(10).to_string(index=False))

print("\nDONE")
