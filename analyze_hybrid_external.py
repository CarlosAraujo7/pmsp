import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# Comparacao final de 4 mecanismos (Task #77, continuacao da Entrada 62):
# apriori_select, racing, hybrid_ql, hybrid_ql_ws -- todos rodados na
# MESMA amostra estratificada de 95 instancias (garantido por construcao:
# validate_hybrid_external.py reusa a identica build_sample() de
# validate_new_mechanisms_external.py), permitindo comparacao linha-a-
# linha direta pela primeira vez no projeto.
# ---------------------------------------------------------------------------

ref = pd.read_csv("_scratch_dup_ratio_validation_full.csv")
new1 = pd.read_csv("_scratch_validate_new_mechanisms_external_results.csv")  # apriori_select, racing
new2 = pd.read_csv("_scratch_validate_hybrid_external_results.csv")          # hybrid_ql, hybrid_ql_ws

piv1 = new1.pivot_table(index="member", columns="method", values="gap_pct", aggfunc="first").reset_index()
piv2 = new2.pivot_table(index="member", columns="method", values="gap_pct", aggfunc="first").reset_index()

df = ref.merge(piv1, on="member", how="inner").merge(piv2, on="member", how="inner")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)

METHODS = ["apriori_select", "racing", "hybrid_ql", "hybrid_ql_ws"]

print(f"total instancias com TODOS os 4 mecanismos + ils_ql/ig_ql: {len(df)}")
print()

# ---------------------------------------------------------------------
# 1. gap_pct medio por familia, os 4 mecanismos + os 2 especialistas +
#    o oraculo, lado a lado
# ---------------------------------------------------------------------
print("=== gap_pct MEDIO por familia -- os 4 mecanismos + referencias ===")
gap_tab = df.groupby("family").agg(
    n=("member", "size"),
    ils_ql=("ils_ql", "mean"),
    ig_ql=("ig_ql", "mean"),
    oracle_best=("oracle_best", "mean"),
    apriori_select=("apriori_select", "mean"),
    racing=("racing", "mean"),
    hybrid_ql=("hybrid_ql", "mean"),
    hybrid_ql_ws=("hybrid_ql_ws", "mean"),
)
print(gap_tab.round(4))
print()
print("=== gap_pct MEDIO GLOBAL (95 instancias) ===")
print(df[["ils_ql", "ig_ql", "oracle_best"] + METHODS].mean().round(4))

# ---------------------------------------------------------------------
# 2. win/tie/lose contra oracle_best, por familia, para os 4 mecanismos
# ---------------------------------------------------------------------
TOL = 1e-9
def wtl(sub, col):
    win = (sub[col] < sub.oracle_best - TOL).sum()
    tie = (abs(sub[col] - sub.oracle_best) <= TOL).sum()
    lose = (sub[col] > sub.oracle_best + TOL).sum()
    return pd.Series({"venceu": win, "empatou": tie, "perdeu": lose, "n": len(sub)})

print()
print("=== venceu/empatou/perdeu contra oracle_best, por familia e GLOBAL ===")
for method in METHODS:
    print(f"\n--- {method} vs oracle_best ---")
    res = df.groupby("family").apply(lambda sub: wtl(sub, method))
    print(res)
    g = wtl(df, method)
    print(f"GLOBAL: venceu={g['venceu']} empatou={g['empatou']} perdeu={g['perdeu']} (n={g['n']})")

# ---------------------------------------------------------------------
# 3. Ranking direto: quantas vezes cada mecanismo e' o MELHOR dos 4
#    (menor gap_pct entre apriori_select/racing/hybrid_ql/hybrid_ql_ws)
# ---------------------------------------------------------------------
print()
print("=== quantas vezes cada mecanismo tem o MENOR gap_pct entre os 4 (por instancia) ===")
df["best_of_4"] = df[METHODS].idxmin(axis=1)
print(df.best_of_4.value_counts())
print()
print("=== idem, por familia ===")
print(df.groupby("family").best_of_4.value_counts().unstack(fill_value=0))

# ---------------------------------------------------------------------
# 4. cpu_time medio -- custo computacional dos 4 mecanismos
# ---------------------------------------------------------------------
cpu1 = new1.pivot_table(index="member", columns="method", values="cpu_time", aggfunc="first").reset_index()
cpu2 = new2.pivot_table(index="member", columns="method", values="cpu_time", aggfunc="first").reset_index()
dfc = df[["member", "family"]].merge(cpu1, on="member").merge(cpu2, on="member")
print()
print("=== cpu_time MEDIO (segundos) por familia ===")
print(dfc.groupby("family")[METHODS].mean().round(3))

# ---------------------------------------------------------------------
# 5. Piores casos de cada mecanismo (nao esconder atras da media)
# ---------------------------------------------------------------------
print()
for method in METHODS:
    df[f"{method}_minus_oracle"] = df[method] - df.oracle_best
    print(f"=== 5 PIORES casos de {method} vs oracle_best ===")
    worst = df.nlargest(5, f"{method}_minus_oracle")[
        ["member", "family", "n", "m", "ils_ql", "ig_ql", method, f"{method}_minus_oracle"]]
    print(worst.to_string(index=False))
    print()

df.to_csv("_scratch_full_4way_comparison.csv", index=False)
print("DONE")
