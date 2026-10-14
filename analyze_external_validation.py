import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

ref = pd.read_csv("_scratch_dup_ratio_validation_full.csv")
new = pd.read_csv("_scratch_validate_new_mechanisms_external_results.csv")

piv_new = new.pivot_table(index="member", columns="method",
                           values="gap_pct", aggfunc="first").reset_index()
fw_new = new.pivot_table(index="member", columns="method",
                          values="framework", aggfunc="first").reset_index()
fw_new = fw_new.rename(columns={"apriori_select": "apriori_fw", "racing": "racing_fw"})

df = ref.merge(piv_new, on="member", how="inner").merge(
    fw_new[["member", "apriori_fw", "racing_fw"]], on="member", how="inner")

df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)

print(f"total instancias na amostra com todos os metodos: {len(df)}")
print()

# ---------------------------------------------------------------------
# 1. Acuracia de escolha de framework (apriori_select via dup_ratio,
#    racing via probe empirico) contra o vencedor real (ils_ql x ig_ql)
# ---------------------------------------------------------------------
print("=== acuracia de escolha de FRAMEWORK, por familia ===")
df["apriori_correct"] = df.apriori_fw == df.actual_winner
df["racing_correct"] = df.racing_fw == df.actual_winner
acc_tab = df.groupby("family").agg(
    n=("member", "size"),
    apriori_acc=("apriori_correct", "mean"),
    racing_acc=("racing_correct", "mean"),
    baseline_classe_majoritaria=("actual_winner", lambda s: s.value_counts(normalize=True).max()),
)
print(acc_tab.round(4))
print()
print("GLOBAL: apriori_acc=", df.apriori_correct.mean().round(4),
      " racing_acc=", df.racing_correct.mean().round(4),
      " baseline_global=", df.actual_winner.value_counts(normalize=True).max().round(4))

print()
print("=== onde racing e apriori_select DISCORDAM na escolha de framework (n=", (df.apriori_fw != df.racing_fw).sum(), ") ===")
disagree = df[df.apriori_fw != df.racing_fw][
    ["member", "family", "n", "m", "dup_ratio", "actual_winner", "apriori_fw", "racing_fw"]]
print(disagree.to_string(index=False))

# ---------------------------------------------------------------------
# 2. gap_pct medio por familia, todos os metodos lado a lado
# ---------------------------------------------------------------------
print()
print("=== gap_pct MEDIO por familia (single-seed novo vs referencia multi-rep) ===")
gap_tab = df.groupby("family").agg(
    n=("member", "size"),
    ils_ql=("ils_ql", "mean"),
    ig_ql=("ig_ql", "mean"),
    oracle_best=("oracle_best", "mean"),
    apriori_select=("apriori_select", "mean"),
    racing=("racing", "mean"),
)
print(gap_tab.round(4))

# ---------------------------------------------------------------------
# 3. taxa de vitoria/empate/derrota contra o oraculo best(ils_ql,ig_ql)
#    e contra cada referencia individual, por familia
# ---------------------------------------------------------------------
print()
print("=== racing vs oracle_best(ils_ql,ig_ql): venceu / empatou / perdeu, por familia ===")
TOL = 1e-9
def wtl(sub, col):
    win = (sub[col] < sub.oracle_best - TOL).sum()
    tie = (abs(sub[col] - sub.oracle_best) <= TOL).sum()
    lose = (sub[col] > sub.oracle_best + TOL).sum()
    return pd.Series({"venceu": win, "empatou": tie, "perdeu": lose, "n": len(sub)})

for method in ["apriori_select", "racing"]:
    print(f"\n--- {method} vs oracle_best ---")
    res = df.groupby("family").apply(lambda sub: wtl(sub, method))
    print(res)

# ---------------------------------------------------------------------
# 4. Piores casos (onde o novo metodo mais perde do oraculo) -- olhar de
#    perto, nao so' a media, pra nao esconder um caso catastrofico atras
#    de uma media favoravel.
# ---------------------------------------------------------------------
print()
print("=== 10 PIORES casos de racing vs oracle_best (maior gap_racing - gap_oracle) ===")
df["racing_minus_oracle"] = df.racing - df.oracle_best
worst = df.nlargest(10, "racing_minus_oracle")[
    ["member", "family", "n", "m", "dup_ratio", "actual_winner", "racing_fw",
     "ils_ql", "ig_ql", "racing", "racing_minus_oracle"]]
print(worst.to_string(index=False))

print()
print("=== 10 PIORES casos de apriori_select vs oracle_best ===")
df["apriori_minus_oracle"] = df.apriori_select - df.oracle_best
worst2 = df.nlargest(10, "apriori_minus_oracle")[
    ["member", "family", "n", "m", "dup_ratio", "actual_winner", "apriori_fw",
     "ils_ql", "ig_ql", "apriori_select", "apriori_minus_oracle"]]
print(worst2.to_string(index=False))

df.to_csv("_scratch_external_validation_merged.csv", index=False)
print("\nDONE")
