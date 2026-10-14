import sys

sys.path.insert(0, ".")

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

df = pd.read_csv("/tmp/pilot_all_mechanisms_combined.csv")

cols_order = [c for c in ["ig_random", "ig_ql", "ils_random", "ils_ql", "hybrid_ql",
                           "hybrid_ql_ws", "apriori_select", "racing"]
              if c in df["method"].unique()]

print("=== gap_pct medio (3 reps), TODOS os metodos ===")
piv_mean = df.pivot_table(index="label", columns="method", values="gap_pct", aggfunc="mean")
piv_mean = piv_mean[cols_order]
print(piv_mean.round(4))

print("\n=== gap_pct desvio-padrao (3 reps) ===")
piv_std = df.pivot_table(index="label", columns="method", values="gap_pct", aggfunc="std")
piv_std = piv_std[cols_order]
print(piv_std.round(4))

print("\n=== cpu_time medio (s), TODOS os metodos ===")
piv_cpu = df.pivot_table(index="label", columns="method", values="cpu_time", aggfunc="mean")
piv_cpu = piv_cpu[cols_order]
print(piv_cpu.round(3))

print("\n=== n_improvements medio ===")
piv_imp = df.pivot_table(index="label", columns="method", values="n_improvements", aggfunc="mean")
piv_imp = piv_imp[cols_order]
print(piv_imp.round(2))

print("\n=== melhor metodo por instancia (menor gap_pct medio) ===")
ranking = piv_mean.idxmin(axis=1)
best_val = piv_mean.min(axis=1)
for label in piv_mean.index:
    row = piv_mean.loc[label].sort_values()
    ordered = "  ".join(f"{m}={v:.4f}" for m, v in row.items())
    print(f"{label}: MELHOR={ranking[label]} (gap%={best_val[label]:.4f})")
    print(f"    ranking completo: {ordered}")

print("\n=== apriori_select vs ils_ql (a referencia mais forte na familia B) ===")
comp = piv_mean[["ils_ql", "apriori_select", "racing"]].copy()
comp["apriori_vs_ils_ql"] = comp["apriori_select"] - comp["ils_ql"]
comp["racing_vs_ils_ql"] = comp["racing"] - comp["ils_ql"]
print(comp.round(4))

print("\n=== apriori_select / racing vs hybrid_ql e hybrid_ql_ws ===")
comp2 = piv_mean[["hybrid_ql", "hybrid_ql_ws", "apriori_select", "racing"]].copy()
comp2["apriori_vs_ws"] = comp2["apriori_select"] - comp2["hybrid_ql_ws"]
comp2["racing_vs_ws"] = comp2["racing"] - comp2["hybrid_ql_ws"]
print(comp2.round(4))

print("\n=== media NAO-PONDERADA entre as 5 instancias (visao grosseira, cuidado) ===")
overall = piv_mean.mean(axis=0).sort_values()
print(overall.round(4))

print("\nDONE")
