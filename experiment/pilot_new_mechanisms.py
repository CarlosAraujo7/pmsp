import sys
import time

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import experiment.config as CFG
from pmsp.instance import PMSPInstance
from pmsp.apriori_select import apriori_run, dup_ratio
from pmsp.racing import racing_run
from external_benchmarks.pcmax_symmetry2025.loader import load_instance_array

MEMBERS = [
    ("B", "B/6/03_028_6_00.dat", "B/6 n=28 m=3  (c=6)"),
    ("B", "B/6/09_059_6_00.dat", "B/6 n=59 m=9  (c=6)"),
    ("B", "B/13/14_171_13_00.dat", "B/13 n=171 m=14 (c=13)"),
    ("B", "B/18/16_261_18_00.dat", "B/18 n=261 m=16 (c=18)"),
    ("C", "C/C5/C_5_070.dat", "C/C5 n=144 m=48"),
]

N_REPS = 3
MAX_ITER = CFG.MAX_ITER  # 1000, mesmo orcamento das Entradas 59/60

results = []
t0 = time.time()
for family, member, label in MEMBERS:
    m, n, p = load_instance_array(family, member)
    inst = PMSPInstance(p=p, m=m, name=member)
    lb = inst.lower_bound
    dr = dup_ratio(inst)
    print(f"\n===== {label}  n={n} m={m} LB={lb} dup_ratio={dr:.4f} =====", flush=True)

    for rep in range(N_REPS):
        seed = 1000 + rep

        r = apriori_run(
            inst, max_iter=MAX_ITER, seed=seed,
            alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA,
            epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
            epsilon_decay=CFG.QL_EPSILON_DECAY, stagnation_window=CFG.STAGNATION_WINDOW,
            perturbation_threshold=CFG.PERTURBATION_THRESHOLD, n_perturb=CFG.N_PERTURB,
        )
        results.append(dict(
            label=label, method="apriori_select", rep=rep, gap_pct=r["gap_pct"],
            n_improvements=r["n_improvements"], cpu_time=r["cpu_time"],
            extra=f"fw={r['framework_escolhido']} dup_ratio={r['dup_ratio']:.4f}",
        ))

        r2 = racing_run(
            inst, max_iter=MAX_ITER, seed=seed,
            alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA,
            epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
            epsilon_decay=CFG.QL_EPSILON_DECAY, stagnation_window=CFG.STAGNATION_WINDOW,
        )
        results.append(dict(
            label=label, method="racing", rep=rep, gap_pct=r2["gap_pct"],
            n_improvements=r2["n_improvements"], cpu_time=r2["cpu_time"],
            extra=(f"fw_venc={r2['framework_vencedor']} "
                   f"probe_ils={r2['probe_ils_makespan']} probe_ig={r2['probe_ig_makespan']}"),
        ))

        print(f"  rep{rep}: apriori_select gap%={r['gap_pct']:.4f} (fw={r['framework_escolhido']})  "
              f"racing gap%={r2['gap_pct']:.4f} (fw_venc={r2['framework_vencedor']})  "
              f"(elapsed {time.time() - t0:.0f}s)", flush=True)

df_new = pd.DataFrame(results)
df_new.to_csv("/tmp/pilot_new_mechanisms_results.csv", index=False)

# ---------------------------------------------------------------------
# Junta com os resultados de referencia JA CALCULADOS (Entradas 59/60):
# ig_ql/ils_ql/hybrid_ql (cold) em pilot_hybrid_results.csv,
# hybrid_ql_ws (warm) em pilot_hybrid_ws_results.csv -- nao roda de novo.
# ---------------------------------------------------------------------
df_cold = pd.read_csv("/tmp/pilot_hybrid_results.csv")
df_ws = pd.read_csv("/tmp/pilot_hybrid_ws_results.csv")
df_all = pd.concat([df_cold, df_ws, df_new], ignore_index=True)
df_all.to_csv("/tmp/pilot_all_mechanisms_combined.csv", index=False)

cols_order = [c for c in ["ig_random", "ig_ql", "ils_random", "ils_ql", "hybrid_ql",
                           "hybrid_ql_ws", "apriori_select", "racing"]
              if c in df_all["method"].unique()]

print("\n\n=== RESUMO COMPARATIVO (gap_pct medio, 3 reps) ===")
piv = df_all.pivot_table(index="label", columns="method", values="gap_pct", aggfunc="mean")
piv = piv[cols_order]
pd.set_option("display.width", 200)
print(piv.round(4))

print("\n=== RESUMO COMPARATIVO (cpu_time medio (s), 3 reps) ===")
piv_cpu = df_all.pivot_table(index="label", columns="method", values="cpu_time", aggfunc="mean")
piv_cpu = piv_cpu[cols_order]
print(piv_cpu.round(3))

print("\n=== detalhe apriori_select: framework escolhido por instancia ===")
sub = df_new[df_new.method == "apriori_select"][["label", "rep", "extra"]]
print(sub.to_string(index=False))

print("\n=== detalhe racing: framework vencedor + probes por instancia ===")
sub2 = df_new[df_new.method == "racing"][["label", "rep", "extra"]]
print(sub2.to_string(index=False))

print(f"\ntotal elapsed: {time.time() - t0:.0f}s")
print("DONE")
