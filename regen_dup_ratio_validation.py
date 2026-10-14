import sys
import time

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from external_benchmarks.pcmax_symmetry2025.loader import load_instance_array

# Regeneracao do dup_ratio_validation_full.csv (Entrada 61) -- a VM do
# device_bash reiniciou entre turnos e o /tmp efemero foi perdido; o
# repositorio real (pasta conectada) esta intacto, so o scratch precisa ser
# reconstruido antes de rodar a validacao em escala maior de apriori_select
# e racing. Mesma logica da Entrada 61 original, so' escrevendo o CSV na
# pasta do repositorio (nao mais em /tmp) -- a VM efemera pode reiniciar de
# novo a qualquer momento, e um resultado em /tmp nao sobrevive a isso.

t0 = time.time()
df = pd.read_csv("external_benchmarks/pcmax_symmetry2025/results_external_symmetry2025.csv")
sub = df[df.method.isin(["ils_ql", "ig_ql"])]
piv = sub.pivot_table(index=["member", "family", "subfolder", "n", "m"],
                       columns="method", values="gap_pct", aggfunc="mean").reset_index()
piv = piv.dropna(subset=["ils_ql", "ig_ql"])
piv["actual_winner"] = np.where(piv.ils_ql < piv.ig_ql, "ils", "ig")
print(f"instancias com ils_ql E ig_ql calculados: {len(piv)}  ({time.time()-t0:.0f}s)")

dup_ratios = []
for i, row in enumerate(piv.itertuples()):
    m, n, p = load_instance_array(row.family, row.member)
    p = np.asarray(p)
    n_unique = len(np.unique(p))
    dup_ratios.append(1 - n_unique / len(p))
    if (i + 1) % 300 == 0:
        print(f"  {i+1}/{len(piv)} instancias lidas ({time.time()-t0:.0f}s)", flush=True)
piv["dup_ratio"] = dup_ratios

for thresh in [0.001, 0.01, 0.05, 0.1]:
    piv[f"pred_{thresh}"] = np.where(piv.dup_ratio > thresh, "ig", "ils")
    acc = (piv[f"pred_{thresh}"] == piv.actual_winner).mean()
    print(f"threshold={thresh:<6} acuracia GLOBAL da regra dup_ratio: {acc:.4f}  (n={len(piv)})")

piv.to_csv("_scratch_dup_ratio_validation_full.csv", index=False)
print(f"\ntotal elapsed: {time.time()-t0:.0f}s")
print("DONE")
