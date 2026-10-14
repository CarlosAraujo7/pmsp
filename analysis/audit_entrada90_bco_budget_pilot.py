"""
analysis/audit_entrada90_bco_budget_pilot.py
Piloto: BCO com o orcamento PUBLICADO (I=100) vs. a regra do projeto I=max(1,1000//(B*n)); 8 instancias x 3 classes, 1 seed.
Diario: Entrada 90 (04/10/2026). Piloto pequeno -- NAO e uma bateria; ver diario para ressalvas.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from pmsp.instance import PMSPInstance
from pmsp.bco import bco_run

ROOT = str(Path(__file__).resolve().parent.parent)
res = pd.read_csv(f"{ROOT}/results/results.csv")
K = 8
classes = ["rand_20j_6m", "rand_50j_10m", "rand_100j_20m"]
for cls in classes:
    rows = []
    for rep in range(K):
        inst = PMSPInstance.load(f"{ROOT}/data/instances/{cls}/rep{rep:02d}.json")
        n = inst.n
        t0 = time.perf_counter()
        r100 = bco_run(inst, max_iter=100 * 5 * n, seed=1000 + rep)  # I = 100 (published)
        t100 = time.perf_counter() - t0
        sel = res[(res.instance == cls) & (res.rep == rep)].set_index("method")
        rows.append(dict(
            rep=rep, I_used_paper=max(1, 1000 // (5 * n)),
            bco_paper=sel.loc["bco", "gap_pct"], bco_I100=r100["gap_pct"],
            ig_ql=sel.loc["ig_ql", "gap_pct"], multifit=sel.loc["multifit", "gap_pct"],
            cpu_bco_paper=sel.loc["bco", "cpu_time"], cpu_bco_I100=t100, cpu_ig_ql=sel.loc["ig_ql", "cpu_time"],
        ))
    d = pd.DataFrame(rows)
    print(f"== {cls}  (n={inst.n}, m={inst.m}, K={K} instances)")
    print(d.mean(numeric_only=True).round(4).to_string())
    print()
