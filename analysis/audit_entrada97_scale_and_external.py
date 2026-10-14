"""
analysis/audit_entrada97_scale_and_external.py
=================================================
Entrada 97. (1) Invariancia a escala: multiplicar todos os p_j por 1000 muda o
resultado (gap_pct) de ig_random / ils_random / lpt? Se nao, o fatorial
"variancia baixa U[1,100] vs alta U[1,1e8]" varia a GRANULARIDADE/duplicacao
dos tempos (dup_ratio), nao a variancia relativa (o CV de uma uniforme e'
escala-invariante). (2) dup_ratio e CV por configuracao do fatorial.
(3) Reprodutibilidade dos numeros da Secao 5.5 (validacao externa):
recomputa as contagens "Holm-significativas" por familia usando a mesma
funcao do report_entrada58 sobre o manifest de 1.500 instancias.
Uso: python -m analysis.audit_entrada97_scale_and_external
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment.runner import run_single  # noqa: E402
from pmsp.apriori_select import dup_ratio  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

print("=" * 90, "\n1. INVARIANCIA A ESCALA (p x 1000, mesmo seed)\n" + "=" * 90)
for method in ["lpt", "ig_random", "ils_random", "ig_ql", "ils_ql", "ils_ucb"]:
    diffs = []
    for rep in range(4):
        base = PMSPInstance.random(n=60, m=12, seed=rep, name="b")
        scaled = PMSPInstance(p=np.asarray(base.p, dtype=np.int64) * 1000, m=12, name="s")
        g1 = run_single(base, method, rep, max_iter=300)["gap_pct"]
        g2 = run_single(scaled, method, rep, max_iter=300)["gap_pct"]
        diffs.append(abs(g1 - g2))
    print(f"  {method:11s}: max |gap(p) - gap(1000p)| = {max(diffs):.6f} pp")

print("\n" + "=" * 90, "\n2. dup_ratio e CV nas configuracoes do fatorial (n=150)\n" + "=" * 90)
for H in [100, 10**8]:
    cvs, dups = [], []
    for seed in range(20):
        inst = PMSPInstance.random(n=150, m=15, seed=seed, high=H)
        p = np.asarray(inst.p, dtype=float)
        cvs.append(p.std() / p.mean()); dups.append(dup_ratio(inst))
    print(f"  U[1,{H:>9}]: CV medio = {np.mean(cvs):.3f}   dup_ratio medio = {np.mean(dups):.3f}   variancia absoluta ~ {np.var(p):.3g}")

print("\n" + "=" * 90, "\n3. SECAO 5.5: contagens Holm por familia externa (manifest 1.500)\n" + "=" * 90)
import analysis.report_entrada58_external_validation as R58  # noqa: E402

R58.MANIFEST_CSV = ROOT / "external_benchmarks" / "pcmax_symmetry2025" / "manifest_pre_6780_backup_20260902.csv"
df = R58.load_clean_external_results()
SOTA = ["multifit", "sa_li_perkins", "bco"]
ABL = ["ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql_flat"]
ILS = ["ils_random", "ils_ucb", "ils_ql", "ils_rf"]
for fam in ["B", "C", "E", "F"]:
    fd = df[df.family == fam]
    out = []
    for tag, others in [("SOTA", SOTA), ("ABL", ABL), ("ILS", ILS)]:
        r = R58.family_by_subfolder(fd, "ig_ql", others, family_name=f"{fam}_{tag}")
        sig = r[r.reject_holm]
        out.append(f"{tag}: {len(sig)}/{int(r.p_raw.notna().sum())} sig (fav {int((sig.direction == 'ref_better').sum())}, contra {int((sig.direction == 'ref_worse').sum())})")
    print(f"  {fam} ({fd.member.nunique()} inst.): " + " | ".join(out))
