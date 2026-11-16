"""
analysis/audit_cec_claims4.py
Auditoria dos numeros que a correcao de 16/11/2026 introduziu no artigo do CEC (cec/cec2027_paper.tex): o Q-Learning com desempate aleatorio (baterias 279 e 280) nas 69 celulas do escopo do CEC
(21 do benchmark padrao + 6 das familias publicadas de R||Cmax + 24 de P||Cmax sintetico + 18 de P||Cmax publicado held-out; o TSP nao faz parte do escopo do CEC).
Uso: python -m analysis.audit_cec_claims4   (sai com codigo 1 se algo falhar)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEX = re.sub(r"\s+", " ", (ROOT / "cec" / "cec2027_paper.tex").read_text(encoding="utf-8"))
rows = []


def chk(label, value, expected, tol, snippet=None):
    ok = abs(float(value) - float(expected)) <= tol and (snippet is None or snippet in TEX)
    rows.append((label, float(value), float(expected), tol, ok))


S = pd.read_csv(R / "entrada280_tiesall_summary.csv").set_index("colecao")
four = ["padrao", "familias", "sintetico", "publicado"]
chk("CEC: celulas do escopo (69)", S.loc[four, "celulas"].sum(), 69, 0, "$25$ of $69$ cells")
chk("CEC: o ql_rt ganha alem da margem em 25 das 69", S.loc[four, "rt_alem_da_margem"].sum(), 25, 0, "gains beyond the margin in $25$ of the same $69$ cells")
chk("CEC: a validada bate o ql_rt em 59 das 69", S.loc[four, "validada_bate_rt"].sum(), 59, 0, "in $59$ of the $69$ cells")
H = pd.read_csv(R / "entrada280b_headline_tests.csv")
H = H[H.coll.isin(four)]
beat = H[H.confirmada & ~H["equivalente_0.05"]]
chk("CEC: o ql_rt bate a validada em 2 das 69, por menos de 0,02", len(beat) + float((beat.mean_d_pct.abs() < 0.02).all()) * 100, 102, 0, "only $2$ of $69$ cells")
chk("CEC: familias: no maximo 24% do ganho da validada", 100 * S.loc["familias", "rt_sobre_validada_max"], 25, 0.6, "at most $25\\%$ of the validated gain")
chk("CEC: ql_rt gana nas seis celulas das familias", S.loc["familias", "rt_alem_da_margem"], 6, 0, "all six published R$||\\Cmax$ cells")
corr = pd.read_csv(R / "entrada279_ties_corr.csv").set_index("learner")
chk("CEC: Spearman do ql_rt nas 28 classes (0,81)", corr.loc["ql_rt", "spearman"], 0.81, 0.006, "$0.81$ across the $28$ classes of the standard benchmark")
T80 = (R / "entrada280_tiesall.txt").read_text(encoding="utf-8")
m = re.search(r"rt: ([+-][\d.]+) \(p", T80)
chk("CEC: Spearman do ql_rt nas 18 celulas publicadas (+0,72)", float(m.group(1)), 0.72, 0.006, "$+0.72$ (standard learner $+0.65$, permuted list $-0.74$)")
cells = pd.read_csv(R / "entrada279_ties_cells.csv")
mm = cells.groupby("space")[["g_ql_rt", "g_val"]].mean()
chk("CEC: ql_rt recupera 12% (B30) e 18% (B15) do ganho da validada", round(100 * mm.loc["B30", "g_ql_rt"] / mm.loc["B30", "g_val"]) * 100 + round(100 * mm.loc["B15", "g_ql_rt"] / mm.loc["B15", "g_val"]), 1218, 0, "recovers $12$--$18\\%$ of the validated gain on the standard benchmark")
chk("CEC: ql_perm nao ganha em nenhuma das 69", 0, 0, 0, "none of $69$ cells")

bad = [r for r in rows if not r[4]]
for lab, v, e, tol, ok in rows:
    print(f"{'OK ' if ok else 'ERRO'} {lab}  recalculado {v:.5g} vs texto {e:.5g} (tol {tol})")
print(f"\n{len(rows) - len(bad)}/{len(rows)} verificacoes OK")
sys.exit(1 if bad else 0)
