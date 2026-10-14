"""
analysis/audit_entrada127_cycle4_claims.py (Entrada 127)
Auditoria numerica das alegacoes do manuscrito baseadas nos experimentos do Ciclo 4 (Entrada 124: lei da folga no R||Cmax e hibrido IG+LS no P||Cmax). Recalcula a partir
dos CSV (brutos quando possivel, primarios pre-registrados quando o calculo e' o proprio teste) e compara com o valor literal do texto. Complementa
analysis/audit_entrada126_final_claims.py. As alegacoes do Ciclo 5 sao auditadas em audit_entrada127_cycle5_claims.py (escrito depois dos resultados).
Uso: python -m analysis.audit_entrada127_cycle4_claims
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
rows = []


def chk(label, value, expected, tol):
    rows.append((label, float(value), float(expected), tol, abs(float(value) - float(expected)) <= tol))


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


# ---------------------------------------------------------------- lei da folga (R||Cmax, 400 instancias novas)
L = pd.read_csv(R / "entrada124_unrelated_law_primary.csv").set_index("id")
chk("lei L1 rho (0,59)", L.loc["L1", "mean_d_pct"], 0.5925, 0.0006)
chk("lei L1 p_holm (0,015)", L.loc["L1", "p_holm"], 0.0148, 0.0006)
chk("lei L2 ql vs random (-0,073)", L.loc["L2", "mean_d_pct"], -0.0727, 0.0006)
chk("lei L2 p_holm (0,21)", L.loc["L2", "p_holm"], 0.2078, 0.006)
chk("lei L3 estado (-0,075)", L.loc["L3", "mean_d_pct"], -0.0751, 0.0006)
chk("lei L4 TD vs bandit (-0,136)", L.loc["L4", "mean_d_pct"], -0.1360, 0.0006)
chk("lei L4 p_holm (0,009)", L.loc["L4", "p_holm"], 0.0094, 0.0006)
chk("lei L5 flat vs random (+0,008)", L.loc["L5", "mean_d_pct"], 0.0075, 0.0006)
chk("lei L6 ucb vs random (+0,098)", L.loc["L6", "mean_d_pct"], 0.0982, 0.0006)
d = pd.read_csv(R / "unrelated_law_entrada124.csv")
w = d.pivot_table(index=["lo", "hi", "n", "m", "rep"], columns="method", values="makespan").reset_index()
chk("lei: instancias novas (400)", len(w), 400, 0)
cell = w.assign(dq=rel(w.ur_igls_ql, w.ur_igls_random), do=rel(w[[f"ur_igls_fix{k}" for k in range(4)]].min(axis=1), w.ur_igls_random)).groupby(["lo", "hi", "n", "m"])[["dq", "do"]].mean()
chk("lei: folga agregada (-0,240)", cell.do.mean(), -0.2404, 0.0015)
chk("lei: politica agregada (-0,073)", cell.dq.mean(), -0.0727, 0.0015)
chk("lei: razao politica/folga (0,30)", cell.dq.mean() / cell.do.mean(), 0.302, 0.01)
rho = cell.dq.rank().corr(cell.do.rank())
chk("lei: Spearman recalculado (0,59)", rho, 0.5925, 0.0006)

# ---------------------------------------------------------------- hibrido IG+LS (P||Cmax, 12 classes x 100)
H = pd.read_csv(R / "entrada124_igls_p_primary.csv").set_index("id")
chk("hibrido P1 busca local (-0,150)", H.loc["P1", "mean_d_pct"], -0.1502, 0.0006)
chk("hibrido P1 melhor (436)", H.loc["P1", "better"], 436, 0)
chk("hibrido P1 pior (16)", H.loc["P1", "worse"], 16, 0)
chk("hibrido P2 vs ILS (-0,722)", H.loc["P2", "mean_d_pct"], -0.7224, 0.0006)
chk("hibrido P3 politica (-0,004)", H.loc["P3", "mean_d_pct"], -0.0037, 0.0006)
chk("hibrido P3 p (0,65)", H.loc["P3", "p_holm"], 0.6463, 0.006)
g = pd.read_csv(R / "igls_p_entrada124.csv")
t = g.pivot_table(index=["instance", "rep"], columns="method", values="makespan").reset_index()
chk("hibrido: pares igls_random (1200)", len(t.dropna(subset=["igls_random"])), 1200, 0)

out = pd.DataFrame(rows, columns=["claim", "recalculado", "no_texto", "tol", "ok"])
pd.set_option("display.width", 200, "display.max_rows", 300, "display.float_format", lambda x: f"{x:.4f}")
print(out.to_string(index=False))
print(f"\n{int(out.ok.sum())}/{len(out)} OK; divergentes: {list(out[~out.ok].claim)}")
(R / "entrada127_cycle4_claims_audit.txt").write_text(out.to_string(index=False) + f"\n\n{int(out.ok.sum())}/{len(out)} OK\n", encoding="utf-8")
