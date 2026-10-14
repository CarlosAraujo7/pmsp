"""
analysis/audit_entrada126_final_claims.py (Entrada 126)
Auditoria numerica das alegacoes do manuscrito escritas apos as Entradas 114-124: recalcula a partir dos CSV (brutos quando possivel, primarios pre-registrados
quando o calculo e' o proprio teste) e compara com o valor literal do texto. Complementa analysis/audit_entrada113_final_claims.py.
Uso: python -m analysis.audit_entrada126_final_claims
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


# ---------------------------------------------------------------- seeds 12 classes
sv = pd.concat([pd.read_csv(R / "seed_variance_entrada109.csv"), pd.read_csv(R / "seed_variance_all_entrada117.csv")])
m = sv.groupby(["instance", "method", "rep"]).makespan.mean().unstack("method").reset_index()
chk("seeds: n instancias", len(m), 240, 0)
chk("seeds: politica d (-0,011)", rel(m.ig_ql, m.ig_random).mean(), -0.0110, 0.0006)
chk("seeds: IG vs ILS random (-0,493)", rel(m.ig_random, m.ils_random).mean(), -0.493, 0.002)
chk("seeds: IG vs ILS ql (-0,532)", rel(m.ig_ql, m.ils_ql).mean(), -0.532, 0.002)
g = m[m.instance == "rand_30j_8m"]
chk("seeds: (30,8) politica (-0,076)", rel(g.ig_ql, g.ig_random).mean(), -0.0764, 0.0006)

# ---------------------------------------------------------------- d como acao
P = pd.read_csv(R / "entrada114_dsel_primary.csv").set_index("id")
chk("dsel A1 (-0,089)", P.loc["A1", "mean_d_pct"], -0.0885, 0.0006)
chk("dsel A2 (+0,015)", P.loc["A2", "mean_d_pct"], 0.0147, 0.0006)
chk("dsel A3 (+0,007)", P.loc["A3", "mean_d_pct"], 0.0069, 0.0006)
chk("dsel A4 (+0,103)", P.loc["A4", "mean_d_pct"], 0.1034, 0.0006)
chk("dsel A4 piores (318)", P.loc["A4", "better"], 57, 0)
chk("dsel A4 decisivas (375)", P.loc["A4", "better"] + P.loc["A4", "worse"], 375, 0)

# ---------------------------------------------------------------- lei do orcamento
B = pd.read_csv(R / "entrada121_budget_law_primary.csv").set_index("id")
chk("budget B1 (-0,011)", B.loc["B1", "mean_d_pct"], -0.0109, 0.0006)
chk("budget B2 (-0,015)", B.loc["B2", "mean_d_pct"], -0.0147, 0.0006)
chk("budget B4 diff (-0,066)", B.loc["B4", "mean_diff"], -0.0657, 0.0006)
Dd = pd.read_csv(R / "entrada121_dsel_budget_primary.csv").set_index("id")
chk("dsel T=5000 D1 (-0,010)", Dd.loc["D1", "mean_d_pct"], -0.0102, 0.0006)
chk("dsel T=5000 D2 (-0,049)", Dd.loc["D2", "mean_d_pct"], -0.0489, 0.0006)
chk("dsel pretreino D3 (-0,032)", Dd.loc["D3", "mean_d_pct"], -0.0323, 0.0006)
chk("dsel pretreino D4 (+0,063)", Dd.loc["D4", "mean_d_pct"], 0.0634, 0.0006)

# ---------------------------------------------------------------- confirmatorio 2
Q = pd.read_csv(R / "entrada115_external_confirm2_primary.csv").set_index("id")
chk("conf2 O_B pp (-0,0147)", Q.loc["O_B", "mean_diff_pp"], -0.01473, 0.00006)
chk("conf2 O_B % (-5,4)", Q.loc["O_B", "rel_gap_pct"], -5.354, 0.06)
chk("conf2 O_C pp (-0,0294)", Q.loc["O_C", "mean_diff_pp"], -0.02938, 0.00006)
chk("conf2 O_C % (-1,8)", Q.loc["O_C", "rel_gap_pct"], -1.846, 0.06)
chk("conf2 O_E pp (-0,0268)", Q.loc["O_E", "mean_diff_pp"], -0.02681, 0.00006)
chk("conf2 O_E % (-3,2)", Q.loc["O_E", "rel_gap_pct"], -3.203, 0.06)
chk("conf2 R_B % (-2,9)", Q.loc["R_B", "rel_gap_pct"], -2.869, 0.06)
chk("conf2 R_C % (-3,1)", Q.loc["R_C", "rel_gap_pct"], -3.126, 0.06)
chk("conf2 R_E % (-3,3)", Q.loc["R_E", "rel_gap_pct"], -3.334, 0.06)

# ---------------------------------------------------------------- ILS semeado
I = pd.read_csv(R / "entrada117_ils_seeded_primary.csv")
g_ = lambda s, i: I[(I.set == s) & (I.id == i)].iloc[0]
chk("ILS+ I1 S (+0,015)", g_("S", "I1").mean_d_pct, 0.0147, 0.0006)
chk("ILS+ I1 X (+0,024)", g_("X", "I1").mean_d_pct, 0.0240, 0.0006)
chk("IG+ vs ILS+ S (-0,093)", g_("S", "I2").mean_d_pct, -0.0929, 0.0006)
chk("IG+ vs ILS+ S ganhos IG (164)", g_("S", "I2").better, 164, 0)
chk("IG+ vs ILS+ X (-0,065)", g_("X", "I2").mean_d_pct, -0.0648, 0.0006)
chk("IG+ vs ILS+ X ILS ganha (554)", g_("X", "I2").worse, 554, 0)
chk("IG+ vs ILS+ X IG ganha (232)", g_("X", "I2").better, 232, 0)

# ---------------------------------------------------------------- igcv
E = pd.read_csv(R / "entrada121_igcv_seed_primary.csv")
g_ = lambda s, i: E[(E.set == s) & (E.id == i)].iloc[0]
for s_, i_, v in [("S", "E1", -0.0342), ("X", "E1", -0.0995), ("S", "E2", -0.2877), ("X", "E2", -0.8489), ("S", "E3", -1.061), ("X", "E3", -1.145), ("S", "E4", -0.2066), ("X", "E4", -0.4202)]:
    chk(f"igcv {s_} {i_}", g_(s_, i_).mean_d_pct, v, 0.002)

# ---------------------------------------------------------------- R||Cmax
M = pd.read_csv(R / "entrada118_unrelated_primary.csv").set_index("id")
chk("R M1 (-4,16)", M.loc["M1", "mean_d_pct"], -4.16, 0.01)
chk("R M1 melhores (384)", M.loc["M1", "better"], 384, 0)
chk("R M2 (-0,145)", M.loc["M2", "mean_d_pct"], -0.1449, 0.001)
chk("R M3 (+0,956)", M.loc["M3", "mean_d_pct"], 0.9558, 0.001)
chk("R M4 (+1,59)", M.loc["M4", "mean_d_pct"], 1.592, 0.005)
d = pd.read_csv(R / "unrelated_entrada118.csv")
w = d.pivot_table(index=["lo", "hi", "n", "m", "rep"], columns="method", values="makespan").reset_index()
chk("R IG puro vs ILS (+1,39)", rel(w.ur_ig_random, w.ur_ils_random).mean(), 1.394, 0.005)
chk("R LS efeito (-5,40)", rel(w.ur_igls_random, w.ur_ig_random).mean(), -5.404, 0.005)
chk("R igls vs mct (-13,1)", rel(w.ur_igls_random, w.mct).mean(), -13.12, 0.01)
chk("R ils vs mct (-9,3)", rel(w.ur_ils_random, w.mct).mean(), -9.319, 0.01)
H = pd.read_csv(R / "entrada118_unrelated_by_cell.csv")
chk("R oraculo agregado (-0,377)", H.d_oracle.mean(), -0.3773, 0.001)
chk("R oraculo maximo (-1,12)", H.d_oracle.min(), -1.119, 0.002)
chk("R captura (0,38)", H.d_ql.mean() / H.d_oracle.mean(), 0.384, 0.005)
ex = pd.read_csv(R / "unrelated_exact_entrada118.csv")
chk("R MILP-60s otimo provado (66,9)", 100 * ex.milp60_proven.mean(), 66.9, 0.1)
chk("R MILP-5s otimo provado (31,9)", 100 * ex.milp5_proven.mean(), 31.9, 0.1)
Bu = pd.read_csv(R / "entrada121_unrelated_budget_primary.csv").set_index("id")
chk("R RB2 (-2,16)", Bu.loc["RB2", "mean_d_pct"], -2.158, 0.005)
chk("R RB3 (-0,087)", Bu.loc["RB3", "mean_d_pct"], -0.0866, 0.001)
chk("R RB4 (+0,95)", Bu.loc["RB4", "mean_d_pct"], 0.9514, 0.005)

out = pd.DataFrame(rows, columns=["claim", "recalculado", "no_texto", "tol", "ok"])
pd.set_option("display.width", 200, "display.max_rows", 300, "display.float_format", lambda x: f"{x:.4f}")
print(out.to_string(index=False))
print(f"\n{int(out.ok.sum())}/{len(out)} OK; divergentes: {list(out[~out.ok].claim)}")
(R / "entrada126_final_claims_audit.txt").write_text(out.to_string(index=False) + f"\n\n{int(out.ok.sum())}/{len(out)} OK\n", encoding="utf-8")
