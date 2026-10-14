"""
analysis/audit_entrada133_cycle5_claims.py (Entrada 133)
Auditoria numerica das alegacoes do manuscrito baseadas nos experimentos do Ciclo 5 (Entradas 127-132: receita com busca local, R||Cmax 1000 x 50 vs MILP, hibrido a CPU igual)
e, se existirem os arquivos, do Ciclo 6 (Entrada 133: politica em regime nao saturado). Recalcula a partir dos CSV brutos e compara com o valor literal do texto.
Complementa audit_entrada126_final_claims.py e audit_entrada127_cycle4_claims.py.
Uso: python -m analysis.audit_entrada133_cycle5_claims
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
rows = []


def chk(label, value, expected, tol):
    rows.append((label, float(value), float(expected), tol, abs(float(value) - float(expected)) <= tol))


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


# ---------------------------------------------------------------- frente 1: receita com busca local
F = pd.read_csv(R / "entrada127_igcvls_primary.csv").set_index("id")
chk("F1 (-0,036)", F.loc["F1", "mean_d_pct"], -0.0361, 0.0006)
chk("F2 (+0,034)", F.loc["F2", "mean_d_pct"], 0.0339, 0.0006)
chk("F2 piores (115)", F.loc["F2", "worse"], 115, 0)
chk("F2 melhores (24)", F.loc["F2", "better"], 24, 0)
chk("F3 (-0,072)", F.loc["F3", "mean_d_pct"], -0.0721, 0.0006)
chk("F4 (-0,245)", F.loc["F4", "mean_d_pct"], -0.2446, 0.0006)
chk("F5 (-0,492)", F.loc["F5", "mean_d_pct"], -0.4918, 0.0006)
chk("F2 p_holm (3e-8)", F.loc["F2", "p_holm"] * 1e8, 2.949, 0.01)
nl = pd.read_csv(R / "igcvls_seed_entrada127.csv")
cv = pd.read_csv(R / "igcv_seed_entrada121.csv")
ref = pd.read_csv(R / "exact_reference_synthetic_entrada101.csv")[["instance", "rep", "lb_lp"]]
res = pd.read_csv(R / "results.csv")
s102 = pd.read_csv(R / "seeded_baselines_entrada102.csv")
keep = ["instance", "rep", "method", "makespan"]
d = pd.concat([res[res.method.isin(["lpt", "multifit"])][keep], s102[keep], cv[keep], nl[keep]]).drop_duplicates(["instance", "rep", "method"])
S = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().merge(ref, on=["instance", "rep"], how="left")
S["bc"] = S[["lpt", "multifit", "ldm"]].min(axis=1, skipna=False)
Sl = S.dropna(subset=["igcvls_seed", "igcv_seed", "igcv_seed_eq", "bc", "lb_lp"])
chk("S: n com LB certificado (1130)", len(Sl), 1130, 0)
for k, v in [("igcvls_seed", 81.2), ("igcv_seed", 74.7), ("igcv_seed_eq", 89.0), ("bc", 55.3)]:
    chk(f"S: otimo provado {k} ({v})", 100 * (Sl[k] == Sl.lb_lp).mean(), v, 0.06)
for k, v in [("igcvls_seed", 0.094), ("igcv_seed", 0.133), ("igcv_seed_eq", 0.058)]:
    chk(f"S: GAP {k} ({v})", 100 * ((Sl[k] - Sl.lb_lp) / Sl.lb_lp).mean(), v, 0.0006)
cpu = pd.concat([cv[["instance", "rep", "method", "cpu_time"]], nl[["instance", "rep", "method", "cpu_time"]]]).pivot(index=["instance", "rep"], columns="method", values="cpu_time")
chk("CPU medio igcv_seed S (0,35)", cpu.igcv_seed.mean(), 0.35, 0.006)
chk("CPU medio igcvls_seed S (6,7)", cpu.igcvls_seed.mean(), 6.69, 0.06)
cx = pd.read_csv(EXT / "results_external_igcv_seed_entrada121.csv").drop_duplicates("member").set_index("member")
nx = pd.read_csv(EXT / "results_external_igcvls_seed_entrada127.csv").drop_duplicates("member").set_index("member")
j = cx.join(nx, lsuffix="_cv", rsuffix="_ls", how="inner")
chk("X': pares (1500)", len(j), 1500, 0)
chk("X': CPU medio igcv_seed (0,37)", j.cpu_time_cv.mean(), 0.37, 0.006)
chk("X': CPU medio igcvls_seed (37,5)", j.cpu_time_ls.mean(), 37.5, 0.1)
ratio = json.loads((R / "igcvls_cpu_ratio_entrada127.json").read_text())
Ks = [v["K"] for v in ratio.values()]
Ts = [v["T_eq"] for v in ratio.values()]
chk("K minimo (2,3)", min(Ks), 2.30, 0.01)
chk("K maximo (82,9)", max(Ks), 82.86, 0.01)
chk("T_eq minimo (2298)", min(Ts), 2298, 0)
chk("T_eq maximo (82855)", max(Ts), 82855, 0)

# ---------------------------------------------------------------- frente 2: R||Cmax 1000 x 50 vs MILP
G = pd.read_csv(R / "entrada127_unrelated_large_primary.csv").set_index("id")
chk("G1 (+2,241)", G.loc["G1", "mean_d_pct"], 2.241, 0.001)
chk("G1 piores/MILP melhor (28)", G.loc["G1", "worse"], 28, 0)
chk("G2 (+0,887)", G.loc["G2", "mean_d_pct"], 0.887, 0.001)
chk("G2 p_holm (0,023)", G.loc["G2", "p_holm"], 0.02296, 0.0005)
dl = pd.read_csv(R / "unrelated_large_entrada127.csv")
ex = pd.read_csv(R / "unrelated_large_exact_entrada127.csv")
w = dl.pivot(index=["lo", "hi", "n", "m", "rep"], columns="method", values="makespan").reset_index().merge(ex, on=["lo", "hi", "n", "m", "rep"])
w["lb_best"] = np.maximum(w.lb_lp, w.milp60_dual)
chk("R grande: instancias (32)", len(w), 32, 0)
chk("R grande: MILP GAP (2,04)", (100 * (w.milp60_inc - w.lb_best) / w.lb_best).mean(), 2.040, 0.006)
chk("R grande: IG T=1000 GAP (4,15)", (100 * (w.ur_igls_ql - w.lb_best) / w.lb_best).mean(), 4.153, 0.006)
chk("R grande: IG T=5000 GAP (2,78)", (100 * (w.ur_igls_ql_T5000 - w.lb_best) / w.lb_best).mean(), 2.782, 0.006)
chk("R grande: politica ql vs random (-0,52)", rel(w.ur_igls_ql, w.ur_igls_random).mean(), -0.523, 0.002)
chk("R grande: MILP min GAP U[10,100] (1,3)", (100 * (w[(w.lo == 10)].milp60_inc - w[(w.lo == 10)].lb_best) / w[(w.lo == 10)].lb_best).min(), 1.32, 0.05)
chk("R grande: MILP max GAP U[10,100] (28,1)", (100 * (w[(w.lo == 10)].milp60_inc - w[(w.lo == 10)].lb_best) / w[(w.lo == 10)].lb_best).max(), 28.07, 0.05)
chk("R grande: CPU IG T=5000 (37)", dl[dl.method == "ur_igls_ql_T5000"].cpu_time.mean(), 37.1, 0.2)
chk("R grande: CPU MILP (51)", ex.milp60_t.mean(), 50.7, 0.2)
chk("R grande: d/n escolhido (0,02)", json.loads((R / "dsel_unrelated_large_entrada127.json").read_text())["frac"], 0.02, 0)

# ---------------------------------------------------------------- frente 3: hibrido a CPU igual
Qp = pd.read_csv(R / "entrada129_igls_eq_primary.csv").set_index("id")
chk("Q1 (-0,010)", Qp.loc["Q1", "mean_d_pct"], -0.0102, 0.0006)
chk("Q1 p (0,015)", Qp.loc["Q1", "p_raw"], 0.0145, 0.0006)
chk("Q1 IC90 sup (-0,002)", Qp.loc["Q1", "ci90_hi"], -0.0019, 0.0006)
chk("Q2 (-0,140)", Qp.loc["Q2", "mean_d_pct"], -0.1399, 0.0006)
hy = pd.read_csv(R / "igls_p_entrada124.csv")
eq = pd.read_csv(R / "igls_eq_entrada129.csv")
keep = ["instance", "rep", "method", "makespan"]
d2 = pd.concat([res[res.method == "ig_random"][keep], hy[hy.method == "igls_random"][keep], eq[keep]]).drop_duplicates(["instance", "rep", "method"])
W = d2.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().merge(ref, on=["instance", "rep"], how="left").dropna(subset=["ig_random", "igls_random", "ig_random_eq", "lb_lp"])
chk("hibrido eq: n (1200)", len(W), 1200, 0)
for k, v in [("ig_random", 36.9), ("igls_random", 55.2), ("ig_random_eq", 54.5)]:
    chk(f"hibrido eq: otimo provado {k} ({v})", 100 * (W[k] == W.lb_lp).mean(), v, 0.06)
chk("hibrido eq: fracao do ganho que some (93%)", 100 * (1 - Qp.loc["Q1", "mean_d_pct"] / rel(W.igls_random, W.ig_random).mean()), 93.2, 0.3)
rat = json.loads((R / "igls_eq_cpu_ratio_entrada129.json").read_text())
chk("hibrido eq: K minimo (2,1)", min(v["K"] for v in rat.values()), 2.064, 0.01)
chk("hibrido eq: K maximo (52,4)", max(v["K"] for v in rat.values()), 52.42, 0.01)
chk("hibrido eq: T_eq minimo (2064)", min(v["T_eq"] for v in rat.values()), 2064, 0)
chk("hibrido eq: T_eq maximo (52425)", max(v["T_eq"] for v in rat.values()), 52425, 0)

# ---------------------------------------------------------------- Ciclo 6 (se existir)
pf = R / "entrada133_unrelated_policy_primary.csv"
if pf.exists():
    P6 = pd.read_csv(pf).set_index("id")
    d6 = pd.read_csv(R / "unrelated_policy_entrada133.csv")
    w6 = d6.pivot(index=["lo", "hi", "n", "m", "rep"], columns="method", values="makespan").reset_index()
    chk("C6 instancias (96)", len(w6), 96, 0)
    chk("C6 Q1 recalculado", rel(w6.ur_igls_ql, w6.ur_igls_random).mean(), P6.loc["Q1", "mean_d_pct"], 1e-9)
    chk("C6 Q2 recalculado", rel(w6.ur_igls_ql, w6.ur_igls_random_eq).mean(), P6.loc["Q2", "mean_d_pct"], 1e-9)
    chk("C6 Q1 (-0,35)", P6.loc["Q1", "mean_d_pct"], -0.353, 0.001)
    chk("C6 Q1 p_holm (0,19)", P6.loc["Q1", "p_holm"], 0.1925, 0.001)
    chk("C6 Q2 (-0,19)", P6.loc["Q2", "mean_d_pct"], -0.190, 0.001)
    chk("C6 Q2 p_holm (0,67)", P6.loc["Q2", "p_holm"], 0.6664, 0.001)
    B6 = pd.read_csv(R / "entrada134_policy_by_dist.csv").set_index("dist")
    chk("C6 U[1,100] ql vs random (-1,73)", B6.loc["U[1,100]", "d_random"], -1.729, 0.001)
    chk("C6 U[1,100] vitorias (17)", B6.loc["U[1,100]", "better_random"], 17, 0)
    chk("C6 U[1,100] derrotas (0)", B6.loc["U[1,100]", "worse_random"], 0, 0)
    chk("C6 U[1,100] melhor braco fixo (-1,66)", B6.loc["U[1,100]", "best_fixed"], -1.657, 0.001)
    chk("C6 U[100,120] ql vs random (+0,10)", B6.loc["U[100,120]", "d_random"], 0.098, 0.001)
    chk("C6 U[100,200] ql vs random (+0,38)", B6.loc["U[100,200]", "d_random"], 0.376, 0.001)
    chk("C6 U[100,120] a CPU igual (+0,11)", B6.loc["U[100,120]", "d_eq"], 0.110, 0.001)
    chk("C6 U[100,200] a CPU igual (+0,89)", B6.loc["U[100,200]", "d_eq"], 0.893, 0.001)
    chk("C6 U[100,200] a CPU igual derrotas (23)", B6.loc["U[100,200]", "worse_eq"], 23, 0)
    chk("C6 U[1,100] p_holm4 (8e-4)", B6.loc["U[1,100]", "p_holm4_random"] * 1e4, 8.269, 0.01)
    chk("C6 razao CPU ql/random geral (1,39)", d6[d6.method == "ur_igls_ql"].cpu_time.mean() / d6[d6.method == "ur_igls_random"].cpu_time.mean(), 1.39, 0.01)
    chk("C6 CPU U[100,200] ql/random (1,63)", B6.loc["U[100,200]", "cpu_ql_over_random"], 1.632, 0.005)
    lowest = min(B6.loc["U[100,120]", [f"fix{k}" for k in range(4)]].min(), B6.loc["U[100,200]", [f"fix{k}" for k in range(4)]].min())
    highest = max(B6.loc["U[100,120]", [f"fix{k}" for k in range(4)]].max(), B6.loc["U[100,200]", [f"fix{k}" for k in range(4)]].max())
    chk("C6 estrategias fixas piores que a mistura, minimo (0,9)", lowest, 0.923, 0.001)
    chk("C6 estrategias fixas piores que a mistura, maximo (4,5)", highest, 4.512, 0.001)
    chk("C6 T_eq (1000, 1164, 1734, 1436)", sum(v["T_eq"] for v in json.loads((R / "unrelated_policy_cpu_ratio_entrada133.json").read_text()).values()), 1000 + 1164 + 1734 + 1436, 0)

pf7 = R / "entrada135_unrelated_valid_primary.csv"
if pf7.exists():
    P7 = pd.read_csv(pf7).set_index("id")
    d7 = pd.read_csv(R / "unrelated_valid_entrada135.csv")
    chk("C7 V2 (-0,46)", P7.loc["V2", "mean_d_pct"], -0.4608, 0.001)
    chk("C7 V2 vitorias (17)", P7.loc["V2", "better"], 17, 0)
    chk("C7 V2 derrotas (0)", P7.loc["V2", "worse"], 0, 0)
    chk("C7 V2 p_holm (8e-4)", P7.loc["V2", "p_holm"] * 1e4, 8.435, 0.01)
    chk("C7 V1 (-0,14)", P7.loc["V1", "mean_d_pct"], -0.135, 0.001)
    chk("C7 V1 IC90 inf (-0,34)", P7.loc["V1", "ci90_lo"], -0.3366, 0.001)
    chk("C7 V1 IC90 sup (+0,07)", P7.loc["V1", "ci90_hi"], 0.0703, 0.001)
    chk("C7 V1 p_holm (0,30)", P7.loc["V1", "p_holm"], 0.2962, 0.001)
    chk("C7 V3 (-0,31)", P7.loc["V3", "mean_d_pct"], -0.3118, 0.001)
    chk("C7 CPU validada (3,3)", d7[d7.method.isin(["ur_igls_random"])].cpu_time.mean(), 3.3, 0.1)
    chk("C7 CPU ql (4,7)", d7[d7.method == "ur_igls_ql"].cpu_time.mean(), 4.7, 0.1)
    w7 = d7.pivot(index=["lo", "hi", "rep"], columns="method", values="makespan").reset_index()
    g1 = w7[w7.lo == 1]
    chk("C7 U[1,100] ql vs fix0 (valida) (-0,45)", rel(g1.ur_igls_ql, g1.ur_igls_fix0).mean(), -0.4465, 0.001)
    g4 = w7[(w7.lo == 100) & (w7.hi == 200)]
    chk("C7 U[100,200] ql vs mistura (+1,1)", rel(g4.ur_igls_ql, g4.ur_igls_random).mean(), 1.103, 0.001)

pf8 = R / "entrada137_rcmax_benchmark_primary.csv"
if pf8.exists():
    P8 = pd.read_csv(pf8).set_index("id")
    d8 = pd.read_csv(R / "rcmax_test_entrada137.csv")
    m8 = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    d8["instance"] = d8.instance.astype(str)
    w8 = d8.pivot(index=["type", "instance", "m", "n"], columns="method", values="makespan").reset_index().merge(m8[["type", "instance", "best_known", "proven_opt", "lr", "gurobi", "cplex"]], on=["type", "instance"])
    chk("C8 instancias de teste (700)", len(w8), 700, 0)
    chk("C8 referencia provada otima (73)", 100 * w8.proven_opt.mean(), 73.4, 0.1)
    chk("C8 B1 recalculado (-2,14)", rel(w8.ur_igls_random, w8.ur_ils_random_eq).mean(), -2.135, 0.001)
    chk("C8 B1 vitorias (529)", P8.loc["B1", "better"], 529, 0)
    chk("C8 B1 derrotas (99)", P8.loc["B1", "worse"], 99, 0)
    chk("C8 B1 a T igual (-3,23)", rel(w8.ur_igls_random, w8.ur_ils_random).mean(), -3.232, 0.001)
    chk("C8 B2a (-0,055)", P8.loc["B2a", "mean_d_pct"], -0.0546, 0.0006)
    chk("C8 B2a p_holm (0,15)", P8.loc["B2a", "p_holm"], 0.1495, 0.001)
    chk("C8 B2b (-0,033)", P8.loc["B2b", "mean_d_pct"], -0.0327, 0.0006)
    chk("C8 B2b p_holm (0,48)", P8.loc["B2b", "p_holm"], 0.4829, 0.001)
    chk("C8 B3 rho (0,74)", P8.loc["B3", "mean_d_pct"], 0.7438, 0.0006)
    chk("C8 B4 (-0,066)", P8.loc["B4", "mean_d_pct"], -0.0663, 0.0006)
    chk("C8 B4 p_holm (0,007)", P8.loc["B4", "p_holm"], 0.00655, 0.0005)
    gap = lambda k, df=w8: 100 * ((df[k] - df.best_known) / df.best_known).mean()
    chk("C8 GAP IG+LS-QL (2,08)", gap("ur_igls_ql"), 2.085, 0.001)
    chk("C8 GAP valida/ILS-eq (4,55)", gap("ur_ils_random_eq"), 4.548, 0.001)
    chk("C8 GAP MCT (14,7)", gap("mct"), 14.712, 0.001)
    chk("C8 GAP Gurobi (0,07)", gap("gurobi"), 0.067, 0.001)
    chk("C8 GAP LR (0,09)", gap("lr"), 0.093, 0.001)
    chk("C8 GAP CPLEX (0,18)", gap("cplex"), 0.183, 0.001)
    chk("C8 IG+LS-QL <= melhor conhecido (10,9%)", 100 * (w8.ur_igls_ql <= w8.best_known).mean(), 10.9, 0.05)
    chk("C8 IG+LS-QL < melhor conhecido (0,1%)", 100 * (w8.ur_igls_ql < w8.best_known).mean(), 0.1, 0.05)
    for tp, v in [("U1a100", 5.638), ("U10a100", 4.172), ("U1000a1100", 0.1325)]:
        chk(f"C8 GAP {tp} ({v})", gap("ur_igls_ql", w8[w8["type"] == tp]), v, 0.001)
    for tp, v, nb, nw in [("U1a100", -0.6135, 46, 9), ("U10a100", -0.1505, 49, 24), ("MaqCorre", 0.2026, 26, 39)]:
        g = w8[w8["type"] == tp]
        x = rel(g.ur_igls_ql, g.ur_igls_random)
        chk(f"C8 politica {tp} ({v})", x.mean(), v, 0.001)
        chk(f"C8 politica {tp} melhor ({nb})", (x < 0).sum(), nb, 0)
        chk(f"C8 politica {tp} pior ({nw})", (x > 0).sum(), nw, 0)
    cal = json.loads((R / "rcmax_cpu_ratio_entrada137.json").read_text())
    chk("C8 T_ils max (8179)", max(v["T_ils"] for v in cal.values()), 8179, 0)
    chk("C8 T_random_eq max (1192)", max(v["T_random_eq"] for v in cal.values()), 1192, 0)

pf9 = R / "entrada141_lpig_primary.csv"
if pf9.exists():
    P9 = pd.read_csv(pf9).set_index("id")
    d9 = pd.read_csv(R / "lpig_test_entrada141.csv")
    m9 = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    d9["instance"] = d9.instance.astype(str)
    w9 = d9.pivot(index=["type", "instance", "m", "n"], columns="method", values="makespan").reset_index().merge(m9[["type", "instance", "best_known", "proven_opt", "lr", "gurobi", "cplex"]], on=["type", "instance"])
    g9 = lambda k, df=w9: 100 * ((df[k] - df.best_known) / df.best_known).mean()
    chk("C9 instancias reservadas (420)", len(w9), 420, 0)
    chk("C9 L1 (-1,18)", P9.loc["L1", "mean_d_pct"], -1.176, 0.001)
    chk("C9 L1 vitorias (308)", P9.loc["L1", "better"], 308, 0)
    chk("C9 L1 derrotas (21)", P9.loc["L1", "worse"], 21, 0)
    chk("C9 L2 (-0,12)", P9.loc["L2", "mean_d_pct"], -0.1236, 0.001)
    chk("C9 L2 vitorias (104)", P9.loc["L2", "better"], 104, 0)
    chk("C9 L3 (+0,055)", P9.loc["L3", "mean_d_pct"], 0.0555, 0.001)
    chk("C9 L3 p_holm (0,12)", P9.loc["L3", "p_holm"], 0.1222, 0.001)
    chk("C9 GAP baseline (2,23)", g9("ur_igls_random"), 2.229, 0.001)
    chk("C9 GAP LP-IG (0,97)", g9("lpig"), 0.973, 0.001)
    chk("C9 GAP LP-IG T5000 (0,85)", g9("lpig_T5000"), 0.846, 0.001)
    chk("C9 GAP LP-IG n=1000 (0,31)", g9("lpig", w9[w9.n == 1000]), 0.3104, 0.001)
    chk("C9 GAP Gurobi (0,10)", g9("gurobi"), 0.101, 0.001)
    chk("C9 GAP LR (0,09)", g9("lr"), 0.090, 0.001)
    chk("C9 iguala/bate T5000 (27%)", 100 * (w9.lpig_T5000 <= w9.best_known).mean(), 27.1, 0.05)
    chk("C9 estritamente melhor T5000 (3,8%)", 100 * (w9.lpig_T5000 < w9.best_known).mean(), 3.8, 0.05)
    chk("C9 CPU LP-IG (3,3)", d9[d9.method == "lpig"].cpu_time.mean(), 3.30, 0.05)
    chk("C9 CPU LP-IG T5000 (15,7)", d9[d9.method == "lpig_T5000"].cpu_time.mean(), 15.68, 0.05)

pf12 = R / "entrada143_lpig_large_primary.csv"
if pf12.exists():
    P12 = pd.read_csv(pf12).set_index("id")
    d12 = pd.read_csv(R / "lpig_large_entrada143.csv")
    ex12 = pd.read_csv(R / "unrelated_large_exact_entrada127.csv")
    w12 = d12.pivot(index=["lo", "hi", "n", "m", "rep"], columns="method", values="makespan").reset_index().merge(ex12, on=["lo", "hi", "n", "m", "rep"])
    w12["lb_best"] = np.maximum(w12.lb_lp, w12.milp60_dual)
    g12 = lambda k, df=w12: 100 * ((df[k] - df.lb_best) / df.lb_best).mean()
    chk("C9b M1 (-0,76)", P12.loc["M1", "mean_d_pct"], -0.7578, 0.001)
    chk("C9b M1 p_holm (0,72)", P12.loc["M1", "p_holm"], 0.7224, 0.001)
    chk("C9b M1 IC90 inf (-2,10)", P12.loc["M1", "ci90_lo"], -2.102, 0.001)
    chk("C9b M1 IC90 sup (+0,13)", P12.loc["M1", "ci90_hi"], 0.1275, 0.001)
    chk("C9b GAP LP-IG (1,09)", g12("lpig"), 1.088, 0.001)
    chk("C9b GAP MILP (2,04)", g12("milp60_inc"), 2.040, 0.001)
    chk("C9b LP-IG <= MILP (62,5%)", 100 * (w12.lpig <= w12.milp60_inc).mean(), 62.5, 0.05)
    chk("C9b CPU LP-IG (6,5)", d12[d12.method == "lpig"].cpu_time.mean(), 6.5, 0.1)
    chk("C9b U[1,100] LP-IG (2,46)", g12("lpig", w12[w12.lo == 1]), 2.457, 0.001)
    chk("C9b U[1,100] MILP (1,23)", g12("milp60_inc", w12[w12.lo == 1]), 1.226, 0.001)
    chk("C9b U[10,100] LP-IG (1,59)", g12("lpig", w12[w12.lo == 10]), 1.589, 0.001)
    chk("C9b U[10,100] MILP (6,25)", g12("milp60_inc", w12[w12.lo == 10]), 6.248, 0.001)

pf13 = R / "entrada139_recipe_budget_table.csv"
if pf13.exists():
    D13 = pd.read_csv(pf13).set_index("classe")
    chk("H3 pooled otimo T=1000 (75,5)", D13.loc["pooled", "opt%@1000"], 75.5, 0.05)
    chk("H3 pooled otimo T=4000 (86,8)", D13.loc["pooled", "opt%@4000"], 86.83, 0.05)
    chk("H3 pooled otimo T=16000 (92,7)", D13.loc["pooled", "opt%@16000"], 92.67, 0.05)
    chk("H3 pooled otimo T=64000 (97,5)", D13.loc["pooled", "opt%@64000"], 97.5, 0.05)
    chk("H3 pooled GAP T=1000 (0,099)", D13.loc["pooled", "GAP@1000"], 0.0994, 0.0006)
    chk("H3 pooled GAP T=64000 (0,009)", D13.loc["pooled", "GAP@64000"], 0.0094, 0.0006)
    chk("H3 (50,10) otimo T=64000 (100)", D13.loc["50x10", "opt%@64000"], 100.0, 0.05)

pf15 = R / "entrada145_dpolicy_primary.csv"
if pf15.exists():
    P15 = pd.read_csv(pf15).set_index("id")
    d15 = pd.read_csv(R / "dpolicy_test_entrada145.csv")
    d15["instance"] = d15.instance.astype(str)
    m15 = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    w15 = d15.pivot(index=["type", "instance", "m", "n"], columns="method", values="makespan").reset_index().merge(m15[["type", "instance", "best_known"]], on=["type", "instance"])
    chk("C9c instancias (700)", len(w15), 700, 0)
    chk("C9c D1 (-0,065)", P15.loc["D1", "mean_d_pct"], -0.0651, 0.001)
    chk("C9c D1 p_holm (0,006)", P15.loc["D1", "p_holm"], 0.00558, 0.0005)
    chk("C9c D2 (+0,152)", P15.loc["D2", "mean_d_pct"], 0.1521, 0.001)
    chk("C9c D3 (+0,360)", P15.loc["D3", "mean_d_pct"], 0.3602, 0.001)
    chk("C9c D4 (+0,249)", P15.loc["D4", "mean_d_pct"], 0.2487, 0.001)
    chk("C9c D5 rho (-0,37)", P15.loc["D5", "mean_d_pct"], -0.3727, 0.001)
    chk("C9c D2 recalculado", rel(w15.ql_d, w15.rand_d).mean(), P15.loc["D2", "mean_d_pct"], 1e-9)
    w15["oracle"] = w15[[f"fix_{k}" for k in range(5)]].min(axis=1)
    chk("C9c oraculo vs rand_d (-0,325)", rel(w15.oracle, w15.rand_d).mean(), -0.3250, 0.001)
    u = w15[w15["type"] == "U1a100"]
    chk("C9c U[1,100] ql vs rand_d (+0,44)", rel(u.ql_d, u.rand_d).mean(), 0.444, 0.001)
    chk("C9c U[1,100] folga (-1,05)", rel(u[[f"fix_{k}" for k in range(5)]].min(axis=1), u.rand_d).mean(), -1.051, 0.001)
    dg = pd.read_csv(R / "dpolicy_diagnostic_entrada147.csv")
    for win, v in [("0-100", 0.206), ("100-400", 0.011), ("400-1000", 0.002)]:
        chk(f"C9c diag recompensa braco 0 janela {win} ({v})", dg[(dg.window == win) & (dg.arm == 0)].mean_reward.mean(), v, 0.0006)
    fin = dg[dg.window == "final"].pivot_table(index=["type", "instance"], columns="arm", values="mean_reward")
    chk("C9c diag final d/n=0,01 vs 0,05 (-0,131)", (100 * (fin[0] - fin[2]) / fin[2]).mean(), -0.131, 0.001)
    chk("C9c diag final d/n=0,20 vs 0,05 (+0,550)", (100 * (fin[4] - fin[2]) / fin[2]).mean(), 0.550, 0.001)

pf17 = R / "entrada149_lpigsr_primary.csv"
if pf17.exists():
    P17 = pd.read_csv(pf17).set_index("id")
    n17 = pd.read_csv(R / "lpigsr_test_entrada149.csv")
    o17 = pd.read_csv(R / "lpig_test_entrada141.csv")
    m17 = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    for dd in (n17, o17):
        dd["instance"] = dd.instance.astype(str)
    dd = pd.concat([n17[["type", "instance", "m", "n", "method", "makespan", "cpu_time"]], o17[["type", "instance", "m", "n", "method", "makespan", "cpu_time"]]])
    w17 = dd.pivot(index=["type", "instance", "m", "n"], columns="method", values="makespan").reset_index().merge(m17[["type", "instance", "best_known", "gurobi", "lr", "cplex"]], on=["type", "instance"])
    g17 = lambda k, df=w17: 100 * ((df[k] - df.best_known) / df.best_known).mean()
    chk("C9d instancias (420)", len(w17), 420, 0)
    chk("C9d S1 (-0,49)", P17.loc["S1", "mean_d_pct"], -0.4892, 0.001)
    chk("C9d S1 vitorias (196)", P17.loc["S1", "better"], 196, 0)
    chk("C9d S1 derrotas (33)", P17.loc["S1", "worse"], 33, 0)
    chk("C9d S2 (-0,675)", P17.loc["S2", "mean_d_pct"], -0.6749, 0.001)
    chk("C9d S3 (-0,55)", P17.loc["S3", "mean_d_pct"], -0.5522, 0.001)
    chk("C9d GAP lpig_sr6 (0,25)", g17("lpig_sr6"), 0.254, 0.001)
    chk("C9d GAP lpig_T8000 (0,78)", g17("lpig_T8000"), 0.775, 0.001)
    chk("C9d GAP CPLEX (0,20)", g17("cplex"), 0.200, 0.001)
    chk("C9d GAP n=1000 (0,11)", g17("lpig_sr6", w17[w17.n == 1000]), 0.1066, 0.001)
    chk("C9d GAP n=100 (0,36)", g17("lpig_sr6", w17[w17.n == 100]), 0.3581, 0.001)
    chk("C9d GAP jobs correlacionados (0,67)", g17("lpig_sr6", w17[w17["type"] == "JobsCorre"]), 0.6734, 0.001)
    chk("C9d iguala/bate (50%)", 100 * (w17.lpig_sr6 <= w17.best_known).mean(), 50.2, 0.05)
    chk("C9d estritamente melhor (4,8%)", 100 * (w17.lpig_sr6 < w17.best_known).mean(), 4.8, 0.05)
    chk("C9d melhora o LP-IG (54,8%)", 100 * (w17.lpig_sr6 < w17.lpig).mean(), 54.8, 0.05)
    chk("C9d integridade lpig_sr6 > lpig (0)", int((w17.lpig_sr6 > w17.lpig).sum()), 0, 0)
    cp = dd.groupby("method").cpu_time.mean()
    chk("C9d CPU lpig_sr6 (23,3)", cp["lpig_sr6"], 23.3, 0.1)
    chk("C9d CPU lpig_T8000 (27,3)", cp["lpig_T8000"], 27.3, 0.1)

pf19 = R / "entrada152_pfsp_primary.csv"
if pf19.exists():
    P19 = pd.read_csv(pf19).set_index("id")
    d19 = pd.read_csv(R / "pfsp_test_entrada152.csv")
    w19 = d19.pivot(index=["n", "m", "rep"], columns="method", values="makespan").reset_index()
    cpu19 = d19.pivot(index=["n", "m", "rep"], columns="method", values="cpu_time")
    chk("C10 instancias PFSP (120)", len(w19), 120, 0)
    chk("C10 P1 (-0,027)", P19.loc["P1", "mean_d_pct"], -0.0273, 0.001)
    chk("C10 P1 p_holm (0,18)", P19.loc["P1", "p_holm"], 0.1845, 0.001)
    chk("C10 P2 (-0,212)", P19.loc["P2", "mean_d_pct"], -0.2116, 0.001)
    chk("C10 P2 vitorias (65)", P19.loc["P2", "better"], 65, 0)
    chk("C10 P2 derrotas (17)", P19.loc["P2", "worse"], 17, 0)
    chk("C10 P3 (+0,142)", P19.loc["P3", "mean_d_pct"], 0.1418, 0.001)
    chk("C10 P4 (-0,161)", P19.loc["P4", "mean_d_pct"], -0.1612, 0.001)
    chk("C10 P5 (-0,020)", P19.loc["P5", "mean_d_pct"], -0.0205, 0.001)
    chk("C10 P3 recalculado", rel(w19.ig_ql, w19.ig_rand).mean(), P19.loc["P3", "mean_d_pct"], 1e-9)
    cc = json.loads((R / "pfsp_choice_entrada152.json").read_text())
    chk("C10 T_nols minimo (5177)", min(cc["T_nols"].values()), 5177, 0)
    chk("C10 T_nols maximo (17076)", max(cc["T_nols"].values()), 17076, 0)
    chk("C10 T_ils minimo (217)", min(cc["T_ils"].values()), 217, 0)
    chk("C10 T_ils maximo (250)", max(cc["T_ils"].values()), 250, 0)

pf20 = R / "synthesis_headroom_table.csv"
if pf20.exists():
    S = pd.read_csv(pf20)
    chk("C10 sintese: ambientes (7)", len(S), 7, 0)
    chk("C10 sintese: oraculo negativo em (6)", int((S.oracle < 0).sum()), 6, 0)
    size_rows = S[S.setting.str.contains("destruction size")]
    chk("C10 sintese: Q-Learning sobre tamanho > 0 nos 3", int((size_rows.ql > 0).sum()), 3, 0)
    strat_rows = S[S.setting.str.contains("strategy")]
    chk("C10 sintese: Q-Learning sobre estrategia <= 0 nos 4", int((strat_rows.ql <= 0).sum()), 4, 0)
    chk("C10 sintese: captura 1000x50 (0,68)", float(S[S.setting.str.contains("1000")].ql_over_validated.iloc[0]), 0.677, 0.005)
    chk("C10 sintese: captura benchmark estrategia (0,44)", float(S[S.setting.str.contains("standard benchmark, destruction strategy")].ql_over_validated.iloc[0]), 0.440, 0.005)
    chk("C10 sintese: Q-Learning tamanho P (+0,015)", float(S[S.setting.str.contains("P.*destruction size", regex=True) & ~S.setting.str.contains("R")].ql.iloc[0]), 0.015, 0.001)

pf21 = R / "entrada156_lprc_primary.csv"
if pf21.exists():
    P21 = pd.read_csv(pf21).set_index("id")
    d21 = pd.read_csv(R / "lprc_test_entrada156.csv")
    d21["instance"] = d21.instance.astype(str)
    m21 = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    w21 = d21.pivot(index=["type", "instance", "m", "n"], columns="method", values="makespan").reset_index().merge(m21[["type", "instance", "best_known"]], on=["type", "instance"])
    g21 = lambda k, df=w21: 100 * ((df[k] - df.best_known) / df.best_known).mean()
    chk("C11 instancias (700)", len(w21), 700, 0)
    chk("C11 R1 (-0,028)", P21.loc["R1", "mean_d_pct"], -0.0275, 0.001)
    chk("C11 R2 (-0,415)", P21.loc["R2", "mean_d_pct"], -0.4153, 0.001)
    chk("C11 R3 (-0,613)", P21.loc["R3", "mean_d_pct"], -0.6131, 0.001)
    chk("C11 GAP sr6 (0,30)", g21("lpig_sr6"), 0.296, 0.001)
    chk("C11 GAP rc6 (0,27)", g21("lpig_rc6"), 0.267, 0.001)
    chk("C11 maquinas correlacionadas sr6 (0,31)", g21("lpig_sr6", w21[w21["type"] == "MaqCorre"]), 0.308, 0.001)
    chk("C11 maquinas correlacionadas rc6 (0,08)", g21("lpig_rc6", w21[w21["type"] == "MaqCorre"]), 0.0769, 0.001)
    chk("C11 U[1,100] sr6 (0,40)", g21("lpig_sr6", w21[w21["type"] == "U1a100"]), 0.3953, 0.001)
    chk("C11 U[1,100] rc6 (0,50)", g21("lpig_rc6", w21[w21["type"] == "U1a100"]), 0.4993, 0.001)
    chk("C11 U[10,100] rc6 (0,64)", g21("lpig_rc6", w21[w21["type"] == "U10a100"]), 0.6388, 0.001)

out = pd.DataFrame(rows, columns=["claim", "recalculado", "no_texto", "tol", "ok"])
pd.set_option("display.width", 200, "display.max_rows", 300, "display.float_format", lambda x: f"{x:.4f}")
print(out.to_string(index=False))
print(f"\n{int(out.ok.sum())}/{len(out)} OK; divergentes: {list(out[~out.ok].claim)}")
(R / "entrada133_cycle5_claims_audit.txt").write_text(out.to_string(index=False) + f"\n\n{int(out.ok.sum())}/{len(out)} OK\n", encoding="utf-8")
