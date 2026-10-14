"""
analysis/audit_entrada113_final_claims.py (Entrada 113)
Auditoria final: recalcula, a partir dos CSV, os numeros que o manuscrito (e o suplemento) afirma nas partes escritas nas
Entradas 101-112 e compara com o valor literal que esta' no texto (tolerancia indicada). Saida: OK / DIVERGE por alegacao.
Uso: python -m analysis.audit_entrada113_final_claims
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada102_seeded_baselines import load_external, load_synthetic, rel  # noqa: E402

rows = []


def chk(label, value, expected, tol):
    ok = abs(value - expected) <= tol
    rows.append((label, value, expected, tol, ok))


# ---------- LB certificado e referencia sintetica
v = pd.read_csv(ROOT / "results" / "entrada101_exact_vs_iaf.csv")
chk("validacao: violacoes LB>otimo (150)", int((v.lb > v.Z).sum()), 0, 0)
chk("validacao: LB==otimo (de 150)", int((v.lb == v.Z).sum()), 148, 0)
chk("validacao: mediana tempo (s)", float(v.t.median()), 0.2, 0.05)
chk("validacao: maximo tempo (s)", float(v.t.max()), 149, 1.5)
ref = pd.read_csv(ROOT / "results" / "exact_reference_synthetic_entrada101.csv")
chk("otimo provado /1200 (903)", int(ref.proven.sum()), 903, 0)
chk("otimo provado % (75,3)", 100 * ref.proven.mean(), 75.3, 0.05)
chk("tempo total LP (s) ~199", float(ref.t.sum()), 199, 1)

# ---------- baselines mais fortes (S: 1130; X: 6780)
S = load_synthetic().dropna(subset=["lpt", "multifit", "ig_ql", "ldm", "ig_ql_seed", "ig_random_seed"])
X = load_external()
chk("|S|", len(S), 1130, 0)
chk("|X|", len(X), 6780, 0)
g = lambda k: 100 * ((S[k] - S.lb_lp) / S.lb_lp).mean()
chk("GAP_AF lpt (2,18)", g("lpt"), 2.18, 0.005)
chk("GAP_AF ldm (1,23)", g("ldm"), 1.23, 0.005)
chk("GAP_AF multifit (0,43)", g("multifit"), 0.43, 0.005)
chk("GAP_AF ig_ql (0,35)", g("ig_ql"), 0.35, 0.005)
chk("GAP_AF ig_ql_seed (0,17)", g("ig_ql_seed"), 0.17, 0.005)
o = lambda k: 100 * (S[k] == S.lb_lp).mean()
chk("opt% multifit (48,5)", o("multifit"), 48.5, 0.05)
chk("opt% ig_ql (34,3)", o("ig_ql"), 34.3, 0.05)
chk("opt% ig_ql_seed (69,3)", o("ig_ql_seed"), 69.3, 0.05)
d = rel(S.ldm, S.multifit)
chk("multifit vence ldm (585)", int((d > 0).sum()), 585, 0)
chk("decisivas ldm vs multifit (725)", int((d != 0).sum()), 725, 0)
bc = S[["lpt", "multifit", "ldm"]].min(axis=1)
chk("ig_ql_seed melhora a melhor construtiva, S % (19)", 100 * (S.ig_ql_seed < bc).mean(), 19, 0.5)
bcx = X[["lpt", "multifit", "ldm"]].min(axis=1)
chk("idem, X % (34)", 100 * (X.ig_ql_seed < bcx).mean(), 34, 0.5)
chk("ig_ql pior que ldm, S % (21)", 100 * (S.ig_ql > S.ldm).mean(), 21, 0.5)
chk("ig_ql pior que ldm, X % (36)", 100 * (X.ig_ql > X.ldm).mean(), 36, 0.5)
for cls, n_exp in [("rand_150j_15m", 67), ("rand_250j_25m", 70), ("rand_300j_20m", 53)]:
    gs = S[S.group == cls]
    chk(f"ig_ql pior que ldm em {cls}", int((gs.ig_ql > gs.ldm).sum()), n_exp, 0)
big = S[S.n >= 100]
chk("n>=100: ig_ql_seed == melhor construtiva em todas", int((big.ig_ql_seed != big[["lpt", "multifit", "ldm"]].min(axis=1)).sum()), 0, 0)
chk("P1 S d (-0,010)", rel(S.ig_ql_seed, S.ig_random_seed).mean(), -0.010, 0.0006)
chk("P1 X d (-0,013)", rel(X.ig_ql_seed, X.ig_random_seed).mean(), -0.013, 0.0006)
s20 = S[S.group == "rand_20j_4m"]
chk("(20,4) gap ig_ql (0,000)", 100 * ((s20.ig_ql - s20.lb_lp) / s20.lb_lp).mean(), 0.0, 0.0005)
chk("(20,4) gap ig_ql_seed (0,026)", 100 * ((s20.ig_ql_seed - s20.lb_lp) / s20.lb_lp).mean(), 0.026, 0.0006)

# ---------- tempos
t = pd.read_csv(ROOT / "results" / "timing_entrada107.csv")
cpu = t.groupby(["instance", "method"]).cpu_time.median().unstack("method")
r = cpu.ils_ql / cpu.ig_ql
chk("ILS-QL/IG-QL min (1,3)", float(r.min()), 1.3, 0.05)
chk("ILS-QL/IG-QL max (24,8)", float(r.max()), 24.8, 0.05)
r3 = cpu.bco_I100 / cpu.ig_ql
chk("BCO100/IG-QL min (3,2)", float(r3.min()), 3.2, 0.05)
chk("BCO100/IG-QL max (16)", float(r3.max()), 16.2, 0.3)
rm = cpu.ig_ql / cpu.multifit
chk("ig_ql/multifit min (~48-50)", float(rm.min()), 48, 4)
chk("ig_ql/multifit max (~500)", float(rm.max()), 500, 40)
chk("ig_ql_seed CPU min (0,2 s)", float(cpu.ig_ql_seed.min()), 0.2, 0.01)
chk("ig_ql_seed CPU max (0,7 s)", float(cpu.ig_ql_seed.max()), 0.7, 0.01)

# ---------- politica: folga, sinal, hiperparametros, seeds (tabela f14 e texto)
sig = pd.read_csv(ROOT / "results" / "signal_headroom_entrada104.csv")
res = pd.read_csv(ROOT / "results" / "results.csv")
cols = ["instance", "rep", "method", "makespan"]
w = pd.concat([res[res.method == "ig_random"][cols], sig[cols]]).drop_duplicates(["instance", "rep", "method"]).pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
chk("dense d (+0,084)", rel(w.ig_ql_dense_cal, w.ig_random).mean(), 0.084, 0.0006)
chk("pretrained d (+0,002)", rel(w.ig_ql_ws_cal, w.ig_random).mean(), 0.0016, 0.0006)
chk("dense+pretrained d (+0,099)", rel(w.ig_ql_dense_ws_cal, w.ig_random).mean(), 0.099, 0.0006)
fix = ["ig_fix0", "ig_fix1", "ig_fix2"]
chk("oraculo por instancia (-0,037)", rel(w[fix].min(axis=1), w.ig_random).mean(), -0.037, 0.0006)
dfix1 = [rel(g_.ig_fix1, g_.ig_random).mean() for _, g_ in w.groupby("instance")]
chk("d_fix1 minimo por classe (0,06)", min(dfix1), 0.061, 0.002)
chk("d_fix1 maximo por classe (3,7)", max(dfix1), 3.73, 0.02)
hp = pd.read_csv(ROOT / "results" / "entrada105_hyperparam_primary.csv")
for f, e in [("ig_ql_d10", -0.112), ("ig_ql_d20", -0.057), ("ig_ql_d40", -0.037)]:
    chk(f"hiper {f}", float(hp[hp["first"] == f].mean_d_pct.iloc[0]), e, 0.0006)
chk("hiper alpha/gamma/decay max |d| (0,023)", float(hp[hp["first"].isin(["ig_ql_a05", "ig_ql_a30", "ig_ql_g50", "ig_ql_g99", "ig_ql_e990", "ig_ql_e999"])].mean_d_pct.abs().max()), 0.0228, 0.0006)
hy = pd.read_csv(ROOT / "results" / "hyperparam_sensitivity_entrada105.csv")
resb = res[res.method == "ig_ql"]
for cls, gd01, gd03 in [("rand_100j_20m", 0.19, 0.62), ("rand_150j_30m", 0.34, 0.84)]:
    chk(f"GAP ig_ql d=0.1 {cls}", float(hy[(hy.instance == cls) & (hy.method == "ig_ql_d10")].gap_pct.mean()), gd01, 0.006)
    chk(f"GAP ig_ql d=0.3 {cls}", float(resb[(resb.instance == cls) & (resb.rep < 50)].gap_pct.mean()), gd03, 0.006)
sv = pd.read_csv(ROOT / "results" / "seed_variance_entrada109.csv")
m = sv.groupby(["instance", "method", "rep"]).makespan.mean().unstack("method").reset_index()
chk("seed-media d ig_ql vs ig_random (-0,045)", rel(m.ig_ql, m.ig_random).mean(), -0.0448, 0.0006)

# ---------- novas distribuicoes
nd = pd.read_csv(ROOT / "results" / "new_distributions_entrada108.csv")
ex = pd.read_csv(ROOT / "results" / "new_distributions_entrada108_exact.csv")
key = ["dist", "n", "m", "rep"]
W = nd.pivot(index=key, columns="method", values="makespan").reset_index().merge(ex[key + ["lb_lp"]], on=key)
chk("|novas instancias|", len(W), 900, 0)
nar = W[W.dist == "u50_100"]
chk("multifit gap u50_100 (3,68)", 100 * ((nar.multifit - nar.lb_lp) / nar.lb_lp).mean(), 3.679, 0.01)
chk("lpt gap u50_100 (2,48)", 100 * ((nar.lpt - nar.lb_lp) / nar.lb_lp).mean(), 2.483, 0.01)
chk("multifit opt% u50_100 (0)", 100 * (nar.multifit == nar.lb_lp).mean(), 0.0, 0.0)
chk("ig_ql_seed melhora a construtiva em 168/900", int((W.ig_ql_seed < W[["lpt", "multifit", "ldm"]].min(axis=1)).sum()), 168, 0)
chk("H1 (a) d (-0,016)", rel(W.ig_ql, W.ig_random).mean(), -0.016, 0.0006)
chk("H1 (b) d (-0,012)", rel(W.ig_ql_seed, W.ig_random_seed).mean(), -0.0122, 0.0006)
eq = pd.read_csv(ROOT / "results" / "entrada110_equal_cpu_primary.csv")
chk("equal CPU E1 d (+0,086)", float(eq[eq.id == "E1"].mean_d_pct.iloc[0]), 0.086, 0.001)
chk("equal CPU E3 d (-0,70)", float(eq[eq.id == "E3"].mean_d_pct.iloc[0]), -0.703, 0.002)

out = pd.DataFrame(rows, columns=["claim", "recalculado", "no_texto", "tol", "ok"])
pd.set_option("display.width", 200, "display.max_rows", 200, "display.float_format", lambda x: f"{x:.4f}")
print(out.to_string(index=False))
print(f"\n{int(out.ok.sum())}/{len(out)} OK; divergentes: {list(out[~out.ok].claim)}")
(ROOT / "results" / "entrada113_final_claims_audit.txt").write_text(out.to_string(index=False) + f"\n\n{int(out.ok.sum())}/{len(out)} OK\n", encoding="utf-8")
