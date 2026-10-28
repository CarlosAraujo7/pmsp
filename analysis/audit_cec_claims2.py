"""
analysis/audit_cec_claims2.py (Entrada 205, Ciclo 18)
=======================================================
Segunda parte da auditoria numerica do artigo do CEC (cec/cec2027_paper.tex), para o texto novo do Ciclo 18: contagens de instancias, pre-teste VNR (calibracao e familia F), seletor offline e controles, Q-Learning ajustado, flowshop (incluindo seu seletor offline) e a contagem
de previsoes. Cada numero da prosa e' recalculado dos CSV/TXT de resultados e conferido (a) com o literal do texto e (b) com a presenca do literal no texto. A primeira parte (analysis/audit_cec_claims.py) cobre o resto do artigo.
Uso: python -m analysis.audit_cec_claims2   (sai com codigo 1 se algo falhar)
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_benchmarks" / "pcmax_symmetry2025"))
TEX = re.sub(r"\s+", " ", (ROOT / "cec" / "cec2027_paper.tex").read_text(encoding="utf-8"))
rows = []


def chk(label, value, expected, tol, snippet=None):
    ok = abs(float(value) - float(expected)) <= tol
    if snippet is not None:
        ok = ok and (snippet in TEX)
    rows.append((label, float(value), float(expected), tol, ok))


def present(label, snippet):
    rows.append((label, float("nan"), float("nan"), 0, snippet in TEX))


rd = lambda f: pd.read_csv(R / f)

# ---------------------------------------------------------------- conjuntos de instancias e a unidade de makespan
from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle17_replication_entrada190 as RH  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402

chk("instancias de ajuste publicadas (525)", len(P.load_set(P.M_CAL)), 525, 0, "$525$ tuning instances")
chk("instancias de descoberta (2.550)", len(P.load_set(P.M_TEST)), 2550, 0, "$2{,}550$ \\emph{discovery}")
chk("instancias held-out (3.950)", len(RH.load_set(RH.M_HOLD)) if hasattr(RH, "load_set") else len(P.load_set(RH.M_HOLD)), 3950, 0, "$3{,}950$ \\emph{held-out}")
chk("instancias de treino do seletor (5.720)", len(S.train_set()), 5720, 0, "$5{,}720$ B, C, E instances")
man = list(csv.DictReader(open(S.M_TRAIN, newline="")))
chk("instancias da familia F (780)", sum(r["family"] == "F" for r in man), 780, 0, "$780$ of family F")
chk("treino bruto B, C, E (6.000)", sum(r["family"] in "BCE" for r in man), 6000, 0, "$6{,}000$ training ones")
chk("excluidas do treino (280)", 6000 - len(S.train_set()), 280, 0, "$280$ of the")
n_raw = sum(len(list(csv.DictReader(open(f, newline="")))) for f in (P.M_CAL, P.M_TEST, RH.M_HOLD))
n_used = len(P.load_set(P.M_CAL)) + len(P.load_set(P.M_TEST)) + len(P.load_set(RH.M_HOLD))
chk("instancias dos tres primeiros conjuntos (7.403)", n_raw, 7403, 0, "$378$ of the $7{,}403$ of")
chk("excluidas dos tres primeiros conjuntos (378)", n_raw - n_used, 378, 0)
d = rd("agents_S5_entrada180.csv")
ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
unit = 100 / ms.rand
chk("1 unidade de makespan, minimo (0,12%)", unit.min(), 0.12, 0.005, "$0.12$--$0.78\\%$")
chk("1 unidade de makespan, maximo (0,78%)", unit.max(), 0.78, 0.005)
chk("1 unidade de makespan, mediana (0,38%)", unit.median(), 0.38, 0.005, "median $0.38\\%$")
ties = [float((ms[k] == ms.rand).mean()) for k in ("ql", "ucb", "pg")]
ties_all = [100 * float(((ms[k] - ms.rand) == 0).mean()) for k in ("ql", "ucb", "pg")]
chk("empates entre pares, minimo (31%)", min(ties_all), 31, 0.6, "$31$--$46\\%$ of the instance pairs tie")
chk("empates entre pares, maximo (46%)", max(ties_all), 46, 0.6)
chk("margem de 0,05 e' ~1/8 da unidade mediana", 0.05 / unit.median(), 0.125, 0.01, "about an eighth of one makespan unit")

# ---------------------------------------------------------------- pre-teste VNR
tau = json.loads((R / "pretest_tau_entrada198.json").read_text())
chk("celulas de calibracao (42)", tau["n_calibration_cells"], 42, 0, None)
from analysis.pretest_vnr_entrada198 import evaluate  # noqa: E402

F = rd("entrada199_pretest_cells.csv")
ev = {sp: evaluate(F[(F.space == sp)].vnr, F[(F.space == sp)].gq, tau["tau"]) for sp in ("S5", "S15")}
chk("familia F: celulas por espaco (21)", ev["S5"]["n"], 21, 0, None)
chk("familia F: 26 classes", F.cell.nunique(), 26, 0, "the $26$ classes")
chk("familia F S5 AUC (0,75)", ev["S5"]["auc"], 0.75, 0.005, "AUC $0.75$ and $0.73$ for ``Q-Learning gains''")
chk("familia F S15 AUC (0,73)", ev["S15"]["auc"], 0.73, 0.005)
chk("familia F S5 acuracia (0,71)", ev["S5"]["acc"], 0.71, 0.005, None)
chk("familia F S15 acuracia (0,67)", ev["S15"]["acc"], 0.67, 0.005)
chk("familia F S5 Spearman (0,65)", ev["S5"]["spearman"], 0.65, 0.006, None)
chk("familia F S15 Spearman (0,348)", ev["S15"]["spearman"], 0.348, 0.0005)
FA = rd("entrada199_famF_Aiter.csv").set_index("id")
FB = rd("entrada199_famF_Bit.csv")

# ---------------------------------------------------------------- seletor offline
Cv = rd("entrada200_selector_Cval.csv")
g = -Cv.mean_d_pct
chk("seletor vs validada, minimo (0,024)", g.min(), 0.024, 0.0006, "$0.024$--$0.044$ points")
chk("seletor vs validada, maximo (0,044)", g.max(), 0.044, 0.0006)
chk("seletor vs validada: p_holm < 1e-38", float(Cv.p_holm.max() < 1e-38), 1, 0, None)
chk("seletor vs validada: dentro da margem exceto S15 held-out", float(list(Cv.equivalente_0_05 if "equivalente_0_05" in Cv else Cv["equivalente_0.05"]).count(False)), 1, 0, "inside the margin, hence parity, except in the larger space on held-out instances")
CE = rd("entrada200_controls_CvE.csv")
BF = rd("entrada200_selector_by_family.csv")
al = BF[BF.fam == "all"]
Cq = rd("entrada200_selector_Cql.csv")
chk("seletor vs Q-Learning, minimo (0,08)", -Cq.mean_d_pct.max(), 0.08, 0.0025, "$0.08$--$0.095$ over Q-Learning")
chk("seletor vs Q-Learning, maximo (0,095)", -Cq.mean_d_pct.min(), 0.095, 0.0006)
Ca = rd("entrada200_selector_Calns.csv")
chk("seletor vs ALNS, minimo (0,04)", -Ca.mean_d_pct.max(), 0.04, 0.003, "and $0.04$--$0.14$ over the ALNS roulette")
chk("seletor vs ALNS, maximo (0,14)", -Ca.mean_d_pct.min(), 0.14, 0.0006)
chk("captura do seletor, minimo (0,79)", al.cap_sel.min(), 0.79, 0.005, "captures $0.79$--$0.84$")
chk("captura do seletor, maximo (0,84)", al.cap_sel.max(), 0.84, 0.005)
chk("captura da validada, minimo (0,58)", al.cap_val.min(), 0.58, 0.005, "against $0.58$--$0.66$")
chk("captura da validada, maximo (0,66)", al.cap_val.max(), 0.66, 0.005)
Sv = rd("entrada200_controls_Svb.csv")
chk("estatica nas mesmas 5.720 vs validada, minimo (0,001)", -Sv.mean_d_pct.max(), 0.001, 0.0006, "only $0.001$--$0.019$ points better")
chk("estatica nas mesmas 5.720 vs validada, maximo (0,019)", -Sv.mean_d_pct.min(), 0.019, 0.0006)
Cc = rd("entrada200_controls_Ccl.csv").set_index("id")
chk("tabela por classe vs seletor, held-out S5 (0,004)", -Cc.loc["Ccl-S5-heldout", "mean_d_pct"], 0.004, 0.0006, "($0.004$ and $0.006$ points, not significant)")
chk("tabela por classe vs seletor, held-out S15 (0,006)", -Cc.loc["Ccl-S15-heldout", "mean_d_pct"], 0.006, 0.0006)
chk("tabela por classe: held-out nao confirmadas", float((~Cc.loc[["Ccl-S5-heldout", "Ccl-S15-heldout"], "confirmada"]).all()), 1, 0)
chk("seletor vs tabela, descoberta S5 (0,017)", -Cc.loc["Ccl-S5-discovery", "mean_d_pct"], 0.017, 0.0006, "$0.014$ and $0.017$")
chk("seletor vs tabela, descoberta S15 (0,014)", -Cc.loc["Ccl-S15-discovery", "mean_d_pct"], 0.014, 0.0006)
import json as _j  # noqa: E402

tabs = _j.loads((R / "selctl_tables_entrada200.json").read_text())["S5"]["valclass"]
cov = {}
for nm, mpath in (("discovery", P.M_TEST), ("heldout", RH.M_HOLD)):
    from experiment import run_cycle18_selector_controls_entrada200 as CT  # noqa: E402

    cof = CT.manifest_class(mpath)
    recs = P.load_set(mpath)
    cov[nm] = 100 * np.mean(["|".join(map(str, cof[r["member"]])) in tabs for r in recs])
chk("cobertura de classes no held-out (100%)", cov["heldout"], 100, 0.5, "where every class was seen in training")
chk("cobertura de classes na descoberta (59%)", cov["discovery"], 59, 0.5, "where $59\\%$ of them were")

# ---------------------------------------------------------------- Q-Learning ajustado e com recompensa de plato
Aq = rd("entrada203_qvariants_Aqt.csv")
chk("qt sobre o padrao, minimo (0,009)", -Aq.mean_d_pct.max(), 0.009, 0.0006, "adds $0.009$--$0.014$ points")
chk("qt sobre o padrao, maximo (0,014)", -Aq.mean_d_pct.min(), 0.014, 0.0006)
Ap = rd("entrada203_qvariants_Aqp.csv").set_index("id")
chk("qtp sobre o padrao em S5 (0,015)", -Ap.loc["Aqp-S5-heldout", "mean_d_pct"], 0.015, 0.0012, "adds $0.015$ with five sizes")
chk("qtp sobre o padrao em S15, minimo (0,007)", Ap.loc[["Aqp-S15-discovery", "Aqp-S15-heldout"], "mean_d_pct"].min(), 0.007, 0.0006, "$0.007$--$0.009$ with fifteen")
chk("qtp sobre o padrao em S15, maximo (0,009)", Ap.loc[["Aqp-S15-discovery", "Aqp-S15-heldout"], "mean_d_pct"].max(), 0.009, 0.0006)
chk("validada vence qt e qtp nos 8 pares", float(rd("entrada203_qvariants_Vqt.csv").confirmada.all() and rd("entrada203_qvariants_Vqp.csv").confirmada.all()), 1, 0, "the validated action beats both in all eight comparisons")
sq = pd.concat([rd("entrada203_qvariants_Sqt.csv"), rd("entrada203_qvariants_Sqp.csv")])
chk("seletor vs qt e qtp, minimo (0,06)", -sq.mean_d_pct.max(), 0.06, 0.0035, "and the selector beats them by $0.06$--$0.10$")
chk("seletor vs qt e qtp, maximo (0,10)", -sq.mean_d_pct.min(), 0.10, 0.003)

# ---------------------------------------------------------------- flowshop
K = json.loads((R / "pfsp_cpu_ratio_entrada197.json").read_text())
kq = [K[sp][c]["ql"] for sp in K for c in K[sp]]
chk("K do Q-Learning no flowshop, minimo (0,7)", min(kq), 0.7, 0.025, None)
chk("K do Q-Learning no flowshop, maximo (0,9)", max(kq), 0.9, 0.03)
Cells = rd("entrada197_pfsp_cells.csv")
Fa = {s: rd(f"entrada197_pfsp_{s}_Aiter.csv").set_index("id") for s in ("syn", "ta")}
chk("flowshop sintetico: Q-Learning F5 (-0,050)", -Fa["syn"].loc["Aiter-syn-F5-ql", "mean_d_pct"], -0.050, 0.0006, "Q-Learning loses $0.05$--$0.08$ points")
chk("flowshop sintetico: Q-Learning F15 (-0,061)", -Fa["syn"].loc["Aiter-syn-F15-ql", "mean_d_pct"], -0.061, 0.0006)
chk("flowshop Taillard: Q-Learning F5 (-0,059)", -Fa["ta"].loc["Aiter-ta-F5-ql", "mean_d_pct"], -0.059, 0.0006, None)
chk("flowshop Taillard: Q-Learning F15 (-0,080)", -Fa["ta"].loc["Aiter-ta-F15-ql", "mean_d_pct"], -0.080, 0.0006)
chk("flowshop: Q-Learning perde com Holm (4)", float(sum(Fa[s].loc[f"Aiter-{s}-{sp}-ql", "mean_d_pct"] > 0 and Fa[s].loc[f"Aiter-{s}-{sp}-ql", "p_holm"] < 0.05 for s in Fa for sp in ("F5", "F15"))), 4, 0)
Fc = rd("entrada197_pfsp_syn_Acpu.csv").set_index("id")
chk("flowshop: Q-Learning a CPU igual F5 (-0,036)", -Fc.loc["Acpu-syn-F5-ql", "mean_d_pct"], -0.036, 0.0006, "and $0.03$--$0.05$ at equal CPU time")
chk("flowshop: Q-Learning a CPU igual F15 (-0,047)", -Fc.loc["Acpu-syn-F15-ql", "mean_d_pct"], -0.047, 0.0006)
oth = Fa["syn"].loc[[i for i in Fa["syn"].index if not i.endswith("-ql")], "mean_d_pct"]
chk("flowshop: outros aprendizes dentro de 0,013", oth.abs().max(), 0.013, 0.0006, None)
Fb = rd("entrada197_pfsp_syn_Bit.csv")
chk("flowshop: validada vence os dez pares a iteracoes iguais", float(Fb.confirmada.all()), 1, 0, "in all ten learner--space pairs at equal iterations")
chk("flowshop: validada vs aprendizes, minimo (0,020)", -Fb.mean_d_pct.max(), 0.020, 0.0006, "(by $0.020$--$0.104$)")
chk("flowshop: validada vs aprendizes, maximo (0,104)", -Fb.mean_d_pct.min(), 0.104, 0.0006)
Fq = rd("entrada197_pfsp_syn_Beq.csv").set_index("id")
chk("flowshop: validada vs Q-Learning a CPU igual F5 (0,019)", -Fq.loc["Beq-syn-F5-ql", "mean_d_pct"], 0.019, 0.0006, None)
chk("flowshop: validada vs Q-Learning a CPU igual F15 (0,064)", -Fq.loc["Beq-syn-F15-ql", "mean_d_pct"], 0.064, 0.0006)
for sp, rho_exp, p_exp in (("F5", -0.93, 0.001), ("F15", -0.78, 0.013)):
    c = Cells[(Cells.set == "syn") & (Cells.space == sp)]
    r = sstats.spearmanr(c.folga, c.gq)
    chk(f"flowshop {sp}: Spearman folga x ganho do Q-Learning", r.statistic, rho_exp, 0.006, "$-0.93$ and $-0.78$" if sp == "F5" else None)
    chk(f"flowshop {sp}: p do Spearman", r.pvalue, p_exp, 0.0006 if sp == "F15" else 0.001)
big = Cells[(Cells.set == "syn") & Cells.cell.isin(["50x10", "50x20", "100x10", "100x20"])]
Wf = {}
from analysis.report_entrada197_pfsp import load_test  # noqa: E402
from analysis.report_entrada180_agents import rel  # noqa: E402

heads = []
for sp in ("F5", "F15"):
    w = load_test(sp)
    w = w[(w.set == "syn") & w.klass.isin(["50x10", "50x20", "100x10", "100x20"])]
    heads.append(float(-rel(w.oracle, w.rand).mean()))
nz = Cells[(Cells.set == "syn")].groupby("cell").noise.mean()
chk("flowshop: ruido nas classes m = 20 e n >= 50, minimo (0,2)", Cells[(Cells.set == "syn") & Cells.cell.isin(["50x20", "100x20"])].noise.min(), 0.2, 0.01, "$0.2$--$0.3\\%$")
chk("flowshop: ruido nas classes m = 20 e n >= 50, maximo (0,3)", Cells[(Cells.set == "syn") & Cells.cell.isin(["50x20", "100x20"])].noise.max(), 0.3, 0.02)
nab = int(((Cells[(Cells.set == "syn")].vnr.notna()) & (Cells[(Cells.set == "syn")].vnr < tau["tau"])).sum())
ev_cells = Cells[(Cells.set == "syn")].vnr.notna().sum()

# ---------------------------------------------------------------- seletor offline do flowshop (Entrada 205) -- preenchido quando o relatorio existir
Cr = rd("entrada205_pfsp_selector_Crand.csv")
chk("seletor do flowshop vs aleatorio, minimo (0,034)", -Cr.mean_d_pct.max(), 0.034, 0.0006, None)
chk("seletor do flowshop vs aleatorio, maximo (0,050)", -Cr.mean_d_pct.min(), 0.050, 0.0006)
chk("seletor do flowshop vs aleatorio: confirmado nos quatro", float(Cr.confirmada.all()), 1, 0)
Cq5 = rd("entrada205_pfsp_selector_Cql.csv")
chk("seletor do flowshop vs Q-Learning, minimo (0,086)", -Cq5.mean_d_pct.max(), 0.086, 0.0006, None)
chk("seletor do flowshop vs Q-Learning, maximo (0,127)", -Cq5.mean_d_pct.min(), 0.127, 0.0006)
Cb5 = rd("entrada205_pfsp_selector_Cvb.csv")
chk("seletor do flowshop vs estatica nas mesmas 540: |d| maximo (0,007)", Cb5.mean_d_pct.abs().max(), 0.007, 0.0006, None)
Cv5 = rd("entrada205_pfsp_selector_Cval.csv")
chk("seletor do flowshop vs validada: nenhum confirmado", float((~Cv5.confirmada).all()), 1, 0, None)
# ---------------------------------------------------------------- validada a CPU igual (Entrada 195/196) e pequenos fatos citados no texto
V = {s: rd(f"entrada195_valeq_{s}.csv").set_index("id") for s in ("syn", "discovery", "heldout")}
chk("validada a CPU igual vence os 54 pares", float(sum(int(v.confirmada.sum()) for v in V.values())), 54, 0, "in all $54$ learner--space--set pairs")
chk("validada a CPU igual vs Q-Learning sintetico, minimo (0,071)", min(-V["syn"].loc["Veq-syn-S5-ql", "mean_d_pct"], -V["syn"].loc["Veq-syn-S15-ql", "mean_d_pct"]), 0.071, 0.0006, None)
chk("validada a CPU igual vs Q-Learning sintetico, maximo (0,097)", max(-V["syn"].loc["Veq-syn-S5-ql", "mean_d_pct"], -V["syn"].loc["Veq-syn-S15-ql", "mean_d_pct"]), 0.097, 0.0006)
chk("validada a CPU igual vs Q-Learning descoberta S5 (0,051)", -V["discovery"].loc["Veq-discovery-S5-ql", "mean_d_pct"], 0.051, 0.0006, None)
chk("validada a CPU igual vs Q-Learning descoberta S15 (0,058)", -V["discovery"].loc["Veq-discovery-S15-ql", "mean_d_pct"], 0.058, 0.0006)
chk("validada a CPU igual vs Q-Learning held-out S5 (0,058)", -V["heldout"].loc["Veq-heldout-S5-ql", "mean_d_pct"], 0.058, 0.0006, None)
chk("validada a CPU igual vs Q-Learning held-out S15 (0,059)", -V["heldout"].loc["Veq-heldout-S15-ql", "mean_d_pct"], 0.059, 0.0006)
chk("validada a CPU igual vs roleta S5 descoberta (0,023)", -V["discovery"].loc["Veq-discovery-S5-alns", "mean_d_pct"], 0.023, 0.0006, "($0.023$ and $0.024$)")
chk("validada a CPU igual vs roleta S5 held-out (0,024)", -V["heldout"].loc["Veq-heldout-S5-alns", "mean_d_pct"], 0.024, 0.0006)
hmax = []
for sp_ in ("S3", "S5", "S15", "S30"):
    dd = rd(f"agents_{sp_}_entrada180.csv")
    mm = dd[~dd.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
    oo = dd[dd.method.str.startswith("arm")].groupby(["cls", "rep"]).makespan.min().rename("oracle").reset_index()
    ww = mm.merge(oo, on=["cls", "rep"])
    hmax.append(float((-100 * (ww.oracle - ww.rand) / ww.rand).groupby(ww.cls).mean().max()))
tcell = Cells[(Cells.set == "ta") & Cells.vnr.notna() & (Cells.gq > 0)]
from experiment import run_cycle18_pfsp_selector_entrada205 as FS  # noqa: E402

chk("treino do seletor do flowshop (540 instancias)", len(FS.train_items()), 540, 0, None)

# ---------------------------------------------------------------- relatorio
bad = [r for r in rows if not r[4]]
for lab, v, e, tol, ok in rows:
    print(f"{'OK ' if ok else 'ERRO'} {lab}" + ("" if np.isnan(v) else f"  recalculado {v:.5g} vs texto {e:.5g} (tol {tol})"))
print(f"\n{len(rows) - len(bad)}/{len(rows)} verificacoes OK")
sys.exit(1 if bad else 0)
