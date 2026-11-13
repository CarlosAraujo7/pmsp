"""
analysis/audit_journal_claims.py (Entrada 257/258)
Auditoria numerica das partes NOVAS do manuscrito de periodico (journal/): recalcula dos CSV/TXT de resultados cada numero citado na prosa e confere (a) o valor com o literal
(tolerancia de arredondamento) e (b) que o literal esta' de fato no arquivo .tex indicado. As partes herdadas do texto do CEC (sec_results, sec_method, sec_intro) tem os numeros
auditados por analysis/audit_cec_claims.py; aqui se conferem: o apendice do estimador (app_estimator), a secao de familias publicadas e do benchmark padrao (sec_external,
sec_fanjul), os contadores do registro (main.tex, app_batteries, sec_discussion).
Uso: python -m analysis.audit_journal_claims   (sai com codigo 1 se algo falhar)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
J = ROOT / "journal"
rows = []


def tex(name):
    return re.sub(r"\s+", " ", (J / name).read_text(encoding="utf-8"))


def chk(label, value, expected, tol, file=None, snippet=None):
    """Valor recalculado ~ literal; `snippet` (opcional) deve aparecer no arquivo `file` (journal/)."""
    ok = abs(float(value) - float(expected)) <= tol
    if snippet is not None:
        ok = ok and (snippet in tex(file))
    rows.append((label, float(value), float(expected), tol, ok))


def present(label, file, snippet):
    rows.append((label, float("nan"), float("nan"), 0, snippet in tex(file)))


# ---------------------------------------------------------------- apendice do estimador (Entrada 257)
S = pd.read_csv(R / "entrada257_sim_crossfit.csv")
E = pd.read_csv(R / "entrada257_empirical_decay.csv").set_index("space")
chk("18 configuracoes da simulacao", len(S), 18, 0, "app_estimator.tex", "$18$ configurations")
chk("E[H_um] >= H em todas as linhas (tol. 0,01)", float((S.H_um >= S.H_true - 0.01).all()), 1, 0)
chk("E[H_cv] <= H em todas as linhas (tol. 0,01)", float((S.H_cv <= S.H_true + 0.01).all()), 1, 0)
chk("|H_cv - (H - arrependimento)| < 0,025 em todas as linhas", float(((S.H_cv - (S.H_true - S.regret_cv)).abs() < 0.025).all()), 1, 0, "app_estimator.tex", "less than $0.025$ in every row")
z = S[(S.A == 5) & (S.tau == 0)].sort_values("sigma")
for v, lit in zip(z.H_um, ("0.58", "1.16", "2.32")):
    chk(f"H_um com tau = 0, A = 5 ({lit})", v, float(lit), 0.005, "app_estimator.tex", f"${lit}$")
chk("vies medio de H_um (+1,31)", (S.H_um - S.H_true).mean(), 1.31, 0.005, "app_estimator.tex", "$+1.31$ points")
chk("vies medio de H_cv (-0,19)", (S.H_cv - S.H_true).mean(), -0.19, 0.005, "app_estimator.tex", "$-0.19$ points")
chk("H_cv com tau = 0: |.| < 0,01", float(S[S.tau == 0].H_cv.abs().max() < 0.01), 1, 0, "app_estimator.tex", "within $0.01$ of zero")
r5 = S[(S.A == 5) & (S.tau == 1.0)].sort_values("sigma").regret_cv.tolist()
for v, lit in zip(r5, ("0.07", "0.21", "0.48")):
    chk(f"arrependimento A = 5, tau = 1 ({lit})", v, float(lit), 0.005, "app_estimator.tex", f"${lit}$")
x = S[(S.A == 15) & (S.tau == 0) & (S.sigma == 1.0)].H_mesmas_k.iloc[0]
chk("melhor de 4 nas mesmas, A = 15, sigma = 1, tau = 0 (0,87)", x, 0.87, 0.005, "app_estimator.tex", "reports $0.87$ points")
for sp, (a, b, c, d) in {"S5": (1.50, 1.32, 1.19, 0.43), "S15": (2.81, 2.63, 2.46, 1.60)}.items():
    r = E.loc[sp]
    chk(f"oraculo de uma semente {sp} ({a})", r.um_semente, a, 0.005)
    chk(f"melhor de 2 nas mesmas {sp} ({b})", r.melhor_de_2_mesmas, b, 0.005)
    chk(f"melhor de 4 nas mesmas {sp} ({c})", r.melhor_de_4_mesmas, c, 0.005)
    chk(f"validacao cruzada {sp} ({d})", r.cv_2mais2, d, 0.005)
for lit in ("$1.50$ and $2.81$ points", "$1.32$ and $2.63$ with two", "$1.19$ and $2.46$ with four", "($0.43$ and $1.60$)"):
    present("texto do apendice: " + lit, "app_estimator.tex", lit)

# ---------------------------------------------------------------- familias publicadas (Entradas 254 e 255)
c254 = pd.read_csv(R / "entrada254_ur_external_cells.csv")
c255 = pd.read_csv(R / "entrada255_ext_competitors_cells.csv")
t254 = pd.read_csv(R / "entrada254_ur_external_tests.csv").set_index("id")
t255 = pd.read_csv(R / "entrada255_ext_competitors_tests.csv").set_index("id")
Cc = c254.merge(c255[["fam", "space", "g_cfg", "g_q", "g_iq", "g_fr"]], on=["fam", "space"])
chk("folga cf minima nas familias publicadas (0,52)", c254.h_cf.min(), 0.52, 0.005, "sec_external.tex", "$0.52$ to $2.31$")
chk("folga cf maxima nas familias publicadas (2,31)", c254.h_cf.max(), 2.31, 0.005)
LE = ("ql", "ucb", "pg", "alns", "qo")
B254 = t254[t254.index.str.startswith("B-")]
chk("30 pares aprendiz-celula na familia B (254)", len(B254), 30, 0)
chk("a validada bate o aprendiz em todos os 30 pares (confirmada)", int(B254.confirmada.sum()), 30, 0)
chk("menor vantagem da validada (0,42)", B254.mean_d_pct.abs().min(), 0.42, 0.005, "sec_external.tex", "$0.42$--$2.64$ points")
chk("maior vantagem da validada (2,64)", B254.mean_d_pct.abs().max(), 2.64, 0.005)
Bp254 = t254[t254.index.str.startswith("B'-")]
chk("nenhum aprendiz bate a validada alem da margem (0 de 30)", int((Bp254.confirmada & ~Bp254["equivalente_0.05"]).sum()), 0, 0)
Aql = t254[t254.index.str.endswith("-ql") & t254.index.str.startswith("A-")]
chk("o Q-Learning ganha alem da margem em 4 das 6 celulas", int((Aql.confirmada & ~Aql["equivalente_0.05"]).sum()), 4, 0, "sec_external.tex", "in four of the six cells")
cm = Cc[Cc.fam == "cm"].set_index("space")
chk("Q-Learning cm S5 (0,67)", cm.loc["S5", "g_ql"], 0.67, 0.005, "sec_external.tex", "$0.67$ and $0.83$ points")
chk("Q-Learning cm S15 (0,83)", cm.loc["S15", "g_ql"], 0.83, 0.005)
chk("fracao do ganho da validada cm S5 (53%)", 100 * cm.loc["S5", "g_ql"] / cm.loc["S5", "g_val"], 53, 0.5, "sec_external.tex", "$53\\%$ and $65\\%$")
chk("fracao do ganho da validada cm S15 (65%)", 100 * cm.loc["S15", "g_ql"] / cm.loc["S15", "g_val"], 65, 0.5)
chk("Q-Learning perde 0,45 em unc S5", -Cc[(Cc.fam == "unc") & (Cc.space == "S5")].g_ql.iloc[0], 0.45, 0.005, "sec_external.tex", "loses $0.45$ points")
chk("Q-Learning em cj S5 (-0,01)", Cc[(Cc.fam == "cj") & (Cc.space == "S5")].g_ql.iloc[0], -0.01, 0.005, "sec_external.tex", "($-0.01$)")
chk("razao maxima melhor aprendiz / validada (65%)", max(100 * Cc[f"g_{k}"].div(Cc.g_val).max() for k in LE), 65, 0.6, "sec_external.tex", "up to $65\\%$")
Bp255 = t255[t255.index.str.startswith("B'-")]
cfg = Bp255[Bp255.index.str.endswith("-cfg")]
chk("cfg do irace: maior diferenca absoluta ao ganho da validada (0,07)", (Cc.g_val - Cc.g_cfg).abs().max(), 0.07, 0.01, "sec_external.tex", "at most $0.07$ points")
chk("cfg nao bate a validada alem da margem (0 de 6)", int((cfg.confirmada & ~cfg["equivalente_0.05"]).sum()), 0, 0)
Aq = t255[t255.index.str.startswith("A-") & t255.index.str.endswith("-q")]
chk("q do irace ganha alem da margem em 5 das 6 celulas", int((Aq.confirmada & ~Aq["equivalente_0.05"]).sum()), 5, 0, "sec_external.tex", "in five of the six cells")
chk("q do irace: maior ganho (0,82)", Cc.g_q.max(), 0.82, 0.005, "sec_external.tex", "up to $0.82$ points")
chk("desenhos publicados: maior ganho (0,54)", max(Cc.g_iq.max(), Cc.g_fr.max()), 0.54, 0.005, "sec_external.tex", "up to $0.54$ points")
B255 = t255[t255.index.str.startswith("B-") & ~t255.index.str.endswith("-cfg")]
chk("18 comparacoes da validada com q, iq, fr", len(B255), 18, 0, "sec_external.tex", "all $18$ comparisons")
chk("a validada vence as 18 (confirmadas)", int(B255.confirmada.sum()), 18, 0)
chk("menor vantagem nas 18 (0,40)", B255.mean_d_pct.abs().min(), 0.40, 0.005, "sec_external.tex", "$0.40$--$2.19$ points")
chk("maior vantagem nas 18 (2,19)", B255.mean_d_pct.abs().max(), 2.19, 0.005)

# ---------------------------------------------------------------- benchmark padrao de Fanjul-Peyro e Ruiz (Entrada 256)
FC = pd.read_csv(R / "entrada256_fanjul_cells.csv")
FT = pd.read_csv(R / "entrada256_fanjul_tests.csv").set_index("id")
FL = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr"]
FA = FT[FT.index.str.startswith("A-")]
FB = FT[FT.index.str.startswith("B-")]
fb = FA.confirmada & ~FA["equivalente_0.05"]
chk("147 pares aprendiz-celula", len(FA), 147, 0, "sec_fanjul.tex", "$147$ learner--cell pairs")
chk("a validada bate os 147 pares (B confirmada)", int(FB.confirmada.sum()), 147, 0)
chk("aprendizes que ganham alem da margem do aleatorio (39)", int(fb.sum()), 39, 0, "sec_fanjul.tex", "$39$ of the $147$")
cnt = {k: int(fb[FA.index.str.endswith(f"-{k}")].sum()) for k in FL}
for k, lit, snip in (("ql", 3, "Q-Learning in $3$ of its $21$ cells"), ("ucb", 1, "UCB1 in one"), ("pg", 0, "the policy gradient in none"), ("alns", 6, "the ALNS roulette in six"), ("qo", 10, "the optimistic Q-Learning in ten"), ("iq", 7, "the episodic design in seven"), ("fr", 12, "the FRRMAB-style design in twelve")):
    chk(f"celulas alem da margem: {k} ({lit})", cnt[k], lit, 0, "sec_fanjul.tex", snip)
qg = FC[FC.g_ql > 0].sort_values("g_ql")
qids = [i for i in fb[fb].index if i.endswith("-ql")]
qgain = sorted(float(FC[(FC.type == i.split("-")[1]) & (FC.space == i.split("-")[2])].g_ql.iloc[0]) for i in qids)
chk("ganhos do Q-Learning alem da margem (0,07; 0,15; 0,39)", float(np.abs(np.array(qgain) - np.array([0.07, 0.15, 0.39])).max()), 0, 0.005, "sec_fanjul.tex", "$0.07$, $0.15$ and $0.39$")
chk("ganho minimo da validada (0,14)", FC.g_val.min(), 0.14, 0.005, "sec_fanjul.tex", "from $0.14$ to $5.08$")
chk("ganho maximo da validada (5,08)", FC.g_val.max(), 5.08, 0.005)
chk("Q-Learning perde ate 1,19", -FC.g_ql.min(), 1.19, 0.005, "sec_fanjul.tex", "up to $1.19$ points")
chk("Spearman celulas (-0,46)", float(__import__("scipy.stats", fromlist=["x"]).spearmanr(FC.g_val, FC.g_ql).statistic), -0.46, 0.005, "sec_fanjul.tex", "$-0.46$")
for sp, lit in (("B5", -0.75), ("B15", 0.14), ("B30", -0.68)):
    x = FC[FC.space == sp]
    chk(f"Spearman em {sp} ({lit})", float(__import__("scipy.stats", fromlist=["x"]).spearmanr(x.g_val, x.g_ql).statistic), lit, 0.006)
for k in FL:
    FC[f"r_{k}"] = FC[f"g_{k}"] / FC.g_val
rmax = max(FC[f"r_{k}"].max() for k in FL)
chk("maior fracao do ganho da validada recuperada por um aprendiz (0,68)", rmax, 0.68, 0.005, "sec_fanjul.tex", "is $0.68$")
rest = FC[~((FC.space == "B5") & (FC.type == "U1000a1100"))]
chk("nas demais celulas no maximo 0,46", max(rest[f"r_{k}"].max() for k in FL), 0.46, 0.006, "sec_fanjul.tex", "at most $0.46$")
H = (R / "entrada256_fanjul.txt").read_text(encoding="utf-8")
for sp, lit in (("B5", 8.32), ("B15", 8.32), ("B30", 4.89)):
    chk(f"lacuna da validada {sp} ({lit})", float(re.search(rf"{sp}: validada ([\d.]+);", H).group(1)), lit, 0.006, "sec_fanjul.tex", f"${lit}\\%$")
for sp, lit in (("B5", 9.35), ("B15", 10.21), ("B30", 7.42)):
    chk(f"lacuna do aleatorio {sp} ({lit})", float(re.search(rf"{sp}: validada [\d.]+; aleatorio ([\d.]+)", H).group(1)), lit, 0.006, "sec_fanjul.tex", f"${lit}\\%$")
chk("previsoes da Entrada 256 (5 de 7)", int(re.search(r"previsoes acertadas: (\d+) de", H).group(1)), 5, 0, "sec_fanjul.tex", "five of seven")
g30 = FC[FC.space == "B30"].set_index("type")
chk("Q-Learning B30 U100a200 (-0,94)", g30.loc["U100a200", "g_ql"], -0.94, 0.005, "sec_fanjul.tex", "$-0.94$ points on $U[100,200]$")
chk("Q-Learning B30 MaqCorre (-0,88)", g30.loc["MaqCorre", "g_ql"], -0.88, 0.005, "sec_fanjul.tex", "$-0.88$ on correlated machines")
chk("lacuna B30 minima por tipo U1000a1100 (0,5)", g30.gap_val.min(), 0.5, 0.05, "sec_fanjul.tex", "$0.5\\%$")
chk("lacuna B30 maxima por tipo U10a100 (9,9)", g30.gap_val.max(), 9.9, 0.05, "sec_fanjul.tex", "$9.9\\%$")

# ---------------------------------------------------------------- sensibilidade do Q-Learning padrao a ordem das acoes (Entradas 258, 260, 261)
import sys as _sys
_sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import rel as _rel

sh = {}
for sp, A in (("B5", 5), ("B15", 15), ("B30", 30)):
    D1 = pd.read_csv(R / f"mech_{sp}_entrada258.csv")
    q = D1[D1.method == "ql"]
    sh[sp] = float((q.used_0 / q[[f"used_{k}" for k in range(A)]].sum(axis=1)).mean())
chk("ql no indice 0: minimo (55%)", min(sh.values()), 0.55, 0.005, "sec_order.tex", "$55$--$73\\%$")
chk("ql no indice 0: maximo (73%)", max(sh.values()), 0.73, 0.005)
rec = {}
for sp in ("B5", "B15", "B30"):
    D2 = pd.read_csv(R / f"mech2_{sp}_entrada258.csv")
    W = D2.groupby(["type", "m", "n", "method"]).makespan.mean().unstack()
    gg = {k: float(-_rel(W[k], W["rand"]).mean()) for k in ("ql_v0", "val")}
    rec[sp] = (gg["ql_v0"], gg["ql_v0"] / gg["val"])
for sp, lit in (("B5", 0.73), ("B15", 1.30), ("B30", 2.02)):
    chk(f"ganho do ql_v0 em {sp} ({lit})", rec[sp][0], lit, 0.005)
chk("ql_v0 recupera 81% (B5)", 100 * rec["B5"][1], 81, 0.5, "sec_order.tex", "$81\\%$, $81\\%$ and $86\\%$")
chk("ql_v0 recupera 81% (B15)", 100 * rec["B15"][1], 81, 0.6)
chk("ql_v0 recupera 86% (B30)", 100 * rec["B30"][1], 86, 0.5)
H1 = (R / "entrada258_mechanism.txt").read_text(encoding="utf-8")
H2 = (R / "entrada258_mechanism2.txt").read_text(encoding="utf-8")
chk("etapa 1: 3 de 7 previsoes", int(re.search(r"previsoes acertadas: (\d+) de", H1).group(1)), 3, 0, "sec_order.tex", "three of the seven predictions")
chk("etapa 2: 5 de 5 previsoes", int(re.search(r"previsoes acertadas: (\d+) de", H2).group(1)), 5, 0, "sec_order.tex", "all five registered predictions")
Cz = pd.read_csv(R / "entrada260_qperm_cells_fanjul.csv")
Cf = pd.read_csv(R / "entrada260_qperm_cells_families.csv")
Cp = pd.read_csv(R / "entrada261_qperm_pcmax_cells.csv")
T260 = pd.read_csv(R / "entrada260_qperm_tests.csv").set_index("id")
chk("ql_perm Fanjul: media nas 21 celulas (-0,30)", Cz.g_qp.mean(), -0.30, 0.005, "sec_order.tex", "mean $-0.30$ points")
chk("ql padrao Fanjul: media nas 21 celulas (-0,25)", Cz.g_ql.mean(), -0.25, 0.005, "sec_order.tex", "$-0.25$ for the standard learner")
tA = T260[T260.index.str.startswith("A-") & ~T260.index.str.contains("-B")]
tAF = T260[T260.index.str.startswith("A-") & T260.index.str.contains("-B")]
chk("ql_perm Fanjul: 0 de 21 celulas alem da margem", int((tAF.confirmada & ~tAF["equivalente_0.05"]).sum()), 0, 0, "sec_order.tex", "none of the $21$ cells")
chk("ql_perm familias: ganho maximo negativo (-0,05)", Cf.g_qp.max(), -0.05, 0.005, "sec_order.tex", "from $-0.05$ to $-0.66$")
chk("ql_perm familias: ganho minimo (-0,66)", Cf.g_qp.min(), -0.66, 0.005)
chk("ql_perm familias: todas negativas (6)", int((Cf.g_qp < 0).sum()), 6, 0)
chk("a validada bate o ql_perm nas 6 celulas das familias", int(tA.shape[0] == 6) and int(T260[T260.index.str.startswith("B-") & ~T260.index.str.contains("-B")].confirmada.sum()), 6, 0)
chk("ql_perm P||Cmax: 0 de 24 celulas", int((Cp.g_qp > 0.05).sum()), 0, 0, "sec_order.tex", "none of the $24$ cells")
T261 = pd.read_csv(R / "entrada261_qperm_pcmax_tests.csv").set_index("id")
chk("a validada bate o ql_perm em 23 de 24 celulas do P||Cmax", int(T261[T261.index.str.startswith("B-")].confirmada.sum()), 23, 0, "sec_order.tex", "beats it in $23$")
H61 = (R / "entrada261_qperm_pcmax.txt").read_text(encoding="utf-8")
mm = re.search(r"S5 ([\d.]+); ([\d.]+) \| S15 ([\d.]+); ([\d.]+)", H61)
chk("razao de avaliacoes ql padrao no P||Cmax: 0,44", min(float(mm.group(2)), float(mm.group(4))), 0.44, 0.005, "sec_order.tex", "$0.44$--$0.46$")
chk("razao de avaliacoes ql padrao no P||Cmax: 0,46", max(float(mm.group(2)), float(mm.group(4))), 0.46, 0.005)
chk("razao de avaliacoes ql_perm: 0,99", min(float(mm.group(1)), float(mm.group(3))), 0.99, 0.005, "sec_order.tex", "$0.99$--$1.01$")
chk("razao de avaliacoes ql_perm: 1,01", max(float(mm.group(1)), float(mm.group(3))), 1.01, 0.005)
H60 = (R / "entrada260_qperm.txt").read_text(encoding="utf-8")
chk("Entrada 260: 4 de 6 previsoes", int(re.search(r"previsoes acertadas: (\d+) de", H60).group(1)), 4, 0, "sec_order.tex", "Four of the six registered predictions")
chk("Entrada 261: 4 de 4 previsoes", int(re.search(r"previsoes acertadas: (\d+) de", H61).group(1)), 4, 0, "sec_order.tex", "all four on $P||C_{\\max}$")

# ---------------------------------------------------------------- horizonte longo (Entrada 258, parte T = 10.000), Q-Learning invariante a ordem a T = 10.000 (Entrada 263) e politica neural (Entradas 259 e 262)
CL = pd.read_csv(R / "entrada258L_longT_cells.csv")
TL = pd.read_csv(R / "entrada258L_longT_tests.csv").set_index("id")
HL = (R / "entrada258L_longT.txt").read_text(encoding="utf-8")
gm = re.search(r"T = 10.000: validada ([\d.]+)%, aleatorio ([\d.]+)%, ql ([\d.]+)% \| T = 1.000: validada ([\d.]+)%, aleatorio ([\d.]+)%, ql ([\d.]+)%", HL)
chk("T longo: lacuna da validada (2,69%)", float(gm.group(1)), 2.69, 0.006, "sec_longneural.tex", "$2.69\\%$")
chk("T longo: lacuna do aleatorio (4,31%)", float(gm.group(2)), 4.31, 0.006, "sec_longneural.tex", "\\to4.31\\%$")
chk("T longo: lacuna do aleatorio a T=1000 (7,42%)", float(gm.group(5)), 7.42, 0.006, "sec_longneural.tex", "$7.42\\%\\to4.31\\%$")
chk("T longo: lacuna do ql (4,23%)", float(gm.group(3)), 4.23, 0.006, "sec_longneural.tex", "$7.91\\%\\to4.23\\%$")
chk("T longo: ganho medio da validada a T=1000 (2,26)", CL.g_val_T1000.mean(), 2.26, 0.006, "sec_longneural.tex", "from $2.26$ to $1.50$ points")
chk("T longo: ganho medio da validada a T=10000 (1,50)", CL.g_val_T10000.mean(), 1.50, 0.006)
LB_ = TL[TL.index.str.startswith("B-")]
LC_ = TL[TL.index.str.startswith("B'-")]
chk("T longo: a validada bate os 49 pares (confirmada)", int(LB_.confirmada.sum()), 49, 0, "sec_longneural.tex", "all $49$ learner--type pairs")
chk("T longo: nenhum aprendiz bate a validada alem da margem", int((LC_.confirmada & ~LC_["equivalente_0.05"]).sum()), 0, 0)
LA_ = TL[TL.index.str.startswith("A-")]
LAb = LA_.confirmada & ~LA_["equivalente_0.05"]
for k, lit, snip in (("alns", 6, "the ALNS roulette gains beyond the margin in six of the seven types"), ("iq", 4, "the episodic published design in four"), ("fr", 5, "the FRRMAB-style design in five"), ("ql", 2, "gains in two types")):
    chk(f"T longo: celulas alem da margem {k} ({lit})", int(LAb[LA_.index.str.endswith(f'-{k}')].sum()), lit, 0, "sec_longneural.tex", snip)
LRN_ = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr"]
rat = CL.best / CL.g_val_T10000
chk("T longo: melhor aprendiz recupera ao menos 12%", 100 * rat.min(), 12, 0.5, "sec_longneural.tex", "from $12\\%$ to $39\\%$")
chk("T longo: melhor aprendiz recupera no maximo 39%", 100 * rat.max(), 39, 0.5)
chk("T longo: maior ganho do alns (1,18 em U10a100)", CL.set_index("type").loc["U10a100", "g_alns"], 1.18, 0.005, "sec_longneural.tex", "up to $1.18$ points on $U[10,100]$")
qlg = sorted(CL.loc[[i for i, t in zip(CL.index, CL.type) if bool(LAb.get(f"A-{t}-ql", False))], "g_ql"])
chk("T longo: ganhos do ql alem da margem (0,07 e 0,45)", float(np.abs(np.array(qlg) - np.array([0.07, 0.45])).max()), 0, 0.006, "sec_longneural.tex", "($0.07$ and $0.45$ points)")
chk("T longo: previsoes 5 de 6", int(re.search(r"previsoes acertadas: (\d+) de", HL).group(1)), 5, 0, "sec_longneural.tex", "five of the six predictions held")

C263 = pd.read_csv(R / "entrada263_qperm_longT_cells.csv")
T263 = pd.read_csv(R / "entrada263_qperm_longT_tests.csv").set_index("id")
H263 = (R / "entrada263_qperm_longT.txt").read_text(encoding="utf-8")
A263 = T263[T263.index.str.startswith("A-")]
chk("Entrada 263: ql_perm ganha alem da margem em 0 de 7", int((A263.confirmada & ~A263["equivalente_0.05"]).sum()), 0, 0, "sec_longneural.tex", "in none of the seven types")
neg = C263[C263.g_qp < -0.05].g_qp
chk("Entrada 263: perde em seis tipos", len(neg), 6, 0, "sec_longneural.tex", "worse than random choice in six")
chk("Entrada 263: perda minima (0,09)", -neg.max(), 0.09, 0.006, "sec_longneural.tex", "$0.09$ to $0.90$ points")
chk("Entrada 263: perda maxima (0,90)", -neg.min(), 0.90, 0.006)
chk("Entrada 263: razao de avaliacoes do ql_perm (1,10)", C263.ev_qp.mean(), 1.10, 0.006, "sec_longneural.tex", "$1.10$ times as many")
chk("Entrada 263: razao de avaliacoes do ql padrao (0,72)", C263.ev_ql.mean(), 0.72, 0.006, "sec_longneural.tex", "$0.72$ for the standard learner")
B263 = T263[T263.index.str.startswith("B-")]
chk("Entrada 263: a validada bate o ql_perm em 7 de 7", int(B263.confirmada.sum()), 7, 0, "sec_longneural.tex", "beats it in all seven types")
chk("Entrada 263: previsoes 4 de 5", int(re.search(r"previsoes acertadas: (\d+) de", H263).group(1)), 4, 0, "sec_longneural.tex", "Four of the five predictions")

CN = pd.read_csv(R / "entrada262_neural_cells.csv")
TN = pd.read_csv(R / "entrada262_neural_tests.csv").set_index("id")
HN = (R / "entrada262_neural.txt").read_text(encoding="utf-8")
NA = TN[TN.index.str.startswith("A-")]
NB = TN[TN.index.str.startswith("B-")]
NC = TN[TN.index.str.startswith("B'-")]
ND = TN[TN.index.str.startswith("D-")]
chk("neural: ganha do aleatorio alem da margem nas 14 celulas", int((NA.confirmada & ~NA["equivalente_0.05"]).sum()), 14, 0, "sec_longneural.tex", "in all $14$ cells")
chk("neural: nunca bate a validada alem da margem", int((NC.confirmada & ~NC["equivalente_0.05"]).sum()), 0, 0)
chk("neural: a validada bate a politica em 11 de 14 (confirmada)", int(NB.confirmada.sum()), 11, 0, "sec_longneural.tex", "in $11$ of the $14$ cells")
chk("neural: ganho medio da validada (1,95)", CN.g_val.mean(), 1.95, 0.006, "sec_longneural.tex", "gains $1.95$ points")
chk("neural: ganho medio da politica (1,31)", CN.g_nn.mean(), 1.31, 0.006, "sec_longneural.tex", "the policy $1.31$")
chk("neural: fracao da politica (67%)", 100 * CN.g_nn.mean() / CN.g_val.mean(), 67, 0.6, "sec_longneural.tex", "$67\\%$")
chk("neural: ganho medio da mistura constante (1,17)", CN.g_nn_const.mean(), 1.17, 0.006, "sec_longneural.tex", "the constant mixture $1.17$")
chk("neural: fracao da mistura constante (60%)", 100 * CN.g_nn_const.mean() / CN.g_val.mean(), 60, 0.6, "sec_longneural.tex", "$60\\%$")
chk("neural: melhor aprendiz online (0,35)", CN.best_online.mean(), 0.35, 0.006, "sec_longneural.tex", "$0.35$ ($18\\%$")
chk("neural: fracao do melhor online (18%)", 100 * CN.best_online.mean() / CN.g_val.mean(), 18, 0.6)
chk("neural: politica contra o melhor online (3,7 vezes)", CN.g_nn.mean() / CN.best_online.mean(), 3.7, 0.06, "sec_longneural.tex", "$3.7$ times")
Db = ND[ND.confirmada & ~ND["equivalente_0.05"]]
chk("neural: bate o nn_const alem da margem em 6 de 14", len(Db), 6, 0, "sec_longneural.tex", "in $6$ of the $14$ cells")
chk("neural: as seis ficam em B30", int(Db.index.str.endswith("B30").sum()), 6, 0, "sec_longneural.tex", "all six are in B30")
chk("neural: menor diferenca (0,04)", Db.mean_d_pct.abs().min(), 0.04, 0.006, "sec_longneural.tex", "$0.04$ to $0.50$ points")
chk("neural: maior diferenca (0,50)", Db.mean_d_pct.abs().max(), 0.50, 0.006)
chk("neural: previsoes 5 de 5", int(re.search(r"previsoes acertadas: (\d+) de", HN).group(1)), 5, 0, "sec_longneural.tex", "All five predictions held")
cost = {}
for sp in ("B15", "B30"):
    for v in ("full", "const"):
        d_ = pd.read_csv(R / f"nn_train_{sp}_{v}_entrada259.csv")
        cost[(sp, v)] = int(len(d_) * d_.runs_per_gen.iloc[0])
chk("neural: custo de treino B30 (138.240)", cost[("B30", "full")], 138240, 0, "sec_longneural.tex", "$138{,}240$ IG runs in B30")
chk("neural: custo de treino B15 (115.200)", cost[("B15", "full")], 115200, 0, "sec_longneural.tex", "$115{,}200$ in B15")
chk("neural: custo dos controles (69.120)", cost[("B30", "const")], 69120, 0, "sec_longneural.tex", "$69{,}120$ for each control")
chk("neural: custo dos controles B15 (69.120)", cost[("B15", "const")], 69120, 0)

# ---------------------------------------------------------------- instancias publicadas held-out de P||Cmax com o Q-Learning invariante a ordem (Entrada 264)
C264 = pd.read_csv(R / "entrada264_qperm_pub_cells.csv")
T264 = pd.read_csv(R / "entrada264_qperm_pub_tests.csv").set_index("id")
H264 = pd.read_csv(R / "entrada264_qperm_pub_h.csv").set_index("id")
H190 = pd.read_csv(R / "entrada190_replication_h.csv").set_index("id")
T264_text = (R / "entrada264_qperm_pub.txt").read_text(encoding="utf-8")
A264 = T264[T264.index.str.startswith("A-")]
B264 = T264[T264.index.str.startswith("B-")]
chk("Entrada 264: ql_perm ganha alem da margem em 0 de 18", int((A264.confirmada & ~A264["equivalente_0.05"]).sum()), 0, 0, "sec_order.tex", "none of the $18$ cells")
chk("Entrada 264: perda do ql_perm na familia C, S15 (0,25)", H264.loc["H1p", "mean_d_pct"], 0.25, 0.006, "sec_order.tex", "$0.25$ ($15$ actions)")
chk("Entrada 264: perda do ql_perm na familia C, S5 (0,12)", H264.loc["H2p", "mean_d_pct"], 0.12, 0.006, "sec_order.tex", "$0.12$ ($5$ actions)")
chk("Entrada 264: perda do ql_perm na familia E, S15 (0,07)", H264.loc["H3bp", "mean_d_pct"], 0.07, 0.006, "sec_order.tex", "it loses $0.07$ and $0.03$ points")
chk("Entrada 264: perda do ql_perm na familia E, S5 (0,03)", H264.loc["H3ap", "mean_d_pct"], 0.03, 0.006)
chk("Entrada 190: ganho do Q padrao na familia C, S15 (0,245)", -H190.loc["H1", "mean_d_pct"], 0.245, 0.0006, "sec_order.tex", "$0.245$ and $0.149$ points")
chk("Entrada 190: ganho do Q padrao na familia C, S5 (0,149)", -H190.loc["H2", "mean_d_pct"], 0.149, 0.0006)
chk("Entrada 190: perda do Q padrao na familia E, S15 (0,05)", H190.loc["H3b", "mean_d_pct"], 0.05, 0.006, "sec_order.tex", "($0.05$ and $0.06$)")
chk("Entrada 190: perda do Q padrao na familia E, S5 (0,06)", H190.loc["H3a", "mean_d_pct"], 0.06, 0.006)
chk("Entrada 264: a validada bate o ql_perm nas 18 celulas", int(B264.confirmada.sum()), 18, 0, "sec_order.tex", "in all $18$ cells")
from scipy import stats as _ss
chk("Entrada 264: Spearman do Q padrao (+0,65)", float(_ss.spearmanr(C264.folga, C264.g_ql).statistic), 0.65, 0.006, "sec_order.tex", "$+0.65$ for the standard learner")
chk("Entrada 264: Spearman do ql_perm (-0,74)", float(_ss.spearmanr(C264.folga, C264.g_qp).statistic), -0.74, 0.006, "sec_order.tex", "is $-0.74$ for the order-invariant one")
chk("Entrada 264: 7 de 7 previsoes", int(re.search(r"previsoes acertadas: (\d+) de", T264_text).group(1)), 7, 0, "sec_order.tex", "all seven predictions held")
ncells = len(Cz) + len(Cf) + len(Cp) + len(C264)
chk("celulas repetidas com o ql_perm (69)", ncells, 69, 0, "sec_order.tex", "none of the $69$ cells where we repeated it")
chk("celulas repetidas com o ql_perm no resumo (69)", ncells, 69, 0, "abstract.tex", "none of $69$ cells")
chk("celulas repetidas com o ql_perm na introducao (69)", ncells, 69, 0, "sec_intro.tex", "none of $69$ cells")
chk("celulas repetidas com o ql_perm na discussao (69)", ncells, 69, 0, "sec_discussion.tex", "all $69$ cells that we repeated")
chk("celulas repetidas com o ql_perm na conclusao (69)", ncells, 69, 0, "sec_discussion.tex", "none of $69$ cells")
chk("Spearman na discussao: de +0,65 a -0,74", float(_ss.spearmanr(C264.folga, C264.g_ql).statistic), 0.65, 0.006, "sec_discussion.tex", "from $+0.65$ to $-0.74$")

# ---------------------------------------------------------------- controle global de falsas descobertas (Entradas 247 e 265)
HG = (R / "entrada265_global.txt").read_text(encoding="utf-8")
mg_ = re.search(r"todas \(ate a Entrada 264\): (\d+) testes em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global \d+ \([\d.]+\) e a Benjamini-Yekutieli global (\d+)", HG)
mo_ = re.search(r"ate a Entrada 255 \(o conjunto do artigo\): (\d+) testes em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global \d+ \([\d.]+\) e a Benjamini-Yekutieli global (\d+)", HG)
chk("BY global: testes (1.844)", int(mg_.group(1)), 1844, 0, "sec_discussion.tex", "$1{,}844$ tests of all registered batteries")
chk("BY global: confirmados (1.048)", int(mg_.group(2)), 1048, 0, "sec_discussion.tex", "$1{,}048$ confirmed tests")
chk("BY global: sobrevivem (1.020)", int(mg_.group(3)), 1020, 0, "sec_discussion.tex", "$1{,}020$ of the")
chk("BY global ate 255: testes (1.169)", int(mo_.group(1)), 1169, 0, "sec_discussion.tex", "$1{,}169$ tests up to Entry 255")
chk("BY global ate 255: confirmados e sobreviventes (685, 663)", int(mo_.group(2)) * 1000 + int(mo_.group(3)), 685663, 0, "sec_discussion.tex", "$663$ of $685$")
chk("BY global: previsoes da Entrada 265 (3 de 3)", int(re.search(r"previsoes acertadas: (\d+) de", HG).group(1)), 3, 0, "sec_discussion.tex", "three predictions of three")

# ---------------------------------------------------------------- contadores do registro
main = (J / "main.tex").read_text(encoding="utf-8")
held = int(re.search(r"\\newcommand\{\\npredheld\}\{(\d+)\}", main).group(1))
total = int(re.search(r"\\newcommand\{\\npredtotal\}\{(\d+)\}", main).group(1))
nbat = int(re.search(r"\\newcommand\{\\nbatteries\}\{(\d+)\}", main).group(1))
share = int(re.search(r"\\newcommand\{\\npredshare\}\{(\d+)\}", main).group(1))
src = (ROOT / "analysis" / "audit_cec_claims.py").read_text(encoding="utf-8")
pred = eval(re.search(r"^PRED = (\[.*?\])\s+#", src, flags=re.M).group(1))   # as 40 baterias do CEC
extra = []
# baterias alem das 40 do CEC: Entrada 256 (benchmark padrao), 258 (mecanismo, etapas 1 e 2; horizonte longo), 260, 261, 262 (politica neural), 263
for fn in ("entrada256_fanjul.txt", "entrada258_mechanism.txt", "entrada258_mechanism2.txt", "entrada258L_longT.txt", "entrada260_qperm.txt", "entrada261_qperm_pcmax.txt",
           "entrada262_neural.txt", "entrada263_qperm_longT.txt", "entrada264_qperm_pub.txt", "entrada265_global.txt"):
    f_ = R / fn
    m = re.search(r"previsoes acertadas: (\d+) de (\d+)", f_.read_text(encoding="utf-8"))
    extra.append((int(m.group(1)), int(m.group(2))))
chk("baterias contadas (CEC + Entradas 256-263)", len(pred) + len(extra), nbat, 0)
chk("previsoes acertadas (CEC + Entradas 256-263)", sum(a for a, _ in pred + extra), held, 0)
chk("previsoes registradas (CEC + Entradas 256-263)", sum(b for _, b in pred + extra), total, 0)
chk("porcentagem de acertos", 100 * held / total, share, 0.5)

bad = [r for r in rows if not r[4]]
for lab, v, e, tol, ok in rows:
    print(f"{'OK ' if ok else 'ERRO'} {lab}" + ("" if np.isnan(v) else f"  recalculado {v:.5g} vs texto {e:.5g} (tol {tol})"))
print(f"\n{len(rows) - len(bad)}/{len(rows)} verificacoes OK")
sys.exit(1 if bad else 0)
