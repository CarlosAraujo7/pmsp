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
ncells = len(Cz) + len(Cf) + len(Cp) + len(C264) + len(pd.read_csv(R / "entrada267_tsp_cells.csv"))
chk("celulas repetidas com o ql_perm (69)", ncells, 85, 0, "sec_order.tex", "none of the $85$ cells where we repeated it")
chk("celulas repetidas com o ql_perm no resumo (69)", ncells, 85, 0, "abstract.tex", "none of $85$ cells")
chk("celulas repetidas com o ql_perm na introducao (85)", ncells, 85, 0, "sec_intro.tex", "none of $85$ cells")
chk("celulas repetidas com o ql_perm na discussao (85)", ncells, 85, 0, "sec_discussion.tex", "all $85$ cells that we repeated")
chk("celulas repetidas com o ql_perm na conclusao (85)", ncells, 85, 0, "sec_discussion.tex", "none of $85$ cells")
chk("Spearman na discussao: de +0,65 a -0,74", float(_ss.spearmanr(C264.folga, C264.g_ql).statistic), 0.65, 0.006, "sec_discussion.tex", "from $+0.65$ to $-0.74$")

# ---------------------------------------------------------------- o valor do conhecimento offline por granularidade (Entradas 266 e 266b)
GCg = pd.read_csv(R / "entrada266_granularity_cells.csv")
G30 = GCg[GCg.space == "B30"]
HK = (R / "entrada266_granularity.txt").read_text(encoding="utf-8")
PH = pd.read_csv(R / "entrada266b_posthoc_cells.csv").set_index("n")
gm_ = lambda c: float(G30[c].mean())
gtn_ = gm_("g_Gtn")
best_ = float(G30[[f"g_{k}" for k in ("ql", "ucb", "pg", "alns", "qo", "iq", "fr")]].max(axis=1).mean())
chk("granularidade: G0 em B30 (1,15)", gm_("g_G0"), 1.15, 0.006, "sec_granularity.tex", "gains $1.15$ points in B30")
chk("granularidade: G0 / Gtn (51%)", 100 * gm_("g_G0") / gtn_, 51, 0.6, "sec_granularity.tex", "$51\\%$ of the class-validated gain ($2.24$)")
chk("granularidade: Gtn (2,24)", gtn_, 2.24, 0.006)
chk("granularidade: melhor aprendiz online (0,39)", best_, 0.39, 0.006, "sec_granularity.tex", "gains $0.39$ ($17\\%$)")
chk("granularidade: melhor aprendiz / Gtn (17%)", 100 * best_ / gtn_, 17, 0.6)
chk("granularidade: Gn (2,00; 89%)", gm_("g_Gn"), 2.00, 0.006, "sec_granularity.tex", "$G_n$ recovers $89\\%$ of the class-validated gain ($2.00$ points)")
chk("granularidade: Gn / Gtn (89%)", 100 * gm_("g_Gn") / gtn_, 89, 0.6)
chk("granularidade: Gt (1,19)", gm_("g_Gt"), 1.19, 0.006, "sec_granularity.tex", "$G_t$: $1.19$")
chk("granularidade: HCF (2,23)", gm_("g_HCF"), 2.23, 0.006, "sec_granularity.tex", "($2.24$ against $2.23$; $K_8$)")
chk("granularidade: HCF / Gtn (99%)", 100 * gm_("g_HCF") / gtn_, 99, 0.6, "sec_granularity.tex", "reaches $99\\%$ of the ceiling")
chk("granularidade: LOTO-G0 (1,00)", gm_("g_LOTO_G0"), 1.00, 0.006, "sec_granularity.tex", "six of seven types ($1.00$ points)")
chk("granularidade: LOTO-Gn / Gtn (87%)", 100 * gm_("g_LOTO_Gn") / gtn_, 87, 0.6, "sec_granularity.tex", "recovers $87\\%$")
chk("granularidade: LOSO-G0 (0,34)", gm_("g_LOSO_G0"), 0.34, 0.006, "sec_granularity.tex", "gain $0.34$ and $0.38$ points ($15\\%$ and $17\\%$)")
chk("granularidade: LOSO-Gt (0,38)", gm_("g_LOSO_Gt"), 0.38, 0.006)
chk("granularidade: LOSO-G0 / Gtn (15%)", 100 * gm_("g_LOSO_G0") / gtn_, 15, 0.6)
chk("granularidade: LOSO-Gt / Gtn (17%)", 100 * gm_("g_LOSO_Gt") / gtn_, 17, 0.6)
chk("granularidade: nn (1,55)", gm_("g_nn"), 1.55, 0.006, "sec_granularity.tex", "gains $1.55$ points ($69\\%$)")
chk("granularidade: nn / Gtn (69%)", 100 * gm_("g_nn") / gtn_, 69, 0.6)


def _k(i):
    return re.search(rf"\(K{i}\)[^\n]*?: (\d+) de (\d+)", HK)


chk("granularidade: K1, tipos em que G0 ganha (6 de 7)", int(_k(1).group(1)), 6, 0, "sec_granularity.tex", "in six of the seven types ($K_1$, $K_2$)")
chk("granularidade: K3, pares aprendiz-tipo (0 de 49)", int(_k(3).group(1)), 0, 0, "sec_granularity.tex", "any of the $49$ learner--type pairs")
chk("granularidade: K5, LOTO-G0 em 6 de 7 tipos", int(_k(5).group(1)), 6, 0)
chk("granularidade: K6, melhor aprendiz contra LOTO-G0 (0 de 7)", int(_k(6).group(1)), 0, 0, "sec_granularity.tex", "the best online learner beats it in none")
chk("granularidade: K7, LOSO-G0 em 2 de 4 tamanhos", int(_k(7).group(1)), 2, 0, "sec_granularity.tex", "in two of four sizes only")
chk("granularidade: K9, a politica neural bate G0 em 4 de 7 tipos", int(_k(9).group(1)), 4, 0, "sec_granularity.tex", "in four of the seven types ($K_9$ failed)")
chk("granularidade: previsoes 7 de 9", int(re.search(r"previsoes acertadas: (\d+) de", HK).group(1)), 7, 0, "sec_granularity.tex", "Seven of the nine predictions held")
chk("granularidade pos hoc: LOSO-G0 em n = 100 (-1,19)", PH.loc[100, "LOSO_G0"], -1.19, 0.006, "sec_granularity.tex", "\\emph{loses} $1.19$ points to random choice")
chk("granularidade pos hoc: fr em n = 100 (+0,07)", PH.loc[100, "fr"], 0.07, 0.006, "sec_granularity.tex", "($+0.07$)")
chk("granularidade pos hoc: qo em n = 1000 (0,83)", PH.loc[1000, "qo"], 0.83, 0.006, "sec_granularity.tex", "gains $0.83$ against $0.37$")
chk("granularidade pos hoc: LOSO-G0 em n = 1000 (0,37)", PH.loc[1000, "LOSO_G0"], 0.37, 0.006)
chk("granularidade pos hoc: LOSO-G0 em n = 200 (0,84)", PH.loc[200, "LOSO_G0"], 0.84, 0.006, "sec_granularity.tex", "($0.84$ and $1.32$ against at most $0.53$)")
chk("granularidade pos hoc: LOSO-G0 em n = 500 (1,32)", PH.loc[500, "LOSO_G0"], 1.32, 0.006)
chk("granularidade pos hoc: melhor aprendiz nos tamanhos intermediarios (0,53)", float(max(PH.loc[200, "g_best_online"], PH.loc[500, "g_best_online"])), 0.53, 0.006)
chk("granularidade pos hoc: melhor aprendiz nas pontas (0,07 e 0,83)", float(PH.loc[100, "g_best_online"]) * 1000 + float(PH.loc[1000, "g_best_online"]), 0.07 * 1000 + 0.83, 6.1, "sec_granularity.tex", "its $0.07$--$0.83$ points")
chk("granularidade pos hoc: Gn matched ao tamanho (2,00)", gm_("g_Gn"), 2.00, 0.006, "sec_granularity.tex", "the $2.00$ that an action matched to the size gives")
PHT = pd.read_csv(R / "entrada266b_posthoc_tests.csv")
chk("granularidade pos hoc: o melhor aprendiz bate LOSO-G0 em 2 de 4 tamanhos", int((PHT[PHT.id.str.startswith("best-vs-LOSO_G0")].confirmada & ~PHT[PHT.id.str.startswith("best-vs-LOSO_G0")]["equivalente_0.05"]).sum()), 2, 0)

chk("resumo: G0 / melhor aprendiz online (tres vezes)", gm_("g_G0") / best_, 3.0, 0.1, "abstract.tex", "gains three times what the best online learner finds")
chk("resumo: Gn / Gtn (nove decimos)", 100 * gm_("g_Gn") / gtn_, 90, 1.5, "abstract.tex", "nine tenths of what class-level knowledge gives")
chk("resumo: a politica neural recupera dois tercos do ganho da validada por classe", 100 * gm_("g_nn") / gtn_, 67, 3.0, "abstract.tex", "recovers about two thirds of it")

# ---------------------------------------------------------------- Q-Learning otimista aquecido por uma priori offline e o controle sem aprendizagem (Entradas 271 e 272)
CW = pd.read_csv(R / "entrada272_frozenqo_cells.csv")
TW = pd.read_csv(R / "entrada271_warmqo_tests.csv")
TZ = pd.read_csv(R / "entrada272_frozenqo_tests.csv")
HW = (R / "entrada271_warmqo.txt").read_text(encoding="utf-8")
HZ = (R / "entrada272_frozenqo.txt").read_text(encoding="utf-8")
mw = {m: float(CW[f"g_{m}"].mean()) for m in ("qo", "qo_warm_loso", "qo_frozen_loso", "qo_warm_full", "qo_frozen_full", "LOSO_Gt", "Gtn")}
chk("aquecido: qo aquecido com a priori dos outros tamanhos (1,07)", mw["qo_warm_loso"], 1.07, 0.006, "sec_granularity.tex", "the warm-started learner gains $1.07$ points, against $0.31$ for the cold learner and $0.38$")
chk("aquecido: qo frio (0,31)", mw["qo"], 0.31, 0.006)
chk("aquecido: LOSO-Gt (0,38)", mw["LOSO_Gt"], 0.38, 0.006)
Ew = TW[TW.id.str.startswith("E-")]
chk("aquecido: o aquecido (LOSO) bate o frio alem da margem em 6 de 7 tipos", int(beyond_ := int((Ew[Ew.id.str.split("-").str[1] == "qo_warm_loso"].confirmada & ~Ew[Ew.id.str.split("-").str[1] == "qo_warm_loso"]["equivalente_0.05"]).sum())), 6, 0, "sec_granularity.tex", "in six of the seven types and the transferred static action at all four sizes")
Bs_ = TW[TW.id.str.startswith("Bs-qo_warm_loso-LOSO_Gt-n")]
chk("aquecido: o aquecido bate a LOSO-Gt em 4 de 4 tamanhos (confirmada)", int(Bs_.confirmada.sum()), 4, 0)
chk("aquecido: em n = 500 a vantagem e' de 0,13 (menor que a margem)", -float(Bs_[Bs_.id.str.endswith("n500")].mean_d_pct.iloc[0]), 0.13, 0.006, "sec_granularity.tex", "at $n=500$ by $0.13$ points, less than the margin")
chk("aquecido: qo aquecido com a priori da classe (1,97)", mw["qo_warm_full"], 1.97, 0.006, "sec_granularity.tex", "it gains $1.97$ points, $88\\%$ of the class-validated gain")
chk("aquecido: qo_warm_full / Gtn (88%)", 100 * mw["qo_warm_full"] / mw["Gtn"], 88, 0.6)
Fw = TW[(TW.id.str.startswith("E-qo_warm_full"))]
chk("aquecido: o aquecido (classe) bate o frio em 7 de 7 tipos", int((Fw.confirmada & ~Fw["equivalente_0.05"]).sum()), 7, 0, "sec_granularity.tex", "beats the cold learner in all seven types")
Sw = TW[TW.id.str.startswith("Bs-qo_warm_full-Gtn-")]
chk("aquecido: o aquecido (classe) nao bate a validada em nenhum dos 7 tipos", int((Sw.confirmada & ~Sw["equivalente_0.05"]).sum()), 0, 0, "sec_granularity.tex", "but never the class-validated action (none of seven)")
chk("controle: qo congelado (LOSO) (0,40)", mw["qo_frozen_loso"], 0.40, 0.006, "sec_granularity.tex", "($0.40$ against $0.38$)")
chk("controle: qo congelado (classe) (2,25)", mw["qo_frozen_full"], 2.25, 0.006, "sec_granularity.tex", "($2.25$ against $2.24$)")
chk("controle: Gtn (2,24)", mw["Gtn"], 2.24, 0.006)
chk("controle: a aprendizagem acrescenta 0,67 quando a priori foi transferida", mw["qo_warm_loso"] - mw["qo_frozen_loso"], 0.67, 0.006, "sec_granularity.tex", "learning adds $0.67$ points when the prior was transferred (six of seven types)")
Lw = TZ[TZ.id.str.startswith("L-loso-")]
chk("controle: a aprendizagem ajuda em 6 de 7 tipos (priori transferida)", int((Lw.confirmada & ~Lw["equivalente_0.05"]).sum()), 6, 0)
chk("controle: a aprendizagem custa 0,28 quando a priori e' exata", mw["qo_frozen_full"] - mw["qo_warm_full"], 0.28, 0.006, "sec_granularity.tex", "costs $0.28$ when the prior was exact")
chk("aquecido: previsoes da Entrada 271 (6 de 7)", int(re.search(r"previsoes acertadas: (\d+) de", HW).group(1)), 6, 0, "sec_granularity.tex", "Six of the seven predictions of the warm-start battery held")
chk("aquecido: previsoes da Entrada 272 (5 de 5)", int(re.search(r"previsoes acertadas: (\d+) de", HZ).group(1)), 5, 0, "sec_granularity.tex", "all five of the control")
chk("aquecido: lacuna de 0,28 ponto a validada (W6)", mw["Gtn"] - mw["qo_warm_full"], 0.28, 0.006, "sec_granularity.tex", "a gap of $0.28$ points to the class-validated action")
chk("discussao: aquecido recupera 48% do ganho por classe", 100 * mw["qo_warm_loso"] / mw["Gtn"], 48, 0.6, "sec_discussion.tex", "it recovers $48\\%$ of the class-level gain there")
chk("discussao: a estatica transferida recupera 17%", 100 * mw["LOSO_Gt"] / mw["Gtn"], 17, 0.6, "sec_discussion.tex", "against $17\\%$ for the transferred static action")
chk("discussao: sem priori recupera 14%", 100 * mw["qo"] / mw["Gtn"], 14, 0.6, "sec_discussion.tex", "without a prior it recovers $14\\%$")
chk("resumo: a aprendizagem acrescenta 0,67 e custa 0,28", round((mw["qo_warm_loso"] - mw["qo_frozen_loso"]) * 100) * 1000 + round((mw["qo_frozen_full"] - mw["qo_warm_full"]) * 100), 67 * 1000 + 28, 0, "abstract.tex", "gains $0.67$ points over the same prior without learning, whereas with an exact prior learning costs $0.28$")

# ---------------------------------------------------------------- segundo dominio: o ILS do TSP (Entrada 267)
CT7 = pd.read_csv(R / "entrada267_tsp_cells.csv")
GT7 = pd.read_csv(R / "entrada267_tsp_gran_cells.csv")
TT7 = pd.read_csv(R / "entrada267_tsp_tests.csv")
HT7 = (R / "entrada267_tsp.txt").read_text(encoding="utf-8")
C15 = CT7[CT7.space == "T15"]
chk("TSP: menor ganho da validada (0,05)", CT7.g_val.min(), 0.05, 0.006, "sec_tsp.tex", "$0.05$ to $0.90$ points of the tour length")
chk("TSP: maior ganho da validada (0,90)", CT7.g_val.max(), 0.90, 0.006)
chk("TSP: lacuna media da validada em T15 (0,26%)", C15.gap_val.mean(), 0.26, 0.006, "sec_tsp.tex", "$0.26\\%$ in T15, against $0.77\\%$")
chk("TSP: lacuna media do aleatorio em T15 (0,77%)", C15.gap_rand.mean(), 0.77, 0.006)
mc_ = re.search(r"celulas em que cada metodo ganha do aleatorio alem da margem \(de 16\): val (\d+); ql (\d+); ucb (\d+); pg (\d+); alns (\d+); ts (\d+); qo (\d+); ql_perm (\d+)", HT7)
chk("TSP: a validada ganha alem da margem nas 16 celulas", int(mc_.group(1)), 16, 0, "sec_tsp.tex", "beyond the margin in all $16$ cells")
chk("TSP: ql ganha em 0 celulas", int(mc_.group(2)), 0, 0)
chk("TSP: ts ganha em 7 celulas", int(mc_.group(6)), 7, 0, "sec_tsp.tex", "Thompson sampling in $7$ of $16$ (up to $0.38$ points), the optimistic Q-Learning in $4$")
chk("TSP: qo ganha em 4 celulas", int(mc_.group(7)), 4, 0)
chk("TSP: ucb e pg ganham em 1 celula cada", int(mc_.group(3)) * 10 + int(mc_.group(4)), 11, 0, "sec_tsp.tex", "UCB1 and the policy gradient in one each")
chk("TSP: ql_perm ganha em 0 celulas", int(mc_.group(8)), 0, 0)
chk("TSP: maior ganho do ts (0,38)", CT7.g_ts.max(), 0.38, 0.006)
Bp7 = TT7[TT7.id.str.startswith("Bp-")]
chk("TSP: nenhum dos 6 aprendizes online bate a validada em 96 pares", int((Bp7[Bp7.id.str.split("-").str[1].isin(["ql", "ucb", "pg", "alns", "ts", "qo"])].confirmada & ~Bp7[Bp7.id.str.split("-").str[1].isin(["ql", "ucb", "pg", "alns", "ts", "qo"])]["equivalente_0.05"]).sum()), 0, 0, "sec_tsp.tex", "any of the $96$ learner--cell pairs")
chk("TSP: pares aprendiz online-celula (96)", len(Bp7[Bp7.id.str.split("-").str[1].isin(["ql", "ucb", "pg", "alns", "ts", "qo"])]), 96, 0)
B7 = TT7[TT7.id.str.startswith("B-")]
nb7 = {m: int(B7[B7.id.str.split("-").str[1] == m].confirmada.sum()) for m in ["ql", "ucb", "pg", "alns", "ts", "qo", "ql_perm"]}
chk("TSP: a validada bate ql, ql_perm, pg, alns, qo em 16 celulas", min(nb7[m] for m in ("ql", "ql_perm", "pg", "alns", "qo")), 16, 0, "sec_tsp.tex", "in all $16$ cells, than UCB1 in $15$ and than Thompson sampling in $12$")
chk("TSP: a validada bate o ucb em 15 celulas", nb7["ucb"], 15, 0)
chk("TSP: a validada bate o ts em 12 celulas", nb7["ts"], 12, 0)
chk("TSP: perda do ql, minima (0,13)", -CT7.g_ql.max(), 0.13, 0.006, "sec_tsp.tex", "($0.13$ to $0.63$ points)")
chk("TSP: perda do ql, maxima (0,63)", -CT7.g_ql.min(), 0.63, 0.006)
ms_ = re.search(r"acao listada primeiro \(T15\): ([\d.]+); \(T5\): ([\d.]+)", HT7)
chk("TSP: fracao do ql na acao 0 em T15 (61%)", 100 * float(ms_.group(1)), 61, 0.6, "sec_tsp.tex", "puts $61\\%$ of its choices in T15 ($65\\%$ in T5)")
chk("TSP: fracao do ql na acao 0 em T5 (65%)", 100 * float(ms_.group(2)), 65, 0.6)
me_ = re.search(r"razao de avaliacoes ao aleatorio: ql ([\d.]+); ql_perm ([\d.]+)", HT7)
chk("TSP: razao de avaliacoes do ql (0,64)", float(me_.group(1)), 0.64, 0.006, "sec_tsp.tex", "evaluates $0.64$ times as many moves")
chk("TSP: razao de avaliacoes do ql_perm (1,11)", float(me_.group(2)), 1.11, 0.006, "sec_tsp.tex", "(which evaluates $1.11$ times as many)")
chk("TSP: ql_perm difere do ql em 15 de 16 celulas", int(re.search(r"\(P6\)[^\n]*?: (\d+) de 16", HT7).group(1)), 15, 0, "sec_tsp.tex", "in $15$ of $16$ cells")
chk("TSP: melhor aprendiz recupera no maximo metade em 12 de 16 celulas", int(re.search(r"\(P9\)[^\n]*?: (\d+) de", HT7).group(1)), 12, 0, "sec_tsp.tex", "in $12$ of the $16$ cells")
chk("TSP: previsoes 10 de 10", int(re.search(r"previsoes acertadas: (\d+) de", HT7).group(1)), 10, 0, "sec_tsp.tex", "All ten predictions held")
chk("TSP: G0 em T15 (0,54)", float(GT7.g_G0.mean()), 0.54, 0.006, "sec_tsp.tex", "gains $0.54$ points in T15")
chk("TSP: Gkn em T15 (0,50)", float(GT7.g_Gkn.mean()), 0.50, 0.006, "sec_tsp.tex", "($0.50$, chosen on")
chk("TSP: LOSO-G0 ganha alem da margem nos 4 tamanhos", int(re.search(r"\(P8\)[^\n]*?: (\d+) de 4", HT7).group(1)), 4, 0, "sec_tsp.tex", "at all four sizes")

# ---------------------------------------------------------------- o protocolo a CPU igual no benchmark de Fanjul-Peyro e Ruiz (Entrada 268)
CE = pd.read_csv(R / "entrada268_eqcpu_fanjul_cells.csv")
KE = pd.read_csv(R / "entrada268_eqcpu_fanjul_K.csv", index_col=0)
TE8 = pd.read_csv(R / "entrada268_eqcpu_fanjul_tests.csv")
HE8 = (R / "entrada268_eqcpu_fanjul.txt").read_text(encoding="utf-8")
LE = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr"]
medK = {k: float(KE[k].median()) for k in LE + ["val"]}
chk("CPU igual: mediana de K do ql (0,74)", medK["ql"], 0.74, 0.006, "sec_eqcpu.tex", "The median $K$ is $0.74$ for the standard Q-Learning")
chk("CPU igual: K minimo dos cinco aprendizes de custo igual (0,98)", min(medK[k] for k in ("ucb", "pg", "alns", "qo", "fr")), 0.98, 0.006, "sec_eqcpu.tex", "between $0.98$ and $1.07$")
chk("CPU igual: K maximo dos cinco aprendizes de custo igual (1,07)", max(medK[k] for k in ("ucb", "pg", "alns", "qo", "fr")), 1.07, 0.006)
chk("CPU igual: mediana de K do iq (1,48)", medK["iq"], 1.48, 0.006, "sec_eqcpu.tex", "$1.48$ for the episodic design")
chk("CPU igual: mediana de K da validada (1,97)", medK["val"], 1.97, 0.006, "sec_eqcpu.tex", "$1.97$ for the validated action")
chk("CPU igual: ganho do ql a iteracoes iguais (-0,46)", CE.dec_ql.mean(), -0.46, 0.006, "sec_eqcpu.tex", "it loses $0.46$ points to random choice")
chk("CPU igual: ganho do ql a CPU igual (-0,03)", CE.cpu_ql.mean(), -0.03, 0.006, "sec_eqcpu.tex", "level with it ($-0.03$)")
chk("CPU igual: efeito de custo do ql (+0,44)", CE.cost_ql.mean(), 0.44, 0.006, "sec_eqcpu.tex", "$+0.44$ points")
ce_ = CE.set_index("type")
chk("CPU igual: ql a CPU igual em U1a100 (+0,55)", ce_.loc["U1a100", "cpu_ql"], 0.55, 0.006, "sec_eqcpu.tex", "$+0.55$ and $+0.42$ points")
chk("CPU igual: ql a CPU igual em U10a100 (+0,42)", ce_.loc["U10a100", "cpu_ql"], 0.42, 0.006)
chk("CPU igual: efeito de decisao do ql em U1a100 (-0,57)", ce_.loc["U1a100", "dec_ql"], -0.57, 0.006, "sec_eqcpu.tex", "decision effects of $-0.57$ and $-0.42$")
chk("CPU igual: efeito de decisao do ql em U10a100 (-0,42)", ce_.loc["U10a100", "dec_ql"], -0.42, 0.006)
Aq8 = TE8[TE8.id.str.startswith("Acpu-ql-")]
chk("CPU igual: ql ganha do aleatorio alem da margem em 2 de 7 tipos", int((Aq8.confirmada & ~Aq8["equivalente_0.05"]).sum()), 2, 0, "sec_eqcpu.tex", "in two of the seven types")
chk("CPU igual: efeitos de custo dos demais (no maximo 0,11)", max(abs(CE[f"cost_{k}"].mean()) for k in ("ucb", "pg", "alns", "qo", "fr")), 0.11, 0.006, "sec_eqcpu.tex", "at most $0.11$ points in absolute value")
chk("CPU igual: efeito de custo do iq (-0,55)", CE.cost_iq.mean(), -0.55, 0.006, "sec_eqcpu.tex", "($-0.55$: it costs $1.48$ per iteration)")
B8 = TE8[TE8.id.str.startswith("Bcpu-")]
chk("CPU igual: a validada bate o aprendiz em 44 de 49 pares", int(B8.confirmada.sum()), 44, 0, "sec_eqcpu.tex", "in $44$ of the $49$ learner--type pairs")
chk("CPU igual: pares (49)", len(B8), 49, 0)
vadv8 = float(np.mean([min(CE.loc[i, f"vgain_{k}"] for k in LE) for i in CE.index]))
chk("CPU igual: vantagem media sobre o aprendiz mais proximo (0,88)", vadv8, 0.88, 0.006, "sec_eqcpu.tex", "by $0.88$ points on average")
P8 = TE8[TE8.id.str.startswith("Bpcpu-")]
P8b = P8[P8.confirmada & ~P8["equivalente_0.05"]]
chk("CPU igual: dois pares em que o aprendiz bate a validada", len(P8b), 2, 0, "sec_eqcpu.tex", "in $47$ of $49$ pairs")
chk("CPU igual: alns contra a validada em MaqCorre (0,29)", -float(P8b[P8b.id == "Bpcpu-alns-MaqCorre"].mean_d_pct.iloc[0]), 0.29, 0.006, "sec_eqcpu.tex", "by $0.29$ and $0.53$ points")
chk("CPU igual: fr contra a validada em MaqCorre (0,53)", -float(P8b[P8b.id == "Bpcpu-fr-MaqCorre"].mean_d_pct.iloc[0]), 0.53, 0.006)
KE["type"] = [i.split("|")[0] for i in KE.index]
mk_ = KE.groupby("type").val.median()
chk("CPU igual: K da validada em MaqCorre (2,50)", float(mk_["MaqCorre"]), 2.50, 0.006, "sec_eqcpu.tex", "median $K=2.50$, against $1.62$--$2.21$ in the other types")
chk("CPU igual: K da validada nos outros tipos, minimo (1,62)", float(mk_.drop("MaqCorre").min()), 1.62, 0.006)
chk("CPU igual: K da validada nos outros tipos, maximo (2,21)", float(mk_.drop("MaqCorre").max()), 2.21, 0.006)
chk("CPU igual: previsoes 2 de 7", int(re.search(r"previsoes acertadas: (\d+) de", HE8).group(1)), 2, 0, "sec_eqcpu.tex", "two held")

# ---------------------------------------------------------------- o ILS do TSP em instancias da TSPLIB (Entrada 270)
C70 = pd.read_csv(R / "entrada270_tsplib_cells.csv").set_index("space")
BR70 = pd.read_csv(R / "entrada270_tsplib_bands.csv").set_index("band")
T70 = pd.read_csv(R / "entrada270_tsplib_tests.csv")
H70 = (R / "entrada270_tsplib.txt").read_text(encoding="utf-8")
chk("TSPLIB: 37 instancias", int(C70.loc["T15", "n"]), 37, 0, "sec_tsp.tex", "$37$ instances of the TSPLIB")
chk("TSPLIB: ganho de Gn_syn em T5 (0,53)", C70.loc["T5", "g_Gn_syn"], 0.53, 0.006, "sec_tsp.tex", "($0.53$ and $0.56$ points of the tour length; one action for all gains $0.53$ and $0.68$)")
chk("TSPLIB: ganho de Gn_syn em T15 (0,56)", C70.loc["T15", "g_Gn_syn"], 0.56, 0.006)
chk("TSPLIB: ganho de G0_syn em T5 (0,53)", C70.loc["T5", "g_G0_syn"], 0.53, 0.006)
chk("TSPLIB: ganho de G0_syn em T15 (0,68)", C70.loc["T15", "g_G0_syn"], 0.68, 0.006)
r6_ = re.search(r"razao melhor aprendiz online / Gn_syn: T5 ([\d.]+), T15 ([\d.]+)", H70)
chk("TSPLIB: o melhor aprendiz recupera 23% em T5", 100 * float(r6_.group(1)), 23, 0.6, "sec_tsp.tex", "recovers $23\\%$ and $43\\%$ of its gain")
chk("TSPLIB: o melhor aprendiz recupera 43% em T15", 100 * float(r6_.group(2)), 43, 0.6)
chk("TSPLIB: ganho do ts em T15 (+0,24)", C70.loc["T15", "g_ts"], 0.24, 0.006, "sec_tsp.tex", "($+0.24$ in T15)")
A70 = T70[T70.id.str.startswith("A-")]
ab70 = A70[A70.confirmada & ~A70["equivalente_0.05"]]
chk("TSPLIB: em T15 so' Gn_syn, G0_syn e ts ganham alem da margem", int(sorted(ab70[ab70.id.str.endswith("-T15")].id.str.split("-").str[1].tolist()) == sorted(["Gn_syn", "G0_syn", "ts"])), 1, 0, "sec_tsp.tex", "is the only one that beats random choice beyond the margin")
chk("TSPLIB: nenhum aprendiz online bate Gn_syn alem da margem", int((T70[T70.id.str.startswith("Bp-")].confirmada & ~T70[T70.id.str.startswith("Bp-")]["equivalente_0.05"]).sum()), 0, 0, "sec_tsp.tex", "no online learner beats it")
chk("TSPLIB: ql em T5 (-0,61)", C70.loc["T5", "g_ql"], -0.61, 0.006, "sec_tsp.tex", "loses ($-0.61$ and $-0.34$)")
chk("TSPLIB: ql em T15 (-0,34)", C70.loc["T15", "g_ql"], -0.34, 0.006)
ms70 = re.search(r"fracao do ql padrao na acao listada primeiro: T15 ([\d.]+)", H70)
chk("TSPLIB: fracao do ql na acao 0 em T15 (63%)", 100 * float(ms70.group(1)), 63, 0.6, "sec_tsp.tex", "puts $63\\%$ of its choices in T15")
chk("TSPLIB: ql_perm em T5 (-0,15)", C70.loc["T5", "g_ql_perm"], -0.15, 0.006, "sec_tsp.tex", "loses less ($-0.15$ and $-0.19$)")
chk("TSPLIB: ql_perm em T15 (-0,19)", C70.loc["T15", "g_ql_perm"], -0.19, 0.006)
E70 = T70[T70.id.str.startswith("E-")]
chk("TSPLIB: o ql_perm difere do ql alem da margem nos dois espacos", int((E70.reject & ~E70["equivalente_0.05"]).sum()), 2, 0, "sec_tsp.tex", "differs from it beyond the margin in both spaces")
chk("TSPLIB: lacuna de Gn_syn em T15 (1,21%)", C70.loc["T15", "gap_Gn_syn"], 1.21, 0.006, "sec_tsp.tex", "$1.21\\%$ above the optimal tour in T15 (random choice: $1.79\\%$)")
chk("TSPLIB: lacuna do aleatorio em T15 (1,79%)", C70.loc["T15", "gap_rand"], 1.79, 0.006)
chk("TSPLIB: lacuna nas instancias pequenas (0,76%)", BR70.loc["small", "gap_Gn_syn"], 0.76, 0.006, "sec_tsp.tex", "$0.76\\%$ and $0.72\\%$ for the small and medium instances")
chk("TSPLIB: lacuna nas instancias medias (0,72%)", BR70.loc["medium", "gap_Gn_syn"], 0.72, 0.006)
chk("TSPLIB: lacuna nas instancias grandes (2,17%)", BR70.loc["large", "gap_Gn_syn"], 2.17, 0.006, "sec_tsp.tex", "$2.17\\%$ for the large ones")
chk("TSPLIB: ganho de Gn_syn nas instancias grandes (1,19)", BR70.loc["large", "g_Gn_syn"], 1.19, 0.006, "sec_tsp.tex", "($1.19$ points; none in the small ones")
chk("TSPLIB: previsoes 9 de 9", int(re.search(r"previsoes acertadas: (\d+) de", H70).group(1)), 9, 0, "sec_tsp.tex", "all nine held")

chk("resumo: dois dos 49 pares a CPU igual vao no sentido contrario", len(P8b), 2, 0, "abstract.tex", "two of $49$ pairs go the other way")

chk("limitacoes: lacuna da validada a T=1000 (4,89%)", float(re.search(r"B30: validada ([\d.]+);", (R / "entrada256_fanjul.txt").read_text(encoding="utf-8")).group(1)), 4.89, 0.006, "sec_discussion.tex", "$4.89\\%$ above the best known solutions at $T=1000$")
chk("limitacoes: lacuna da validada a T=10^4 (2,69%)", float(gm.group(1)), 2.69, 0.006, "sec_discussion.tex", "$2.69\\%$ at $T=10^4$")
chk("limitacoes: lacuna da validada na TSPLIB (1,21%)", C70.loc["T15", "gap_Gn_syn"], 1.21, 0.006, "sec_discussion.tex", "$1.21\\%$ above the optimum of the TSPLIB instances (T15)")

# ---------------------------------------------------------------- Q-Learning otimista aquecido por uma priori offline no TSP (Entradas 274 e 275)
import glob as _glob

from scipy.stats import spearmanr as _spearmanr

C74 = pd.read_csv(R / "entrada274_warmtsp_cells.csv")
T74 = pd.read_csv(R / "entrada274_warmtsp_tests.csv")
H74 = (R / "entrada274_warmtsp.txt").read_text(encoding="utf-8")
H75 = (R / "entrada275_warmtsplib.txt").read_text(encoding="utf-8")
K75 = pd.read_csv(R / "entrada275_warmtsplib_cells.csv").set_index("space")
B75 = pd.read_csv(R / "entrada275_warmtsplib_bands.csv").set_index("band")
T75 = pd.read_csv(R / "entrada275_warmtsplib_tests.csv")
D74 = pd.concat([pd.read_csv(f) for f in sorted(_glob.glob(str(R / "tspwarm_test_s*of*_entrada274.csv")))], ignore_index=True)
m74 = lambda sp: C74[C74.space == sp].drop(columns=["space", "kind", "n"]).mean()
a74, b74 = m74("T15"), m74("T5")
nb = lambda F_, pre, suf: int((F_[F_.id.str.startswith(pre) & F_.id.str.endswith(suf)].confirmada & ~F_[F_.id.str.startswith(pre) & F_.id.str.endswith(suf)]["equivalente_0.05"]).sum())
TU = pd.concat([pd.read_csv(f) for f in sorted(_glob.glob(str(R / "tsp_tune_s*of*_entrada267.csv")))], ignore_index=True)
MM = TU.groupby(["cls", "arm"]).ratio.mean().unstack("arm")
rho_min = min(float(_spearmanr(MM.loc[i], MM.loc[j])[0]) for i in MM.index for j in MM.index if i < j)
chk("TSP aquecido: menor correlacao de Spearman entre os rankings das classes (>= 0,92)", float(np.floor(rho_min * 100) / 100), 0.92, 0, "sec_tsp.tex", "Spearman correlation at least $0.92$")
chk("TSP aquecido: qo_warm_full T15 (0,49)", a74.g_warm_full, 0.49, 0.006, "sec_tsp.tex", "gains $0.49$ points in T15, level with the class-validated action ($0.50$)")
chk("TSP aquecido: validada T15 (0,50)", a74.g_val, 0.50, 0.006)
chk("TSP aquecido: qo frio T15 (0,06)", a74.g_qo, 0.06, 0.006, "sec_tsp.tex", "eight times the cold learner ($0.06$)")
chk("TSP aquecido: qo_warm_full / qo frio (8)", a74.g_warm_full / a74.g_qo, 8.0, 0.5)
chk("TSP aquecido: qo_warm_full bate o frio alem da margem em 8 de 8 classes (T15)", nb(T74, "E-warm_full-", "-T15"), 8, 0, "sec_tsp.tex", "which it beats beyond the margin in all eight classes")
chk("TSP aquecido: qo_frozen_full T15 (0,50)", a74.g_frozen_full, 0.50, 0.006, "sec_tsp.tex", "the same prior without learning gains $0.50$")
chk("TSP aquecido: fracao na melhor acao da priori, congelado (95%)", 100 * float(D74[(D74.space == "T15") & (D74.method == "frozen_full")].share_top.mean()), 95, 0.6, "sec_tsp.tex", "in $95\\%$ of the iterations")
chk("TSP aquecido: fracao na melhor acao da priori, aquecido (39%)", 100 * float(D74[(D74.space == "T15") & (D74.method == "warm_full")].share_top.mean()), 39, 0.6, "sec_tsp.tex", "against $39\\%$ for the learner")
chk("TSP aquecido: qo_warm_loso T15 (0,48)", a74.g_warm_loso, 0.48, 0.006, "sec_tsp.tex", "the figures are $0.48$ and $0.53$, against $0.54$")
chk("TSP aquecido: qo_frozen_loso T15 (0,53)", a74.g_frozen_loso, 0.53, 0.006)
chk("TSP aquecido: LOSO-G0 T15 (0,54)", a74.g_LOSO_G0, 0.54, 0.006)
chk("TSP aquecido: custo da aprendizagem com a priori da classe (0,005)", a74.g_frozen_full - a74.g_warm_full, 0.005, 0.0006, "sec_tsp.tex", "$0.005$ and $0.046$ points in T15")
chk("TSP aquecido: custo da aprendizagem com a priori dos outros tamanhos (0,046)", a74.g_frozen_loso - a74.g_warm_loso, 0.046, 0.0006)
chk("TSP aquecido: custo alem da margem em uma classe de oito (full e loso)", nb(T74, "Lp-full-", "-T15") * 10 + nb(T74, "Lp-loso-", "-T15"), 11, 0, "sec_tsp.tex", "beyond the margin in one class of eight each")
chk("TSP aquecido: custo em T5 (0,023 e 0,016)", round((b74.g_frozen_full - b74.g_warm_full) * 1000) * 1000 + round((b74.g_frozen_loso - b74.g_warm_loso) * 1000), 23016, 0, "sec_tsp.tex", "$0.023$ and $0.016$ in T5")
chk("TSP aquecido: o custo previsto de >= 0,10 nao se materializou (S3)", float(a74.g_frozen_loso - a74.g_warm_loso < 0.10), 1.0, 0, "sec_tsp.tex", "did not materialise")
chk("TSP aquecido: priori invertida, a aprendizagem acrescenta 0,72 (T15)", a74.g_warm_rev - a74.g_frozen_rev, 0.72, 0.006, "sec_tsp.tex", "learning adds $0.72$ points")
chk("TSP aquecido: priori invertida, ajuda alem da margem em 8 de 8 (T15)", nb(T74, "L-rev-", "-T15"), 8, 0, "sec_tsp.tex", "in T15, beyond the margin in all eight classes")
chk("TSP aquecido: priori invertida, T5 (0,46)", b74.g_warm_rev - b74.g_frozen_rev, 0.46, 0.006, "sec_tsp.tex", "($0.46$ in T5)")
chk("TSP aquecido: priori invertida congelada (-2,03)", a74.g_frozen_rev, -2.03, 0.006, "sec_tsp.tex", "the frozen prior loses $2.03$ points and the learner $1.32$")
chk("TSP aquecido: priori invertida aquecida (-1,32)", a74.g_warm_rev, -1.32, 0.006)
ev74 = lambda m: float(D74[(D74.space == "T15") & (D74.method == m)].evals.mean())
chk("TSP aquecido: avaliacoes congelado / aquecido, priori da classe (1,38) e dos outros tamanhos (1,59)", round(ev74("frozen_full") / ev74("warm_full") * 100) * 1000 + round(ev74("frozen_loso") / ev74("warm_loso") * 100), 138159, 0, "sec_tsp.tex", "$1.38$ to $1.59$ times as many moves")
chk("TSP aquecido: previsoes da Entrada 274 (6 de 9)", int(re.search(r"previsoes acertadas: (\d+) de", H74).group(1)), 6, 0, "sec_tsp.tex", "Six of the nine predictions held")
chk("TSPLIB aquecido: warm_synn T15 (0,56)", K75.loc["T15", "g_warm_synn"], 0.56, 0.006, "sec_tsp.tex", "gains $0.56$ points with the nearest-size prior and $0.58$ with the global one, against $0.07$")
chk("TSPLIB aquecido: warm_syn0 T15 (0,58)", K75.loc["T15", "g_warm_syn0"], 0.58, 0.006)
chk("TSPLIB aquecido: qo frio T15 (0,07)", K75.loc["T15", "g_qo"], 0.07, 0.006)
chk("TSPLIB aquecido: Gn_syn T15 (0,56) e G0_syn (0,68)", round(K75.loc["T15", "g_Gn_syn"] * 100) * 100 + round(K75.loc["T15", "g_G0_syn"] * 100), 5668, 0, "sec_tsp.tex", "$0.56$ and $0.68$ for the corresponding static actions")
chk("TSPLIB aquecido: o aquecido bate o frio alem da margem com as duas prioris nos dois espacos (4 de 4)", sum(nb(T75, f"E-{m}-", f"-{sp}") for m in ("warm_synn", "warm_syn0") for sp in ("T15", "T5")), 4, 0, "sec_tsp.tex", "beyond the margin with both priors in both spaces")
chk("TSPLIB aquecido: nenhuma aquecida bate a estatica correspondente (M5)", int(re.search(r"\(M5\)[^:]*: (\d+) de 2", H75).group(1)), 0, 0, "sec_tsp.tex", "(none of two)")
chk("TSPLIB aquecido: faixa pequena, Gn_syn (-0,02)", B75.loc["small", "g_Gn_syn"], -0.02, 0.006, "sec_tsp.tex", "does not gain ($-0.02$)")
chk("TSPLIB aquecido: faixa pequena, aquecido (0,21)", B75.loc["small", "g_warm_synn"], 0.21, 0.006, "sec_tsp.tex", "the warm-started learner gains $0.21$ points")
chk("TSPLIB aquecido: faixa pequena, aquecido - congelado (0,07; 0,11 com a global)", round(B75.loc["small", "d_synn"] * 100) * 100 + round(B75.loc["small", "d_syn0"] * 100), 711, 0, "sec_tsp.tex", "$0.07$ more than the same prior without learning ($0.11$ more with the global prior)")
chk("TSPLIB aquecido: faixa pequena, aquecido - Gn_syn (0,24)", B75.loc["small", "g_warm_synn"] - B75.loc["small", "g_Gn_syn"], 0.24, 0.006, "sec_tsp.tex", "$0.24$ more than the static action")
chk("TSPLIB aquecido: faixas media e grande, Gn_syn (0,96)", B75.loc["medium+large", "g_Gn_syn"], 0.96, 0.006, "sec_tsp.tex", "where the static action gains $0.96$, learning costs $0.16$ points")
chk("TSPLIB aquecido: faixas media e grande, custo da aprendizagem (0,16; 0,24 com a global)", round(-B75.loc["medium+large", "d_synn"] * 100) * 100 + round(-B75.loc["medium+large", "d_syn0"] * 100), 1624, 0, "sec_tsp.tex", "($0.24$ with the global prior)")
chk("TSPLIB aquecido: previsoes da Entrada 275 (6 de 7)", int(re.search(r"previsoes acertadas: (\d+) de", H75).group(1)), 6, 0, "sec_tsp.tex", "Six of the seven predictions held")
chk("TSPLIB aquecido: warm_synn 0,56 e Gn_syn 0,56 (M6 falhou)", round(K75.loc["T15", "g_warm_synn"] * 100) * 100 + round(K75.loc["T15", "g_Gn_syn"] * 100), 5656, 0, "sec_tsp.tex", "the two gains are $0.56$ and $0.56$")
chk("discussao TSP: o custo da aprendizagem e' de no maximo 0,05 com a priori quase exata", max(a74.g_frozen_full - a74.g_warm_full, a74.g_frozen_loso - a74.g_warm_loso), 0.05, 0.006, "sec_discussion.tex", "learning costs at most $0.05$ points against a nearly exact prior")
chk("discussao TSP: a priori invertida acrescenta 0,72 e a TSPLIB 0,07 e custa 0,16", round((a74.g_warm_rev - a74.g_frozen_rev) * 100) * 10000 + round(B75.loc["small", "d_synn"] * 100) * 100 + round(-B75.loc["medium+large", "d_synn"] * 100), 720716, 0, "sec_discussion.tex", "adds $0.72$ points against a reversed one and, on the TSPLIB, adds $0.07$ points on the small instances")

# ---------------------------------------------------------------- controle global de falsas descobertas (Entradas 247 e 265)
HG = (R / "entrada265_global.txt").read_text(encoding="utf-8")
HG9 = (R / "entrada269_global.txt").read_text(encoding="utf-8")
HG73 = (R / "entrada273_global.txt").read_text(encoding="utf-8")
HG76 = (R / "entrada276_global.txt").read_text(encoding="utf-8")
pat_ = r"\(ate a Entrada \d+[^)]*\): (\d+) testes em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global \d+ \([\d.]+\) e a Benjamini-Yekutieli global (\d+)"
mg_ = re.search(r"todas \(ate a Entrada 275\): (\d+) testes em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global \d+ \([\d.]+\) e a Benjamini-Yekutieli global (\d+)", HG76)
m72_ = re.search(r"todas \(ate a Entrada 272\): (\d+) testes em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global \d+ \([\d.]+\) e a Benjamini-Yekutieli global (\d+)", HG73)
m70_ = re.search(r"todas \(ate a Entrada 270\): (\d+) testes em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global \d+ \([\d.]+\) e a Benjamini-Yekutieli global (\d+)", HG9)
m64_ = re.search(r"todas \(ate a Entrada 264\): (\d+) testes em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global \d+ \([\d.]+\) e a Benjamini-Yekutieli global (\d+)", HG)
mo_ = re.search(r"ate a Entrada 255 \(o conjunto do artigo\): (\d+) testes em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global \d+ \([\d.]+\) e a Benjamini-Yekutieli global (\d+)", HG)
chk("BY global: testes (3.578)", int(mg_.group(1)), 3578, 0, "sec_discussion.tex", "$3{,}578$ tests of all registered batteries")
chk("BY global ate 272: testes, confirmados e sobreviventes (3.318; 1.550, 1.486)", int(m72_.group(1)) * 100000000 + int(m72_.group(2)) * 10000 + int(m72_.group(3)), 3318 * 100000000 + 1550 * 10000 + 1486, 0, "sec_discussion.tex", "$1{,}486$ of $1{,}550$ over the $3{,}318$ tests up to Entry 272")
chk("BY global ate 270: testes (3.240)", int(m70_.group(1)), 3240, 0, "sec_discussion.tex", "$3{,}240$ tests up to Entry 270")
chk("BY global: confirmados (1.674)", int(mg_.group(2)), 1674, 0, "sec_discussion.tex", "$1{,}674$ confirmed tests")
chk("BY global ate 270: confirmados e sobreviventes (1.497, 1.435)", int(m70_.group(2)) * 10000 + int(m70_.group(3)), 1497 * 10000 + 1435, 0, "sec_discussion.tex", "$1{,}435$ of $1{,}497$")
chk("BY global: sobrevivem (1.606)", int(mg_.group(3)), 1606, 0, "sec_discussion.tex", "$1{,}606$ of the")
chk("BY global ate 264: testes (1.844)", int(m64_.group(1)), 1844, 0, "sec_discussion.tex", "$1{,}844$ tests up to Entry 264")
chk("BY global ate 264: confirmados e sobreviventes (1.048, 1.020)", int(m64_.group(2)) * 10000 + int(m64_.group(3)), 1048 * 10000 + 1020, 0, "sec_discussion.tex", "$1{,}020$ of $1{,}048$")
chk("BY global ate 255: testes (1.169)", int(mo_.group(1)), 1169, 0, "sec_discussion.tex", "$1{,}169$ tests up to Entry 255")
chk("BY global ate 255: confirmados e sobreviventes (685, 663)", int(mo_.group(2)) * 1000 + int(mo_.group(3)), 685663, 0, "sec_discussion.tex", "$663$ of $685$")
chk("BY global: previsoes das Entradas 265, 269, 273 e 276 (3 de 3 cada)", int(re.search(r"previsoes acertadas: (\d+) de", HG).group(1)) + int(re.search(r"previsoes acertadas: (\d+) de", HG9).group(1)) + int(re.search(r"previsoes acertadas: (\d+) de", HG73).group(1)) + int(re.search(r"previsoes acertadas: (\d+) de", HG76).group(1)), 12, 0, "sec_discussion.tex", "three predictions of three")

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
           "entrada262_neural.txt", "entrada263_qperm_longT.txt", "entrada264_qperm_pub.txt", "entrada265_global.txt",
           "entrada266_granularity.txt", "entrada267_tsp.txt", "entrada268_eqcpu_fanjul.txt", "entrada270_tsplib.txt", "entrada269_global.txt",
           "entrada271_warmqo.txt", "entrada272_frozenqo.txt", "entrada273_global.txt",
           "entrada274_warmtsp.txt", "entrada275_warmtsplib.txt", "entrada276_global.txt"):
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
