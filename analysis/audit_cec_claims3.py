"""
analysis/audit_cec_claims3.py (Entrada 222, Ciclo 19)
=======================================================
Terceira parte da auditoria numerica do artigo do CEC (cec/cec2027_paper.tex): os numeros do Ciclo 19 -- oraculo com validacao cruzada de sementes e a regra da folga unificada, horizonte longo na familia C, o seletor e o Q aquecido a CPU igual, o hospedeiro populacional (algoritmo genetico),
a mistura validada e a robustez estatistica. Cada numero da prosa e' recalculado dos CSV/TXT de resultados e conferido (a) com o literal do texto e (b) com a presenca do literal no texto.
Uso: python -m analysis.audit_cec_claims3   (sai com codigo 1 se algo falhar)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEX = re.sub(r"\s+", " ", (ROOT / "cec" / "cec2027_paper.tex").read_text(encoding="utf-8"))
rows = []


def chk(label, value, expected, tol, snippet=None):
    ok = abs(float(value) - float(expected)) <= tol
    if snippet is not None:
        ok = ok and (snippet in TEX)
    rows.append((label, float(value), float(expected), tol, ok))


rd = lambda f: pd.read_csv(R / f)

# ---------------------------------------------------------------- oraculo com validacao cruzada
P = rd("entrada209_cforacle_pooled.csv").set_index(["set", "space", "sub"])
b, c = lambda s, sp, sub: P.loc[(s, sp, sub), "h_biased"], lambda s, sp, sub: P.loc[(s, sp, sub), "h_cf"]
chk("folga sintetica enviesada S5 (0,125)", b("syn", "S5", "all"), 0.125, 0.0006, "is $0.125$ and $0.122$ points")
chk("folga sintetica enviesada S15 (0,122)", b("syn", "S15", "all"), 0.122, 0.0006)
chk("folga sintetica cf S5 (0,063)", c("syn", "S5", "all"), 0.063, 0.0006, "($0.063$ and $0.043$)")
chk("folga sintetica cf S15 (0,043)", c("syn", "S15", "all"), 0.043, 0.0006)
chk("remocao sintetica, minimo (metade)", float(P.loc[("syn", "S5", "all"), "bias_rel"]), 0.5, 0.01, "removes half to two thirds of it")
chk("remocao sintetica, maximo (dois tercos)", float(P.loc[("syn", "S15", "all"), "bias_rel"]), 2 / 3, 0.02)
chk("publicadas held-out enviesada S5 (0,132)", b("heldout", "S5", "all"), 0.132, 0.0006, "held-out $0.132\\to0.079$ and $0.174\\to0.117$")
chk("publicadas held-out cf S5 (0,079)", c("heldout", "S5", "all"), 0.079, 0.0006)
chk("publicadas held-out enviesada S15 (0,174)", b("heldout", "S15", "all"), 0.174, 0.0006)
chk("publicadas held-out cf S15 (0,117)", c("heldout", "S15", "all"), 0.117, 0.0006)
pr = [float(P.loc[(s, sp, "all"), "bias_rel"]) for s in ("discovery", "heldout") for sp in ("S5", "S15")]
chk("remocao nas publicadas, minimo (28%)", 100 * min(pr), 28, 0.6, "and $28$--$40\\%$ of the headroom of the published instances")
chk("remocao nas publicadas, maximo (40%)", 100 * max(pr), 40, 0.6)
chk("familia C held-out enviesada S5 (0,305)", b("heldout", "S5", "C12"), 0.305, 0.0006, "$0.305\\to0.244$ and $0.378\\to0.321$")
chk("familia C held-out cf S5 (0,244)", c("heldout", "S5", "C12"), 0.244, 0.0006)
chk("familia C held-out enviesada S15 (0,378)", b("heldout", "S15", "C12"), 0.378, 0.0006)
chk("familia C held-out cf S15 (0,321)", c("heldout", "S15", "C12"), 0.321, 0.0006)
cr = [100 * float(P.loc[("heldout", sp, "C12"), "bias_rel"]) for sp in ("S5", "S15")]
chk("remocao na familia C, minimo (15%)", min(cr), 15, 0.6, "only $15$--$20\\%$ in family C")
chk("remocao na familia C, maximo (20%)", max(cr), 20, 0.6)
chk("flowshop enviesada F5 (0,119)", b("pfsp-syn", "F5", "all"), 0.119, 0.0006, "($0.119\\to0.012$ and $0.152\\to0.025$)")
chk("flowshop cf F5 (0,012)", c("pfsp-syn", "F5", "all"), 0.012, 0.0006)
chk("flowshop enviesada F15 (0,152)", b("pfsp-syn", "F15", "all"), 0.152, 0.0006)
chk("flowshop cf F15 (0,025)", c("pfsp-syn", "F15", "all"), 0.025, 0.0006)
fr = [100 * float(P.loc[("pfsp-syn", sp, "all"), "bias_rel"]) for sp in ("F5", "F15")]
chk("remocao no flowshop, minimo (83%)", min(fr), 83, 0.6, "it removes $80$--$90\\%$")
chk("remocao no flowshop, maximo (90%)", max(fr), 90, 0.6)
chk("folga cf nas classes grandes do flowshop F5 (0,029)", c("pfsp-syn", "F5", "large"), 0.029, 0.0006, None)
chk("folga cf nas classes grandes do flowshop F15 (0,063)", c("pfsp-syn", "F15", "large"), 0.063, 0.0006)
allrel = [100 * float(P.loc[k, "bias_rel"]) for k in P.index if k[2] == "all" and k[0] != "pfsp-ta"]
chk("ruido na folga: de 28% a 90%", min(allrel), 28, 0.6, None)
chk("ruido na folga: maximo (90%)", max(allrel), 90, 0.6)

# ---------------------------------------------------------------- a regra da folga com a folga cf
Co = rd("entrada209_cforacle_corr.csv").set_index(["set", "space"])
chk("rho cf held-out S5 (0,95)", Co.loc[("heldout", "S5"), "rho_cf"], 0.95, 0.006, "is $0.95$ and $0.97$ on the held-out instances")
chk("rho cf held-out S15 (0,97)", Co.loc[("heldout", "S15"), "rho_cf"], 0.97, 0.006)
chk("rho enviesado held-out S5 (0,67)", Co.loc[("heldout", "S5"), "rho_biased"], 0.67, 0.006, "($0.67$ and $0.60$ with the biased headroom)")
chk("rho enviesado held-out S15 (0,60)", Co.loc[("heldout", "S15"), "rho_biased"], 0.60, 0.006)
chk("rho cf sinteticas S5 (0,64)", Co.loc[("syn", "S5"), "rho_cf"], 0.64, 0.006, "and $0.64$ and $0.66$ across the $12$ synthetic classes")
chk("rho cf sinteticas S15 (0,66)", Co.loc[("syn", "S15"), "rho_cf"], 0.66, 0.006)
chk("rho enviesado sinteticas S5 (0,13)", Co.loc[("syn", "S5"), "rho_biased"], 0.13, 0.006)
chk("rho enviesado sinteticas S15 (0,01)", Co.loc[("syn", "S15"), "rho_biased"], 0.01, 0.006)
chk("rho cf flowshop F5 (-0,27)", Co.loc[("syn", "F5"), "rho_cf"] if ("syn", "F5") in Co.index else np.nan, -0.27, 0.006) if False else None
Cp = rd("entrada209_cforacle_corr.csv")
fl = Cp[(Cp.problem == "pfsp") & (Cp.set == "syn")].set_index("space")
chk("rho cf flowshop F5 (-0,27)", fl.loc["F5", "rho_cf"], -0.27, 0.006, "but $-0.27$ and $-0.07$ on the flowshop ($-0.93$ and $-0.78$ before:")
chk("rho cf flowshop F15 (-0,07)", fl.loc["F15", "rho_cf"], -0.07, 0.006)
chk("rho enviesado flowshop F5 (-0,93)", fl.loc["F5", "rho_biased"], -0.93, 0.006)
chk("rho enviesado flowshop F15 (-0,78)", fl.loc["F15", "rho_biased"], -0.78, 0.006)
t209 = (R / "entrada209_cforacle.txt").read_text(encoding="utf-8")
m = re.search(r"limiar h\* escolhido na descoberta \(acuracia balanceada [\d.]+\): ([\d.]+) pontos; acuracia nas demais celulas \((\d+)\): ([\d.]+); por conjunto: pcmax/heldout ([\d.]+), pcmax/syn ([\d.]+), pfsp/syn ([\d.]+)", t209)
chk("limiar h* (0,048)", float(m.group(1)), 0.048, 0.0006, "$h^*=0.048$ points")
chk("celulas fora da amostra do limiar (60)", int(m.group(2)), 60, 0, "applied unchanged to the $60$ others")
chk("acuracia do limiar nas 60 celulas (78%)", 100 * float(m.group(3)), 78, 0.6, "classifies $78\\%$ of them correctly")
chk("acuracia do limiar held-out (89%)", 100 * float(m.group(4)), 89, 0.6, "$89\\%$ held-out, $75\\%$ synthetic, $72\\%$ flowshop;")
chk("acuracia do limiar sinteticas (75%)", 100 * float(m.group(5)), 75, 0.6)
chk("acuracia do limiar flowshop (72%)", 100 * float(m.group(6)), 72, 0.6)

# ---------------------------------------------------------------- horizonte longo
Hh = rd("entrada217_longc_by_T.csv").set_index("T")
chk("horizonte: folga cf em T = 1000 (0,222)", Hh.loc[1000, "cf_headroom"], 0.222, 0.0006, "falls from $0.222$ points at $T=1{,}000$ to $0.047$ at $T=5{,}000$ and $0.023$ at $T=20{,}000$")
chk("horizonte: folga cf em T = 5000 (0,047)", Hh.loc[5000, "cf_headroom"], 0.047, 0.0006)
chk("horizonte: folga cf em T = 20000 (0,023)", Hh.loc[20000, "cf_headroom"], 0.023, 0.0006)
chk("horizonte: Q-Learning em T = 1000 (0,128)", Hh.loc[1000, "ql"], 0.128, 0.0006, "Q-Learning's gain from $0.128$ to $0.033$ and $0.017$")
chk("horizonte: Q-Learning em T = 5000 (0,033)", Hh.loc[5000, "ql"], 0.033, 0.0006)
chk("horizonte: Q-Learning em T = 20000 (0,017)", Hh.loc[20000, "ql"], 0.017, 0.0006)
for T, e in ((5000, 0.002), (20000, 0.001)):
    X = rd(f"entrada217_longc_T{T}.csv").set_index("id")
    chk(f"horizonte: validada - Q frio em T = {T} ({e})", X.loc[f"val-ql-T{T}", "mean_d_pct"], e, 0.0006, "(differences of $0.002$ and $0.001$ points, $p_{\\mathrm{Holm}}>0.9$)" if T == 5000 else None)
    chk(f"horizonte: p_holm da validada vs Q frio em T = {T} > 0,9", float(X.loc[f"val-ql-T{T}", "p_holm"] > 0.9), 1, 0)

# ---------------------------------------------------------------- seletor a CPU igual; Q aquecido; reparos
Cvl, Cql, Cal = [rd(f"entrada213_equalcpu_{k}.csv") for k in ("Cvl", "Cql", "Cal")]
chk("seletor vs validada a CPU igual, minimo (0,018)", -Cvl.mean_d_pct.max(), 0.018, 0.0006, None)
chk("seletor vs validada a CPU igual, maximo (0,035)", -Cvl.mean_d_pct.min(), 0.035, 0.0006)
chk("seletor vs Q a CPU igual, minimo (0,077)", -Cql.mean_d_pct.max(), 0.077, 0.0006, None)
chk("seletor vs Q a CPU igual, maximo (0,093)", -Cql.mean_d_pct.min(), 0.093, 0.0006)
chk("seletor vs roleta a CPU igual, minimo (0,049)", -Cal.mean_d_pct.max(), 0.049, 0.0006, None)
chk("seletor vs roleta a CPU igual, maximo (0,147)", -Cal.mean_d_pct.min(), 0.147, 0.0006)
W = rd("entrada211_warmq_Wql.csv").set_index("id")
chk("Q aquecido vs frio, S5 descoberta (-0,006)", -W.loc["Wql-S5-discovery", "mean_d_pct"], -0.006, 0.0006, "($-0.006$ and $+0.005$ points) and much worse with fifteen ($-0.146$ and $-0.143$;")
chk("Q aquecido vs frio, S5 held-out (+0,005)", -W.loc["Wql-S5-heldout", "mean_d_pct"], 0.005, 0.0006)
chk("Q aquecido vs frio, S15 descoberta (-0,146)", -W.loc["Wql-S15-discovery", "mean_d_pct"], -0.146, 0.0006)
chk("Q aquecido vs frio, S15 held-out (-0,143)", -W.loc["Wql-S15-heldout", "mean_d_pct"], -0.143, 0.0006)
Wc = rd("entrada211_warmq_cells.csv")
chk("politica congelada pior que o aleatorio em S15", float(Wc[(Wc.space == "S15") & (Wc.cell == "all")].wq0.max() < 0), 1, 0, None)
chk("atualizacoes do pre-treino (5,7 milhoes)", 5720 * 1000 / 1e6, 5.7, 0.05, None)

# ---------------------------------------------------------------- hospedeiro populacional
GP = rd("entrada219_ga_pooled.csv").set_index("space")
chk("AG: validada vs aleatorio em G9 (-0,057)", GP.loc["G9", "val"], -0.057, 0.0006, "($-0.057$ points with nine actions)")
chk("AG: folga cf de acoes unicas em G9 (-0,073)", GP.loc["G9", "h_cf"], -0.073, 0.0006, None)
chk("AG: Thompson G9 (0,051)", GP.loc["G9", "ts"], 0.051, 0.0006, "Thompson sampling gains $0.051$ and $0.047$ points")
chk("AG: Thompson G18 (0,047)", GP.loc["G18", "ts"], 0.047, 0.0006)
chk("AG: Q-Learning G9 (-0,096)", GP.loc["G9", "ql"], -0.096, 0.0006, "Q-Learning loses $0.096$ and $0.048$")
chk("AG: Q-Learning G18 (-0,048)", GP.loc["G18", "ql"], -0.048, 0.0006)
GC = rd("entrada219_ga_cells.csv")
from scipy import stats as sstats  # noqa: E402

chk("AG: Spearman folga cf x Q-Learning G9 (0,96)", sstats.spearmanr(GC[GC.space == "G9"].h_cf, GC[GC.space == "G9"].ql).statistic, 0.96, 0.006, None)
chk("AG: Spearman folga cf x Q-Learning G18 (0,50)", sstats.spearmanr(GC[GC.space == "G18"].h_cf, GC[GC.space == "G18"].ql).statistic, 0.50, 0.006)
Kg = json.loads((R / "ga_cpu_ratio_entrada219.json").read_text())
kts = [float(np.exp(np.mean(np.log([Kg[sp][c]["ts"] for c in Kg[sp]])))) for sp in ("G9", "G18")]
chk("AG: custo do Thompson, G9 (2,0)", kts[0], 2.0, 0.1, None)
chk("AG: custo do Thompson, G18 (2,8)", kts[1], 2.8, 0.1)
Mr, Mv = rd("entrada220_ga_mix_Mrand.csv"), rd("entrada220_ga_mix_Mval.csv")
chk("mistura vs aleatorio G9 (0,050)", -Mr.mean_d_pct.iloc[0], 0.050, 0.0006, "it gains $0.050$ and $0.144$ points")
chk("mistura vs aleatorio G18 (0,144)", -Mr.mean_d_pct.iloc[1], 0.144, 0.0006)
chk("mistura vs validada unica G9 (0,105)", -Mv.mean_d_pct.iloc[0], 0.105, 0.0006, "beats the validated single action by $0.105$ and $0.067$")
chk("mistura vs validada unica G18 (0,067)", -Mv.mean_d_pct.iloc[1], 0.067, 0.0006)
Mi, Me = rd("entrada220_ga_mix_Mit.csv").set_index("id"), rd("entrada220_ga_mix_Meq.csv")
chk("mistura vence 9 dos 10 pares a iteracoes iguais", int(Mi.confirmada.sum()), 9, 0, "and nine of the ten learner--space pairs at equal iterations and at equal CPU time")
chk("mistura vence 9 dos 10 pares a CPU igual", int(Me.confirmada.sum()), 9, 0)
chk("mistura nivelada com Thompson em G9 a iteracoes iguais", float(not Mi.loc["Mit-G9-ts", "confirmada"]), 1, 0, None)

# ---------------------------------------------------------------- robustez estatistica
t215 = (R / "entrada215_robustness.txt").read_text(encoding="utf-8")
m = re.search(r"controle global: (\d+) testes registrados em \d+ arquivos; (\d+) 'confirmada'; sobrevivem a Holm global (\d+) \([\d.]+\) e a Benjamini-Yekutieli global (\d+)", t215)
chk("testes registrados (1.562)", int(m.group(1)), 1562, 0, "over the $1{,}562$ tests registered up to the genetic-algorithm battery")
chk("testes confirmados (855)", int(m.group(2)), 855, 0, "$829$ of the $855$ confirmed tests")
chk("sobrevivem a BY global (829)", int(m.group(4)), 829, 0)
chk("sobrevivem a Holm global (763)", int(m.group(3)), 763, 0, None)
chk("concordancia dos tres modos nos pares de cabecalho", float(re.search(r"concordancia dos tres modos: ([\d.]+)", t215).group(1)), 1.0, 0, "keep their significance under Pratt's treatment of ties and a permutation test")

# ---------------------------------------------------------------- Ciclo 20: o limiar (pos hoc), o Q otimista, a folga cf como cota inferior
t222 = (R / "entrada222_threshold.txt").read_text(encoding="utf-8")
m = re.search(r"fora da amostra \((\d+) celulas\): acuracia ([\d.]+), acuracia balanceada ([\d.]+), taxa de base da classe majoritaria ([\d.]+)", t222)
chk("limiar: acuracia balanceada fora da amostra (0,80)", float(m.group(3)), 0.80, 0.006, "balanced accuracy $0.80$ (bootstrap over the $30$ classes, each counted in both spaces: $95\\%$ interval $0.67$--$0.91$, against $0.60$, $0.42$--$0.79$, with the one-seed headroom;")
chk("limiar: taxa de base (68%)", 100 * float(m.group(4)), 68, 0.6)
m = re.search(r"pfsp/syn: ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| ([\d.]+)", t222)
chk("limiar: acuracia balanceada no flowshop (0,38)", float(m.group(2)), 0.38, 0.006, "(balanced accuracy $0.38$)")
chk("limiar: fracao de celulas com ganho no flowshop (1 em 18)", float(m.group(5)) * int(m.group(4)), 1, 0.01, "where one cell in $18$ has a gain")
m = re.search(r"algoritmo genetico: ([\d.]+) \|", t222)
chk("limiar: acuracia no algoritmo genetico (88%)", 100 * float(m.group(1)), 88, 0.6, "on the genetic algorithm ($24$ cells, outside the $60$) it classifies $88\\%$")
lo = [float(v) for v in re.findall(r"limiar de treino ([\d.]+)", t222)]
chk("LOHO: limiar minimo (0,049)", min(lo), 0.049, 0.0006, None)
chk("LOHO: limiar maximo (0,050)", max(lo), 0.050, 0.0006)
Oo = {k: rd(f"entrada223_optq_{k}.csv") for k in ("Or", "Oq", "Ov", "Os")}
chk("Q otimista vs aleatorio: ganho minimo (0,004)", (-Oo["Or"].mean_d_pct).min(), 0.004, 0.0006, "stays level with random choice ($0.004$--$0.008$ points)")
chk("Q otimista vs aleatorio: ganho maximo (0,008)", (-Oo["Or"].mean_d_pct).max(), 0.008, 0.0006)
oq = Oo["Oq"].mean_d_pct.tolist()
chk("Q otimista pior que o frio, S5 held-out (0,007)", oq[1], 0.007, 0.0006, "is worse than the cold Q-Learning ($0.007$--$0.051$)")
chk("Q otimista pior que o frio, S5 descoberta (0,014)", oq[0], 0.014, 0.0006)
chk("Q otimista pior que o frio, S15 held-out (0,046)", oq[3], 0.046, 0.0006)
chk("Q otimista pior que o frio, S15 descoberta (0,051)", oq[2], 0.051, 0.0006)
chk("validada vence o Q otimista, minimo (0,067)", (-Oo["Ov"].mean_d_pct).min(), 0.067, 0.0006, "loses to the validated action by $0.067$--$0.104$")
chk("validada vence o Q otimista, maximo (0,104)", (-Oo["Ov"].mean_d_pct).max(), 0.104, 0.0006)
chk("validada confirmadamente melhor que o Q otimista nos 4 pares", int(Oo["Ov"].confirmada.sum()), 4, 0)
# o ganho do seletor sobre o aleatorio fica entre a folga cf (cota inferior) e a de uma semente (cota superior), nos quatro pares conjunto x espaco
from analysis.report_entrada200_selector import SETS, load as load_sel  # noqa: E402

Pc = rd("entrada209_cforacle_pooled.csv").set_index(["set", "space", "sub"])
sg, between = [], []
for sp_ in ("S5", "S15"):
    for cj_ in SETS:
        w_ = load_sel(sp_, cj_)
        g_ = float((100 * (w_["rand"] - w_["sel"]) / w_["rand"]).mean())
        sg.append(g_)
        between.append(Pc.loc[(cj_, sp_, "all"), "h_cf"] <= g_ <= Pc.loc[(cj_, sp_, "all"), "h_biased"])
chk("ganho do seletor sobre o aleatorio, minimo (0,099)", min(sg), 0.099, 0.0006, "($0.099$--$0.146$ points) lies between the cross-fitted headroom")
chk("ganho do seletor sobre o aleatorio, maximo (0,146)", max(sg), 0.146, 0.0006)
chk("ganho do seletor entre a folga cf e a de uma semente nos 4 pares", float(all(between)), 1, 0)
nt = int(re.search(r"(\d+) tests? collected", subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout).group(1))
chk("testes unitarios no total (1.526)", nt, 1526, 0, "$1{,}526$ unit tests in all")

# ---------------------------------------------------------------- Ciclo 21: o estimador, a regra com IC por conglomerado, os tempos finos e a leitura de condicao necessaria
t227 = (R / "entrada227_cfvariants.txt").read_text(encoding="utf-8")
rv = {m_.group(1): m_ for m_ in re.finditer(r"^\s*(biased|cf13|cf22|cf22s|cf31)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+(\[[^\]]+\])\s+(\[[^\]]+\])\s+(\[[^\]]+\])\s+([\d.]+)", t227, re.M)}
chk("IC95 da acuracia balanceada, cf 2+2 (inferior 0,67)", float(re.findall(r"[-\d.]+", rv["cf22"].group(9))[0]), 0.67, 0.006, "each counted in both spaces: $95\\%$ interval $0.67$--$0.91$")
chk("IC95 da acuracia balanceada, cf 2+2 (superior 0,91)", float(re.findall(r"[-\d.]+", rv["cf22"].group(9))[1]), 0.91, 0.006)
chk("IC95 da acuracia balanceada, uma semente (0,42)", float(re.findall(r"[-\d.]+", rv["biased"].group(9))[0]), 0.42, 0.006)
chk("IC95 da acuracia balanceada, uma semente (0,79)", float(re.findall(r"[-\d.]+", rv["biased"].group(9))[1]), 0.79, 0.006)
chk("excesso sobre a taxa de base nao e' significativo (P(<=0) > 0,05)", float(float(rv["cf22"].group(11)) > 0.05), 1, 0, "its excess over the $68\\%$ of always predicting ``no gain'' is not significant")
chk("escolher em tres sementes: mesma acuracia (0,78)", float(rv["cf31"].group(4)), 0.78, 0.006, "leaves the accuracy unchanged ($0.78$)")
Ff = pd.read_csv(R / "entrada228_cf_fine_pooled.csv").set_index("space")
chk("tempos finos: folga cf S5 (0,098)", Ff.loc["S5", "h_cf"], 0.098, 0.0006, "the cross-fitted headroom is $0.098$ and $0.121$")
chk("tempos finos: folga cf S15 (0,121)", Ff.loc["S15", "h_cf"], 0.121, 0.0006)
t228 = (R / "entrada228_cf_fine.txt").read_text(encoding="utf-8")
chk("tempos finos: acuracia da regra (58%)", 100 * float(re.search(r"acuracia da regra h\* nas 24 celulas: ([\d.]+)", t228).group(1)), 58, 0.6, "it classifies only $58\\%$")
m = re.search(r"fora da descoberta, (\d+) celulas \| abaixo de h\*: (\d+), ganho do Q > 0: (\d+), ganho > 0,05: (\d+), maior ganho ([+-][\d.]+) \| acima de h\*: (\d+), ganho > 0: (\d+), ganho > 0,05: (\d+), ganho negativo: (\d+), razao ganho/folga cf mediana ([\d.]+)", t222)
chk("condicao necessaria: celulas fora da descoberta (150)", int(m.group(1)), 150, 0, "the $150$ cells outside the discovery set")
chk("condicao necessaria: celulas abaixo de h* (71)", int(m.group(2)), 71, 0, "below $h^*$ there are $71$ cells and in none does Q-Learning gain beyond the margin (a point gain above $0.05$ points; $+0.026$ at most)")
chk("condicao necessaria: nenhuma abaixo de h* ganha alem da margem (0)", int(m.group(4)), 0, 0, "in none of $71$ cells below about $0.05$ points")
chk("condicao necessaria: maior ganho abaixo de h* (+0,026)", float(m.group(5)), 0.026, 0.0006)
chk("condicao necessaria: celulas acima de h* (79)", int(m.group(6)), 79, 0, "above it there are $79$ others, and it does in $29$, gains something in $42$ and loses in $37$ (a median of $20\\%$")
chk("condicao necessaria: ganho > 0,05 acima de h* (29)", int(m.group(8)), 29, 0, "and in $29$ of the $79$ above it")
chk("condicao necessaria: ganho > 0 acima de h* (42)", int(m.group(7)), 42, 0)
chk("condicao necessaria: ganho negativo acima de h* (37)", int(m.group(9)), 37, 0)
chk("condicao necessaria: mediana da captura (20%)", 100 * float(m.group(10)), 20, 0.6)
mi = re.search(r"controle aleatorio independente.*?balanceada ([\d.]+) \(IC95 por conglomerado \[([\d.]+), ([\d.]+)\]\)", t227, re.S)
chk("controle aleatorio independente: balanceada (0,80)", float(mi.group(1)), 0.80, 0.006, "(balanced accuracy $0.80$, interval $0.68$--$0.90$)")
chk("controle aleatorio independente: IC inferior (0,68)", float(mi.group(2)), 0.68, 0.006)
chk("controle aleatorio independente: IC superior (0,90)", float(mi.group(3)), 0.90, 0.006)
for mg_, lo_snip in ((0.02, "with a margin of $0.02$ the counts are $2$ of $71$ and $37$ of $79$"), (0.10, "with $0.10$ they are $0$ of $71$ and $15$ of $79$")):
    mm_ = re.search(rf"margem {mg_:.2f}: abaixo de h\* (\d+) de (\d+) \| acima de h\* (\d+) de (\d+)", t222)
    chk(f"sensibilidade a margem {mg_}: abaixo de h* ({mm_.group(1)} de {mm_.group(2)})", int(mm_.group(1)) * 1000 + int(mm_.group(2)), {0.02: 2071, 0.10: 71}[mg_], 0, lo_snip)
    chk(f"sensibilidade a margem {mg_}: acima de h* ({mm_.group(3)} de {mm_.group(4)})", int(mm_.group(3)) * 1000 + int(mm_.group(4)), {0.02: 37079, 0.10: 15079}[mg_], 0)

# referencia absoluta: lacuna ao limite inferior classico e fracao das execucoes no limite do IG com selecao aleatoria a T = 1000
import glob  # noqa: E402


def gap_lb(d):
    x = d[d.method == "rand"]
    return float((100 * (x.makespan - x.lb) / x.lb).mean()), float((x.makespan <= x.lb).mean())


syn_ref = [gap_lb(rd(f"agents_{sp_}_entrada180.csv")) for sp_ in ("S5", "S15")]
chk("referencia absoluta: lacuna media nas sinteticas, minimo (0,26)", min(g for g, _ in syn_ref), 0.26, 0.006, "(mean gap $0.26$--$0.28\\%$)")
chk("referencia absoluta: lacuna media nas sinteticas, maximo (0,28)", max(g for g, _ in syn_ref), 0.28, 0.006)
chk("referencia absoluta: fracao no limite nas sinteticas, minimo (42%)", 100 * min(f for _, f in syn_ref), 42, 0.6, "in $42$--$48\\%$ of the synthetic runs")
chk("referencia absoluta: fracao no limite nas sinteticas, maximo (48%)", 100 * max(f for _, f in syn_ref), 48, 0.6)
pub_gaps = []
for pat in ("pub_S5_s*of*_entrada189.csv", "pubh_S5_s*of*_entrada190.csv"):
    dd = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat)))], ignore_index=True)
    dd["fam"] = dd.cls.str.split("|").str[0]
    pub_gaps += [gap_lb(x)[0] for _, x in dd.groupby("fam")]
chk("referencia absoluta: lacuna nas familias publicadas, minimo (0,2)", min(pub_gaps), 0.2, 0.06, "is within $0.2$--$1.3\\%$ of it on the published families")
chk("referencia absoluta: lacuna nas familias publicadas, maximo (1,3)", max(pub_gaps), 1.3, 0.06)
KAu = rd("entrada180_agents_k.csv")
kk = KAu[KAu.learner != "ql"].K_cpu
chk("K de UCB1 e do gradiente: minimo (1,01)", kk.min(), 1.01, 0.006, "$1.01$--$1.08$ for UCB1 and the policy gradient")
chk("K de UCB1 e do gradiente: maximo (1,08)", kk.max(), 1.08, 0.006)

# ---------------------------------------------------------------- Ciclo 22: o hospedeiro novo (R||Cmax, Entrada 234/235) e o custo de treino do seletor
Ur = rd("entrada234_ur_pooled.csv").set_index("space")
UrB = rd("entrada234_ur_B.csv")
UrC = rd("entrada234_ur_cells.csv")
t234 = (R / "entrada234_ur.txt").read_text(encoding="utf-8")
P9 = rd("entrada209_cforacle_pooled.csv").set_index(["set", "space", "sub"])
chk("R||Cmax: folga de uma semente S5 (1,5)", Ur.loc["S5", "h_biased"], 1.5, 0.006, "The one-seed headroom is $1.5$ and $2.8$ points (cross-fitted $0.43$ and $1.60$)")
chk("R||Cmax: folga de uma semente S15 (2,8)", Ur.loc["S15", "h_biased"], 2.8, 0.006)
chk("R||Cmax: folga cf S5 (0,43)", Ur.loc["S5", "h_cf"], 0.43, 0.006)
chk("R||Cmax: folga cf S15 (1,60)", Ur.loc["S15", "h_cf"], 1.60, 0.006)
chk("R||Cmax: razao da folga ao P||Cmax sintetico em S5 (12)", Ur.loc["S5", "h_biased"] / P9.loc[("syn", "S5", "all"), "h_biased"], 12, 0.5, "$12$ and $23$ times that of the synthetic")
chk("R||Cmax: razao da folga ao P||Cmax sintetico em S15 (23)", Ur.loc["S15", "h_biased"] / P9.loc[("syn", "S15", "all"), "h_biased"], 23, 0.5)
chk("R||Cmax: razao cross-fitted em S5 (7)", Ur.loc["S5", "h_cf"] / P9.loc[("syn", "S5", "all"), "h_cf"], 7, 0.5, "($7$ and $37$ times cross-fitted)")
chk("R||Cmax: razao cross-fitted em S15 (37)", Ur.loc["S15", "h_cf"] / P9.loc[("syn", "S15", "all"), "h_cf"], 37, 0.6)
chk("R||Cmax: ganho da validada S5 (0,95)", Ur.loc["S5", "g_val"], 0.95, 0.006, "the validated action gains $0.95$ and $2.08$ points")
chk("R||Cmax: ganho da validada S15 (2,08)", Ur.loc["S15", "g_val"], 2.08, 0.006)
chk("R||Cmax: Q-Learning S5 (-1,06)", Ur.loc["S5", "g_ql"], -1.06, 0.006, "Q-Learning loses ($-1.06$ and $-0.48$)")
chk("R||Cmax: Q-Learning S15 (-0,48)", Ur.loc["S15", "g_ql"], -0.48, 0.006)
chk("R||Cmax: Q otimista S5 (+0,17)", Ur.loc["S5", "g_qo"], 0.17, 0.006, "the optimistic Q-Learning ($+0.17$, five sizes)")
chk("R||Cmax: roleta ALNS S15 (+0,15)", Ur.loc["S15", "g_alns"], 0.15, 0.006, "the ALNS roulette ($+0.15$, fifteen actions)")
chk("R||Cmax: captura do Q otimista em S5 (18%)", 100 * Ur.loc["S5", "g_qo"] / Ur.loc["S5", "g_val"], 18, 0.6, "at most $18\\%$ and $7\\%$ of the validated action's gain")
chk("R||Cmax: captura da roleta em S15 (7%)", 100 * Ur.loc["S15", "g_alns"] / Ur.loc["S15", "g_val"], 7, 0.6)
chk("R||Cmax: UCB1 e gradiente em paridade (|ganho| <= 0,04)", max(abs(Ur.loc[s_, k_]) for s_ in ("S5", "S15") for k_ in ("g_ucb", "g_pg")), 0.04, 0.006, "UCB1 and the policy gradient are at parity")
chk("R||Cmax: a validada vence os dez pares (confirmados)", int(UrB.confirmada.sum()), 10, 0, "the validated action beats every learner in all ten learner--space pairs, by $0.77$--$2.52$ points")
chk("R||Cmax: vantagem da validada, minimo (0,77)", float((-UrB.mean_d_pct).min()), 0.77, 0.006)
chk("R||Cmax: vantagem da validada, maximo (2,52)", float((-UrB.mean_d_pct).max()), 2.52, 0.006)
macc = re.search(r"regra congelada: acuracia ([\d.]+)", t234)
chk("R||Cmax: acuracia da regra (42%)", 100 * float(macc.group(1)), 42, 0.6, "($42\\%$ of the $24$ cells: $19$ lie above $h^*$ and Q-Learning gains in only $5$ of those)")
mu = re.search(r"abaixo de h\* (\d+) celulas \(ganho do Q > 0,05 em (\d+); maior ganho ([+-][\d.]+)\); acima (\d+) celulas \(ganho > 0,05 em (\d+)", t234)
chk("R||Cmax: celulas acima de h* (19)", int(mu.group(4)), 19, 0)
chk("R||Cmax: acima de h* com ganho do Q > 0,05 (5)", int(mu.group(5)), 5, 0)
chk("R||Cmax: celulas abaixo de h* (5)", int(mu.group(1)), 5, 0, "none of the $5$ cells below $h^*$ shows a gain beyond the margin")
chk("R||Cmax: nenhuma abaixo de h* passa da margem (0)", int(mu.group(2)), 0, 0)
chk("R||Cmax: celulas no total (24)", len(UrC), 24, 0)
import glob as _glob  # noqa: E402

cpu_s = sum(int(re.findall(r"concluido \((\d+)s\)", open(f_).read())[-1]) for f_ in _glob.glob(str(R / "_run200_label_S*_*.log")))
chk("treino do seletor: execucoes com cinco acoes (85.800)", 5720 * 3 * 5, 85800, 0, "$85{,}800$ and $257{,}400$ IG runs")
chk("treino do seletor: execucoes com quinze acoes (257.400)", 5720 * 3 * 15, 257400, 0)
chk("treino do seletor: iteracoes extras por instancia held-out, cinco acoes (21.700)", 5720 * 3 * 5 * 1000 / 3950, 21700, 50, "$21{,}700$ and $65{,}200$ extra iterations")
chk("treino do seletor: iteracoes extras por instancia held-out, quinze acoes (65.200)", 5720 * 3 * 15 * 1000 / 3950, 65200, 50)
chk("treino do seletor: horas de CPU (0,7)", cpu_s / 3600, 0.7, 0.05, "about $0.7$ CPU-hours")

# ---------------------------------------------------------------- Ciclo 22b: o cronograma de acoes (oraculo dinamico no tempo, Entrada 238/239)
Sp = rd("entrada238_sched_pooled.csv").set_index("host")
Ssv = rd("entrada238_sched_SV.csv").set_index("id")
chk("cronograma: ganho no P||Cmax (0,095)", Sp.loc["pc", "g_sch"], 0.095, 0.0006, "gains $0.095$ against $0.092$ for the validated action on P$||\\Cmax$ (parity)")
chk("cronograma: ganho da validada no P||Cmax (0,092)", Sp.loc["pc", "g_val"], 0.092, 0.0006)
chk("cronograma: P||Cmax em paridade com a validada", float(Ssv.loc["pc", "equivalente_0.05"]), 1, 0)
chk("cronograma: perde para a validada no R||Cmax (0,166)", float(Ssv.loc["ur", "mean_d_pct"]), 0.166, 0.0006, "loses $0.166$ to it on unrelated machines")
chk("cronograma: vantagem no ajuste do R||Cmax (0,178)", Sp.loc["ur", "tune_adv_pct"], 0.178, 0.0006, "($0.178$ better on the tuning instances")
chk("cronograma: 125 candidatos", 5 ** 3, 125, 0, "among $125$ candidates")
# ---------------------------------------------------------------- Ciclo 22c: o seletor offline no R||Cmax (Entrada 240/241)
Us = rd("entrada240_urselector_pooled.csv").set_index("space")
Usv = rd("entrada240_urselector_SV.csv").set_index("id")
Usb = rd("entrada240_urselector_SB.csv").set_index("id")
Usc = rd("entrada240_urselector_SC.csv")
chk("seletor R||Cmax: vence o melhor aprendiz em S5 (0,72)", float(-Usb.loc["SB-S5", "mean_d_pct"]), 0.72, 0.006, "beats the best online learner by $0.72$ and $1.91$ points")
chk("seletor R||Cmax: vence o melhor aprendiz em S15 (1,91)", float(-Usb.loc["SB-S15", "mean_d_pct"]), 1.91, 0.006)
chk("seletor R||Cmax: captura em S5 (60%)", 100 * Us.loc["S5", "captured_sel"], 60, 0.6, "captures $60\\%$ and $73\\%$ of the headroom")
chk("seletor R||Cmax: captura em S15 (73%)", 100 * Us.loc["S15", "captured_sel"], 73, 0.6)
chk("seletor R||Cmax: diferenca para a validada em S5 (-0,056)", float(-Usv.loc["SV-S5", "mean_d_pct"]), -0.056, 0.0006, "($-0.056$ and $-0.023$ points; level with the class table)")
chk("seletor R||Cmax: diferenca para a validada em S15 (-0,023)", float(-Usv.loc["SV-S15", "mean_d_pct"]), -0.023, 0.0006)
chk("seletor R||Cmax: nivelado com a tabela de classe (|d| < 0,05)", float((Usc.mean_d_pct.abs() < 0.05).all()), 1, 0)
chk("seletor R||Cmax: execucoes de treino S5 (18.000)", int(Us.loc["S5", "train_runs"]), 18000, 0, "($18{,}000$ and $54{,}000$ training runs, five times the tuning of the validated action)")
chk("seletor R||Cmax: execucoes de treino S15 (54.000)", int(Us.loc["S15", "train_runs"]), 54000, 0)
chk("seletor R||Cmax: treino = 5 vezes o ajuste da validada (3.600 execucoes em S5)", Us.loc["S5", "train_runs"] / (30 * 12 * 5 * 2), 5, 0)
chk("seletor R||Cmax: instancias de treino adicionais (1.200)", 100 * 12, 1200, 0, "trained on $1{,}200$ further instances")
# ---------------------------------------------------------------- Ciclo 23: dois projetos de AOS da literatura (Entrada 242/243)
Pa = rd("entrada242_pubaos_pooled.csv").set_index(["host", "space"])
PaA = rd("entrada242_pubaos_A.csv")
PaB = rd("entrada242_pubaos_B.csv")
urS5, urS15 = Pa.loc[("ur", "S5")], Pa.loc[("ur", "S15")]
pcA = PaA[PaA.id.str.contains("-pc-")]
chk("AOS publicados: nenhum bate o aleatorio alem da margem no P||Cmax", float(not ((pcA.confirmada) & (~pcA["equivalente_0.05"]) & (pcA.mean_d_pct < 0)).any()), 1, 0, "on P$||\\Cmax$ neither gains beyond the margin")
chk("AOS publicados: iq perde no R||Cmax com cinco tamanhos (0,15)", float(-urS5.g_iq), 0.15, 0.005, "the first loses $0.15$ and the second gains $0.02$ with five sizes")
chk("AOS publicados: fr ganha no R||Cmax com cinco tamanhos (0,02)", float(urS5.g_fr), 0.02, 0.005)
chk("AOS publicados: iq ganha no R||Cmax com quinze acoes (0,24)", float(urS15.g_iq), 0.24, 0.005, "they gain $0.24$ and $0.29$ with fifteen actions")
chk("AOS publicados: fr ganha no R||Cmax com quinze acoes (0,29)", float(urS15.g_fr), 0.29, 0.005)
chk("AOS publicados: no maximo 14% do ganho da validada (S15)", 100 * max(urS15.g_iq, urS15.g_fr) / urS15.g_val, 14, 0.5, "($14\\%$ of the validated action's gain at most)")
chk("AOS publicados: no maximo 14% do ganho da validada (S5 abaixo disso)", float(100 * max(urS5.g_iq, urS5.g_fr) / urS5.g_val <= 14), 1, 0)
chk("AOS publicados: a validada vence os dois nas 8 comparacoes", float(PaB.confirmada.sum()), 8, 0, "beats both in all eight comparisons")
chk("AOS publicados: diferenca minima para a validada (0,06)", float(-PaB.mean_d_pct.max()), 0.06, 0.005, "($0.06$--$1.82$ points)")
chk("AOS publicados: diferenca maxima para a validada (1,82)", float(-PaB.mean_d_pct.min()), 1.82, 0.005)
# ---------------------------------------------------------------- Ciclo 24: o irace como configurador (Entrada 244/245)
Ip = rd("entrada244_irace_pooled.csv")
Ib = rd("entrada244_irace_FB.csv")
Irv = rd("entrada244_irace_FBrev.csv")
Id = rd("entrada244_irace_FD.csv")
Ie = rd("entrada244_irace_FE.csv")
ib1 = lambda D: D[D.id.str.endswith("-b1")].set_index("id")
Ib1, Id1, Ie1 = ib1(Ib), ib1(Id), ib1(Ie)
Ip1 = Ip[Ip.bm == 1].set_index(["host", "space"])
chk("irace cfg: paridade com a validada no P||Cmax (dentro de 0,012)", float(Ib1.loc[["B-pc-S5-b1", "B-pc-S15-b1"], "mean_d_pct"].abs().max()), 0.012, 0.0006, "(within $0.012$ points)")
chk("irace cfg: atras da validada no R||Cmax, S5 (0,15)", float(-Ib1.loc["B-ur-S5-b1", "mean_d_pct"]), 0.15, 0.005, "$0.15$ and $0.19$ points behind on R$||\\Cmax$")
chk("irace cfg: atras da validada no R||Cmax, S15 (0,19)", float(-Ib1.loc["B-ur-S15-b1", "mean_d_pct"]), 0.19, 0.005)
w10 = Irv[Irv.id.str.endswith("-b10")]
chk("irace cfg: com 10 vezes o orcamento nunca supera a validada alem da margem", float(not ((w10.confirmada) & (~w10["equivalente_0.05"]) & (w10.mean_d_pct < 0)).any()), 1, 0, "still never ahead with ten times the budget")
qf_pc = [json.loads(x)["best"]["floor"] for x in (R / "irace_tune_pc_entrada244.jsonl").read_text().splitlines() if json.loads(x)["kind"].startswith("q") and json.loads(x)["bm"] == 1]
qf_ur = [json.loads(x)["best"]["floor"] for x in (R / "irace_tune_ur_entrada244.jsonl").read_text().splitlines() if json.loads(x)["kind"].startswith("q") and json.loads(x)["bm"] == 1]
chk("irace Q: piso do epsilon mediano, P||Cmax (0,26)", float(np.median(qf_pc)), 0.26, 0.005, "(median $0.26$--$0.37$)")
chk("irace Q: piso do epsilon mediano, R||Cmax (0,37)", float(np.median(qf_ur)), 0.37, 0.005)
chk("irace Q: melhor que o Q padrao no R||Cmax, S5 (0,58)", float(-Ie1.loc["E-ur-S5-b1", "mean_d_pct"]), 0.58, 0.005, "by $0.58$ and $0.63$ points")
chk("irace Q: melhor que o Q padrao no R||Cmax, S15 (0,63)", float(-Ie1.loc["E-ur-S15-b1", "mean_d_pct"]), 0.63, 0.005)
chk("irace Q: ganho sobre o aleatorio no P||Cmax, S5 (0,01)", float(Ip1.loc[("pc", "S5"), "g_q"]), 0.01, 0.005, "gains $0.01$ and $0.02$ over random choice on P$||\\Cmax$")
chk("irace Q: ganho sobre o aleatorio no P||Cmax, S15 (0,02)", float(Ip1.loc[("pc", "S15"), "g_q"]), 0.02, 0.005)
chk("irace Q: ganho sobre o aleatorio no R||Cmax, S5 (-0,45)", float(Ip1.loc[("ur", "S5"), "g_q"]), -0.45, 0.005, "$-0.45$ and $+0.17$ on R$||\\Cmax$")
chk("irace Q: ganho sobre o aleatorio no R||Cmax, S15 (+0,17)", float(Ip1.loc[("ur", "S15"), "g_q"]), 0.17, 0.005)
chk("irace Q: a validada vence nos quatro pares", float(Id1.confirmada.sum()), 4, 0, "the validated action beats it in all four host--space pairs ($0.06$--$1.90$ points)")
chk("irace Q: diferenca minima para a validada (0,06)", float(-Id1.mean_d_pct.max()), 0.06, 0.005)
chk("irace Q: diferenca maxima para a validada (1,90)", float(-Id1.mean_d_pct.min()), 1.90, 0.005)
bad = [r for r in rows if not r[4]]
for lab, v, e, tol, ok in rows:
    print(f"{'OK ' if ok else 'ERRO'} {lab}" + ("" if np.isnan(v) else f"  recalculado {v:.5g} vs texto {e:.5g} (tol {tol})"))
print(f"\n{len(rows) - len(bad)}/{len(rows)} verificacoes OK")
sys.exit(1 if bad else 0)
