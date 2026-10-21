"""
analysis/audit_entrada171_claims.py (Entrada 171)
Auditoria numerica das alegacoes do manuscrito que vem do Ciclo 14, frente H (implementacao compilada e contagem de avaliacoes): recalcula dos CSV e compara com o valor literal do texto.
Uso: python -m analysis.audit_entrada171_claims
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
rows = []


def chk(label, value, expected, tol):
    rows.append((label, float(value), float(expected), tol, abs(float(value) - float(expected)) <= tol))


P = pd.read_csv(R / "entrada170_numba_primary.csv").set_index("id")
K = pd.read_csv(R / "entrada170_numba_k.csv")
txt = (R / "entrada170_numba.txt").read_text(encoding="utf-8")
ratio = json.loads((R / "numba_cpu_ratio_entrada170.json").read_text())

# ---- texto do artigo (exp_ident, tabela f38, resumo)
chk("C1 d (0,115)", -P.loc["C1", "mean_d_pct"], 0.115, 0.0006)
chk("C1 IC lo (0,104)", -P.loc["C1", "ci90_hi"], 0.104, 0.0006)
chk("C1 IC hi (0,126)", -P.loc["C1", "ci90_lo"], 0.126, 0.0006)
chk("C1 vitorias (348)", P.loc["C1", "better"], 348, 0)
chk("C1 derrotas (13)", P.loc["C1", "worse"], 13, 0)
chk("C2 d (0,080)", -P.loc["C2", "mean_d_pct"], 0.080, 0.0006)
chk("C3 d (-0,006)", P.loc["C3", "mean_d_pct"], -0.006, 0.0006)
chk("C3 IC lo (-0,012)", P.loc["C3", "ci90_lo"], -0.012, 0.0006)
chk("C3 IC hi (+0,001)", P.loc["C3", "ci90_hi"], 0.001, 0.0006)
chk("C4 d (+0,018)", P.loc["C4", "mean_d_pct"], 0.018, 0.0006)
chk("C4 vitorias (21)", P.loc["C4", "better"], 21, 0)
chk("C4 derrotas (71)", P.loc["C4", "worse"], 71, 0)
b_cpu = float(re.search(r"B_cpu = (\d\.\d+)", txt).group(1))
b_ev = float(re.search(r"B_ev = (\d\.\d+)", txt).group(1))
chk("B_cpu (28%)", 100 * b_cpu, 28, 0.6)
chk("B_ev (50%)", 100 * b_ev, 50, 0.6)
kh, kr = K[K.cenario == "H"], K[K.cenario == "R"]
chk("K_cpu H min (1,7)", kh.K_cpu.min(), 1.7, 0.05)
chk("K_cpu H max (2,4)", kh.K_cpu.max(), 2.4, 0.05)
chk("K_cpu R min (1,7)", kr.K_cpu.min(), 1.7, 0.05)
chk("K_cpu R max (6,3)", kr.K_cpu.max(), 6.3, 0.05)
chk("K_ev H min (1,9)", kh.K_ev.min(), 1.9, 0.05)
chk("K_ev H max (9,1)", kh.K_ev.max(), 9.1, 0.05)
chk("K_cpu H media geometrica (2,0)", float(np.exp(np.log(kh.K_cpu).mean())), 2.0, 0.05)
chk("Spearman K_cpu vs K_ev (0,71)", sstats.spearmanr(K.K_cpu, K.K_ev).statistic, 0.71, 0.006)
chk("n classes-cenarios (24)", len(K), 24, 0)
sens = {m: float(re.search(rf"H\s+{m}\s+(-?\d\.\d+)", txt).group(1)) for m in ("ig_eqev_w0.25", "ig_eqev_w4")}
chk("sens w=0,25 H (-0,026)", sens["ig_eqev_w0.25"], -0.026, 0.0006)
chk("sens w=4 H (-0,124)", sens["ig_eqev_w4"], -0.124, 0.0006)
# equivalencia compilado vs Python: |diferenca| < 0,01 nas quatro verificacoes (legenda da tabela f38)
for line in txt.splitlines():
    m = re.match(r"(V[1-4])\s+.*?\s+1200\s+(-?[\d.e-]+)\s", line)
    if m:
        chk(f"{m.group(1)} |diferenca| < 0,01", abs(float(m.group(2))), 0.0, 0.01)
n_v = sum(1 for r in rows if r[0].startswith("V") and "diferenca" in r[0])
chk("quatro verificacoes de equivalencia", n_v, 4, 0)
# 403 testes unitarios do numba
out = subprocess.run([sys.executable, "-m", "pytest", "tests/test_ig_numba.py", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
chk("testes de ig_numba (403)", int(re.search(r"(\d+) passed", out).group(1)), 403, 0)
# razoes de CPU da Entrada 162 e 129 citadas no metodo (9,6 e 1,8) e a compilada (2,0)
k129 = json.loads((R / "igls_eq_cpu_ratio_entrada129.json").read_text())
k162 = json.loads((R / "igls_fast_cpu_ratio_entrada162.json").read_text())
gm = lambda d: float(np.exp(np.mean([np.log(v["K"]) for v in d.values()])))
chk("K loop (9,6)", gm(k129), 9.6, 0.05)
chk("K vetorizada (1,8)", gm(k162["H"]), 1.8, 0.05)
chk("K compilada (2,0)", gm(ratio["H"]), 2.0, 0.05)

# ---- auditoria de literatura (rodadas 1 e 2, amostra agrupada; Entrada 172): numeros citados em related.tex e s_litaudit.tex
pool = pd.read_csv(R / "litaudit_pooled_entrada172.csv", dtype=str, keep_default_na=False)
cn = lambda col, v: int((pool[col] == v).sum())
chk("lit: artigos agrupados (23)", len(pool), 23, 0)
chk("lit: controle aleatorio (10)", cn("random_control", "Y"), 10, 0)
chk("lit: variante sem a politica (15)", cn("ablation_policy", "Y"), 15, 0)
chk("lit: limite de tempo comum (6)", cn("budget_basis", "CPU"), 6, 0)
chk("lit: concorrente estatico offline (3)", cn("offline_tuned_static", "Y"), 3, 0)
chk("lit: separacao ajuste/teste (6)", cn("tune_test_split", "Y"), 6, 0)
chk("lit: testam (10)", cn("stat_test", "Y"), 10, 0)
chk("lit: multiplicidade (2)", cn("multiplicity", "Y"), 2, 0)
chk("lit: concluem que ajuda (22)", cn("claim_policy_helps", "Y") + cn("claim_policy_helps", "MIXED"), 22, 0)
chk("lit: evidencia mista (5)", cn("claim_policy_helps", "MIXED"), 5, 0)
chk("lit: rodada 2 (11)", int((pool["round"] == "2").sum()), 11, 0)

# ---- frente I (Entrada 173-175): DQN maior e espaco conjunto (exp_unrelated.tex, f42_dqn.tex)
D = pd.read_csv(R / "entrada173_dqn_primary.csv").set_index("id")
cap = pd.read_csv(R / "entrada173_dqn_capture.csv")
hd = pd.read_csv(R / "entrada173_dqn_headroom.csv").set_index("ambiente")
chk("DQN R: ganho sobre aleatorio (0,025)", -D.loc["I1R", "mean_d_pct"], 0.025, 0.0006)
chk("DQN R IC lo (-0,100)", D.loc["I1R", "ci90_lo"], -0.100, 0.0006)
chk("DQN R IC hi (+0,047)", D.loc["I1R", "ci90_hi"], 0.047, 0.0006)
chk("DQN P5 d (+0,006)", D.loc["I1P5", "mean_d_pct"], 0.006, 0.0006)
chk("DQN P15 d (+0,023)", D.loc["I1P15", "mean_d_pct"], 0.023, 0.0006)
chk("DQN P15 p_holm (0,006)", D.loc["I1P15", "p_holm"], 0.006, 0.0006)
chk("validado vs DQN R (0,169)", -D.loc["I2R", "mean_d_pct"], 0.169, 0.0006)
chk("validado vs DQN P5 (0,094)", -D.loc["I2P5", "mean_d_pct"], 0.094, 0.0006)
chk("validado vs DQN P15 (0,094)", -D.loc["I2P15", "mean_d_pct"], 0.094, 0.0006)
chk("p_holm I2R <= 2e-16", D.loc["I2R", "p_holm"] * 1e16, 1.1, 0.9)
chk("captura DQN R (0,12)", cap[(cap.ambiente == "R") & (cap.aprendiz == "dqn")].captura.iloc[0], 0.12, 0.006)
chk("folga P15 (0,11)", -hd.loc["P15", "folga_media"], 0.11, 0.006)
chk("folga P5 (0,13)", -hd.loc["P5", "folga_media"], 0.13, 0.006)
chk("Spearman R (+0,16)", hd.loc["R", "spearman_folga_ganho_dqn"], 0.16, 0.006)
chk("Spearman P5 (-0,43)", hd.loc["P5", "spearman_folga_ganho_dqn"], -0.43, 0.006)
chk("Spearman P15 (-0,52)", hd.loc["P15", "spearman_folga_ganho_dqn"], -0.52, 0.006)
chk("CPU DQN R (0,8)", hd.loc["R", "razao_cpu_dqn_aleatorio"], 0.8, 0.05)
chk("CPU DQN P5 (1,8)", hd.loc["P5", "razao_cpu_dqn_aleatorio"], 1.8, 0.05)
chk("CPU DQN P15 (2,6)", hd.loc["P15", "razao_cpu_dqn_aleatorio"], 2.6, 0.05)
t15 = (R / "entrada173_dqn.txt").read_text(encoding="utf-8")
chk("P15 ql15 vs aleatorio (+0,013)", float(re.search(r"P15: ql15 vs rand\s+(-?\d\.\d+)", t15).group(1)), 0.013, 0.0006)
chk("P15 linucb vs aleatorio (+0,004)", float(re.search(r"P15: linucb15 vs rand\s+(-?\d\.\d+)", t15).group(1)), 0.004, 0.0006)
chk("P15 val vs ql15 (-0,083 -> 0,075-0,094)", -float(re.search(r"P15: val vs ql15\s+(-?\d\.\d+)", t15).group(1)), 0.083, 0.0106)
chk("P15 val vs linucb (0,075)", -float(re.search(r"P15: val vs linucb15\s+(-?\d\.\d+)", t15).group(1)), 0.075, 0.0006)
chk("DQN grande vs pequeno (-0,018)", float(re.search(r"nq_d pequeno: d = ([+-]\d\.\d+)", t15).group(1)), -0.018, 0.0006)

# ---- frente I2 (Entrada 176-177): aprendizes maiores sobre os quatro operadores do ILS (exp_unrelated.tex)
K = pd.read_csv(R / "entrada176_ilsdqn_primary.csv").set_index("id")
t176 = (R / "entrada176_ilsdqn.txt").read_text(encoding="utf-8")
chk("ILS DQN vs aleatorio (+0,041)", K.loc["K1", "mean_d_pct"], 0.041, 0.0006)
chk("ILS LinUCB vs aleatorio (+0,066)", K.loc["K2", "mean_d_pct"], 0.066, 0.0006)
chk("ILS DQN p_holm (4e-4)", K.loc["K1", "p_holm"] * 1e4, 4.2, 0.6)
chk("ILS LinUCB p_holm (1e-6)", K.loc["K2", "p_holm"] * 1e6, 1.2, 0.6)
chk("ILS validado vs aleatorio (+0,062)", float(re.search(r"val_ils vs aleatorio \(operador unico validado\)\s+(-?\d\.\d+)", t176).group(1)), 0.062, 0.0006)
chk("ILS DQN vs validado (-0,021)", float(re.search(r"dqn_ils vs val_ils\s+(-?\d\.\d+)", t176).group(1)), -0.021, 0.0006)
chk("ILS DQN vs validado p (0,063)", float(re.search(r"dqn_ils vs val_ils\s+(?:\S+\s+){5}(\d\.\d+)", t176).group(1)), 0.063, 0.0006)
chk("ILS Q tabular vs aleatorio, 780 (+0,044)", float(re.search(r"Q-learning tabular vs aleatorio ([+-]\d\.\d+)%", t176).group(1)), 0.044, 0.0006)
chk("ILS oraculo de operadores unicos (+0,038)", float(re.search(r"quatro operadores unicos vs aleatorio ([+-]\d\.\d+)%", t176).group(1)), 0.038, 0.0006)
chk("ILS CPU dqn (3,3)", float(re.search(r"dqn ([\d.]+), linucb", t176).group(1)), 3.3, 0.05)
chk("ILS CPU linucb (1,6)", float(re.search(r"linucb ([\d.]+)\s*$", t176.strip()[:t176.strip().rfind("Leitura")].strip()).group(1)), 1.6, 0.05)
cl176 = json.loads((R / "ilsdqn_choice_entrada176.json").read_text())
chk("ILS operador validado sempre a inter-troca (12 classes com 0)", sum(1 for v in cl176["val"].values() if v == 0), 12, 0)

# ---- frente J2 (Entrada 178-179): IG vs ILS vetorizado a CPU igual, tempos finos e grossos (supplementary_material.tex, s_eqcpu_fast.tex)
E = pd.read_csv(R / "entrada178_eqcpu_fast_primary.csv").set_index("id")
chk("eqcpu fino ql (+0,21)", E.loc["E1", "mean_d_pct"], 0.21, 0.005)
chk("eqcpu fino random (+0,22)", E.loc["E2", "mean_d_pct"], 0.22, 0.005)
chk("eqcpu grosso ql (-0,65)", E.loc["E3", "mean_d_pct"], -0.65, 0.005)
chk("eqcpu grosso random (-0,59)", E.loc["E4", "mean_d_pct"], -0.59, 0.005)
chk("eqcpu p_holm maximo fino (<1e-14)", max(E.loc["E1", "p_holm"], E.loc["E2", "p_holm"]) * 1e14, 0.62, 0.38)
chk("eqcpu p_holm maximo grosso (<1e-9)", max(E.loc["E3", "p_holm"], E.loc["E4", "p_holm"]) * 1e9, 0.51, 0.49)
chk("eqcpu T_ILS maximo (2.222: 2,2 vezes)", E.T_ils_max.max() / 1000, 2.2, 0.05)

# ---- frente L/M (Entrada 180-181): agentes compilados e espacos de ate' 30 acoes (exp_unrelated.tex, f43_agents.tex)
AI = pd.read_csv(R / "entrada180_agents_aiter.csv").set_index("id")
AC = pd.read_csv(R / "entrada180_agents_acpu.csv").set_index("id")
BB = pd.read_csv(R / "entrada180_agents_b.csv").set_index("id")
KA = pd.read_csv(R / "entrada180_agents_k.csv")
ta = (R / "entrada180_agents.txt").read_text(encoding="utf-8")
vv = [float(x) for x in re.findall(r"^V\d\s+S\d \w+: compilado vs Python\s+\d+\s+(-?\d\.\d+)", ta, flags=re.M)]
chk("agentes: cinco verificacoes (diferenca minima -0,003)", min(vv), -0.003, 0.0006)
chk("agentes: cinco verificacoes (diferenca maxima +0,012)", max(vv), 0.012, 0.0006)
chk("agentes: numero de verificacoes (5)", len(vv), 5, 0)
chk("agentes: maior ganho a iteracoes iguais (0,016)", -AI.mean_d_pct.min(), 0.016, 0.0006)
chk("agentes: Q S30 iteracoes iguais (+0,017)", AI.loc["Aiter-S30-ql", "mean_d_pct"], 0.017, 0.0006)
chk("agentes: Q S5 CPU igual (-0,059)", AC.loc["Acpu-S5-ql", "mean_d_pct"], -0.059, 0.0006)
chk("agentes: Q S15 CPU igual (-0,049)", AC.loc["Acpu-S15-ql", "mean_d_pct"], -0.049, 0.0006)
vb = BB.loc[[f"B-{sp}-{k}" for sp in ("S5", "S15") for k in ("ql", "ucb", "pg")], "mean_d_pct"]
chk("agentes: validado vs aprendiz S5/S15, minimo (0,076)", -vb.max(), 0.076, 0.0006)
chk("agentes: validado vs aprendiz S5/S15, maximo (0,095)", -vb.min(), 0.095, 0.0006)
chk("agentes: p_holm validado S5/S15 < 1e-58", BB.loc[[f"B-{sp}-{k}" for sp in ("S5", "S15") for k in ("ql", "ucb", "pg")], "p_holm"].max() * 1e58, 0.9, 0.1)
vo = BB.loc[[f"B-{sp}-{k}" for sp in ("S3", "S30") for k in ("ql", "ucb", "pg")], "mean_d_pct"].abs()
chk("agentes: validado vs aprendiz em S3/S30, maximo (<0,03)", vo.max(), 0.0285, 0.0006)
sob = KA[KA.learner.isin(["ucb", "pg"])].sobrecusto_por_avaliacao
chk("agentes: sobrecusto ucb e pg minimo (1,0)", sob.min(), 1.0, 0.01)
chk("agentes: sobrecusto ucb e pg maximo (1,09)", sob.max(), 1.09, 0.006)
sq = KA[KA.learner == "ql"].sobrecusto_por_avaliacao
chk("agentes: sobrecusto Q minimo (1,07)", sq.min(), 1.07, 0.006)
chk("agentes: sobrecusto Q maximo (1,33)", sq.max(), 1.33, 0.006)
kq = KA[(KA.learner == "ql") & KA.space.isin(["S5", "S15", "S30"])].K_cpu
chk("agentes: K_CPU Q grandes minimo (0,50)", kq.min(), 0.50, 0.006)
chk("agentes: K_CPU Q grandes maximo (0,60)", kq.max(), 0.60, 0.006)
ke = KA[(KA.learner == "ql") & KA.space.isin(["S5", "S15", "S30"])].K_avaliacoes
chk("agentes: avaliacoes Q minimo (0,38)", ke.min(), 0.38, 0.006)
chk("agentes: avaliacoes Q maximo (0,50)", ke.max(), 0.50, 0.006)
hh = {sp: -float(v) for sp, v in re.findall(r"^\s*(S\d+)\s+(-?\d\.\d+)\s*$", ta.split("folga: oraculo")[1].split("S30: fracao")[0], flags=re.M)}
chk("agentes: folga S30 (0,05)", hh["S30"], 0.05, 0.006)
chk("agentes: folga S5 e S15 minimo (0,12)", min(hh["S5"], hh["S15"]), 0.12, 0.006)
chk("agentes: folga S5 e S15 maximo (0,13)", max(hh["S5"], hh["S15"]), 0.13, 0.006)
chk("agentes: Spearman 48 celulas (0,09)", float(re.search(r"em 48 celulas: rho = (-?\d\.\d+)", ta).group(1)), 0.09, 0.006)

bad = [r for r in rows if not r[4]]
for lab, v, e, t, ok in rows:
    print(f"{lab:62s} {v:12.4f} {e:12.4f} {t:7.4f} {ok}")
print(f"\n{len(rows) - len(bad)}/{len(rows)} OK; divergentes: {[r[0] for r in bad]}")
sys.exit(1 if bad else 0)
