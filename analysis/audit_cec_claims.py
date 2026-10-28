"""
analysis/audit_cec_claims.py (Entrada 182)
Auditoria numerica do artigo do CEC (cec/cec2027_paper.tex): recalcula dos CSV/TXT de resultados cada numero citado na prosa e confere (a) o valor com o literal do texto
(tolerancia de arredondamento) e (b) que o literal esta' de fato no texto do artigo. Numeros herdados do manuscrito do periodico (auditado por audit_short_numbers.py e
audit_entrada171_claims.py) sao conferidos por presenca nas fontes do periodico. As tabelas do CEC sao geradas por analysis/make_table_cec.py (nao tem numeros digitados).
Uso: python -m analysis.audit_cec_claims   (sai com codigo 1 se algo falhar)
"""
from __future__ import annotations

import glob
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEX = re.sub(r"\s+", " ", (ROOT / "cec" / "cec2027_paper.tex").read_text(encoding="utf-8"))
JOURNAL = "\n".join(re.sub(r"\s+", " ", Path(f).read_text(encoding="utf-8", errors="ignore")) for f in sorted(glob.glob(str(ROOT / "paper" / "*.tex")) + glob.glob(str(ROOT / "paper" / "short" / "*.tex"))))
rows = []


def chk(label, value, expected, tol, snippet=None):
    """Valor recalculado ~ literal; `snippet` (opcional) deve aparecer no texto do CEC."""
    ok = abs(float(value) - float(expected)) <= tol
    if snippet is not None:
        ok = ok and (snippet in TEX)
    rows.append((label, float(value), float(expected), tol, ok))


def present(label, snippet):
    rows.append((label, float("nan"), float("nan"), 0, snippet in TEX))


def journal(label, snippet):
    """Numero herdado do periodico: o literal tem de estar no texto do CEC e nas fontes do periodico."""
    rows.append((label, float("nan"), float("nan"), 0, snippet in TEX and snippet in JOURNAL))


rd = lambda f: pd.read_csv(R / f).set_index("id")

# ---------------------------------------------------------------- resumo / introducao / resultados A (mecanismo)
P162 = rd("entrada162_ils_fast_primary.csv")
chk("E1 IG vs ILS CPU igual (0,46%)", -P162.loc["E1", "mean_d_pct"], 0.46, 0.005, "$0.46\\%$")
chk("E1 vitorias (557)", P162.loc["E1", "better"], 557, 0, None)
chk("E1 derrotas (87)", P162.loc["E1", "worse"], 87, 0)
chk("E2 IG vs ILS iteracoes iguais (0,50%)", -P162.loc["E2", "mean_d_pct"], 0.50, 0.005, None)
chk("E1 p_holm < 1e-80", float(P162.loc["E1", "p_holm"] < 1e-80), 1, 0, "p_{\\mathrm{Holm}}<10^{-80}")
chk("E2 p_holm < 1e-80", float(P162.loc["E2", "p_holm"] < 1e-80), 1, 0)
t162 = (R / "entrada162_ils_fast.txt").read_text(encoding="utf-8")
chk("ILS vetorizado 6,2x mais rapido", float(re.search(r"razao de CPU padrao/rapido (\d\.\d+)", t162).group(1)), 6.2, 0.05, "$6.2$ times faster")
kils = [float(v) for v in re.findall(r"^\s*\d+x\d+\s+(\d\.\d+)\s+\d+\s*$", t162.split("--- calibracao ---")[1].split("razao de CPU realizada")[0], flags=re.M)]
chk("K_ils minimo (0,4)", min(kils), 0.4, 0.05, "$0.4$ to $1.9$")
chk("K_ils maximo (1,9)", max(kils), 1.9, 0.05)
chk("12 classes de K_ils", len(kils), 12, 0)
cls = []
for ln in t162.split("por classe")[1].splitlines():
    m = re.match(r"\s*rand_(\d+)j_(\d+)m\s+(-?\d\.\d+)\s+(-?\d\.\d+)", ln)
    if m:
        cls.append((int(m.group(1)), int(m.group(2)), -float(m.group(3)), -float(m.group(4))))
chk("12 classes na tabela por classe", len(cls), 12, 0)
chk("IG vence o ILS em todas as 12 classes (CPU igual)", sum(c[2] > 0 for c in cls), 12, 0, "in all twelve classes")
chk("menor ganho por classe (40x5: 0,004)", min(c[2] for c in cls), 0.004, 0.0006, None)
chk("maior ganho por classe (60x20: 2,84)", max(c[2] for c in cls), 2.84, 0.005, None)
low = sorted(c[2] for c in cls if c[0] / c[1] <= 5 and (c[0], c[1]) != (20, 4))

seed = (R / "entrada117_seed_variance_all.txt").read_text(encoding="utf-8")
m = re.search(r"S1' politica .*?d = (-?\d\.\d+)% IC90 \[(-?\d\.\d+), (-?\d\.\d+)\]", seed)
chk("seed variance ganho (0,011)", -float(m.group(1)), 0.011, 0.0006, "$0.011\\%$ ($90\\%$ interval $0.003$ to $0.019$")
chk("seed variance IC (0,019)", -float(m.group(2)), 0.019, 0.0006)
chk("seed variance IC (0,003)", -float(m.group(3)), 0.003, 0.0006)
m = re.search(r"S3' ig_random vs ils_random: d = (-?\d\.\d+)%", seed)
chk("IG vs ILS sem aprendizado, seed (0,49)", -float(m.group(1)), 0.49, 0.005, "by $0.49\\%$, against $0.53\\%$")
m = re.search(r"S3' ig_ql vs ils_ql: d = (-?\d\.\d+)%", seed)
chk("IG vs ILS aprendido, seed (0,53)", -float(m.group(1)), 0.53, 0.005, "$0.53\\%$")
d_sa = {}
for ln in seed.splitlines():
    m = re.match(r"\s*rand_(\d+)j_(\d+)m ig_random vs ils_random\s+(-?\d\.\d+)", ln)
    if m:
        d_sa[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
exc = sorted(k for k, v in d_sa.items() if v > -0.01)
chk("10 de 12 classes (excecoes 20x4 e 40x5, |d| < 0,01)", 12 - len(exc), 10, 0, "$10$ of $12$ classes")
chk("excecoes sao 20x4 e 40x5", float(exc == [(20, 4), (40, 5)]), 1, 0)

fv = pd.read_csv(R / "fig_thesis_values_entrada162.csv")
fv = fv[fv.problem.str.startswith("Identical")].gain_pp.astype(float).tolist()
chk("Fig. 1a mecanismo (0,462)", fv[0], 0.462, 0.0006)
chk("Fig. 1a busca local (0,105)", fv[1], 0.105, 0.0006)

# validacao externa (Tabela I do CEC)
H90 = pd.read_csv(R / "entrada90_family_level_holdout.csv")
H90 = H90[(H90.comparator == "ig_random") & (H90.holm_family.str.startswith("P"))].set_index("family")
C115 = rd("entrada115_external_confirm2_primary.csv")
chk("held-out |d| minimo (0,009)", -H90.mean_diff_pp.max(), 0.009, 0.0006, "by only $0.009$ to $0.029$ points")
chk("held-out |d| maximo (0,026)", -H90.mean_diff_pp.min(), 0.026, 0.0006)
chk("held-out 4.160 instancias", H90.n.sum(), 4160, 0, "two other sets of $4{,}160$ and $2{,}690$ published instances")
O = C115.loc[["O_B", "O_C", "O_E"]]
chk("confirmatorio |d| minimo (0,015)", -O.mean_diff_pp.max(), 0.015, 0.0006, None)
chk("confirmatorio |d| maximo (0,029)", -O.mean_diff_pp.min(), 0.029, 0.0006)
chk("confirmatorio 2.690 instancias", O.n.sum(), 2690, 0, "$2{,}690$")
chk("seis testes Holm significativos", int((H90.p_holm < 0.05).sum() + (O.p_holm < 0.05).sum()), 6, 0, "all six Holm-adjusted tests significant")
t115 = (R / "entrada115_external_confirm2.txt").read_text(encoding="utf-8")
m = re.search(r"^\s*ALL\s+\d+\s+\d+\s+\d+\s+\d+\s+(-?\d\.\d+)\s+(-?\d\.\d+)\s+(-?\d\.\d+)\s+(\d\.\d+)\s*$", t115, flags=re.M)
chk("estado informativo, d pooled (-0,006)", float(m.group(1)), -0.006, 0.0006, "$-0.006$ points pooled")
chk("estado informativo, p (0,77)", float(m.group(4)), 0.77, 0.005, "$p=0.77$")

# ---------------------------------------------------------------- resultados B (orcamento)
P124 = rd("entrada124_igls_p_primary.csv")
Q129 = rd("entrada129_igls_eq_primary.csv")
P170 = rd("entrada170_numba_primary.csv")
PD162 = rd("entrada162_igls_fast_primary.csv")
t170 = (R / "entrada170_numba.txt").read_text(encoding="utf-8")
m = re.search(r"Iteracoes iguais \(T = 1000\): H d = (-?\d\.\d+)%", t170)
d_it_c = -float(m.group(1))
d_it_py = -P124.loc["P1", "mean_d_pct"]
chk("P1 descida a iteracoes iguais (0,150)", d_it_py, 0.150, 0.0006, "$0.150\\%$")
chk("Q1 CPU igual, lacos (0,010)", -Q129.loc["Q1", "mean_d_pct"], 0.010, 0.0006, "$0.010\\%$")
chk("Q1 parcela de orcamento (93%)", 100 * (1 - (-Q129.loc["Q1", "mean_d_pct"]) / d_it_py), 93, 0.5, "$93\\%$")
chk("D1 CPU igual, vetorizada (0,105)", -PD162.loc["D1", "mean_d_pct"], 0.105, 0.0006, "$0.105\\%$")
chk("D1 parcela (30%)", 100 * (1 - (-PD162.loc["D1", "mean_d_pct"]) / d_it_py), 30, 0.5, "($30\\%$ budget)")
chk("C1 CPU igual, compilada (0,115)", -P170.loc["C1", "mean_d_pct"], 0.115, 0.0006, "$0.115\\%$")
chk("C1 parcela (28%)", 100 * (1 - (-P170.loc["C1", "mean_d_pct"]) / d_it_c), 28, 0.5, "($28\\%$)")
chk("C2 contagem de avaliacoes (0,080)", -P170.loc["C2", "mean_d_pct"], 0.080, 0.0006, "$0.080\\%$")
chk("C2 parcela (50%)", 100 * (1 - (-P170.loc["C2", "mean_d_pct"]) / d_it_c), 50, 0.5, "$50\\%$ budget")
k129 = pd.read_json(R / "igls_eq_cpu_ratio_entrada129.json", typ="series")
chk("K lacos (9,6)", np.exp(np.mean(np.log([v["K"] for v in k129.values]))), 9.6, 0.05, "($K=9.6$)")
import json

k162 = json.loads((R / "igls_fast_cpu_ratio_entrada162.json").read_text())
chk("K vetorizada (1,8)", np.exp(np.mean(np.log([v["K"] for v in k162["H"].values()]))), 1.8, 0.05, "cuts $K$ to $1.8$")
K170 = pd.read_csv(R / "entrada170_numba_k.csv")
KH = K170[K170.cenario == "H"]
chk("K compilada (2,0)", np.exp(np.mean(np.log(KH.K_cpu))), 2.0, 0.05, "gives $K=2.0$")
w = {(c, mt): float(v) for c, mt, v in re.findall(r"^\s*([HR])\s+(ig_eqev_w[\d.]+)\s+(-?\d\.\d+)", t170, flags=re.M) for mt in [mt]}
chk("peso 0,25: vantagem (0,026)", -w[("H", "ig_eqev_w0.25")], 0.026, 0.0006, None)
chk("peso 4: vantagem (0,124)", -w[("H", "ig_eqev_w4")], 0.124, 0.0006)
chk("D2 receita vetorizada (-0,008)", PD162.loc["D2", "mean_d_pct"], -0.008, 0.0006, "$-0.008\\%$ and $-0.006\\%$")
chk("C3 receita compilada (-0,006)", P170.loc["C3", "mean_d_pct"], -0.006, 0.0006)

# ---------------------------------------------------------------- resultados C (agentes)
Ai, Ac, B = rd("entrada180_agents_aiter.csv"), rd("entrada180_agents_acpu.csv"), rd("entrada180_agents_b.csv")
KA = pd.read_csv(R / "entrada180_agents_k.csv")
ta = (R / "entrada180_agents.txt").read_text(encoding="utf-8")
chk("12 comparacoes a iteracoes iguais", len(Ai), 12, 0)
chk("maior ganho de aprendiz a iteracoes iguais (0,016)", -Ai.mean_d_pct.min(), 0.016, 0.0006, "more than $0.016\\%$")
chk("Q-Learning CPU igual S5 (0,059)", -Ac.loc["Acpu-S5-ql", "mean_d_pct"], 0.059, 0.0006, "$0.059\\%$ and $0.049\\%$")
chk("Q-Learning CPU igual S15 (0,049)", -Ac.loc["Acpu-S15-ql", "mean_d_pct"], 0.049, 0.0006)
nonq = Ac[~Ac.index.str.endswith("-ql")]
chk("UCB1 e gradiente a CPU igual: |d| <= 0,008", nonq.mean_d_pct.abs().max(), 0.008, 0.0006, None)
bsel = B[B.index.str.contains("S5-|S15-")]
A5b_ = pd.read_csv(R / "entrada185_aos_bval.csv")
A8b_ = pd.read_csv(R / "entrada188_new_bval.csv")
b9 = np.concatenate([-bsel.mean_d_pct.to_numpy(), -A5b_.mean_d_pct.to_numpy(), -A8b_.mean_d_pct.to_numpy()])
chk("validada vs nove metodos S5/S15: minimo (0,053)", b9.min(), 0.053, 0.0006, "$0.053$--$0.147$ points")
chk("validada vs nove metodos S5/S15: maximo (0,147)", b9.max(), 0.147, 0.0006)
chk("validada vs nove metodos: p_holm < 1e-57", float(max(bsel.p_holm.max(), A5b_.p_holm.max(), A8b_.p_holm.max()) < 1e-57), 1, 0, "p_{\\mathrm{Holm}}<10^{-57}")
chk("validada vs os 3 aprendizes originais: minimo (0,076; nao citado)", -bsel.mean_d_pct.max(), 0.076, 0.0006)
bout = B[B.index.str.contains("S3-|S30-")]
ku = lambda sp, k: float(KA[(KA.space == sp) & (KA.learner == k)].K_cpu.iloc[0])
ke = lambda sp, k: float(KA[(KA.space == sp) & (KA.learner == k)].K_avaliacoes.iloc[0])
over = KA[KA.learner != "ql"].sobrecusto_por_avaliacao
chk("sobrecusto UCB1/PG minimo (1,0)", over.min(), 1.0, 0.05, None)
chk("sobrecusto UCB1/PG maximo (1,09)", over.max(), 1.09, 0.005)
oq = KA[KA.learner == "ql"].sobrecusto_por_avaliacao
chk("sobrecusto Q minimo (1,07)", oq.min(), 1.07, 0.005, None)
chk("sobrecusto Q maximo (1,33)", oq.max(), 1.33, 0.005, "at most $1.33$ times that of random choice")
kq = [ku(s, "ql") for s in ("S5", "S15", "S30")]
chk("K_CPU do Q em S5/S15/S30: minimo (0,50)", min(kq), 0.50, 0.005, "$0.50$--$0.60$ for Q-Learning in the three larger spaces")
chk("K_CPU do Q em S5/S15/S30: maximo (0,60)", max(kq), 0.60, 0.005)
evq = [ke(s, "ql") for s in ("S5", "S15", "S30")]
hd = {sp: -float(v) for sp, v in re.findall(r"^\s*(S\d+)\s+(-?\d\.\d+)\s*$", ta.split("folga: oraculo")[1].split("S30: fracao")[0], flags=re.M)}
chk("folga S30 (0,05)", hd["S30"], 0.05, 0.005)
chk("folga S5 (0,12-0,13)", hd["S5"], 0.125, 0.0051)
chk("folga S15 (0,12-0,13)", hd["S15"], 0.125, 0.0051)

# 48 celulas (Fig. 2b)
cells = []
for sp in ("S3", "S5", "S15", "S30"):
    d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
    orc = d[d.method.str.startswith("arm")].groupby(["cls", "rep"]).makespan.min().rename("oracle").reset_index()
    w_ = ms.merge(orc, on=["cls", "rep"])
    cells.append(pd.DataFrame({"h": -100 * (w_.oracle - w_.rand) / w_.rand, "gain": -100 * (w_.ql - w_.rand) / w_.rand}).groupby(w_.cls).mean().assign(sp=sp))
c = pd.concat(cells)
chk("48 celulas", len(c), 48, 0)   # a frase 'Across the 48 cells ... rho = 0.09' saiu do artigo (Ciclo 20): a folga enviesada contradizia a Fig. 1
rho = sstats.spearmanr(c.h, c.gain).statistic
chk("Spearman folga x ganho do Q (0,09)", rho, 0.09, 0.005, None)
hi = c[c.h > 0.15]

# ---------------------------------------------------------------- resultados D (aprendizes maiores)
P176, P173, P167 = rd("entrada176_ilsdqn_primary.csv"), rd("entrada173_dqn_primary.csv"), rd("entrada167_online_primary.csv")
t176 = (R / "entrada176_ilsdqn.txt").read_text(encoding="utf-8")
chk("DQN-ILS vs mistura (0,041)", P176.loc["K1", "mean_d_pct"], 0.041, 0.0006, "($0.041\\%$ and $0.066\\%$")
chk("LinUCB-ILS vs mistura (0,066)", P176.loc["K2", "mean_d_pct"], 0.066, 0.0006)
mv = re.search(r"val_ils vs aleatorio \(operador unico validado\)\s+(-?[\d.]+)", t176)
chk("operador validado vs mistura (0,062)", float(mv.group(1)), 0.062, 0.0006, "as does the operator validated per class ($0.062\\%$)")
m = re.search(r"Q-learning tabular vs aleatorio \+(\d\.\d+)%; DQN \+\d\.\d+%; LinUCB \+\d\.\d+%; oraculo por instancia entre os quatro operadores unicos vs aleatorio \+(\d\.\d+)%", t176)
m = re.search(r"medias geometricas: dqn (\d\.\d+), linucb (\d\.\d+)", t176)
chk("DQN em 15 acoes vs aleatorio (+0,023)", P173.loc["I1P15", "mean_d_pct"], 0.023, 0.0006, None)
chk("validada vs DQN, 15 acoes (0,094)", -P173.loc["I2P15", "mean_d_pct"], 0.094, 0.0006, None)
vs = [-P173.loc["I2P5", "mean_d_pct"], -P167.loc["P-G3", "mean_d_pct"], -P167.loc["P-G4", "mean_d_pct"]]


# ---------------------------------------------------------------- RQ4: poder, controle positivo, AOS classicos, tempos finos (Entradas 184-185)
PW = pd.read_csv(R / "entrada184_power.csv")
chk("MDE80 analitico minimo (0,006)", PW.mde80.min(), 0.006, 0.0006, "$0.006$--$0.014$ points")
chk("MDE80 analitico maximo (0,014)", PW.mde80.max(), 0.014, 0.0006)
cl = PW[PW.mean_d.abs() < 0.005]
chk("10 comparacoes com efeito observado desprezivel", len(cl), 10, 0, "the ten comparisons with a negligible observed effect")
chk("poder em 0,02: minimo (58%)", 100 * cl["power_0.02"].min(), 58, 0.5, "$58$--$99.8\\%$ power at $0.02$ points")
chk("poder em 0,02: maximo (99,8%)", 100 * cl["power_0.02"].max(), 99.8, 0.05)
chk("poder em 0,03: minimo >= 97%", 100 * cl["power_0.03"].min(), 97, 0.5, "at least $97\\%$ at $0.03$")
PC = pd.read_csv(R / "entrada184_power_control.csv").set_index("space")
chk("controle positivo real S5 (0,092)", -PC.loc["S5", "mean_d"], 0.092, 0.0006, "$0.092$ and $0.080$ points ($p<10^{-76}$)")
chk("controle positivo real S15 (0,080)", -PC.loc["S15", "mean_d"], 0.080, 0.0006)
chk("controle positivo real: p < 1e-76", float(PC.loc[["S5", "S15"], "p_raw"].max() < 1e-76), 1, 0)
PL = pd.read_csv(R / "entrada185_plant_shares.csv")
m1 = PL[PL["mode"] == 1]
chk("plantio, bonus 0,5: minimo >= 1,5 vez", m1[m1.bonus == 0.5].x_uniforme.min(), 1.5, 0.05, "at least $1.5$ times more often than uniform")
chk("plantio, bonus 5: minimo (2,5)", m1[m1.bonus == 5.0].x_uniforme.min(), 2.5, 0.05, None)
chk("plantio, bonus 5: maximo (13,7)", m1[m1.bonus == 5.0].x_uniforme.max(), 13.7, 0.05)
m2 = PL[(PL["mode"] == 2) & (PL.bonus == 5.0) & (PL.space == "S5")].set_index("learner").x_uniforme

AIi, AIc, AIb, AIk = (pd.read_csv(R / f"entrada185_aos_{x}.csv").set_index("id") if x != "k" else pd.read_csv(R / "entrada185_aos_k.csv") for x in ("aiter", "acpu", "bval", "k"))
chk("ALNS S5 a iteracoes iguais (0,039)", -AIi.loc["Aiter-S5-alns", "mean_d_pct"], 0.039, 0.0006, "$0.039$ points at equal iterations with five sizes")
chk("ALNS S5 efeito de custo (0,029)", AIc.loc["Acpu-S5-alns", "mean_d_pct"] * -1 + AIi.loc["Aiter-S5-alns", "mean_d_pct"], 0.029, 0.0006, None)
chk("ALNS S5 a CPU igual (0,068)", -AIc.loc["Acpu-S5-alns", "mean_d_pct"], 0.068, 0.0006)
chk("ALNS S15 a iteracoes iguais (+0,002)", AIi.loc["Aiter-S15-alns", "mean_d_pct"], 0.002, 0.0006, None)
pmap = AIi[AIi.index.str.contains("-pm|-ap")]
chk("PM e AP a iteracoes iguais: |d| <= 0,01", pmap.mean_d_pct.abs().max(), 0.0095, 0.0006, None)
dd = {}
for sp in ("S5", "S15"):
    di, dc = Ai.loc[f"Aiter-{sp}-ql", "mean_d_pct"], Ac.loc[f"Acpu-{sp}-ql", "mean_d_pct"]
    dd[sp] = (-di, di - dc)
chk("decisao do Q-Learning S5 (-0,003)", dd["S5"][0], -0.003, 0.0006, "its decision effect is nil ($-0.003$ and $-0.004$ points)")
chk("decisao do Q-Learning S15 (-0,004)", dd["S15"][0], -0.004, 0.0006)
chk("custo do Q-Learning S5 (0,062)", dd["S5"][1], 0.062, 0.0006, "its cost effect is $0.062$ and $0.053$")
chk("custo do Q-Learning S15 (0,053)", dd["S15"][1], 0.053, 0.0006)

N8 = pd.read_csv(R / "entrada188_new_aiter.csv").set_index("id")
chk("Thompson e DMAB a iteracoes iguais: |d| maximo (0,013)", N8[N8.index.str.contains("-ts|-dmab")].mean_d_pct.abs().max(), 0.0134, 0.0006, "the others stay within $0.013$ points")
chk("corrida a iteracoes iguais S5 (+0,012)", N8.loc["Aiter-S5-race", "mean_d_pct"], 0.012, 0.0006, "($-0.012$ and $-0.067$ points)")
chk("corrida a iteracoes iguais S15 (+0,067)", N8.loc["Aiter-S15-race", "mean_d_pct"], 0.067, 0.0006)
FI = {x: pd.read_csv(R / f"entrada185_fine_{x}.csv") for x in ("aiter", "acpu", "bval", "k", "headroom")}
FIi, FIc, FIb = (FI[x].set_index("id") for x in ("aiter", "acpu", "bval"))
chk("tempos finos: maior ganho a iteracoes iguais (0,015)", -FIi.mean_d_pct.min(), 0.015, 0.0006, "no learner gains more than $0.015$ points at equal iterations")
FK = FI["k"].set_index(["space", "learner"]).K_cpu
chk("tempos finos: validada vs aprendiz, minimo (0,105)", -FIb.mean_d_pct.max(), 0.105, 0.0006, "$0.105$--$0.136$ points")
chk("tempos finos: validada vs aprendiz, maximo (0,136)", -FIb.mean_d_pct.min(), 0.136, 0.0006)
FH = FI["headroom"].set_index("space")
chk("tempos finos: folga S5 (0,142)", -FH.loc["S5", "folga_fina_pct"], 0.142, 0.0006, "($0.142$ and $0.175$ for one seed)")
chk("tempos finos: folga S15 (0,175)", -FH.loc["S15", "folga_fina_pct"], 0.175, 0.0006)
chk("tempos finos: validada nunca atinge o limite inferior", FH.frac_validada_no_lb_fina.max(), 0.0, 0, None)

LA = pd.read_csv(R / "entrada185_long_aiter.csv")
LB = pd.read_csv(R / "entrada185_long_bval.csv")
LH = pd.read_csv(R / "entrada185_long_headroom.csv").set_index("T")
LA["T"] = LA.id.str.extract(r"T(\d+)").astype(int)
LB["T"] = LB.id.str.extract(r"T(\d+)").astype(int)
mid = LA[LA["T"].isin([5000, 20000])]
chk("horizonte 5000 e 20000: |d| maximo dos aprendizes (0,005)", mid.mean_d_pct.abs().max(), 0.0045, 0.0006, None)
chk("folga do oraculo T = 1000 (0,125)", -LH.loc[1000, "folga_oraculo_pct"], 0.125, 0.0006, "$0.125\\%$ at $T=1{,}000$ to")
chk("folga do oraculo T = 5000 (0,060)", -LH.loc[5000, "folga_oraculo_pct"], 0.060, 0.0006, None)
chk("folga do oraculo T = 20000 (0,036)", -LH.loc[20000, "folga_oraculo_pct"], 0.036, 0.0006)
b5 = -LB[LB["T"] == 5000].mean_d_pct
b20 = -LB[LB["T"] == 20000].mean_d_pct
a100 = LA[LA["T"] == 100000].set_index("id")
chk("T = 100000: |d| maximo dos aprendizes (0,006)", a100.mean_d_pct.abs().max(), 0.006, 0.0009, "the learners stay within $0.006$ points of random choice")
chk("T = 100000: folga do oraculo (0,015)", -LH.loc[100000, "folga_oraculo_pct"], 0.015, 0.0006, "to $0.015\\%$, and the validated action still leads them")
chk("T = 100000: razao da folga (0,12)", LH.loc[100000, "folga_oraculo_pct"] / LH.loc[1000, "folga_oraculo_pct"], 0.12, 0.005)
chk("T = 100000: fracao da validada no limite inferior (92%)", 100 * LH.loc[100000, "frac_validada_no_lb"], 92, 0.5, None)
b100 = -LB[LB["T"] == 100000].mean_d_pct
chk("T = 100000: vantagem da validada, minimo (0,012)", b100.min(), 0.012, 0.0006, "by $0.012$--$0.017$ points at $T=100{,}000$ ($p<10^{-12}$)")
chk("T = 100000: vantagem da validada, maximo (0,017)", b100.max(), 0.017, 0.0006)
chk("T = 100000: p_holm da validada < 1e-12", float(LB[LB["T"] == 100000].p_holm.max() < 1e-12), 1, 0)



# ---------------------------------------------------------------- RQ5: instancias publicadas (Entradas 189-192)
GP = json.loads((R / "pub_groups_entrada189.json").read_text())
GH = json.loads((R / "pubh_groups_entrada190.json").read_text())
chk("instancias de teste do estagio 2 (2.550)", len(GP["test"]), 2550, 0, "$2{,}550$ of the published instances")
chk("excluidas no ajuste (28)", GP["excluded"]["calib"]["B"], 28, 0)
chk("instancias held-out (3.950)", len(GH["test"]), 3950, 0, "$3{,}950$ held-out instances")
chk("fracao excluida (5,1%)", 100 * 350 / 6850, 5.1, 0.05)
PSU = pd.read_csv(R / "entrada192_published_summary.csv")
pick = lambda st, sp, sub: PSU[(PSU.set == st) & (PSU.space == sp) & (PSU.subset == sub)].iloc[0]
PCE = pd.read_csv(R / "entrada192_published_cells.csv")
chk("publicadas: Q-Learning a iteracoes iguais S15 (0,058)", -pick("discovery", "S15", "all").ql_iter, 0.058, 0.0006, "Q-Learning gains $0.058$ points at equal iterations with fifteen actions")
chk("publicadas: roleta a iteracoes iguais S5 (0,057)", -pick("discovery", "S5", "all").alns_iter, 0.057, 0.0006, "the ALNS roulette $0.057$ with five sizes")
chk("publicadas: Q-Learning a CPU igual S5 (0,121)", -pick("discovery", "S5", "all").ql_cpu, 0.121, 0.0006, None)
chk("publicadas: Q-Learning a CPU igual S15 (0,137)", -pick("discovery", "S15", "all").ql_cpu, 0.137, 0.0006)
chk("familia C 1-2: Q-Learning S5 (0,16)", -pick("discovery", "S5", "C12").ql_iter, 0.16, 0.005, "Q-Learning gains $0.16$ and $0.24$ points")
chk("familia C 1-2: Q-Learning S15 (0,24)", -pick("discovery", "S15", "C12").ql_iter, 0.24, 0.005)
rr = [pick("discovery", sp, "C12").ql_iter / pick("discovery", sp, "C12").val_iter for sp in ("S5", "S15")]
chk("Q-Learning recupera de 4/5 a 9/10 do ganho da validada (minimo 0,80)", min(rr), 0.80, 0.02, None)
chk("Q-Learning recupera de 4/5 a 9/10 do ganho da validada (maximo 0,90)", max(rr), 0.90, 0.02)
chk("familia E: Q-Learning perde S5 e S15 (0,06-0,07)", min(pick("discovery", "S5", "E").ql_iter, pick("discovery", "S15", "E").ql_iter), 0.058, 0.0006, "it \\emph{loses} $0.06$--$0.07$.")
dsc = PCE[PCE.set == "discovery"]
rep_ = PCE[PCE.set == "replication"]
for sp_, rho_, p_ in (("S5", 0.37, None), ("S15", 0.57, None)):
    x_ = dsc[dsc.space == sp_]
    chk(f"correlacao descoberta dentro de {sp_} (nove celulas)", sstats.spearmanr(-x_.folga, -x_.ql).statistic, rho_, 0.005)
for sp_, rho_, p_ in (("S5", 0.67, 0.05), ("S15", 0.60, 0.09)):
    x_ = rep_[rep_.space == sp_]
    r_ = sstats.spearmanr(-x_.folga, -x_.ql)
    chk(f"correlacao held-out dentro de {sp_} (nove celulas)", r_.statistic, rho_, 0.005)
    chk(f"p da correlacao held-out dentro de {sp_}", r_.pvalue, p_, 0.006)
H190 = pd.read_csv(R / "entrada190_replication_h.csv").set_index("id")
chk("H1 Q-Learning S15 familia C 1-2 (0,245)", -H190.loc["H1", "mean_d_pct"], 0.245, 0.0006, "gains $0.245$ and $0.149$ points")
chk("H2 Q-Learning S5 familia C 1-2 (0,149)", -H190.loc["H2", "mean_d_pct"], 0.149, 0.0006)
chk("H3b Q-Learning perde em E, S15 (0,052)", H190.loc["H3b", "mean_d_pct"], 0.052, 0.0006, "loses $0.052$ and $0.059$ in family E")
chk("H3a Q-Learning perde em E, S5 (0,059)", H190.loc["H3a", "mean_d_pct"], 0.059, 0.0006)
chk("H4 roleta S5 (0,056)", -H190.loc["H4", "mean_d_pct"], 0.056, 0.0006, "the ALNS roulette gains $0.056$ with five sizes")
chk("as cinco hipoteses confirmatorias confirmadas (5)", int(H190.confirmada.sum()), 5, 0, "All five confirmatory tests held")
B190 = pd.read_csv(R / "entrada190_replication_bval.csv").set_index("id")
chk("a validada vence os 18 pares nas held-out", int(B190.confirmada.sum()), 18, 0, None)

# ---------------------------------------------------------------- previsoes registradas e auditoria da literatura
# Previsoes pontuais registradas por bateria (acertos, total): Entradas 170 (16, 16), 173 (16, 18), 176 (4, 6), 178 (5, 7), 180 (16, 19) -- contagens do diario; 184 A1 (3, 5), 185 B (3, 8), A2 (4, 4), D (8, 8), C (6, 6) -- relatorios.
PRED = [(16, 16), (16, 18), (4, 6), (5, 7), (16, 19), (3, 5), (3, 8), (4, 4), (8, 8), (6, 6), (7, 9), (5, 9), (11, 12), (8, 9), (7, 7), (5, 5), (9, 14), (4, 6), (4, 9), (6, 6), (1, 6), (3, 5), (2, 4), (4, 6), (6, 6), (4, 7), (4, 5), (5, 6), (4, 5), (5, 6)]   # + Ciclo 18: 196; (4, 5) = Entrada 223 (Q otimista); (5, 6) = Entrada 228 (cf nos tempos finos) valeq 8/9, 200 seletor 7/7, controles 5/5, 199 familia F 9/14, 203 Q ajustado 4/6, 197 flowshop 4/9, 205 seletor do flowshop 6/6
chk("baterias que contaram previsoes (30)", len(PRED), 30, 0, "in $30$ batteries")
chk("previsoes que acertaram (185)", sum(a for a, _ in PRED), 185, 0, "$185$ of $239$ point predictions held")
chk("previsoes registradas (239)", sum(b for _, b in PRED), 239, 0)

L = pd.read_csv(R / "litaudit_pooled_entrada172.csv")
L = L[L.included == "Y"]
chk("23 artigos", len(L), 23, 0, "$23$ recent open-access papers")
chk("controle aleatorio (10)", (L.random_control == "Y").sum(), 10, 0, "$10$ include a random-selection control")
chk("limite de CPU comum (6)", (L.budget_basis == "CPU").sum(), 6, 0, "$6$ compare at a common time limit")
chk("estatico ajustado offline (3)", (L.offline_tuned_static == "Y").sum(), 3, 0, "$3$ include a static configuration")
chk("beneficio do componente (22)", L.claim_policy_helps.isin(["Y", "MIXED"]).sum(), 22, 0, "$22$ report a benefit")
chk("dos quais parcial (5)", (L.claim_policy_helps == "MIXED").sum(), 5, 0, "$5$ of them only partly")

# ---------------------------------------------------------------- numeros herdados do periodico (presenca nas fontes do periodico)
for label, snip in [("75,3% de otimalidade provada", "$75.3\\%$"), ("120 de 120 execucoes identicas", "$120$ of $120$")]:
    journal(label, snip)


def n_tests(f):
    out = subprocess.run([sys.executable, "-m", "pytest", f, "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    return int(re.search(r"(\d+) tests? collected", out).group(1))


chk("testes unitarios do ILS/descida vetorizados (357)", n_tests("tests/test_operators_fast.py"), 357, 0, None)
chk("testes unitarios do nucleo compilado (403)", n_tests("tests/test_ig_numba.py"), 403, 0, None)
chk("testes unitarios dos agentes compilados (35)", n_tests("tests/test_ig_numba_agents.py"), 35, 0)
journal("72 de 96 comparacoes significativas contra LPT e sete ILS", "$72$ comparisons")

# ---------------------------------------------------------------- relatorio
bad = [r for r in rows if not r[4]]
for lab, v, e, tol, ok in rows:
    print(f"{'OK ' if ok else 'ERRO'} {lab}" + ("" if np.isnan(v) else f"  recalculado {v:.5g} vs texto {e:.5g} (tol {tol})"))
print(f"\n{len(rows) - len(bad)}/{len(rows)} verificacoes OK")
sys.exit(1 if bad else 0)
