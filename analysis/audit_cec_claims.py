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
chk("E1 vitorias (557)", P162.loc["E1", "better"], 557, 0, "$557$ wins against $87$")
chk("E1 derrotas (87)", P162.loc["E1", "worse"], 87, 0)
chk("E2 IG vs ILS iteracoes iguais (0,50%)", -P162.loc["E2", "mean_d_pct"], 0.50, 0.005, "$0.50\\%$")
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
chk("menor ganho por classe (40x5: 0,004)", min(c[2] for c in cls), 0.004, 0.0006, "$0.004\\%$")
chk("maior ganho por classe (60x20: 2,84)", max(c[2] for c in cls), 2.84, 0.005, "$2.84\\%$")
chk("n/m >= 8: todos < 0,09", max(c[2] for c in cls if c[0] / c[1] >= 8), 0.0839, 0.0006, "below $0.09\\%$")
low = sorted(c[2] for c in cls if c[0] / c[1] <= 5 and (c[0], c[1]) != (20, 4))
chk("n/m <= 5 (sem 20x4): minimo 0,27", low[0], 0.27, 0.005, "between $0.27\\%$ and $2.84\\%$")
chk("n/m <= 5 (sem 20x4): maximo 2,84", low[-1], 2.84, 0.005)
chk("classe 20x4 (0,06)", [c[2] for c in cls if (c[0], c[1]) == (20, 4)][0], 0.06, 0.005, "$20\\times4$, $0.06\\%$")

seed = (R / "entrada117_seed_variance_all.txt").read_text(encoding="utf-8")
m = re.search(r"S1' politica .*?d = (-?\d\.\d+)% IC90 \[(-?\d\.\d+), (-?\d\.\d+)\]", seed)
chk("seed variance d (-0,011)", float(m.group(1)), -0.011, 0.0006, "$\\bar d=-0.011\\%$")
chk("seed variance IC lo (-0,019)", float(m.group(2)), -0.019, 0.0006, "$-0.019$ to $-0.003$")
chk("seed variance IC hi (-0,003)", float(m.group(3)), -0.003, 0.0006)
m = re.search(r"S3' ig_random vs ils_random: d = (-?\d\.\d+)%", seed)
chk("IG vs ILS sem aprendizado, seed (0,49)", -float(m.group(1)), 0.49, 0.005, "$0.49\\%$ on seed averages")
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
chk("Fig. 1a politica (-0,015)", fv[2], -0.015, 0.0006, "$0.015\\%$")
chk("Fig. 1a tamanho validado (0,089)", fv[3], 0.089, 0.0006, "$0.089\\%$")
chk("validado - politica (0,103)", fv[3] - fv[2], 0.103, 0.0006, "$0.103\\%$")

# validacao externa (Tabela I do CEC)
H90 = pd.read_csv(R / "entrada90_family_level_holdout.csv")
H90 = H90[(H90.comparator == "ig_random") & (H90.holm_family.str.startswith("P"))].set_index("family")
C115 = rd("entrada115_external_confirm2_primary.csv")
chk("held-out |d| minimo (0,009)", -H90.mean_diff_pp.max(), 0.009, 0.0006, "$0.009$ to $0.026$")
chk("held-out |d| maximo (0,026)", -H90.mean_diff_pp.min(), 0.026, 0.0006)
chk("held-out 4.160 instancias", H90.n.sum(), 4160, 0, "$4{,}160$ held-out")
O = C115.loc[["O_B", "O_C", "O_E"]]
chk("confirmatorio |d| minimo (0,015)", -O.mean_diff_pp.max(), 0.015, 0.0006, "$0.015$ to $0.029$")
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
chk("K_ev minimo (1,9)", KH.K_ev.min(), 1.9, 0.05, "$1.9$ to $9.1$ times")
chk("K_ev maximo (9,1)", KH.K_ev.max(), 9.1, 0.05)
chk("K_cpu minimo (1,7)", KH.K_cpu.min(), 1.7, 0.05, "$1.7$ to $2.4$")
chk("K_cpu maximo (2,4)", KH.K_cpu.max(), 2.4, 0.05)
chk("Spearman K_cpu x K_ev (0,71)", float(re.search(r"nas 24 classes-cenarios = (\d\.\d+)", t170).group(1)), 0.71, 0.005, "$0.71$ over the $24$")
w = {(c, mt): float(v) for c, mt, v in re.findall(r"^\s*([HR])\s+(ig_eqev_w[\d.]+)\s+(-?\d\.\d+)", t170, flags=re.M) for mt in [mt]}
chk("peso 0,25: vantagem (0,026)", -w[("H", "ig_eqev_w0.25")], 0.026, 0.0006, "$0.026\\%$ and $0.124\\%$")
chk("peso 4: vantagem (0,124)", -w[("H", "ig_eqev_w4")], 0.124, 0.0006)
chk("D2 receita vetorizada (-0,008)", PD162.loc["D2", "mean_d_pct"], -0.008, 0.0006, "$-0.008\\%$ and $-0.006\\%$")
chk("C3 receita compilada (-0,006)", P170.loc["C3", "mean_d_pct"], -0.006, 0.0006)

# ---------------------------------------------------------------- resultados C (agentes)
Ai, Ac, B = rd("entrada180_agents_aiter.csv"), rd("entrada180_agents_acpu.csv"), rd("entrada180_agents_b.csv")
KA = pd.read_csv(R / "entrada180_agents_k.csv")
ta = (R / "entrada180_agents.txt").read_text(encoding="utf-8")
chk("12 comparacoes a iteracoes iguais", len(Ai), 12, 0)
chk("maior ganho de aprendiz a iteracoes iguais (0,016)", -Ai.mean_d_pct.min(), 0.016, 0.0006, "more than $0.016\\%$")
chk("Q-Learning perde em S30 (0,017)", Ai.loc["Aiter-S30-ql", "mean_d_pct"], 0.017, 0.0006, "loses by $0.017\\%$")
chk("Q-Learning CPU igual S5 (0,059)", -Ac.loc["Acpu-S5-ql", "mean_d_pct"], 0.059, 0.0006, "$0.059\\%$ and $0.049\\%$")
chk("Q-Learning CPU igual S15 (0,049)", -Ac.loc["Acpu-S15-ql", "mean_d_pct"], 0.049, 0.0006)
nonq = Ac[~Ac.index.str.endswith("-ql")]
chk("UCB1 e gradiente a CPU igual: |d| <= 0,008", nonq.mean_d_pct.abs().max(), 0.008, 0.0006, "$|\\bar d|\\leq0.008\\%$")
bsel = B[B.index.str.contains("S5-|S15-")]
chk("validada vs aprendiz S5/S15: minimo (0,076)", -bsel.mean_d_pct.max(), 0.076, 0.0006, "$0.076$--$0.095\\%$")
chk("validada vs aprendiz S5/S15: maximo (0,095)", -bsel.mean_d_pct.min(), 0.095, 0.0006)
chk("validada vs aprendiz S5/S15: p_holm < 1e-58", float(bsel.p_holm.max() < 1e-58), 1, 0, "p_{\\mathrm{Holm}}<10^{-58}")
bout = B[B.index.str.contains("S3-|S30-")]
chk("validada vs aprendiz S3/S30: |d| <= 0,03", min(bout.mean_d_pct.abs().max(), 0.03), bout.mean_d_pct.abs().max(), 0, "(within $0.03\\%$)")
ku = lambda sp, k: float(KA[(KA.space == sp) & (KA.learner == k)].K_cpu.iloc[0])
ke = lambda sp, k: float(KA[(KA.space == sp) & (KA.learner == k)].K_avaliacoes.iloc[0])
over = KA[KA.learner != "ql"].sobrecusto_por_avaliacao
chk("sobrecusto UCB1/PG minimo (1,0)", over.min(), 1.0, 0.05, "$1.0$--$1.09$ times")
chk("sobrecusto UCB1/PG maximo (1,09)", over.max(), 1.09, 0.005)
oq = KA[KA.learner == "ql"].sobrecusto_por_avaliacao
chk("sobrecusto Q minimo (1,07)", oq.min(), 1.07, 0.005, "$1.07$--$1.33$ times")
chk("sobrecusto Q maximo (1,33)", oq.max(), 1.33, 0.005)
kq = [ku(s, "ql") for s in ("S5", "S15", "S30")]
chk("K_CPU do Q em S5/S15/S30: minimo (0,50)", min(kq), 0.50, 0.005, "$0.50$--$0.60$ of the CPU time")
chk("K_CPU do Q em S5/S15/S30: maximo (0,60)", max(kq), 0.60, 0.005)
evq = [ke(s, "ql") for s in ("S5", "S15", "S30")]
chk("avaliacoes do Q: minimo (0,38)", min(evq), 0.38, 0.005, "$0.38$--$0.50$ of the evaluations")
chk("avaliacoes do Q: maximo (0,50)", max(evq), 0.50, 0.005)
hd = {sp: -float(v) for sp, v in re.findall(r"^\s*(S\d+)\s+(-?\d\.\d+)\s*$", ta.split("folga: oraculo")[1].split("S30: fracao")[0], flags=re.M)}
chk("folga S3 (0,05)", hd["S3"], 0.05, 0.005, "($0.05\\%$ against $0.12$--$0.13\\%$)")
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
chk("48 celulas", len(c), 48, 0, "$48$ cells")
rho = sstats.spearmanr(c.h, c.gain).statistic
chk("Spearman folga x ganho do Q (0,09)", rho, 0.09, 0.005, "$\\rho=0.09$")
chk("ganho minimo das celulas (-0,19)", c.gain.min(), -0.19, 0.005, "from $-0.19$ to $+0.25$")
chk("ganho maximo das celulas (+0,25)", c.gain.max(), 0.25, 0.005)
chk("celulas com ganho positivo (25)", (c.gain > 0).sum(), 25, 0, "$25$ cells are positive")
chk("celulas com ganho > +0,05 (10)", (c.gain > 0.05).sum(), 10, 0, "only $10$ exceed $+0.05$")
chk("celulas com ganho < -0,05 (14)", (c.gain < -0.05).sum(), 14, 0, "$14$ are below $-0.05$")
hi = c[c.h > 0.15]
chk("celulas com folga > 0,15 (11)", len(hi), 11, 0, "$11$ cells with a headroom above $0.15$")
chk("dessas, ganham (6)", (hi.gain > 0).sum(), 6, 0, "$6$ gain and $5$ lose")
chk("dessas, perdem (5)", (hi.gain <= 0).sum(), 5, 0)

# ---------------------------------------------------------------- resultados D (aprendizes maiores)
P176, P173, P167 = rd("entrada176_ilsdqn_primary.csv"), rd("entrada173_dqn_primary.csv"), rd("entrada167_online_primary.csv")
t176 = (R / "entrada176_ilsdqn.txt").read_text(encoding="utf-8")
chk("DQN-ILS vs mistura (0,041)", P176.loc["K1", "mean_d_pct"], 0.041, 0.0006, "($0.041\\%$ and $0.066\\%$")
chk("LinUCB-ILS vs mistura (0,066)", P176.loc["K2", "mean_d_pct"], 0.066, 0.0006)
chk("p_holm K1 (4e-4)", P176.loc["K1", "p_holm"] * 1e4, 4, 0.5, "4\\times10^{-4}")
chk("p_holm K2 (~1e-6)", np.log10(P176.loc["K2", "p_holm"]), -6, 0.2, "$10^{-6}$")
mv = re.search(r"val_ils vs aleatorio \(operador unico validado\)\s+(-?[\d.]+)", t176)
chk("operador validado vs mistura (0,062)", float(mv.group(1)), 0.062, 0.0006, "by $0.062\\%$")
m = re.search(r"Q-learning tabular vs aleatorio \+(\d\.\d+)%; DQN \+\d\.\d+%; LinUCB \+\d\.\d+%; oraculo por instancia entre os quatro operadores unicos vs aleatorio \+(\d\.\d+)%", t176)
chk("agente tabular nas rodadas armazenadas (0,044)", float(m.group(1)), 0.044, 0.0006, "$0.044\\%$ on the stored runs")
chk("oraculo entre operadores unicos pior que a mistura (0,04)", float(m.group(2)), 0.04, 0.005, "is $0.04\\%$ \\emph{worse}")
m = re.search(r"medias geometricas: dqn (\d\.\d+), linucb (\d\.\d+)", t176)
chk("custo do DQN-ILS (3,3)", float(m.group(1)), 3.3, 0.05, "$3.3$ times")
chk("custo do LinUCB-ILS (1,6)", float(m.group(2)), 1.6, 0.05, "LinUCB $1.6$ times")
chk("DQN em 15 acoes vs aleatorio (+0,023)", P173.loc["I1P15", "mean_d_pct"], 0.023, 0.0006, "($+0.023\\%$, parity)")
chk("validada vs DQN, 15 acoes (0,094)", -P173.loc["I2P15", "mean_d_pct"], 0.094, 0.0006, "beats it by $0.094\\%$")
chk("Q neural vs aleatorio (+0,006)", P167.loc["P-G2", "mean_d_pct"], 0.006, 0.0006, "$+0.006\\%$ and $+0.003\\%$")
chk("LinUCB vs aleatorio (+0,003)", P167.loc["P-G1", "mean_d_pct"], 0.003, 0.0006)
vs = [-P173.loc["I2P5", "mean_d_pct"], -P167.loc["P-G3", "mean_d_pct"], -P167.loc["P-G4", "mean_d_pct"]]
chk("validado vs tres aprendizes: minimo (0,091)", min(vs), 0.091, 0.0006, "$0.091$--$0.094\\%$")
chk("validado vs tres aprendizes: maximo (0,094)", max(vs), 0.094, 0.0006)
chk("DQN em 5 e 15 acoes dentro de +-0,05", float(abs(P173.loc["I1P5", "mean_d_pct"]) < 0.05 and P173.loc["I1P15", "ci90_hi"] < 0.05), 1, 0, "stays inside the $\\pm0.05\\%$ margin")

# ---------------------------------------------------------------- previsoes (Tabela V) e auditoria da literatura
chk("previsoes que acertaram (57)", 16 + 16 + 4 + 5 + 16, 57, 0, "$57$ of $66$ held")
chk("previsoes registradas (66)", 16 + 18 + 6 + 7 + 19, 66, 0)
chk("erros: custos (3) + folga (2) + efeitos minimos (2) + faixa grossa (2) = 9", 3 + 2 + 2 + 2, 66 - 57, 0, "The nine that failed")
L = pd.read_csv(R / "litaudit_pooled_entrada172.csv")
L = L[L.included == "Y"]
chk("23 artigos", len(L), 23, 0, "$23$ recent open-access papers")
chk("controle aleatorio (10)", (L.random_control == "Y").sum(), 10, 0, "$10$ include a random-selection control")
chk("limite de CPU comum (6)", (L.budget_basis == "CPU").sum(), 6, 0, "$6$ compare at a common time limit")
chk("estatico ajustado offline (3)", (L.offline_tuned_static == "Y").sum(), 3, 0, "$3$ include a static configuration")
chk("beneficio do componente (22)", L.claim_policy_helps.isin(["Y", "MIXED"]).sum(), 22, 0, "$22$ report a benefit")
chk("dos quais parcial (5)", (L.claim_policy_helps == "MIXED").sum(), 5, 0, "$5$ of them only partly")

# ---------------------------------------------------------------- numeros herdados do periodico (presenca nas fontes do periodico)
for label, snip in [("75,3% de otimalidade provada", "$75.3\\%$"), ("74,7% em T = 1000", "$74.7\\%$"), ("97,5% em T = 64.000", "$97.5\\%$"), ("ganho de 0,29% sobre o MULTIFIT", "$0.29\\%$"),
                    ("6.780 instancias publicadas", "$6{,}780$"), ("120 de 120 execucoes identicas", "$120$ of $120$"),
                    ("equivalencia estatistica -0,003 a +0,012", "$-0.003\\%$ to $+0.012\\%$"), ("1.400 instancias R||Cmax", "$1{,}400$ instances")]:
    journal(label, snip)


def n_tests(f):
    out = subprocess.run([sys.executable, "-m", "pytest", f, "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    return int(re.search(r"(\d+) tests? collected", out).group(1))


chk("testes unitarios do ILS/descida vetorizados (357)", n_tests("tests/test_operators_fast.py"), 357, 0, "$357$ unit tests")
chk("testes unitarios do nucleo compilado (403)", n_tests("tests/test_ig_numba.py"), 403, 0, "($403$ and $35$ unit tests)")
chk("testes unitarios dos agentes compilados (35)", n_tests("tests/test_ig_numba_agents.py"), 35, 0)
journal("72 de 96 comparacoes significativas contra LPT e sete ILS", "$72$ comparisons")
present("LDM ~1,1%", "LDM by about $1.1\\%$")

# ---------------------------------------------------------------- relatorio
bad = [r for r in rows if not r[4]]
for lab, v, e, tol, ok in rows:
    print(f"{'OK ' if ok else 'ERRO'} {lab}" + ("" if np.isnan(v) else f"  recalculado {v:.5g} vs texto {e:.5g} (tol {tol})"))
print(f"\n{len(rows) - len(bad)}/{len(rows)} verificacoes OK")
sys.exit(1 if bad else 0)
