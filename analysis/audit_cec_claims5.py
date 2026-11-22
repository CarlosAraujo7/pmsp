"""
analysis/audit_cec_claims5.py (Ciclo 51)
=========================================
Auditoria dos numeros NOVOS do artigo do CEC reescrito em 20/11/2026 (cec/cec2027_paper.tex): a ablacao dos componentes do IG (Entrada 284), a reanalise de captura nas 69 celulas e o diagnostico da primeira acao (Entradas 283 e 283b),
a tabela de classes sinteticas, o preco do conhecimento offline (Entradas 277 e 281) e o hospedeiro de terceiros (ALNS do CVRP, Entrada 282). Cada numero e' recalculado dos CSV/TXT e conferido (a) com o literal do texto e (b) com a presenca do trecho.
Uso: python -m analysis.audit_cec_claims5   (sai com codigo 1 se algo falhar)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEX = re.sub(r"\s+", " ", (ROOT / "cec" / "cec2027_paper.tex").read_text(encoding="utf-8"))
rows = []


def chk(label, value, expected, tol, snippet=None):
    ok = abs(float(value) - float(expected)) <= tol and (snippet is None or snippet in TEX)
    rows.append((label, float(value), float(expected), tol, ok))


# ---------------------------------------------------------------- ablacao dos componentes do IG (Entrada 284)
K = pd.read_csv(R / "entrada284_contrasts.csv", index_col=0)
F5 = pd.read_csv(R / "entrada285_contrasts.csv", index_col=0)
F5b = pd.read_csv(R / "entrada285b_contrasts.csv", index_col=0)
chk("ablacao: C1 gulosa contra reinsercao aleatoria, sem descida (1,7)", K.loc["C1", "ganho_medio"], 1.7, 0.05, "costs $1.7$ points without a descent")
chk("ablacao: C1 em todas as doze classes", K.loc["C1", "vence_alem"], 12, 0, "in all twelve classes, from $0.10$ to $5.3$")
chk("ablacao: C1 menor ganho por classe (0,10)", K.loc["C1", "ganho_min"], 0.10, 0.006)
chk("ablacao: C1 maior ganho por classe (5,3)", K.loc["C1", "ganho_max"], 5.3, 0.06)
chk("ablacao: C1c a CPU igual (1,7)", K.loc["C1c", "ganho_medio"], 1.7, 0.05, "$1.7$ at equal CPU time")
chk("ablacao: C2 com a descida curta (0,9)", K.loc["C2", "ganho_medio"], 0.9, 0.05, "$0.9$ with the short one and $0.7$ with the full one (nine classes)")
chk("ablacao: C2 em dez classes", K.loc["C2", "vence_alem"], 10, 0)
chk("ablacao: F3 com a descida completa (0,7)", F5.loc["F3", "ganho_medio"], 0.7, 0.05)
chk("ablacao: F3 em nove classes", F5.loc["F3", "vence_alem"], 9, 0)
chk("ablacao: C1 mais de quarenta vezes o maior ganho de um aprendiz (0,039)", K.loc["C1", "ganho_medio"] / 0.039, 44, 4, "more than forty times the largest gain of a learner")
chk("ablacao: F4 IG simples contra o ILS classico, iteracoes iguais (0,65)", F5.loc["F4", "ganho_medio"], 0.65, 0.006, "The plain IG beats the classical ILS by $0.65$ points at equal iterations and $0.79$ at equal CPU time")
chk("ablacao: F4c a CPU igual (0,79)", F5.loc["F4c", "ganho_medio"], 0.79, 0.006)
Kr = json.loads((R / "fulldescent_cpu_ratio_entrada285.json").read_text(encoding="utf-8"))
kr = [v["R2/G0"] for v in Kr.values()]
chk("ablacao: o ILS custa de 5 a 13 vezes (minimo)", min(kr), 5, 0.4, "the ILS costs $5$--$13$ times as much per run")
chk("ablacao: o ILS custa de 5 a 13 vezes (maximo)", max(kr), 13, 0.4)
chk("ablacao: 285b, IG simples contra o ILS de melhor tamanho, iteracoes iguais (0,45)", F5b.loc["S1", "ganho_medio"], 0.45, 0.006, "and still by $0.45$ and $0.57$ when the ILS uses the best fixed kick size of each class, chosen on the test instances in its favour (it is ahead in three small classes at equal iterations, never at equal CPU time)")
chk("ablacao: 285b, a CPU igual (0,57)", F5b.loc["S2", "ganho_medio"], 0.57, 0.006)
chk("ablacao: 285b, o ILS vence em tres classes a iteracoes iguais", F5b.loc["S1", "perde_alem"], 3, 0)
chk("ablacao: 285b, nunca a CPU igual", F5b.loc["S2", "perde_alem"], 0, 0)
chk("ablacao: F2 descida completa sobre a gulosa (+0,003)", F5.loc["F2", "ganho_medio"], 0.003, 0.0006, "adds nothing to the greedy reconstruction ($+0.003$)")
t285 = (R / "entrada285_fulldescent.txt").read_text(encoding="utf-8")
mvm = re.search(r"movimentos por descida do R2: ([\d.]+); do G2: ([\d.]+)", t285)
chk("ablacao: movimentos por descida do G2 (0,9)", float(mvm.group(2)), 0.9, 0.06, "($0.9$ moves per descent against $5.7$ after a random reinsertion)")
chk("ablacao: movimentos por descida do R2 (5,7)", float(mvm.group(1)), 5.7, 0.06)
chk("ablacao: F1 descida completa sobre a curta na aleatoria (+0,16)", F5.loc["F1", "ganho_medio"], 0.16, 0.006, "and little to random reinsertion ($+0.16$)")
chk("resumo: ILS classico e aprendizado", F5.loc["F4c", "ganho_medio"], 0.8, 0.05, "a classical ILS stays $0.6$--$0.8$ percentage points of makespan behind at equal CPU time, while online learning adds at most $0.04$")
chk("resumo: limite inferior do intervalo (0,6)", F5b.loc["S2", "ganho_medio"], 0.6, 0.05)
chk("tabela de respostas: RQ1 (ILS classico)", F5.loc["F4c", "ganho_medio"], 0.8, 0.05, "a classical ILS is $0.6$--$0.8$ points behind at equal CPU time")

# ---------------------------------------------------------------- captura, correlacao e margem (Entradas 283 e 283b)
Cp = pd.read_csv(R / "entrada283b_capture69.csv")
g = lambda grupo, ap, col: float(Cp[(Cp.grupo == grupo) & (Cp.aprendiz == ap)][col].iloc[0])
rt = "desempate aleatorio"
chk("captura agrupada do desempate aleatorio (0,16)", g("todas as 69", rt, "captura"), 0.16, 0.006, "$0.16$ pooled over the cells (90\\% interval by class $0.14$--$0.19$)")
chk("captura agrupada, limite inferior (0,14)", g("todas as 69", rt, "cap_lo"), 0.14, 0.006)
chk("captura agrupada, limite superior (0,19)", g("todas as 69", rt, "cap_hi"), 0.19, 0.006)
chk("captura nas sinteticas (0,06)", g("sintetico", rt, "captura"), 0.06, 0.006, "and $0.06$, $0.16$, $0.17$ and $0.28$ in the synthetic, benchmark, family and published P$||\\Cmax$ cells")
chk("captura no benchmark (0,16)", g("padrao", rt, "captura"), 0.16, 0.006)
chk("captura nas familias (0,17)", g("familias", rt, "captura"), 0.17, 0.006)
chk("captura nas publicadas (0,28)", g("publicado", rt, "captura"), 0.28, 0.006)
chk("Spearman do desempate aleatorio com a validada (0,78)", g("todas as 69", rt, "spearman"), 0.78, 0.006, "with a rank correlation of $0.78$ over the cells (interval by class $0.57$--$0.89$; $0.49$--$0.94$ within each collection)")
chk("Spearman, limite inferior (0,57)", g("todas as 69", rt, "sp_lo"), 0.57, 0.006)
chk("Spearman, limite superior (0,89)", g("todas as 69", rt, "sp_hi"), 0.89, 0.006)
within = [g(k, rt, "spearman") for k in ("padrao", "familias", "sintetico", "publicado")]
chk("Spearman por colecao, minimo (0,49)", min(within), 0.49, 0.006)
chk("Spearman por colecao, maximo (0,94)", max(within), 0.94, 0.006)
chk("Spearman do padrao (-0,06)", g("todas as 69", "padrao", "spearman"), -0.06, 0.006, "against $-0.06$ for the standard learner and $-0.64$ for the permuted one")
chk("Spearman da permutada (-0,64)", g("todas as 69", "permutada", "spearman"), -0.64, 0.006)
chk("resumo: captura de 16% e correlacao 0,78", g("todas as 69", rt, "captura"), 0.16, 0.006, "it recovers about $16\\%$ of the static action's gain, in step with it (rank correlation $0.78$)")
chk("tabela de respostas: RQ3, captura de 16%", g("todas as 69", rt, "captura"), 0.16, 0.006, "it captures $16\\%$ of the validated gain, in step with it")
C = pd.read_csv(R / "entrada283_cells.csv")
C = C[C.coll != "tsp"]
low = C[C.g_val <= 0.05]
chk("celulas com a validada no maximo 0,05 (18)", len(low), 18, 0, "in none of the $18$ cells where the validated gain is at most $0.05$")
chk("nenhuma delas com ganho alem da margem", int(low["rt_beyond_0.05"].sum()), 0, 0)
chk("margens de 0,02 e 0,10 (29 e 19)", int(C["rt_beyond_0.02"].sum()) * 100 + int(C["rt_beyond_0.10"].sum()), 2919, 0, "$29$ and $19$ cells with margins of $0.02$ and $0.10$")
by = C.groupby("coll")["rt_beyond_0.05"].sum()
chk("25 de 69: benchmark 15, familias 6, publicadas 4, sinteticas 0", int(by["padrao"]) * 1000000 + int(by["familias"]) * 10000 + int(by["publicado"]) * 100 + int(by["sintetico"]), 15060400, 0, "$15$ of $21$ benchmark cells, $4$ of $18$ published P$||\\Cmax$, none of the $24$ synthetic ones")
T = pd.read_csv(R / "entrada280_tiesall_tests.csv")
Bt = T[T.id.str.startswith("B-")].copy()
Bt["cell"] = Bt.id.str[2:]
Bt = Bt.merge(C[["coll", "cell"]], on=["coll", "cell"])
chk("a validada bate o desempate aleatorio em 59 das 69", int(Bt.confirmada.sum()), 59, 0, "beats the random-tie learner in $59$ of the $69$ cells")
F = pd.read_csv(R / "entrada283_first_action.csv")
from scipy import stats as ss  # noqa: E402
chk("48 celulas sinteticas", len(F), 48, 0, "across $48$ cells (twelve classes, four action spaces)")
chk("Spearman do Q-Learning padrao com a primeira acao (0,94)", ss.spearmanr(F.g_arm0, F.g_ql).statistic, 0.94, 0.006, "their rank correlation is $0.94$ (median absolute difference $0.04$ points), against $0.03$ and $0.07$ for UCB1 and the policy gradient")
chk("diferenca absoluta mediana (0,04)", (F.g_arm0 - F.g_ql).abs().median(), 0.04, 0.006)
chk("Spearman do UCB1 com a primeira acao (0,03)", ss.spearmanr(F.g_arm0, F.g_ucb).statistic, 0.03, 0.006)
chk("Spearman do gradiente com a primeira acao (0,07)", ss.spearmanr(F.g_arm0, F.g_pg).statistic, 0.07, 0.006)
O = pd.read_csv(R / "entrada283_order.csv")
rev = O[O.variante == "lista invertida"].fracao_na_primeira_acao
chk("lista invertida: fracao na nova primeira acao, minimo (63%)", 100 * rev.min(), 63, 0.6, "reversing the list moves the lock-in to the new first action ($63$--$72\\%$)")
chk("lista invertida: fracao na nova primeira acao, maximo (72%)", 100 * rev.max(), 72, 0.6)
V = pd.read_csv(R / "entrada283_cf_vs_val.csv")
chk("o ganho da validada supera a folga cruzada em 41 de 60", int(V.val_maior.sum()), 41, 0, "exceeds the cross-fitted headroom in $41$ of the $60$ P$||\\Cmax$ cells")
chk("60 celulas de P||Cmax", len(V), 60, 0)

# ---------------------------------------------------------------- tabela de classes (Entrada 283)
K5 = pd.read_csv(R / "entrada283_classtable.csv")
K5 = K5[K5.espaco == "S5"]
chk("n/m de 3 a 15 (minimo)", K5.n_sobre_m.min(), 3, 0, "with $n/m$ from $3$ to $15$")
chk("n/m de 3 a 15 (maximo)", K5.n_sobre_m.max(), 15, 0)
chk("o aleatorio no limite em 7 a 96% das execucoes (minimo)", 100 * K5.no_limite.min(), 7, 0.6, "random choice is at the bound in $7$--$96\\%$ of the runs")
chk("o aleatorio no limite em 7 a 96% das execucoes (maximo)", 100 * K5.no_limite.max(), 96, 0.6)
chk("legenda: lacuna 0,04 a 1,24 unidades (minimo)", K5.lacuna_unidades.min(), 0.04, 0.006)
chk("legenda: lacuna 0,04 a 1,24 unidades (maximo)", K5.lacuna_unidades.max(), 1.24, 0.006)
chk("legenda: ganho da validada de 0,01 a 0,62 unidades (minimo)", K5.g_val_unidades.min(), 0.01, 0.006)
chk("legenda: ganho da validada de 0,01 a 0,62 unidades (maximo)", K5.g_val_unidades.max(), 0.62, 0.006)

# ---------------------------------------------------------------- preco do conhecimento offline (Entradas 277 e 281)
t277 = (R / "entrada277_cost.txt").read_text(encoding="utf-8")
chk("preco: 600 execucoes de ajuste por classe", float(re.search(r"^\s*10\s+(\d+)", t277, flags=re.M).group(1)), 600, 0, "$600$ tuning runs per class, $11.4$ times the test run of the class ($25$ instances)")
chk("preco: 11,4 vezes o custo do teste", float(re.search(r"= ([\d.]+) vezes o custo de rodar a validada", t277).group(1)), 11.4, 0.05)
chk("preco: uma instancia rende 81% do ganho", 100 * float(re.search(r"G_tn com k = 1 / com k = 10: [\d.]+ / [\d.]+ = ([\d.]+)", t277).group(1)), 81, 0.6, "yields $81\\%$ of the gain")
chk("preco: ponto de equilibrio medido (70)", float(re.search(r"N_10k \(k = 10\): ([\d.]+)", t277).group(1)), 70, 2, "about $70$ (measured) to $130$ (interpolated) test instances per class")
chk("preco: ponto de equilibrio interpolado (130)", float(re.search(r"N\* interpolado \(k = 10\): (\d+)", t277).group(1)), 130, 1)
t281 = (R / "entrada281_scarce.txt").read_text(encoding="utf-8")
m = re.search(r"k = 1: ganho do aquecido ([\d.]+) contra o da estatica ([\d.]+)", t281)
chk("preco: aquecido com k = 1 (1,86)", float(m.group(1)), 1.86, 0.006, "($1.86$ against $1.92$ points)")
chk("preco: estatica com k = 1 (1,92)", float(m.group(2)), 1.92, 0.006)

# ---------------------------------------------------------------- hospedeiro de terceiros: ALNS do CVRP (Entrada 282)
cc = pd.read_csv(R / "entrada282_cvrp_cells.csv")
tt = pd.read_csv(R / "entrada282_cvrp_tests.csv")
At = tt[tt.id.str.startswith("A-")].copy()
At["method"] = At.id.str.split("-").str[1]
At["cls"] = At.id.str.split("-").str[2]
At["beyond"] = At.confirmada & ~At["equivalente_0.05"]
val_b = At[At.method == "val"].set_index("cls")
chk("CVRP: a validada ganha alem da margem em 7 de 8 classes", int(val_b.beyond.sum()), 7, 0, "in $7$ of the $8$ classes ($0.6$--$1.1$ points of the cost)")
gv = cc.set_index("cls").g_val[val_b.beyond[val_b.beyond].index]
chk("CVRP: ganho da validada, minimo (0,6)", gv.min(), 0.6, 0.05)
chk("CVRP: ganho da validada, maximo (1,1)", gv.max(), 1.1, 0.05)
srw = At[At.method == "srw"].set_index("cls")
chk("CVRP: a roleta segmentada ganha alem da margem em 5 classes", int(srw.beyond.sum()), 5, 0, "gains beyond the margin in $5$ classes ($0.33$--$0.52$)")
gs = cc.set_index("cls").g_srw[srw.beyond[srw.beyond].index]
chk("CVRP: ganho da roleta segmentada, minimo (0,33)", gs.min(), 0.33, 0.006)
chk("CVRP: ganho da roleta segmentada, maximo (0,52)", gs.max(), 0.52, 0.006)
c2 = cc.set_index("cls")
best = c2[["g_rw", "g_srw", "g_aucb", "g_mab_ucb1", "g_mab_eps"]].max(axis=1)
use = best[(best > 0) & (c2.g_val > 0.05)]
ratio = 100 * use / c2.g_val[use.index]
chk("CVRP: o melhor seletor captura 23% a 61% (minimo)", ratio.min(), 23, 0.6, "captures $23$--$61\\%$ (mean $43\\%$) of the validated gain where any selector gains")
chk("CVRP: o melhor seletor captura 23% a 61% (maximo)", ratio.max(), 61, 0.6)
chk("CVRP: media de 43%", ratio.mean(), 43, 0.6)
Bp = tt[tt.id.str.startswith("Bp-")]
chk("CVRP: nenhum seletor bate a validada em nenhuma classe (0 de 40)", int(Bp.confirmada.sum()), 0, 0, "($0$ of $40$)")
chk("CVRP: 40 pares seletor-classe", len(Bp), 40, 0)
chk("CVRP: o bandido UCB1 perde ate' 3,9", -c2.g_mab_ucb1.min(), 3.9, 0.06, "the UCB1 bandit loses up to $3.9$")
Pm = tt[tt.id.str.startswith("P-")]
chk("CVRP: permutar a lista nao muda nenhum seletor", int((Pm.reject & ~Pm["equivalente_0.05"]).sum()), 0, 0, "permuting the operator list changes no selector")
tx = (R / "entrada282_cvrp.txt").read_text(encoding="utf-8")
mg = re.search(r"aleatorio ([\d.]+); Gnr ([\d.]+); G0 ([\d.]+); rw ([\d.]+); srw ([\d.]+)", tx)
chk("CVRP, instancias X: lacuna do aleatorio (2,51%)", float(mg.group(1)), 2.51, 0.006, "the mean gaps to the best known solutions are $2.51\\%$ (random choice), $2.38\\%$ (the operator chosen per class on the synthetic classes), $1.89\\%$ (the single best operator) and $2.00\\%$ (segmented roulette)")
chk("CVRP, instancias X: lacuna da acao da classe (2,38%)", float(mg.group(2)), 2.38, 0.006)
chk("CVRP, instancias X: lacuna da acao global (1,89%)", float(mg.group(3)), 1.89, 0.006)
chk("CVRP, instancias X: lacuna da roleta segmentada (2,00%)", float(mg.group(5)), 2.00, 0.006)
chk("resumo: nenhum seletor da biblioteca bate a validada em 40 comparacoes", int(Bp.confirmada.sum()), 0, 0, "no selector of the library did in $40$ comparisons")

# ---------------------------------------------------------------- controle positivo e demais numeros novos
PL = pd.read_csv(R / "entrada185_plant_shares.csv")
chk("plantio: maior multiplo do uniforme (13,7)", PL.x_uniforme.max(), 13.7, 0.06, "$1.5$--$13.7$ times as often as uniform choice would")
m2 = PL[(PL["mode"] == 2) & (PL.space == "S5") & (PL.bonus == 5.0)].set_index("learner").x_uniforme
chk("plantio com o bonus dependente do estado: Q-Learning (3,8)", m2["ql"], 3.8, 0.06, "Q-Learning and the policy gradient choose the best action $3.8$ and $2.5$ times as often (five actions)")
chk("plantio com o bonus dependente do estado: gradiente (2,5)", m2["pg"], 2.5, 0.06)

bad = [r for r in rows if not r[4]]
for lab, v, e, tol, ok in rows:
    print(f"{'OK ' if ok else 'ERRO'} {lab}  recalculado {v:.5g} vs texto {e:.5g} (tol {tol})")
print(f"\n{len(rows) - len(bad)}/{len(rows)} verificacoes OK")
sys.exit(1 if bad else 0)
