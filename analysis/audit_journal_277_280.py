"""
analysis/audit_journal_277_280.py
Auditoria dos numeros das Entradas 277-280 (e 280b) citados no manuscrito de periodico (journal/): o preco do conhecimento offline, o teto das classes sinteticas e a margem,
os aprendizes sem travamento e a ordem das acoes, e o Q-Learning com desempate aleatorio nas 85 celulas. Chamado por analysis/audit_journal_claims.py: run(chk, present).
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def _read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def _f(rx, text, g=1):
    m = re.search(rx, text)
    assert m, rx
    return float(m.group(g).replace(",", "."))


def run(chk, present):
    # ------------------------------------------------------------ Entrada 277: o preco do conhecimento offline
    T7 = (R / "entrada277_cost.txt").read_text(encoding="utf-8")
    C = pd.read_csv(R / "entrada277_cost_curve.csv").set_index("k")
    B = pd.read_csv(R / "entrada277_cost_breakeven.csv").set_index("k")
    f = "sec_granularity.tex"
    chk("277: custo do ajuste por classe (8.482 M)", _f(r"por classe: (\d+) M avaliacoes", T7), 8482, 0.5, f, "$8{,}482$ million moves per class")
    chk("277: custo do teste da classe (747 M)", _f(r"\((\d+) M\)\n", T7), 747, 0.5, f, "$747$ million of running")
    chk("277: razao ajuste / teste (11,4)", _f(r"= ([\d.]+) vezes o custo", T7), 11.4, 0.05, f, "$11.4$ times")
    chk("277: ganho de G_tn com k = 1 (1,81)", C.loc[1, "Gtn"], 1.81, 0.005, f, "gains $1.81$ points")
    chk("277: ganho de G_tn com k = 10 (2,24)", C.loc[10, "Gtn"], 2.24, 0.005, f, "of the $2.24$ obtained with ten")
    chk("277: G_tn(1) / G_tn(10) (81%)", 100 * C.loc[1, "Gtn"] / C.loc[10, "Gtn"], 81, 0.5, f, "$81\\%$ of the $2.24$")
    chk("277: G_n com k = 1 (1,97) e k = 10 (2,00)", round(C.loc[1, "Gn"], 2) * 100 + round(C.loc[10, "Gn"], 2), 197 + 2.00, 0.001, f, "gains $1.97$ with one instance per class and $2.00$ with ten")
    chk("277: lacuna do aleatorio a T = 1.000 (7,42)", _f(r"aleatorio T=1\.000 ([\d.]+)%;", T7), 7.42, 0.005, f, "$7.42\\%$ at $10^3$")
    chk("277: lacuna do aleatorio a T = 10.000 (4,31)", _f(r"aleatorio T=10\.000 ([\d.]+)%;", T7), 4.31, 0.005, f, "$4.31\\%$ at $10^4$")
    chk("277: lacuna da validada a T = 1.000 (4,89)", _f(r"validada T=1\.000 ([\d.]+)%;", T7), 4.89, 0.005, f, "$4.89\\%$ for the validated action")
    chk("277: avaliacoes do aleatorio a T = 10.000 (148 M)", _f(r"aleatorio T=10\.000 ([\d.]+) M", T7), 148, 0.5, f, "$148$ million moves per run")
    chk("277: avaliacoes da validada a T = 1.000 (29,9 M)", _f(r"validada T=1\.000 ([\d.]+) M;", T7), 29.9, 0.05, f, "$29.9+8{,}482/N$")
    chk("277: cruzamento medido N_10k (72)", B.loc[10, "N_10k"], 72, 0.5, f, "$N=72$")
    chk("277: custo total igual a N = 25: cerca de 25.000 iteracoes", B.loc[10, "T_equal_total_N25"], 25000, 100, f, "about $25{,}000$ iterations")
    chk("277: lacuna do aleatorio com esse custo (3,1)", B.loc[10, "gap_random_at_that_T"], 3.1, 0.05, f, "near $3.1\\%$")
    chk("277: N* interpolado (130) e com k = 1 (22)", round(B.loc[10, "Nstar_interp"]) * 1000 + round(B.loc[1, "Nstar_interp"]), 130022, 0, f, "the interpolated break-even is $130$ instances per class, and $22$ with one tuning instance")
    chk("277: aprendizes a T = 10.000: roleta (3,89) e FRRMAB (4,15)", _f(r"alns ([\d.]+)%", T7) * 100 + _f(r"fr ([\d.]+)%", T7) / 100, 3.89 * 100 + 4.15 / 100, 0.001, f, "($3.89\\%$ and $4.15\\%$)")
    chk("277: previsoes acertadas (5 de 7)", _f(r"previsoes acertadas: (\d+) de", T7), 5, 0, f, "Five of the seven predictions")
    chk("277: G_n(3) 2,00 e G_tn(3) 2,15 (previsao C7 errada)", round(C.loc[3, "Gn"], 2) * 100 + round(C.loc[3, "Gtn"], 2), 200 + 2.15, 0.011, f, "($2.00$ against $2.15$)")
    present("277: o resumo cita custo 11 vezes, 81% e 72 a 130", "abstract.tex", "its tuning costs $11$ times its test run, one tuning instance per class gives $81\\%$ of its gain")
    present("277: o resumo cita 72 a 130", "abstract.tex", "about $72$ to $130$ instances")
    # ------------------------------------------------------------ Entrada 278: o teto e a margem
    T8 = (R / "entrada278_ceiling.txt").read_text(encoding="utf-8")
    f = "sec_results.tex"
    gaps = re.findall(r"aleatorio ([\d.]+)%; validada ([\d.]+)%; fracao das execucoes que atingem o limite inferior: aleatorio ([\d.]+); validada ([\d.]+)", T8)
    allhit = re.findall(r"TODOS os \d+ bracos fixos atingem o limite inferior: ([\d.]+); em que ALGUM braco o atinge: ([\d.]+)", T8)
    chk("278: lacuna do aleatorio ao limite inferior (0,28 e 0,26)", round(float(gaps[0][0]), 2) * 100 + round(float(gaps[1][0]), 2), 28 + 0.26, 0.001, f, "random choice ends $0.28\\%$ and $0.26\\%$ above the lower bound")
    chk("278: execucoes do aleatorio no limite (42% e 48%)", round(100 * float(gaps[0][2])) * 100 + round(100 * float(gaps[1][2])), 4248, 0, f, "reaches it in $42\\%$ and $48\\%$ of the runs")
    chk("278: execucoes da validada no limite (60%)", round(100 * float(gaps[0][3])) + round(100 * float(gaps[1][3])), 120, 0, f, "the validated action in $60\\%$")
    chk("278: todos os bracos no limite (12% e 4%)", round(100 * float(allhit[0][0])) * 100 + round(100 * float(allhit[1][0])), 1204, 0, f, "in only $12\\%$ and $4\\%$ of the instances")
    chk("278: algum braco no limite (66% e 70%)", round(100 * float(allhit[0][1])) * 100 + round(100 * float(allhit[1][1])), 6670, 0, f, "(some action does in $66\\%$ and $70\\%$)")
    ml = re.search(r"aprendizes \(de 72\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", T8)
    mv = re.search(r"validada \(de 24\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", T8)
    chk("278: aprendizes alem da margem 0,01/0,02/0,05/0,10 (10, 10, 8, 6)", int("".join(f"{int(x):02d}" for x in ml.groups())), 10100806, 0, f, "in $10$, $10$, $8$ and $6$ of the $72$")
    chk("278: validada alem da margem (19, 18, 15, 12)", int("".join(f"{int(x):02d}" for x in mv.groups())), 19181512, 0, f, "in $19$, $18$, $15$ and $12$ of the $24$ cells")
    chk("278: previsoes acertadas (1 de 6)", _f(r"previsoes acertadas: (\d+) de", T8), 1, 0, f, "Only one of the six predictions")
    present("278: texto curto (teto)", "s_results.tex", "random choice ends $0.28\\%$ and $0.26\\%$ above the lower bound")
    # ------------------------------------------------------------ Entrada 279: aprendizes sem travamento e ordem
    T9 = (R / "entrada279_ties.txt").read_text(encoding="utf-8")
    cells = pd.read_csv(R / "entrada279_ties_cells.csv")
    corr = pd.read_csv(R / "entrada279_ties_corr.csv").set_index("learner")
    tests = pd.read_csv(R / "entrada279_ties_tests.csv")
    f = "sec_order.tex"
    mean = cells.groupby("space")[[c for c in cells.columns if c.startswith("g_")]].mean()
    chk("279: ganho do ql_rt em B30 (0,28) e B15 (0,30)", round(mean.loc["B30", "g_ql_rt"], 2) * 100 + round(mean.loc["B15", "g_ql_rt"], 2), 28 + 0.30, 0.001, f, "$0.28$ points in B30 and $0.30$ in B15")
    chk("279: razao ql_rt / validada em B30 (12%) e B15 (18%)", round(100 * mean.loc["B30", "g_ql_rt"] / mean.loc["B30", "g_val"]) * 100 + round(100 * mean.loc["B15", "g_ql_rt"] / mean.loc["B15", "g_val"]), 1218, 0, f, "$12\\%$ and $18\\%$ of the validated gain")
    chk("279: Spearman folga-ganho do ql_rt (0,81; IC90 0,66-0,89)", round(corr.loc["ql_rt", "spearman"], 2) * 10000 + round(corr.loc["ql_rt", "ci90_lo"], 2) * 100 + round(corr.loc["ql_rt", "ci90_hi"], 2), 81 * 100 + 66 + 0.89, 0.001, f, "is $0.81$ ($90\\%$ interval $0.66$--$0.89$)")
    chk("279: Spearman do ql padrao (-0,14) e do ql_perm (-0,55)", round(corr.loc["ql", "spearman"], 2) * 100 + round(corr.loc["ql_perm", "spearman"], 2), -14 - 0.55, 0.001, f, "against $-0.14$ for the standard learner and $-0.55$ for the permuted-list learner")
    chk("279: Spearman de UCB1 (0,39), roleta (0,47), otimista (0,54) e FRRMAB (0,82)", round(corr.loc["ucb", "spearman"], 2) * 1e6 + round(corr.loc["alns", "spearman"], 2) * 1e4 + round(corr.loc["qo", "spearman"], 2) * 1e2 + round(corr.loc["fr", "spearman"], 2), 0.39e6 + 0.47e4 + 0.54e2 + 0.82, 0.001, f, "from $0.39$ (UCB1) and $0.47$ (ALNS roulette) to $0.54$ (optimistic Q-Learning) and $0.82$ (FRRMAB-style learner)")
    pairs = [("ucb", "ucb_perm"), ("pg", "pg_perm"), ("alns", "alns_perm"), ("qo", "qo_perm"), ("iq", "iq_perm")]
    dmax = max(abs(mean.loc[sp, f"g_{a}"] - mean.loc[sp, f"g_{b}"]) for sp in ("B30", "B15") for a, b in pairs)
    chk("279: a permutacao muda o ganho medio em no maximo 0,04", float(dmax <= 0.04), 1, 0, f, "differs by at most $0.04$ points")
    FB = tests[tests.id.str.startswith("B-")].copy()
    FB["variant"] = FB.id.str.split("-").str[1]
    FB["diff"] = FB.reject & ~FB["equivalente_0.05"]
    cnt = FB.groupby("variant")["diff"].sum()
    chk("279: iq, ucb, pg, alns, qo permutados diferem alem da margem em 0 das 14 celulas", int(cnt[["iq_perm", "ucb_perm", "pg_perm", "alns_perm", "qo_perm"]].sum()), 0, 0, f, "by more than the margin in none of the $14$ type--space cells")
    chk("279: FRRMAB permutado difere em 3 das 14", int(cnt["fr_perm"]), 3, 0, f, "only the FRRMAB-style learner differs, in $3$ of them")
    chk("279: previsoes acertadas (2 de 9)", _f(r"previsoes acertadas: (\d+) de", T9), 2, 0, f, "two of nine held")
    # custo por iteracao do ql_rt: B5, B15, B30 e TSP
    ev = {}
    for sp in ("B30", "B15"):
        d = _read(f"fr_test_{sp}_s*of4_entrada256.csv")
        t = _read(f"ties_fanjul_{sp}_s*of12_entrada279.csv")
        ev[sp] = t[t.method == "ql_rt"].evals.mean() / d[d.method == "rand"].evals.mean()
    d5 = _read("fr_test_B5_s*of4_entrada256.csv")
    ev["B5"] = _read("ties280_b5_B5_s*of12_entrada280.csv").evals.mean() / d5[d5.method == "rand"].evals.mean()
    chk("279/280: avaliacoes do ql_rt / aleatorio em B5, B15, B30 (0,89; 0,96; 1,31)", round(ev["B5"], 2) * 1e4 + round(ev["B15"], 2) * 1e2 + round(ev["B30"], 2), 0.89e4 + 0.96e2 + 1.31, 0.001, f, "$0.89$, $0.96$ and $1.31$ times")
    # ------------------------------------------------------------ Entrada 280: as 85 celulas
    S = pd.read_csv(R / "entrada280_tiesall_summary.csv").set_index("colecao")
    T80 = (R / "entrada280_tiesall.txt").read_text(encoding="utf-8")
    chk("280: celulas em que o ql_rt ganha alem da margem (32 de 85)", int(S.rt_alem_da_margem.sum()) * 1000 + int(S.celulas.sum()), 32085, 0, f, "gains beyond the equivalence margin in $32$ of the $85$ cells")
    chk("280: padrao 15 de 21; familias 6; TSP 7; publicado 4; sintetico 0", int(S.loc["padrao", "rt_alem_da_margem"]) * 1e5 + int(S.loc["familias", "rt_alem_da_margem"]) * 1e4 + int(S.loc["tsp", "rt_alem_da_margem"]) * 1e2 + int(S.loc["publicado", "rt_alem_da_margem"]) * 10 + int(S.loc["sintetico", "rt_alem_da_margem"]), 1500000 + 60000 + 700 + 40 + 0, 0, f, "$15$ of the $21$ cells of the standard benchmark")
    chk("280: ganho medio do ql_rt no padrao (0,25) e razao com a validada (16%)", round(S.loc["padrao", "ql_rt"], 2) * 100 + round(100 * S.loc["padrao", "ql_rt"] / S.loc["padrao", "validada"]), 25 + 16, 0.5, f, "mean gain $0.25$ points, $16\\%$ of the validated action's $1.60$")
    chk("280: validada no padrao (1,60)", S.loc["padrao", "validada"], 1.60, 0.005, f, "$1.60$")
    chk("280: familias: maior razao ql_rt / validada (24%)", 100 * S.loc["familias", "rt_sobre_validada_max"], 25, 0.6, f, "at most $25\\%$ of the validated gain")
    chk("280: sintetico: ganho da validada (0,09)", S.loc["sintetico", "validada"], 0.09, 0.005, f, "gains only $0.09$ points")
    chk("280: a validada bate o ql_rt em 74 de 85 (27 do padrao e das familias)", int(S.validada_bate_rt.sum()) * 100 + int(S.loc["padrao", "validada_bate_rt"] + S.loc["familias", "validada_bate_rt"]), 7427, 0, f, "beats the random-tie learner in $74$ of the $85$ cells (all $27$ cells")
    chk("280: TSP: ganho medio do ql_rt (0,09) e da validada (0,44)", round(S.loc["tsp", "ql_rt"], 2) * 100 + round(S.loc["tsp", "validada"], 2), 9 + 0.44, 0.001, "sec_tsp.tex", "(mean $0.09$ points, against $0.44$ for the validated action; battery 280)")
    m = re.search(r"ql: ([+-][\d.]+) \(p = [^)]*\); qp: ([+-][\d.]+) \(p = [^)]*\); rt: ([+-][\d.]+)", T80)
    chk("280: Spearman nas 18 celulas publicadas: ql +0,65, ql_perm -0,74, ql_rt +0,72", round(float(m.group(1)), 2) * 1e4 + round(float(m.group(2)), 2) * 1e2 + round(float(m.group(3)), 2), 0.65e4 - 0.74e2 + 0.72, 0.001, f, "it is $+0.72$ (standard $+0.65$, permuted list $-0.74$)")
    tsp_s, tsp_r = _read("tsp_test_s*of*_entrada267.csv"), _read("ties280_tsp_s*of12_entrada280.csv")
    evt = {sp: tsp_r[tsp_r.space == sp].evals.mean() / tsp_s[(tsp_s.space == sp) & (tsp_s.method == "rand")].evals.mean() for sp in ("T5", "T15")}
    sh = {sp: tsp_r[tsp_r.space == sp].share0.mean() for sp in ("T5", "T15")}
    chk("280: TSP: avaliacoes do ql_rt / aleatorio (1,12 e 1,28)", round(evt["T5"], 2) * 100 + round(evt["T15"], 2), 112 + 1.28, 0.001, "sec_tsp.tex", "evaluates $1.12$ and $1.28$ times as many moves as random choice (T5, T15)")
    chk("280: TSP: parcela do ql_rt na primeira acao (14% e 4%)", round(100 * sh["T5"]) * 100 + round(100 * sh["T15"]), 1404, 0, "sec_tsp.tex", "puts $14\\%$ and $4\\%$ of its choices on the action listed first")
    chk("280: TSP: parcela do ql_rt na primeira acao em T15 (4%)", round(100 * sh["T15"]), 4, 0, "s_external.tex", "puts $4\\%$ of its choices on the first action in T15")
    chk("280: previsoes acertadas (6 de 8)", _f(r"previsoes acertadas: (\d+) de", T80), 6, 0, f, "battery 280: six of eight")
    # ------------------------------------------------------------ Entrada 280b: o ql_rt contra a validada
    H = pd.read_csv(R / "entrada280b_headline_tests.csv")
    beat = H[H.confirmada & ~H["equivalente_0.05"]]
    chk("280b: o ql_rt bate a validada alem da margem em 2 celulas", len(beat), 2, 0, f, "beats the validated action beyond the margin in $2$")
    chk("280b: ganhos da validada nessas celulas (-0,016 e +0,008)", round(float(beat.g_val.min()), 3) * 1000 + round(float(beat.g_val.max()), 3), -16 + 0.008, 0.0011, f, "($-0.016$ and $+0.008$ points)")
    chk("280b: a diferenca e' menor que 0,02", float((beat.mean_d_pct.abs() < 0.02).all()), 1, 0, f, "the difference is below $0.02$ points")
    # ------------------------------------------------------------ aquecimento no TSP: efeito liquido na TSPLIB
    T75 = (R / "entrada275_warmtsplib.txt").read_text(encoding="utf-8")
    g = lambda nm: _f(rf"^g_{nm}\s+([\d.]+)", T75) if re.search(rf"^g_{nm}\s", T75, re.M) else None
    wsn = re.search(r"^g_warm_synn\s+([\d.]+)", T75, re.M); fsn = re.search(r"^g_frozen_synn\s+([\d.]+)", T75, re.M)
    ws0 = re.search(r"^g_warm_syn0\s+([\d.]+)", T75, re.M); fs0 = re.search(r"^g_frozen_syn0\s+([\d.]+)", T75, re.M)
    chk("275: efeito liquido da aprendizagem na TSPLIB (0,56 contra 0,63; 0,58 contra 0,68)", round(float(wsn.group(1)), 2) * 1e6 + round(float(fsn.group(1)), 2) * 1e4 + round(float(ws0.group(1)), 2) * 1e2 + round(float(fs0.group(1)), 2),
        0.56e6 + 0.63e4 + 0.58e2 + 0.68, 0.001, "sec_tsp.tex", "$0.56$ points against $0.63$ for the same prior without learning ($0.58$ against $0.68$ with the global prior)")
    # ------------------------------------------------------------ controle positivo (Entrada 185)
    T85 = (R / "entrada185_plant.txt").read_text(encoding="utf-8")
    vals = []
    for blk in re.split(r"\n--- ", T85):
        if "modo 1" in blk.split("\n")[0]:
            rows = [ln.split() for ln in blk.split("\n")[3:] if ln.strip() and ln.split()[0] in ("alns", "ap", "pg", "pm", "ql", "ucb")]
            vals += [[float(x) for x in r[1:]] for r in rows]
    a = np.array(vals)
    sel = a[:, 2:]          # os dois maiores bonus (2 e 5)
    chk("185: razao da acao plantada ao uniforme, minimo (2,2) e maximo (13,7), bonus 2 e 5, mesma acao em todo estado", round(float(sel.min()), 1) * 100 + round(float(sel.max()), 1), 2.2 * 100 + 13.7, 0.001, "sec_results.tex", "from $2.2$ to $13.7$ times as often as uniform choice would")
    present("185: texto curto do controle positivo", "s_results.tex", "from $2.2$ to $13.7$ times as often as uniform choice would")
