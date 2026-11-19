"""
analysis/audit_journal_281_282.py
Auditoria dos numeros das Entradas 281 (poucos dados de ajuste) e 282 (CVRP com a biblioteca de terceiros; 282b pos hoc) citados no manuscrito de periodico. Chamado por analysis/audit_journal_claims.py: run(chk, present).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
LEARN = ["rw", "srw", "aucb", "mab_ucb1", "mab_eps"]


def _f(rx, text, g=1):
    m = re.search(rx, text)
    assert m, rx
    return float(m.group(g))


def run(chk, present):
    # ------------------------------------------------------------ Entrada 281
    T = pd.read_csv(R / "entrada281_scarce_by_k.csv").set_index("k")
    T81 = (R / "entrada281_scarce.txt").read_text(encoding="utf-8")
    f = "sec_granularity.tex"
    chk("281: k = 1: aquecido 1,86, estatica 1,92, congelado 1,98", round(T.loc[1, "warm"], 2) * 1e4 + round(T.loc[1, "static"], 2) * 1e2 + round(T.loc[1, "frozen"], 2), 1.86e4 + 1.92e2 + 1.98, 0.001, f,
        "it gains $1.86$ points, against $1.92$ for the static action and $1.98$ for the frozen prior")
    chk("281: k = 1: aquecido bate a estatica em 1 e e' batido em 2 dos 7 tipos", int(T.loc[1, "warm_beats_static"]) * 10 + int(T.loc[1, "static_beats_warm"]), 12, 0, f, "beats the static action beyond the margin in $1$ of the $7$ instance types and is beaten in $2$")
    chk("281: custo da aprendizagem contra o congelado: k = 1 (0,12), 2 (0,21), 10 (0,28)", round(-T.loc[1, "warm_minus_frozen"], 2) * 1e4 + round(-T.loc[2, "warm_minus_frozen"], 2) * 1e2 + round(-T.loc[10, "warm_minus_frozen"], 2), 0.12e4 + 0.21e2 + 0.28, 0.001, f,
        "Learning costs $0.12$ points against the frozen prior at $k=1$, $0.21$ at $k=2$ and $0.28$ at $k=10$")
    chk("281: queda de k = 10 a k = 1: aquecido 0,10, estatica 0,33, congelado 0,27", round(T.loc[10, "warm"] - T.loc[1, "warm"], 2) * 1e4 + round(T.loc[10, "static"] - T.loc[1, "static"], 2) * 1e2 + round(T.loc[10, "frozen"] - T.loc[1, "frozen"], 2),
        0.10e4 + 0.33e2 + 0.27, 0.001, f, "its gain falls by $0.10$ points, that of the static action by $0.33$ and that of the frozen prior by $0.27$")
    chk("281: sd entre os 4 sorteios em k = 1: aquecido 0,020, estatica 0,044, congelado 0,052", round(T.loc[1, "sd_warm"], 3) * 1e6 + round(T.loc[1, "sd_static"], 3) * 1e3 + round(T.loc[1, "sd_frozen"], 3),
        0.020e6 + 0.044e3 + 0.052, 0.0001, f, "is $0.020$, against $0.044$ for the static action and $0.052$ for the frozen prior")
    chk("281: o aquecido bate o otimista frio nos 7 tipos em todo k", float((T.warm_beats_cold == 7).all()), 1, 0, f, "in all seven types at every $k$")
    chk("281: previsoes acertadas (6 de 7)", _f(r"previsoes acertadas: (\d+) de", T81), 6, 0, f, "Six of the seven predictions registered for this battery held")
    present("281: texto curto", "s_external.tex", "$0.020$ against $0.044$ and $0.052$")
    # ------------------------------------------------------------ Entrada 282
    C = pd.read_csv(R / "entrada282_cvrp_cells.csv")
    tests = pd.read_csv(R / "entrada282_cvrp_tests.csv")
    T82 = (R / "entrada282_cvrp.txt").read_text(encoding="utf-8")
    t = tests.copy()
    t["kind"] = t.id.str.split("-").str[0]
    t["m"] = t.id.str.split("-").str[1]
    beyond = (t.confirmada & ~t["equivalente_0.05"])
    f = "sec_tsp.tex"
    A = t[t.kind == "A"]
    cnt = {m: int((A.m == m).mul(beyond.loc[A.index]).sum()) for m in ["val"] + LEARN}
    chk("282: a validada ganha alem da margem em 7 das 8 classes", cnt["val"], 7, 0, f, "beyond the margin in $7$ of the $8$ classes")
    pos = C[C.g_val > 0.3]
    chk("282: ganho da validada nas classes com ganho: 0,61 a 1,14; na oitava -0,03", round(pos.g_val.min(), 2) * 1e4 + round(pos.g_val.max(), 2) * 1e2 + round(C[C.g_val <= 0.3].g_val.iloc[0], 2), 0.61e4 + 1.14e2 - 0.03, 0.001, f, "($0.61$ to $1.14$ points;")
    chk("282: classes com ganho alem da margem: SRW 5, AlphaUCB 2, MAB-eps 1, RW 0, MAB-UCB1 0", cnt["srw"] * 10000 + cnt["aucb"] * 1000 + cnt["mab_eps"] * 100 + cnt["rw"] * 10 + cnt["mab_ucb1"], 52100, 0, f, "gains beyond the margin in $5$ classes")
    mean = {m: C[f"g_{m}"].mean() for m in LEARN}
    chk("282: ganhos medios SRW 0,29, AlphaUCB -0,07, MAB-eps 0,01, RW 0,11, MAB-UCB1 -1,95", round(mean["srw"], 2) * 1e8 + round(mean["aucb"], 2) * 1e6 + round(mean["mab_eps"], 2) * 1e4 + round(mean["rw"], 2) * 1e2 + round(mean["mab_ucb1"], 2),
        0.29e8 - 0.07e6 + 0.01e4 + 0.11e2 - 1.95, 0.001, f, "($-0.07$)")
    chk("282: o MAB-UCB1 perde de 0,46 a 3,89 em toda classe", round(-C.g_mab_ucb1.max(), 2) * 100 + round(-C.g_mab_ucb1.min(), 2), 0.46e2 + 3.89, 0.001, f, "loses $0.46$ to $3.89$ points")
    bests = [max(r[f"g_{m}"] for m in LEARN) / r.g_val for _, r in C.iterrows() if r.g_val > 0.3 and max(r[f"g_{m}"] for m in LEARN) > 0]
    chk("282: o melhor aprendiz recupera 23% a 61% (media 43%) em 6 classes", round(100 * min(bests)) * 1e4 + round(100 * max(bests)) * 1e2 + round(100 * np.mean(bests)), 23e4 + 61e2 + 43, 0, f, "recovers $23\\%$ to $61\\%$ of the validated gain (mean $43\\%$)")
    B = t[t.kind == "B"]
    cb = {m: int((B.m == m).mul(beyond.loc[B.index]).sum()) for m in LEARN}
    chk("282: a validada bate RW, SRW, AlphaUCB, MAB-UCB1 e MAB-eps em 7, 7, 7, 8, 7 classes", cb["rw"] * 10000 + cb["srw"] * 1000 + cb["aucb"] * 100 + cb["mab_ucb1"] * 10 + cb["mab_eps"], 77787, 0, f, "in $7$, $7$, $7$, $8$ and $7$ classes")
    Bp = t[t.kind == "Bp"]
    chk("282: nenhum aprendiz bate a validada (0 de 40)", int(beyond.loc[Bp.index].sum()), 0, 0, f, "in any of the $40$ (selector, class) pairs")
    P = t[t.kind == "P"]
    chk("282: a permutacao nao muda nenhum aprendiz em nenhuma classe", int((P.reject & ~P["equivalente_0.05"]).sum()), 0, 0, f, "changes none of the five selectors beyond the margin in any class")
    tm = pd.read_csv(R / "entrada282_cvrp_time.csv").set_index("method").ratio_to_random
    chk("282: razoes de tempo AlphaUCB 0,73, RW 1,16, SRW 1,14, MAB-eps 1,38, MAB-UCB1 1,44", round(tm["aucb"], 2) * 1e8 + round(tm["rw"], 2) * 1e6 + round(tm["srw"], 2) * 1e4 + round(tm["mab_eps"], 2) * 1e2 + round(tm["mab_ucb1"], 2),
        0.73e8 + 1.16e6 + 1.14e4 + 1.38e2 + 1.44, 0.001, f, "$0.73$ times as long as random choice")
    gx = re.search(r"aleatorio ([\d.]+); Gnr ([\d.]+); G0 ([\d.]+); rw ([\d.]+); srw ([\d.]+); aucb ([\d.]+); mab_ucb1 ([\d.]+); mab_eps ([\d.]+)", T82)
    g = [round(float(x), 2) for x in gx.groups()]
    chk("282: X: lacunas aleatorio 2,51; Gnr 2,38; G0 1,89; RW 2,29; SRW 2,00; AlphaUCB 2,29; UCB1 5,00; eps 2,21", sum(v * 100 ** (7 - i) for i, v in enumerate(g)), sum(v * 100 ** (7 - i) for i, v in enumerate([2.51, 2.38, 1.89, 2.29, 2.00, 2.29, 5.00, 2.21])), 0.001, f,
        "$2.51\\%$ for random choice, $2.38\\%$ for the action of the (size, route length) class chosen on the synthetic tuning instances, $1.89\\%$ for the single action")
    X = tests[tests.id.str.startswith("XA-")].set_index("id")
    chk("282: X: G0 e SRW ganham alem da margem; a Gnr nao", float(bool(X.loc["XA-g0", "confirmada"]) and bool(X.loc["XA-srw", "confirmada"]) and not bool(X.loc["XA-gnr", "confirmada"])), 1, 0, f, "does not gain over random choice there")
    XB = tests[tests.id.str.startswith("XB-")]
    chk("282: X: nenhum aprendiz bate a Gnr", int((XB.confirmada & ~XB["equivalente_0.05"]).sum()), 0, 0, f, "no selector beats the class action (none of $5$)")
    P82b = pd.read_csv(R / "entrada282b_xposthoc_tests.csv").set_index("id")
    chk("282b: SRW contra a G0: +0,12 (IC90 -0,17 a +0,41)", round(P82b.loc["XG-srw", "mean_d_pct"], 2) * 1e4 + round(P82b.loc["XG-srw", "ci90_lo"], 2) * 1e2 + round(P82b.loc["XG-srw", "ci90_hi"], 2), 0.12e4 - 0.17e2 + 0.41, 0.001, f, "($+0.12$ points, interval $-0.17$ to $+0.41$)")
    chk("282b: nenhum aprendiz bate a G0; a G0 bate RW e UCB1", float((~P82b.confirmada).all() and P82b.loc["XG-rw", "mean_d_pct"] > 0 and P82b.loc["XG-rw", "p_holm"] < 0.05 and P82b.loc["XG-mab_ucb1", "p_holm"] < 0.05), 1, 0, f, "no selector beats it, it beats the roulette wheel and the UCB1 bandit")
    chk("282: previsoes acertadas (4 de 9)", _f(r"previsoes acertadas: (\d+) de", T82), 4, 0, f, "Four of the nine predictions registered for this battery held")
    present("282: texto curto", "s_external.tex", "recovers $23\\%$ to $61\\%$ of the validated gain (mean $43\\%$)")
    present("282: resumo", "abstract.tex", "recovers $23$--$61\\%$ of the validated gain")
