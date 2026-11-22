"""
analysis/audit_cec_v8.py (rascunho v8 do artigo do CEC, 22/11/2026)
=====================================================================
AUDITORIA NUMERICA do rascunho cec/cec2027_v8.tex. Dois niveis:
 (1) os numeros NOVOS dos Ciclos 53 a 56 (estimandos, teste de ordem, deplecao, evolucao diferencial, credito alinhado) sao recalculados dos arquivos de resultado e conferidos, com o trecho exato do texto, na tolerancia do ultimo digito;
 (2) todo numero do v8 que NAO esta no v7 (cec/cec2027_paper.tex, auditado por analysis/audit_cec_claims*.py) e NAO esta na lista do nivel (1) e' listado para conferencia manual (aviso, nao falha).
Uso: python -m analysis.audit_cec_v8
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
V8 = (ROOT / "cec" / "cec2027_v8.tex").read_text(encoding="utf-8")
V7 = (ROOT / "cec" / "cec2027_paper.tex").read_text(encoding="utf-8")


def strip(t):
    t = "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("%"))
    t = re.sub(r"\\input\{[^}]*\}", " ", t)
    return t


TEX8 = strip(V8)
CHECKS = []
SHOWN = set()


def chk(label, snippet, got, shown, tol):
    for s in re.findall(r"\d+(?:\.\d+)?", snippet):
        SHOWN.add(s)
    CHECKS.append((label, snippet, float(got), float(shown), float(tol)))


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(sstats.spearmanr(a[ok], b[ok]).statistic)


def main():
    E = pd.read_csv(R / "entrada286_estimators.csv")
    g = lambda est: E[(E.conjunto == "69 CEC") & (E.estimador == est)].iloc[0]
    r = g("razao_de_somas"); f = g("fieller_sobre_totais_por_cluster"); sl = g("inclinacao_pela_origem"); pr = g("premio_offline_medio_pp")
    mc = g([k for k in E.estimador if k.startswith("captura_por_celula")][0])
    chk("captura", r"captures $0.157$ of the validated gain", r.estimativa, 0.157, 0.0006)
    chk("IC cluster", r"cluster bootstrap $0.137$--$0.189$", r.lo, 0.137, 0.0006); chk("IC cluster hi", r"cluster bootstrap $0.137$--$0.189$", r.hi, 0.189, 0.0006)
    chk("Fieller", r"Fieller $0.136$--$0.192$", f.lo, 0.136, 0.0006); chk("Fieller hi", r"Fieller $0.136$--$0.192$", f.hi, 0.192, 0.0006)
    chk("inclinacao", r"through the origin $0.141$", sl.estimativa, 0.141, 0.0006); chk("mediana por celula", r"median per cell $0.141$", mc.estimativa, 0.141, 0.0006)
    C = pd.read_csv(R / "entrada286_cells.csv")
    C69 = C[C.coll != "tsp"]
    byc = {k: x.g_rt.sum() / x.g_val.sum() for k, x in C69.groupby("coll")}
    for k, shown in (("sintetico", 0.06), ("padrao", 0.16), ("familias", 0.17), ("publicado", 0.28)):
        chk(f"captura {k}", r"$0.06$, $0.16$, $0.17$ and $0.28$", byc[k], shown, 0.0056)
    chk("premio", r"offline premium is $0.55$ points ($0.29$--$0.83$)", pr.estimativa, 0.55, 0.006); chk("premio lo", r"($0.29$--$0.83$)", pr.lo, 0.29, 0.006); chk("premio hi", r"($0.29$--$0.83$)", pr.hi, 0.83, 0.006)
    CP = pd.read_csv(R / "entrada286_coupling.csv"); c = CP[(CP.conjunto == "69 CEC") & (CP.escopo == "todas")].iloc[0]
    chk("Spearman", r"rank correlation $0.78$, $0.77$ when", c.S_completo, 0.78, 0.006); chk("Spearman desacoplado", r"rank correlation $0.78$, $0.77$ when", c.S_desacoplado_meio, 0.77, 0.006)
    RU = pd.read_csv(R / "entrada286_rules.csv"); ru = RU[(RU.conjunto == "69 CEC") & np.isclose(RU.margem, 0.05)].iloc[0]
    chk("celulas alem da margem (estrita)", r"beyond the margin in $20$ of the $69$ cells", ru.rt_alem_estrita, 20, 0); chk("regra antiga", r"counts $25$ cells instead of $20$", ru.rt_alem_antiga, 25, 0)
    chk("validada bate em 59", r"beats it in $59$ of the $69$ cells ($45$ beyond the margin)", C69.b_conf.sum(), 59, 0); chk("validada bate estrito 45", r"($45$ beyond the margin)", ru.val_bate_rt_estrita, 45, 0)
    P = pd.read_csv(R / "entrada286_pools.csv"); S = pd.read_csv(R / "entrada286b_pool_sensitivity.csv")
    chk("pool acao unica", r"($-0.07$ points)", P[(P.espaco == "S5") & (P.metodo == "pool")].ganho_pp.iloc[0], -0.07, 0.006)
    chk("pool retrospectivo", r"($-0.06$)", S[(S.espaco == "S5") & S.criterio.str.startswith("c3") & (S.metrica == "pool")].ganho_pp.iloc[0], -0.06, 0.006)
    chk("rotulo da classe", r"($+0.16$ over the best single action)", P[(P.espaco == "S5") & (P.metodo == "val_vs_pool")].ganho_pp.iloc[0], 0.16, 0.006)
    # ---- DE
    G = pd.read_csv(R / "entrada289_cells.csv").set_index("fun")
    for fun, v in ((8, 8.4), (9, 7.1), (21, 3.4), (18, 3.1), (10, 2.1), (17, 1.8), (12, 1.3), (7, 1.2)):
        chk(f"ganho validada f{fun}", r"$8.4$, $7.1$, $3.4$, $3.1$, $2.1$, $1.8$, $1.3$ and $1.2$ decades", G.loc[fun, "gain_val"], v, 0.051)
    hi = G[G.gain_val >= 0.1]
    chk("funcoes com ganho >= 0,1", r"in $10$ of the $24$ the gain is at least $0.1$ decade; median $1.95$", len(hi), 10, 0); chk("mediana", r"median $1.95$", hi.gain_val.median(), 1.95, 0.006)
    chk("spread max", r"up to $9.2$ decades", G.spread.max(), 9.2, 0.051); chk("spread min", r"by $0.09$ to $9.2$ decades", G.spread.min(), 0.09, 0.006)
    chk("sucessos min", r"$177$ to $12{,}910$ successes", G.succ_per_arm_run.min(), 177, 0.5); chk("sucessos max", r"$177$ to $12{,}910$ successes", G.succ_per_arm_run.max(), 12910, 1)
    L = pd.read_csv(R / "entrada289c_learners.csv").set_index("aprendiz")
    chk("mediana de ganho max", r"($-0.11$ to $-0.60$ decades)", L.mediana_ganho.max(), -0.11, 0.006); chk("mediana de ganho min", r"($-0.11$ to $-0.60$ decades)", L.mediana_ganho.min(), -0.60, 0.006)
    chk("funcoes negativas min", r"in $15$ to $20$ functions", L.funcoes_ganho_negativo.min(), 15, 0); chk("funcoes negativas max", r"in $15$ to $20$ functions", L.funcoes_ganho_negativo.max(), 20, 0)
    chk("postos sinalizados", r"($p<0.03$)", L.p_postos_sinalizados.max(), 0.0277, 0.003)
    chk("UCB1 mais de uma decada", r"in $9$ and $10$ functions", L.loc["ucb", "pior_que_aleatorio_em_mais_de_1_decada"], 9, 0); chk("AP mais de uma decada", r"in $9$ and $10$ functions", L.loc["ap", "pior_que_aleatorio_em_mais_de_1_decada"], 10, 0)
    chk("validada bate ql_rt", r"in $5$ functions beyond the margin", G.val_beats_qlrt.sum(), 5, 0)
    # ---- credito alinhado
    S1 = pd.read_csv(R / "entrada291_summary.csv").set_index(["credito", "aprendiz"])
    chk("ql_rt log", r"end at $0.00$, $0.00$ and $-0.03$ decades", S1.loc[("log_incumbent", "ql_rt"), "mediana_ganho"], 0.0, 0.006); chk("pg log", r"end at $0.00$, $0.00$ and $-0.03$ decades", S1.loc[("log_incumbent", "pg"), "mediana_ganho"], 0.0, 0.006)
    chk("ql_std log", r"end at $0.00$, $0.00$ and $-0.03$ decades", S1.loc[("log_incumbent", "ql_std"), "mediana_ganho"], -0.03, 0.006)
    chk("captura maxima log", r"(capture at most $0.007$)", S1.loc["log_incumbent"].mediana_captura.max(), 0.007, 0.0006)
    pw = S1.loc["log_incumbent"].loc[["ucb", "pm", "ap"]].p_postos.max()
    chk("p dos piores", r"($p\le0.018$)", pw, 0.018, 0.0006)
    txt = (R / "entrada291_credit.txt").read_text(encoding="utf-8")
    m = re.search(r"sucesso ([\d.]+); log_incumbent ([\d.]+); rel_incumbent ([\d.]+)", txt)
    chk("fracao credito alinhado = validada (4 de 24)", r"in $4$ of $24$ functions", float(m.group(2)) * 24, 4, 0.01)
    chk("UCB1 na validada (log)", r"spends $0.205$ of its decisions", S1.loc[("log_incumbent", "ucb"), "na_validada"], 0.205, 0.0006)
    # ---- mecanismos
    O = pd.concat([pd.read_csv(f) for f in sorted((R).glob("order288_*.csv"))], ignore_index=True)
    OC = pd.read_csv(R / "entrada288_cells.csv"); c0, c1 = OC[OC.tie == 0], OC[OC.tie == 1]
    chk("celulas com efeito de ordem", r"in $17$ of the $19$ cells", (c0.p < 0.05).sum(), 17, 0); chk("sem efeito com desempate aleatorio", r"and in none with random ties", (c1.p < 0.05).sum(), 0, 0)
    chk("linha zerada", r"($72\%$ of the decisions are taken in a zero row)", O[O.tie == 0].share_zero.mean() * 100, 72, 0.5)
    chk("cota", r"this bound averages $0.631$ against an observed share of $0.751$", O[O.tie == 0].bound.mean(), 0.631, 0.0006); chk("participacao", r"observed share of $0.751$", O[O.tie == 0].share_first.mean(), 0.751, 0.0006)
    chk("84%", r"so it explains $84\%$ of the lock-in", O[O.tie == 0].bound.mean() / O[O.tie == 0].share_first.mean() * 100, 84, 0.6); chk("1/K", r"the share is $0.2003=1/K$", O[O.tie == 1].share_first.mean(), 0.2003, 0.00006)
    chk("execucoes", r"In $27{,}360$ runs", len(O), 27360, 0)
    D = pd.read_csv(R / "entrada287_cells.csv"); M = D[D.g_val >= 0.05]
    chk("positivas por acao por execucao", r"only $0.58$ positive rewards per action per run ($5.3$ per run", D.pos_per_arm_run.median(), 0.58, 0.006); chk("positivas por execucao", r"($5.3$ per run", D.pos_per_run.median(), 5.3, 0.06)
    chk("bandido superestima", r"overstates the gain of a fixed action $5.75$ times", (1000 * D.delta_cf / D.g_val)[(D.g_val >= 0.05) & (D.delta_cf > 0)].median(), 5.75, 0.006)
    byc2 = {k: (1000 * x.delta_cf / x.g_val)[(x.g_val >= 0.05) & (x.delta_cf > 0)].median() for k, x in D.groupby("coll")}
    chk("bandido por colecao (min)", r"($4.5$ to $20$ by collection)", min(byc2.values()), 4.5, 0.06); chk("bandido por colecao (max)", r"($4.5$ to $20$ by collection)", max(byc2.values()), 20, 0.2)
    chk("execucoes de controle aleatorio", r"$34{,}500$ random-control runs", len(D) * 500, 34500, 0)
    chk("Spearman L", r"(rank correlations $-0.24$ and $-0.17$", sp(M.L, M.kap_rt), -0.24, 0.006); chk("Spearman kappa_ws", r"(rank correlations $-0.24$ and $-0.17$", sp(M.kappa_ws, M.kap_rt), -0.17, 0.006)
    H = pd.read_csv(R / "entrada290_cells.csv"); piv = H.pivot(index="type", columns="T", values="pos_per_arm_run")
    chk("sucessos x horizonte", r"by only $3.1$ and the capture does not rise (medians $0.33$, $0.36$, $0.18$)", (piv[16000] / piv[1000]).median(), 3.1, 0.06)
    for T, shown in ((1000, 0.33), (4000, 0.36), (16000, 0.18)):
        chk(f"captura T={T}", r"(medians $0.33$, $0.36$, $0.18$)", H[H["T"] == T].cap_rt.median(), shown, 0.006)
    AL = pd.read_csv(R / "entrada289c_alignment.csv")
    chk("sucesso maximo = validada", r"only $5$ of the $24$ functions", AL.coincidem.sum(), 5, 0)
    chk("UCB1 na de maior sucesso", r"UCB1 puts $37.7\%$ of its decisions", AL.ucb_na_sucesso.mean() * 100, 37.7, 0.06); chk("UCB1 na validada", r"and $17.9\%$ on the validated one", AL.ucb_na_val.mean() * 100, 17.9, 0.06)
    SM = pd.read_csv(R / "entrada289_summary.csv", index_col=0)
    chk("captura de recompensa do ql_rt", r"is $+0.19$ in the reward but $-0.18$ in the error", SM.loc["ql_rt", "mediana_captura_recompensa"], 0.19, 0.006); chk("captura de erro do ql_rt", r"$-0.18$ in the error", SM.loc["ql_rt", "mediana_captura"], -0.18, 0.006)
    chk("participacao da acao 0 no ql_std (DE)", r"puts $0.63$ of its decisions on the first action ($0.44$--$0.79$ by function)", G.u0_qlstd.mean(), 0.63, 0.006); chk("min", r"($0.44$--$0.79$ by function)", G.u0_qlstd.min(), 0.44, 0.006); chk("max", r"($0.44$--$0.79$ by function)", G.u0_qlstd.max(), 0.79, 0.006)
    chk("rotulo contra aprendiz", r"($0.16$ points against $0.01$)", P[(P.espaco == "S5") & (P.metodo == "val_vs_pool")].ganho_pp.iloc[0], 0.16, 0.006); chk("ganho do rt sobre o aleatorio (pool S5)", r"($0.16$ points against $0.01$)", P[(P.espaco == "S5") & (P.metodo == "rt")].ganho_pp.iloc[0], 0.01, 0.006)
    chk("previsoes 27 de 56", r"$27$ of $56$ predictions of the last six batteries", 10 + 2 + 6 + 3 + 2 + 4, 27, 0); chk("previsoes total", r"$27$ of $56$", 12 + 11 + 7 + 12 + 6 + 8, 56, 0)
    # ---- avaliacao
    bad = 0
    for label, snippet, got, shown, tol in CHECKS:
        in_tex = snippet in TEX8
        ok = in_tex and abs(got - shown) <= tol + 1e-12
        if not ok:
            bad += 1
            print(f"FALHA  {label}: {'trecho NAO esta no v8; ' if not in_tex else ''}calculado {got:.6g}, no texto {shown:.6g} (tolerancia {tol:g}) | {snippet[:70]}")
    print(f"{len(CHECKS) - bad}/{len(CHECKS)} verificacoes OK")
    # ---- nivel 2: numeros do v8 que nao estao no v7 nem na lista verificada
    toks8 = set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\\(?:ref|label|cite|includegraphics|bibliography\w*|usepackage|documentclass)\{[^}]*\}", " ", TEX8)))
    toks7 = set(re.findall(r"\d+(?:\.\d+)?", strip(V7)))
    pend = sorted(t for t in toks8 if t not in toks7 and t not in SHOWN and len(t) > 0)
    print(f"numeros do v8 fora do v7 e da lista verificada (conferir a mao; {len(pend)}): {' '.join(pend)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
