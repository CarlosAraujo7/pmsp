"""
analysis/audit_entradas286_290.py (Ciclos 53 a 55, 22/11/2026)
================================================================
AUDITORIA NUMERICA das Entradas 286 a 290: recalcula, a partir dos arquivos de resultado, os numeros que o diario (diario_pesquisa.md, secoes das Entradas 286, 287, 288, 289 e 290) e revisao_cega_cec_2026-11-22.md atribuem a elas, e
confere cada um com a tolerancia do ultimo digito escrito. Nao toca no artigo (que nao foi alterado neste ciclo). Uso: python -m analysis.audit_entradas286_290
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
CHECKS: list[tuple[str, float, float, float]] = []


def chk(label, got, exp, tol):
    CHECKS.append((label, float(got), float(exp), float(tol)))


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(sstats.spearmanr(a[ok], b[ok]).statistic)


def audit_286():
    E = pd.read_csv(R / "entrada286_estimators.csv")
    g = lambda conj, est: E[(E.conjunto == conj) & (E.estimador == est)].iloc[0]
    r = g("69 CEC", "razao_de_somas")
    chk("286 captura agregada (69)", r.estimativa, 0.157, 0.0005); chk("286 IC cluster inferior", r.lo, 0.137, 0.0006); chk("286 IC cluster superior", r.hi, 0.189, 0.0006)
    f = g("69 CEC", "fieller_sobre_totais_por_cluster")
    chk("286 Fieller inferior", f.lo, 0.136, 0.0006); chk("286 Fieller superior", f.hi, 0.192, 0.0006)
    s = g("69 CEC", "inclinacao_pela_origem")
    chk("286 inclinacao pela origem", s.estimativa, 0.141, 0.0006); chk("286 inclinacao inferior", s.lo, 0.132, 0.0006); chk("286 inclinacao superior", s.hi, 0.174, 0.0006)
    p = g("69 CEC", "premio_offline_medio_pp")
    chk("286 premio offline", p.estimativa, 0.547, 0.0006); chk("286 premio inferior", p.lo, 0.287, 0.0006); chk("286 premio superior", p.hi, 0.832, 0.0006)
    m = [k for k in E.estimador if k.startswith("captura_por_celula")][0]
    mm = g("69 CEC", m)
    chk("286 captura por celula (mediana)", mm.estimativa, 0.141, 0.0006); chk("286 captura por celula (q25)", mm.lo, 0.078, 0.0006); chk("286 captura por celula (q75)", mm.hi, 0.227, 0.0006)
    CP = pd.read_csv(R / "entrada286_coupling.csv")
    c = CP[(CP.conjunto == "69 CEC") & (CP.escopo == "todas")].iloc[0]
    chk("286 Spearman completo", c.S_completo, 0.781, 0.0006); chk("286 acoplado a meio", c.S_acoplado_meio, 0.780, 0.0006); chk("286 desacoplado a meio", c.S_desacoplado_meio, 0.768, 0.0006)
    RU = pd.read_csv(R / "entrada286_rules.csv")
    for mg, old, strict, par in ((0.02, 29, 25, 19), (0.05, 25, 20, 36), (0.10, 19, 14, 47)):
        row = RU[(RU.conjunto == "69 CEC") & (np.isclose(RU.margem, mg))].iloc[0]
        chk(f"286 regra antiga {mg}", row.rt_alem_antiga, old, 0); chk(f"286 regra estrita {mg}", row.rt_alem_estrita, strict, 0); chk(f"286 TOST {mg}", row.rt_paridade_TOST, par, 0)
    PL = pd.read_csv(R / "entrada286_pools.csv")
    q = lambda sp_, me: PL[(PL.espaco == sp_) & (PL.metodo == me)].iloc[0].ganho_pp
    chk("286 pool S5 acao unica", q("S5", "pool"), -0.0718, 0.0006); chk("286 pool S15 acao unica", q("S15", "pool"), -0.0809, 0.0006)
    chk("286 pool S5 validada", q("S5", "val"), 0.092, 0.0006); chk("286 pool S5 rt", q("S5", "rt"), 0.0095, 0.0006); chk("286 pool S5 rt contra acao unica", q("S5", "rt_vs_pool"), 0.0797, 0.0006)
    chk("286 pool S5 validada contra acao unica", q("S5", "val_vs_pool"), 0.1622, 0.0006)
    S = pd.read_csv(R / "entrada286b_pool_sensitivity.csv")
    qb = lambda sp_, cr, me: S[(S.espaco == sp_) & S.criterio.str.startswith(cr) & (S.metrica == me)].iloc[0].ganho_pp
    chk("286b S5 retrospectivo", qb("S5", "c3", "pool"), -0.0633, 0.0006); chk("286b S15 retrospectivo", qb("S15", "c3", "pool"), -0.0809, 0.0006); chk("286b S5 balanceado", qb("S5", "c1", "pool"), -0.1084, 0.0006)
    chk("286 cota do vies em 18 configuracoes", pd.read_csv(R / "entrada286_sandwich_check.csv").ok_cota.sum(), 18, 0)


def audit_287():
    D = pd.read_csv(R / "entrada287_cells.csv")
    M = D[D.g_val >= 0.05]
    chk("287 celulas", len(D), 69, 0); chk("287 celulas com g_val >= 0,05", len(M), 51, 0)
    chk("287 taxa mediana de recompensa positiva", D.pos_rate.median(), 0.00533, 0.000006); chk("287 positivas por acao por execucao (mediana)", D.pos_per_arm_run.median(), 0.58, 0.005)
    chk("287 positivas por execucao (mediana)", D.pos_per_run.median(), 5.3, 0.05); chk("287 L mediano", D.L.median(), 0.21, 0.005); chk("287 fracao com L < 0,1", (D.L < 0.1).mean(), 0.275, 0.001)
    chk("287 Spearman kappa_ws x captura", sp(M.kappa_ws, M.kap_rt), -0.166, 0.0006); chk("287 Spearman L x captura", sp(M.L, M.kap_rt), -0.238, 0.0006)
    chk("287 mediana captura / kappa_ws", (M.kap_rt / M.kappa_ws)[M.kappa_ws > 0].median(), 0.43, 0.005)
    ok = D.g_ucb.notna()
    chk("287 Spearman ganho do UCB1 gemeo x observado", sp(D.tw_gain_ucb[ok], D.g_ucb[ok]), 0.354, 0.0006)
    chk("287 fracao kappa QL sem estados >= captura", (M.tw_kappa_ql >= M.kap_rt).mean(), 0.647, 0.001); chk("287 fracao G_max - G_U >= g_val", (D.tw_oracle >= D.g_val).mean(), 1.0, 0)
    chk("287 mediana tw_oracle / g_val", (D.tw_oracle / D.g_val)[D.g_val >= 0.05].median(), 6.1, 0.05)
    chk("287 fracao folga de fase > 0", (D.H_phase > 0).mean(), 0.754, 0.001); chk("287 fracao folga de estado > 0", (D.H_state > 0).mean(), 0.667, 0.001)
    chk("287 fidelidade das acoes fixas", D.static_fidelity.mean(), 0.483, 0.0006)
    chk("287 mediana folga de fase / vantagem", (D.H_phase / D.adv_static)[D.adv_static > 0].median(), 0.106, 0.0006); chk("287 mediana folga de estado / vantagem", (D.H_state / D.adv_static)[D.adv_static > 0].median(), 0.113, 0.0006)
    # 287b
    D["adv_cf"] = 1000 * D.delta_cf
    chk("287b T Delta_cf / g_val (mediana)", (D.adv_cf / D.g_val)[(D.g_val >= 0.05) & (D.adv_cf > 0)].median(), 5.75, 0.005)
    chk("287b Spearman positivas por acao x captura", sp(M.pos_per_arm_run, M.kap_rt), 0.546, 0.0006); chk("287b Spearman K x captura", sp(M.K, M.kap_rt), -0.123, 0.0006)
    q = pd.qcut(M["pos_per_arm_run"].rank(method="first"), 3, labels=["a", "b", "c"])
    caps = M.groupby(q, observed=True).kap_rt.median()
    chk("287b captura no terco baixo", caps["a"], 0.036, 0.0006); chk("287b captura no terco medio", caps["b"], 0.164, 0.0006); chk("287b captura no terco alto", caps["c"], 0.208, 0.0006)


def audit_288():
    C = pd.read_csv(R / "entrada288_cells.csv")
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "order288_*.csv")))], ignore_index=True)
    c0, c1 = C[C.tie == 0], C[C.tie == 1]
    chk("288 execucoes", len(D), 27360, 0); chk("288 celulas", c0.cell.nunique(), 19, 0)
    chk("288 R||Cmax com p < 0,05 (tie 0)", (c0[c0.coll == "padrao"].p < 0.05).sum(), 6, 0); chk("288 sinteticas com p < 0,05 (tie 0)", (c0[c0.coll == "sintetico"].p < 0.05).sum(), 11, 0)
    chk("288 celulas com Holm (tie 0)", c0.holm.sum(), 17, 0); chk("288 celulas com p < 0,05 (tie 1)", (c1.p < 0.05).sum(), 0, 0)
    chk("288 participacao do indice 0 (tie 0)", D[D.tie == 0].share_first.mean(), 0.751, 0.0006); chk("288 participacao do indice 0 (tie 1)", D[D.tie == 1].share_first.mean(), 0.2003, 0.00006)
    chk("288 cota media (tie 0)", D[D.tie == 0].bound.mean(), 0.631, 0.0006); chk("288 cota / participacao", D[D.tie == 0].bound.mean() / D[D.tie == 0].share_first.mean(), 0.84, 0.005)
    chk("288 fracao em linha zerada (tie 0)", D[D.tie == 0].share_zero.mean(), 0.719, 0.0006)
    chk("288 DP da ordem mediano (tie 0)", c0.sd_order.median(), 0.111, 0.0006); chk("288 DP da ordem mediano (tie 1)", c1.sd_order.median(), 0.029, 0.0006)
    chk("288 cota nunca excede a participacao por celula", (c0.bound <= c0.share_first).sum(), 19, 0)


def audit_289_290():
    f = R / "entrada289_cells.csv"
    if f.exists():
        import re
        G = pd.read_csv(f)
        S = pd.read_csv(R / "entrada289_summary.csv", index_col=0)
        L = pd.read_csv(R / "entrada289c_learners.csv").set_index("aprendiz")
        AL = pd.read_csv(R / "entrada289c_alignment.csv")
        txt = (R / "entrada289_de.txt").read_text(encoding="utf-8")
        nrows = len(pd.read_csv(R / "de_test_base_entrada289.csv")) + len(pd.read_csv(R / "de_test_learn_entrada289.csv"))
        chk("289 funcoes", len(G), 24, 0); chk("289 execucoes de teste", nrows, 9360, 0)
        G = G.set_index("fun")
        for fun, v in ((8, 8.435), (9, 7.063), (21, 3.392), (18, 3.113), (10, 2.111), (17, 1.789), (12, 1.331), (7, 1.210), (13, -0.538), (22, -0.408), (20, -0.339)):
            chk(f"289 ganho da validada f{fun}", G.loc[fun, "gain_val"], v, 0.0006)
        hi = G[G.gain_val >= 0.1]
        chk("289 funcoes com ganho da validada >= 0,1", len(hi), 10, 0); chk("289 mediana do ganho da validada nessas", hi.gain_val.median(), 1.95, 0.005)
        chk("289 minimo de sucessos por acao por execucao", G.succ_per_arm_run.min(), 177.3, 0.05); chk("289 maximo de sucessos por acao por execucao", G.succ_per_arm_run.max(), 12910, 1)
        chk("289 menor spread", G.spread.min(), 0.09, 0.005); chk("289 maior spread", G.spread.max(), 9.2, 0.05)
        chk("289 validada bate o ql_rt (estrito)", G.val_beats_qlrt.sum(), 5, 0); chk("289 validada alem da margem sobre o aleatorio (estrito)", G.val_beats_rand.sum(), 2, 0)
        chk("289 mediana da captura de erro do ql_rt", S.loc["ql_rt", "mediana_captura"], -0.179, 0.0006); chk("289 mediana da captura de recompensa do ql_rt", S.loc["ql_rt", "mediana_captura_recompensa"], 0.194, 0.0006)
        for m, med in (("ql_std", -0.11), ("ql_rt", -0.18), ("ucb", -0.60), ("pg", -0.11), ("pm", -0.27), ("ap", -0.44), ("etc", -0.28)):
            chk(f"289c mediana do ganho {m}", L.loc[m, "mediana_ganho"], med, 0.006)
        for m, neg, pv in (("ql_std", 20, 0.0043), ("ql_rt", 15, 0.028), ("ucb", 20, 0.0), ("pg", 17, 0.017), ("pm", 16, 0.0006), ("ap", 20, 0.0001), ("etc", 16, 0.0060)):
            chk(f"289c funcoes com ganho negativo {m}", L.loc[m, "funcoes_ganho_negativo"], neg, 0); chk(f"289c postos sinalizados {m}", L.loc[m, "p_postos_sinalizados"], pv, 0.0006)
        chk("289c IC90 abaixo da margem UCB1", L.loc["ucb", "IC90_abaixo_da_margem"], 11, 0); chk("289c IC90 abaixo da margem AP", L.loc["ap", "IC90_abaixo_da_margem"], 14, 0)
        chk("289c pior que o aleatorio em mais de 1 decada UCB1", L.loc["ucb", "pior_que_aleatorio_em_mais_de_1_decada"], 9, 0); chk("289c idem AP", L.loc["ap", "pior_que_aleatorio_em_mais_de_1_decada"], 10, 0)
        chk("289c acao de maior sucesso = validada", AL.coincidem.sum(), 5, 0); chk("289c UCB1 na acao de maior sucesso", AL.ucb_na_sucesso.mean(), 0.377, 0.0006); chk("289c UCB1 na validada", AL.ucb_na_val.mean(), 0.179, 0.0006)
        chk("289 participacao da acao 0 no ql_std", G.u0_qlstd.mean(), 0.630, 0.0006); chk("289 fracao |ql_rt - ql_std| <= 0,05", ((G.g_ql_rt - G.g_ql_std).abs() <= 0.05).mean(), 0.208, 0.001)
        m = re.search(r"ucb ([+-]\d\.\d+); pm ([+-]\d\.\d+); ap ([+-]\d\.\d+); etc ([+-]\d\.\d+); ql1 ([+-]\d\.\d+)", txt)
        for k, v in zip(("ucb", "pm", "ap", "etc", "ql1"), (0.472, 0.353, 0.483, 0.529, 0.630)):
            chk(f"289 Spearman gemeo x observado {k}", float(m.group(("ucb", "pm", "ap", "etc", "ql1").index(k) + 1)), v, 0.0006)
    f = R / "entrada290_cells.csv"
    if f.exists():
        C = pd.read_csv(f)
        chk("290 celulas (tipo, T)", len(C), 21, 0); chk("290 celulas com ganho da validada > 0", (C.g_val > 0).sum(), 21, 0)
        g = C.groupby("T").agg(pos=("pos_per_arm_run", "median"), gval=("g_val", "median"), grt=("g_ql_rt", "median"), cap=("cap_rt", "median"), beats=("val_beats_rt_strict", "sum"))
        for T, pos, gval, grt, cap, beats in ((1000, 2.15, 0.93, 0.159, 0.331, 6), (4000, 4.22, 1.12, 0.155, 0.363, 6), (16000, 7.97, 0.91, 0.164, 0.179, 5)):
            r = g.loc[T]
            chk(f"290 sucessos por acao por execucao T={T}", r.pos, pos, 0.005); chk(f"290 ganho da validada T={T}", r.gval, gval, 0.006); chk(f"290 ganho do ql_rt T={T}", r.grt, grt, 0.0006)
            chk(f"290 captura do ql_rt T={T}", r.cap, cap, 0.0006); chk(f"290 validada bate o ql_rt (estrito) T={T}", r.beats, beats, 0)
        piv = C.pivot(index="type", columns="T", values="pos_per_arm_run")
        chk("290 razao dos sucessos 16000 / 1000", (piv[16000] / piv[1000]).median(), 3.1, 0.05)
        capp = C.pivot(index="type", columns="T", values="cap_rt")
        chk("290 tipos com captura maior em T = 16000", (capp[16000] > capp[1000]).sum(), 3, 0)
        ok = C.cap_rt.notna()
        chk("290 Spearman log sucessos x captura", sp(np.log(C.pos_per_arm_run[ok]), C.cap_rt[ok]), 0.296, 0.0006)


def audit_291():
    f = R / "entrada291_summary.csv"
    if not f.exists():
        return
    S = pd.read_csv(f).set_index(["credito", "aprendiz"])
    for m, v in (("ql_std", -0.035), ("ql_rt", -0.005), ("ucb", -0.109), ("pg", 0.000), ("pm", -0.119), ("ap", -0.150), ("etc", -0.078)):
        chk(f"291 log_incumbent mediana do ganho {m}", S.loc[("log_incumbent", m), "mediana_ganho"], v, 0.0006)
    for m, v in (("ql_std", -0.097), ("ql_rt", -0.050), ("ucb", -0.108), ("pg", -0.006), ("pm", -0.213), ("ap", -0.257), ("etc", -0.089)):
        chk(f"291 rel_incumbent mediana do ganho {m}", S.loc[("rel_incumbent", m), "mediana_ganho"], v, 0.0006)
    chk("291 maior captura (log_incumbent)", S.loc["log_incumbent"].mediana_captura.max(), 0.007, 0.0006)
    for m, p in (("ql_rt", 0.31), ("pg", 0.95), ("ql_std", 0.12), ("ucb", 0.002), ("pm", 0.003), ("ap", 0.018)):
        chk(f"291 postos sinalizados {m} (log_incumbent)", S.loc[("log_incumbent", m), "p_postos"], p, 0.006 if p > 0.05 else 0.0006)
    chk("291 rel_incumbent ql_rt p", S.loc[("rel_incumbent", "ql_rt"), "p_postos"], 0.042, 0.0006)
    chk("291 participacao do UCB1 na validada (log)", S.loc[("log_incumbent", "ucb"), "na_validada"], 0.205, 0.0006)
    import re
    txt = (R / "entrada291_credit.txt").read_text(encoding="utf-8")
    m = re.search(r"sucesso ([\d.]+); log_incumbent ([\d.]+); rel_incumbent ([\d.]+)", txt)
    chk("291 fracao com credito = validada (sucesso)", float(m.group(1)), 0.208, 0.0006); chk("291 fracao (log_incumbent)", float(m.group(2)), 0.167, 0.0006); chk("291 fracao (rel_incumbent)", float(m.group(3)), 0.167, 0.0006)
    chk("291 execucoes novas", len(pd.read_csv(R / "de_credit_learn_entrada291.csv")), 10080, 0)


def main():
    audit_286(); audit_287(); audit_288(); audit_289_290(); audit_291()
    bad = 0
    for label, got, exp, tol in CHECKS:
        ok = abs(got - exp) <= tol + 1e-12
        bad += (not ok)
        if not ok:
            print(f"FALHA  {label}: calculado {got:.6g}, no diario {exp:.6g} (tolerancia {tol:g})")
    print(f"{len(CHECKS) - bad}/{len(CHECKS)} verificacoes OK")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
