"""
analysis/report_entrada286_estimands.py (Entrada 286, Ciclo 53)
================================================================
ESTIMANDOS, INFERENCIA E ACOPLAMENTO, em dados existentes (nada novo e' rodado). Resposta a quatro pontos da segunda revisao cega do artigo do CEC (revisao_cega_cec_2026-11-22.md, S3, S4, S5, S9):
a razao de captura e' mal condicionada, a correlacao de Spearman de 0,78 pode estar inflada porque o ganho do aprendiz e o da validada sao diferencas do MESMO braco aleatorio nas mesmas instancias,
'alem da margem' e' frouxo (Holm + IC90 'nao contido em +-m' conta um ganho de 0,02 com IC [0,004; 0,06]) e o competidor estatico usa o rotulo da classe.

Fontes (todas congeladas): as tabelas por instancia das 85 celulas das Entradas 280/283 (analysis/cells_instance_tables.py, copia da carga de report_entrada280_tiesall.py; reproduz os 85 ganhos de results/entrada283_cells.csv
sem diferenca), results/entrada283_cells.csv (coluna 'cluster'), results/agents_tune_S{5,15}_entrada180.csv (ajuste, 30 instancias por classe), results/entrada257_sim_crossfit.csv (simulacao do estimador, Entrada 257).
Conjunto principal: as 69 celulas do artigo do CEC (todas menos o TSP); as 85 celulas vao como verificacao.

Estimandos (ganho = -100 (m - m_aleatorio)/m_aleatorio em pontos, positivo = melhor; 'aprendiz' = Q-Learning com desempate aleatorio, rt):
  premio offline por celula  P_c = g_validada - g_rt  (diferenca, estimando primario); media sobre as celulas com IC por bootstrap de CLUSTER (coluna 'cluster'; 2.000 reamostras, IC90).
  captura agregada  R = soma g_rt / soma g_validada  (razao de somas), com (i) IC90 por bootstrap de cluster e (ii) intervalo de Fieller sobre os totais por cluster (aproximacao: cluster = unidade amostral).
  inclinacao pela origem  b = soma(g_rt g_val) / soma(g_val^2), IC90 por bootstrap de cluster.
  captura por celula so' onde g_validada >= 0,05 (denominador afastado de zero): mediana e quartis.
Acoplamento: divisao aleatoria das INSTANCIAS de cada celula em duas metades A e B (200 divisoes); Spearman entre celulas de (i) ganhos completos, (ii) acoplado a meio tamanho (g_rt(A) com g_val(A)), (iii) DESACOPLADO a meio tamanho
(g_rt(A) com g_val(B), e o inverso): (ii) menos (iii) mede o acoplamento ao mesmo ruido amostral; (i) menos (ii) mede a perda por reduzir a amostra.
Regras: para margem m em {0,02; 0,05; 0,10}: regra antiga = Holm (Wilcoxon, por colecao) e IC90 do ganho NAO contido em (-m, m); regra ESTRITA = Holm e IC90 inteiro alem da margem (limite inferior do ganho > m);
paridade (TOST) = IC90 contido em (-m, m), com ou sem significancia. Mesmo para a validada contra o aprendiz.
Pools sem rotulo de classe (P||Cmax sintetico, S5 e S15; nas demais colecoes nao ha' braco fixo por instancia no teste): acao estatica UNICA para as 12 classes juntas, escolhida nas instancias de ajuste (menor lacuna media ao limite inferior,
todas as classes, mesmo peso por instancia), contra a validada por classe, o rt, o UCB1 e o Q-Learning padrao; ganho do pool = media das 12 classes; IC90 por bootstrap ESTRATIFICADO (instancias dentro de cada classe; 2.000).
Cota do vies do oraculo de uma semente (Entrada 257): para cada uma das 18 configuracoes, vies_um <= sigma raiz(2 ln A) (maximo esperado de A ruidos sub-gaussianos de desvio sigma).

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hashes em results/entrada286_prereg_hashes.txt). Probabilidades do autor. Ressalva de transparencia: os ganhos por celula, a captura agregada (0,157; IC por classe
0,137-0,187 nas 85 celulas) e o Spearman de 0,78 ja' eram conhecidos (Entrada 283), entao P1 e P3 sao informadas por eles; o acoplamento, a regra estrita, o TOST, o intervalo de Fieller, a inclinacao, o premio e os pools NAO tinham sido calculados.
(P1) Nas 69 celulas do CEC, o IC90 por cluster da captura agregada (razao de somas) fica inteiro em [0,10; 0,25]. (80%)
(P2) O intervalo de Fieller e o IC por cluster concordam a menos de 0,03 nos dois extremos. (70%)
(P3) O Spearman DESACOPLADO a meio tamanho entre as 69 celulas e' de pelo menos 0,60 (media das 200 divisoes). (70%)
(P4) O acoplamento infla o Spearman em no maximo 0,10: |(ii) - (iii)| <= 0,10 nas 69 celulas. (65%)
(P5) Regra estrita com margem 0,05: o rt ganha do aleatorio alem da margem em no maximo 20 das 69 celulas (regra antiga: 25). (70%)
(P6) Paridade TOST com margem 0,05: o IC90 do ganho do rt fica dentro de +-0,05 em pelo menos 35 das 69 celulas. (60%)
(P7) O premio offline medio (g_validada - g_rt) sobre as 69 celulas tem IC90 por cluster acima de zero. (95%)
(P8) A inclinacao pela origem do ganho do rt sobre o da validada nas 69 celulas tem IC90 inteiro em [0,08; 0,30]. (75%)
(P9) A cota sigma raiz(2 ln A) vale nas 18 configuracoes da simulacao da Entrada 257 (vies_um <= cota). (95%)
(P10) No pool do P||Cmax sintetico S5, a acao estatica unica retem pelo menos 90% do ganho medio da validada por classe (razao de somas sobre as 12 classes). (60%)
(P11) Nos pools S5 e S15, o rt NAO ganha da acao estatica unica alem da margem 0,05 (IC90 do ganho relativo inteiro acima de +0,05 em nenhum dos dois). (80%)
(P12) No pool S5, o ganho medio da validada por classe supera o da acao estatica unica com IC90 estratificado acima de zero (o rotulo da classe vale algo). (50%)
Uso: python -m analysis.report_entrada286_estimands
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.cells_instance_tables import load_cells  # noqa: E402
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
MARGINS = (0.02, 0.05, 0.10)
NB = 2000
Z90 = 1.6448536269514722
pd.set_option("display.width", 220)


# ---------------------------------------------------------------------------------------------- utilidades
def boot_cluster(cluster, fn, nb=NB, seed=286):
    """IC90 por bootstrap de cluster: reamostra clusters com reposicao; fn recebe o vetor de indices das celulas escolhidas."""
    keys = pd.unique(cluster)
    idx = [np.where(cluster == k)[0] for k in keys]
    rng = np.random.default_rng(seed)
    vals = np.empty(nb)
    for b in range(nb):
        pick = rng.integers(0, len(keys), len(keys))
        vals[b] = fn(np.concatenate([idx[i] for i in pick]))
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def fieller(a_c, b_c, z=Z90):
    """Intervalo de Fieller para A/B, com A = soma a_c, B = soma b_c e os clusters como unidades amostrais independentes."""
    a_c, b_c = np.asarray(a_c, float), np.asarray(b_c, float)
    n = len(a_c)
    A, B = a_c.sum(), b_c.sum()
    C = np.cov(np.vstack([a_c, b_c]), ddof=1) * n
    vaa, vab, vbb = C[0, 0], C[0, 1], C[1, 1]
    qa, qb, qc = B * B - z * z * vbb, -2.0 * (A * B - z * z * vab), A * A - z * z * vaa
    if qa <= 0:
        return float("nan"), float("nan")
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return float("nan"), float("nan")
    r1, r2 = (-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)
    return float(min(r1, r2)), float(max(r1, r2))


def spearman(a, b):
    return float(sstats.spearmanr(a, b).statistic)


# ---------------------------------------------------------------------------------------------- testes por celula e familias de Holm (como a Entrada 280)
def cell_table(cells):
    rows = []
    for coll in dict.fromkeys(c["coll"] for c in cells):
        cs = [c for c in cells if c["coll"] == coll]
        TA = [test(c["x"].rt, c["x"].rand) for c in cs]
        TB = [test(c["x"].val, c["x"].rt) for c in cs]
        FA, FB = family(TA), family(TB)
        for c, (_, fa), (_, fb) in zip(cs, FA.iterrows(), FB.iterrows()):
            x = c["x"]
            rows.append(dict(coll=coll, cell=c["cell"], n=len(x), g_val=float(-rel(x.val, x.rand).mean()), g_rt=float(-rel(x.rt, x.rand).mean()),
                             a_lo=float(fa.ci90_lo), a_hi=float(fa.ci90_hi), a_conf=bool(fa.confirmada), b_lo=float(fb.ci90_lo), b_hi=float(fb.ci90_hi), b_conf=bool(fb.confirmada)))
    T = pd.DataFrame(rows)
    cl = pd.read_csv(R / "entrada283_cells.csv")[["coll", "cell", "cluster"]]
    T = T.merge(cl, on=["coll", "cell"], how="left")
    assert T.cluster.notna().all()
    return T


def rules_table(T, label):
    out = []
    for m in MARGINS:
        old = (T.a_conf & ~((T.a_lo > -m) & (T.a_hi < m))).sum()
        strict = (T.a_conf & (T.a_hi < -m)).sum()
        parity = ((T.a_lo > -m) & (T.a_hi < m)).sum()
        worse = (T.a_lo > m).sum()
        vold = (T.b_conf & ~((T.b_lo > -m) & (T.b_hi < m))).sum()
        vstrict = (T.b_conf & (T.b_hi < -m)).sum()
        rt_beats = (T.b_lo > m).sum()
        out.append(dict(conjunto=label, margem=m, rt_alem_antiga=int(old), rt_alem_estrita=int(strict), rt_paridade_TOST=int(parity), rt_pior_que_aleatorio_alem=int(worse),
                        val_bate_rt_antiga=int(vold), val_bate_rt_estrita=int(vstrict), rt_bate_val_alem=int(rt_beats), celulas=len(T)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- estimandos
def estimators(T, label):
    gv, gr, cl = T.g_val.to_numpy(), T.g_rt.to_numpy(), T.cluster.to_numpy()
    res = {}
    ratio = lambda ix: gr[ix].sum() / gv[ix].sum()
    slope = lambda ix: (gr[ix] * gv[ix]).sum() / (gv[ix] ** 2).sum()
    prem = lambda ix: (gv[ix] - gr[ix]).mean()
    allix = np.arange(len(T))
    res["razao_de_somas"] = (ratio(allix),) + boot_cluster(cl, ratio)
    ks = pd.unique(cl)
    a_c = np.array([gr[cl == k].sum() for k in ks])
    b_c = np.array([gv[cl == k].sum() for k in ks])
    res["fieller_sobre_totais_por_cluster"] = (ratio(allix),) + fieller(a_c, b_c)
    res["inclinacao_pela_origem"] = (slope(allix),) + boot_cluster(cl, slope)
    res["premio_offline_medio_pp"] = (prem(allix),) + boot_cluster(cl, prem)
    mask = gv >= 0.05
    cap = gr[mask] / gv[mask]
    res["captura_por_celula_gval>=0,05 (mediana, q25, q75)"] = (float(np.median(cap)), float(np.percentile(cap, 25)), float(np.percentile(cap, 75)))
    rows = [dict(conjunto=label, estimador=k, estimativa=v[0], lo=v[1], hi=v[2], n_celulas=len(T), n_clusters=len(ks), n_cap=int(mask.sum())) for k, v in res.items()]
    return pd.DataFrame(rows), res


def coupling(cells, label, nsplit=200, seed=286):
    rng = np.random.default_rng(seed)
    data = []
    for c in cells:
        x = c["x"]
        data.append((c["coll"], np.asarray(-rel(x.val, x.rand)), np.asarray(-rel(x.rt, x.rand))))
    colls = np.array([d[0] for d in data])
    full_v = np.array([d[1].mean() for d in data])
    full_r = np.array([d[2].mean() for d in data])
    S_full = spearman(full_r, full_v)
    acc = {"cH": [], "dH": []}
    per = {k: {"cH": [], "dH": []} for k in dict.fromkeys(colls)}
    for _ in range(nsplit):
        aV, aR, bV, bR = [], [], [], []
        for (_, v, r) in data:
            p = rng.permutation(len(v))
            h = len(v) // 2
            A, B = p[:h], p[h:2 * h]
            aV.append(v[A].mean()); aR.append(r[A].mean()); bV.append(v[B].mean()); bR.append(r[B].mean())
        aV, aR, bV, bR = map(np.array, (aV, aR, bV, bR))
        acc["cH"].append(0.5 * (spearman(aR, aV) + spearman(bR, bV)))
        acc["dH"].append(0.5 * (spearman(aR, bV) + spearman(bR, aV)))
        for k in per:
            mk = colls == k
            per[k]["cH"].append(0.5 * (spearman(aR[mk], aV[mk]) + spearman(bR[mk], bV[mk])))
            per[k]["dH"].append(0.5 * (spearman(aR[mk], bV[mk]) + spearman(bR[mk], aV[mk])))
    rows = [dict(conjunto=label, escopo="todas", n=len(data), S_completo=S_full, S_acoplado_meio=float(np.mean(acc["cH"])), S_desacoplado_meio=float(np.mean(acc["dH"])),
                 dH_p05=float(np.percentile(acc["dH"], 5)), dH_p95=float(np.percentile(acc["dH"], 95)))]
    for k in per:
        mk = colls == k
        rows.append(dict(conjunto=label, escopo=k, n=int(mk.sum()), S_completo=spearman(full_r[mk], full_v[mk]), S_acoplado_meio=float(np.mean(per[k]["cH"])),
                         S_desacoplado_meio=float(np.mean(per[k]["dH"])), dH_p05=float(np.percentile(per[k]["dH"], 5)), dH_p95=float(np.percentile(per[k]["dH"], 95))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- pools sem rotulo de classe
def pools(cells, nb=NB, seed=286):
    rng = np.random.default_rng(seed)
    rows, detail = [], {}
    for sp in ("S5", "S15"):
        cs = [c for c in cells if c["coll"] == "sintetico" and c["space"] == sp]
        tune = pd.read_csv(R / f"agents_tune_{sp}_entrada180.csv")
        tune = tune[tune.method.str.startswith("arm")].copy()
        tune["a"] = tune.method.str[3:].astype(int)
        tune["gap"] = 100.0 * (tune.makespan - tune.lb) / tune.lb
        a_pool = int(tune.groupby("a").gap.mean().idxmin())
        a_cls = {cl: int(g.groupby("a").gap.mean().idxmin()) for cl, g in tune.groupby("cls")}
        series = {}
        for c in cs:
            x = c["x"]
            cl = c["cell"].split("|", 1)[1]
            series[cl] = dict(pool=np.asarray(-rel(x[f"arm{a_pool}"], x.rand)), val=np.asarray(-rel(x.val, x.rand)), rt=np.asarray(-rel(x.rt, x.rand)),
                              ucb=np.asarray(-rel(x.ucb, x.rand)), ql=np.asarray(-rel(x.ql, x.rand)), rt_vs_pool=np.asarray(-rel(x.rt, x[f"arm{a_pool}"])),
                              val_vs_pool=np.asarray(-rel(x.val, x[f"arm{a_pool}"])))
        cls_ = sorted(series)
        # bootstrap estratificado: reamostra instancias dentro de cada classe; o ganho do pool e' a media das classes
        idx = {cl: rng.integers(0, len(series[cl]["pool"]), size=(nb, len(series[cl]["pool"]))) for cl in cls_}
        for me in ("pool", "val", "rt", "ucb", "ql", "rt_vs_pool", "val_vs_pool"):
            pt = float(np.mean([series[cl][me].mean() for cl in cls_]))
            bs = np.mean([series[cl][me][idx[cl]].mean(axis=1) for cl in cls_], axis=0)
            lo, hi = np.percentile(bs, [5, 95])
            rows.append(dict(espaco=sp, metodo=me, ganho_pp=pt, ci90_lo=float(lo), ci90_hi=float(hi), acao_unica=a_pool, n_classes=len(cls_)))
        detail[sp] = dict(a_pool=a_pool, a_cls=a_cls, n_distintas=len(set(a_cls.values())))
    return pd.DataFrame(rows), detail


def sandwich_check():
    D = pd.read_csv(R / "entrada257_sim_crossfit.csv")
    D["cota"] = D.sigma * np.sqrt(2.0 * np.log(D.A))
    D["ok_cota"] = D.bias_um <= D.cota
    D["ok_um"] = D.H_um >= D.H_true - 0.01
    D["ok_cv"] = D.H_cv <= D.H_true + 0.01
    return D


# ---------------------------------------------------------------------------------------------- principal
def main():
    L = ["=" * 118, "ESTIMANDOS, INFERENCIA E ACOPLAMENTO EM DADOS EXISTENTES (Entrada 286)", "=" * 118]
    cells = load_cells(include_tsp=True)
    T85 = cell_table(cells)
    T69 = T85[T85.coll != "tsp"].reset_index(drop=True)
    cells69 = [c for c in cells if c["coll"] != "tsp"]
    L += [f"celulas: {len(T85)} (69 do CEC + 16 do TSP); instancias por celula: min {T85.n.min()}, mediana {int(T85.n.median())}, max {T85.n.max()}", ""]
    old = pd.read_csv(R / "entrada283_cells.csv")
    chk = {m: int(old[f"rt_beyond_{m:.2f}"].sum()) for m in MARGINS}
    # ---- regras
    RU = pd.concat([rules_table(T69, "69 CEC"), rules_table(T85, "85")])
    L += ["--- regras de decisao (rt contra o aleatorio; validada contra o rt) ---", RU.to_string(index=False), ""]
    mine = {m: int((T85.a_conf & ~((T85.a_lo > -m) & (T85.a_hi < m))).sum()) for m in MARGINS}
    L += [f"conferencia da regra antiga com a Entrada 283 (85 celulas; margens {MARGINS}): aqui {mine}; na Entrada 283 {chk}", ""]
    # ---- estimandos
    E69, e69 = estimators(T69, "69 CEC")
    E85, e85 = estimators(T85, "85")
    EE = pd.concat([E69, E85])
    L += ["--- estimandos (ganho em pontos; IC90 por bootstrap de cluster, Fieller sobre os totais por cluster) ---", EE.round(4).to_string(index=False), ""]
    pc = T69.assign(premio=T69.g_val - T69.g_rt, captura=lambda d: np.where(d.g_val >= 0.05, d.g_rt / d.g_val, np.nan))
    byc = pc.groupby("coll").agg(celulas=("cell", "size"), g_val=("g_val", "mean"), g_rt=("g_rt", "mean"), premio=("premio", "mean"), razao_somas=("g_rt", "sum")).reset_index()
    byc["razao_somas"] = pc.groupby("coll").g_rt.sum().to_numpy() / pc.groupby("coll").g_val.sum().to_numpy()
    L += ["--- por colecao (69 celulas) ---", byc.round(4).to_string(index=False), ""]
    # ---- acoplamento
    CP = pd.concat([coupling(cells69, "69 CEC"), coupling(cells, "85")])
    L += ["--- acoplamento: Spearman entre celulas do ganho do rt e do ganho da validada (200 divisoes das instancias) ---", CP.round(3).to_string(index=False), ""]
    # ---- pools
    PL, pdet = pools(cells)
    L += ["--- pools sem rotulo de classe (P||Cmax sintetico; ganho medio das 12 classes, pontos; IC90 estratificado) ---", PL.round(4).to_string(index=False),
          f"acoes unicas escolhidas no ajuste: {pdet['S5']['a_pool']} (S5), {pdet['S15']['a_pool']} (S15); acoes validadas distintas entre as 12 classes: {pdet['S5']['n_distintas']} (S5), {pdet['S15']['n_distintas']} (S15)", ""]
    # ---- cota do vies do oraculo
    SW = sandwich_check()
    L += ["--- cota do vies do oraculo de uma semente (Entrada 257): vies_um <= sigma raiz(2 ln A) ---",
          SW[["A", "tau", "sigma", "bias_um", "cota", "ok_cota", "bias_cv", "ok_um", "ok_cv"]].round(4).to_string(index=False), ""]
    # ---- previsoes
    r = e69["razao_de_somas"]
    f = e69["fieller_sobre_totais_por_cluster"]
    cp = CP[(CP.conjunto == "69 CEC") & (CP.escopo == "todas")].iloc[0]
    st = RU[(RU.conjunto == "69 CEC") & (RU.margem == 0.05)].iloc[0]
    sl = e69["inclinacao_pela_origem"]
    pr = e69["premio_offline_medio_pp"]
    g = lambda sp, me: PL[(PL.espaco == sp) & (PL.metodo == me)].iloc[0]
    ret5 = g("S5", "pool").ganho_pp / g("S5", "val").ganho_pp
    rtp = [g(sp, "rt_vs_pool") for sp in ("S5", "S15")]
    vp5 = g("S5", "val_vs_pool")
    yn = lambda b: "SIM" if b else "NAO"
    preds = [
        ("P1", f"capture agregada (69 celulas): {r[0]:.3f}, IC90 por cluster [{r[1]:.3f}; {r[2]:.3f}] (previsto inteiro em [0,10; 0,25])", 0.10 <= r[1] and r[2] <= 0.25),
        ("P2", f"Fieller [{f[1]:.3f}; {f[2]:.3f}] contra cluster [{r[1]:.3f}; {r[2]:.3f}] (previsto |dif| < 0,03 nos dois extremos)", abs(f[1] - r[1]) < 0.03 and abs(f[2] - r[2]) < 0.03),
        ("P3", f"Spearman desacoplado a meio tamanho (69): {cp.S_desacoplado_meio:.3f} (previsto >= 0,60)", cp.S_desacoplado_meio >= 0.60),
        ("P4", f"acoplamento = acoplado - desacoplado a meio tamanho: {cp.S_acoplado_meio - cp.S_desacoplado_meio:+.3f} (previsto |.| <= 0,10)", abs(cp.S_acoplado_meio - cp.S_desacoplado_meio) <= 0.10),
        ("P5", f"regra estrita, margem 0,05: rt alem da margem em {int(st.rt_alem_estrita)} de 69 (antiga: {int(st.rt_alem_antiga)}) (previsto <= 20)", st.rt_alem_estrita <= 20),
        ("P6", f"paridade TOST, margem 0,05: {int(st.rt_paridade_TOST)} de 69 (previsto >= 35)", st.rt_paridade_TOST >= 35),
        ("P7", f"premio offline medio (69): {pr[0]:.3f} pp, IC90 por cluster [{pr[1]:.3f}; {pr[2]:.3f}] (previsto limite inferior > 0)", pr[1] > 0),
        ("P8", f"inclinacao pela origem (69): {sl[0]:.3f}, IC90 [{sl[1]:.3f}; {sl[2]:.3f}] (previsto inteiro em [0,08; 0,30])", 0.08 <= sl[1] and sl[2] <= 0.30),
        ("P9", f"cota sigma raiz(2 ln A) vale em {int(SW.ok_cota.sum())} de {len(SW)} configuracoes (previsto 18 de 18)", bool(SW.ok_cota.all())),
        ("P10", f"pool S5: a acao unica retem {ret5:.3f} do ganho medio da validada por classe (previsto >= 0,90)", ret5 >= 0.90),
        ("P11", f"rt contra a acao unica: S5 IC90 [{rtp[0].ci90_lo:.3f}; {rtp[0].ci90_hi:.3f}], S15 [{rtp[1].ci90_lo:.3f}; {rtp[1].ci90_hi:.3f}] (previsto limite inferior <= 0,05 nos dois)", rtp[0].ci90_lo <= 0.05 and rtp[1].ci90_lo <= 0.05),
        ("P12", f"pool S5: validada contra acao unica {vp5.ganho_pp:+.3f} pp, IC90 [{vp5.ci90_lo:+.3f}; {vp5.ci90_hi:+.3f}] (previsto limite inferior > 0)", vp5.ci90_lo > 0),
    ]
    L += ["--- previsoes ---"] + [f"({k}) {t}: {yn(ok)}" for k, t, ok in preds] + [f"previsoes acertadas: {sum(ok for _, _, ok in preds)} de {len(preds)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada286_estimands.txt").write_text(text + "\n", encoding="utf-8")
    T85.to_csv(R / "entrada286_cells.csv", index=False)
    RU.to_csv(R / "entrada286_rules.csv", index=False)
    EE.to_csv(R / "entrada286_estimators.csv", index=False)
    CP.to_csv(R / "entrada286_coupling.csv", index=False)
    PL.to_csv(R / "entrada286_pools.csv", index=False)
    SW.to_csv(R / "entrada286_sandwich_check.csv", index=False)


if __name__ == "__main__":
    main()
