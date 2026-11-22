"""
analysis/report_entrada283_reanalysis.py
=========================================
Entrada 283 (Ciclo 51): REANALISE, SEM EXECUCAO NOVA, PARA A REVISAO DO ARTIGO DO CEC (resposta a revisao duplo-cega de 18/11/2026, pontos P1, P2, P5, P6, P9). Responde a quatro criticas com dados que ja existem:
(a) "a folga e' condicao necessaria" era quase uma definicao e a folga com validacao cruzada nao e' um limite (o ganho da validada a supera nas 10 colunas): troca-se a folga pela RAZAO DE CAPTURA (ganho do aprendiz / ganho da validada) e pelo ganho
da validada como medida do que se pode obter sem aprender dentro da execucao; (b) inferencia: celulas que compartilham instancias (S5 e S15 da mesma classe) nao sao independentes: bootstrap por CLASSE (cluster) nos intervalos e nas correlacoes;
(c) sensibilidade da margem (0,02; 0,05; 0,10) para o aprendiz de desempate aleatorio; (d) o experimento da ordem das acoes numa so tabela (padrao, lista invertida, validada em primeiro, permutada por execucao, desempate aleatorio), mais a
equivalencia "Q-Learning padrao = repetir a primeira acao" nas classes sinteticas e a tabela de classes (n, m, n/m, otimo provado, lacuna absoluta ao limite, ganho da validada em unidades de makespan).
Fontes (todas congeladas): results/entrada280_tiesall_cells.csv e _tests.csv (85 celulas), results/mech_*_entrada258.csv e mech2_*_entrada258.csv (diagnostico de ordem na R||Cmax de Fanjul-Peyro: Q-Learning padrao, lista invertida, validada
em primeiro), results/entrada260_qperm_cells_fanjul.csv, results/agents_S{3,5,15,30}_entrada180.csv (bracos fixos, aprendizes originais, validada), results/exact_reference_synthetic_entrada101.csv, results/entrada209_cforacle_cells.csv,
results/entrada192_published_cells.csv.

ESTATUTO: os valores por celula das Entradas 258, 260 e 280 ja tinham sido vistos quando as previsoes abaixo foram escritas; o que NAO tinha sido calculado e' cada estatistica abaixo (razao de captura agrupada, bootstrap por classe,
sensibilidade da margem, equivalencia com a primeira acao, tabela de classes). As previsoes sao, portanto, de reanalise e NAO entram no total das previsoes registradas do manuscrito.

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hashes em results/entrada283_prereg_hashes.txt):
(W1) A razao de captura agrupada do aprendiz de desempate aleatorio (soma dos ganhos / soma dos ganhos da validada, nas 85 celulas) esta entre 0,08 e 0,30. (70%)
(W2) O limite superior do IC90 dessa razao (bootstrap por classe, 39 classes) e' menor que 0,40. (75%)
(W3) A correlacao de Spearman agrupada entre o ganho do desempate aleatorio e o ganho da validada nas 85 celulas e' >= 0,60 e o limite inferior do IC90 (bootstrap por classe) e' > 0,30. (65%)
(W4) Entre as celulas em que a validada ganha no maximo 0,05, o desempate aleatorio ganha alem da margem em no maximo 3. (65%)
(W5) A razao de captura agrupada da lista permutada por execucao e' negativa. (95%)
(W6) O Spearman agrupado do Q-Learning padrao com o ganho da validada e' menor que o do desempate aleatorio em pelo menos 0,15. (60%)
(W7) Sensibilidade da margem: o desempate aleatorio ganha alem da margem em pelo menos 40 das 85 celulas com margem 0,02 e em no maximo 25 com margem 0,10. (60%)
(W8) Nas classes sinteticas (48 celulas: 12 classes x 4 espacos), a mediana de |ganho do Q-Learning - ganho da acao 0| e' menor que a mediana de |ganho do UCB1 - ganho da acao 0| e e' no maximo 0,05. (60%)
(W9) O ganho da validada supera a folga com validacao cruzada em mais da metade das 60 celulas de P||Cmax (24 sinteticas, 18 de descoberta e 18 held-out publicadas). (80%)
(W10) Na R||Cmax de Fanjul-Peyro (B5, B15, B30): a fracao de escolhas do Q-Learning padrao na primeira acao listada esta entre 0,55 e 0,75 e a da lista invertida na (nova) primeira acao listada e' >= 0,50, nos tres espacos; a validada em primeiro recupera >= 0,75
do ganho da validada nos tres. (75%)
Uso: python -m analysis.report_entrada283_reanalysis
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import rel  # noqa: E402

R = ROOT / "results"
B_BOOT = 4000
RNG_SEED = 283
MARGINS = (0.02, 0.05, 0.10)
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)


def cluster_of(coll: str, cell: str) -> str:
    """Unidade de replicacao: a classe (as celulas S5/S15 -- B5/B15/B30, T5/T15 -- de uma mesma classe compartilham instancias)."""
    parts = cell.split("|")
    if coll == "padrao":      # B5|U1a100
        return f"padrao|{parts[1]}"
    if coll == "familias":    # S5|unc
        return f"familias|{parts[1]}"
    if coll == "sintetico":   # S5|rand_100j_20m
        return f"sintetico|{parts[1]}"
    if coll == "publicado":   # S5|B|0
        return f"publicado|{parts[1]}|{parts[2]}"
    return f"tsp|{parts[1]}|{parts[2]}"  # T5|clust|200


def boot(df: pd.DataFrame, fn, seed: int = RNG_SEED):
    """Ponto e IC90 percentil da estatistica fn(d) (d = DataFrame) com reamostragem de conglomerados (coluna 'cluster'), B_BOOT replicas. As colunas numericas sao pre-extraidas como matriz para a reamostragem ser rapida."""
    rng = np.random.default_rng(seed)
    cols = [c for c in df.columns if c != "cluster" and pd.api.types.is_numeric_dtype(df[c])]
    arr = df[cols].to_numpy(float)
    groups = {c: np.flatnonzero(df.cluster.to_numpy() == c) for c in df.cluster.unique()}
    keys = list(groups)
    point = fn(df)
    vals = []
    for _ in range(B_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        v = fn(pd.DataFrame(arr[idx], columns=cols))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [5, 95]) if len(vals) > 50 else (np.nan, np.nan)
    return float(point), float(lo), float(hi), len(keys)


def cap(col):
    return lambda d: float(d[col].sum() / d.g_val.sum()) if d.g_val.sum() != 0 else np.nan


def spear(col):
    def f(d):
        x, y = d[col].to_numpy(), d.g_val.to_numpy()
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            return np.nan
        return float(ss.spearmanr(x, y).statistic)
    return f


def load_cells() -> pd.DataFrame:
    C = pd.read_csv(R / "entrada280_tiesall_cells.csv")
    T = pd.read_csv(R / "entrada280_tiesall_tests.csv")
    A = T[T.id.str.startswith("A-")].copy()
    A["cell"] = A.id.str[2:]
    A = A.set_index(["coll", "cell"])
    C["cluster"] = [cluster_of(a, b) for a, b in zip(C.coll, C.cell)]
    for m in MARGINS:
        eq = (A["ci90_lo"] > -m) & (A["ci90_hi"] < m)
        beyond = (A["confirmada"] & ~eq)
        C[f"rt_beyond_{m:.2f}"] = [bool(beyond.loc[(a, b)]) for a, b in zip(C.coll, C.cell)]
    return C


def part_capture(C: pd.DataFrame, L: list, store: dict):
    L += ["", "-" * 110, "1. RAZAO DE CAPTURA (soma dos ganhos do aprendiz / soma dos ganhos da validada) e correlacao com o ganho da validada; IC90 por bootstrap de conglomerados (classes)", "-" * 110]
    rows = []
    groups = [("todas as 85", C)] + [(k, C[C.coll == k]) for k in ("padrao", "familias", "sintetico", "publicado", "tsp")]
    for name, d in groups:
        for lab, col in (("padrao", "g_ql"), ("permutada", "g_qp"), ("desempate aleatorio", "g_rt")):
            pt, lo, hi, nc = boot(d, cap(col))
            sp, slo, shi, _ = boot(d, spear(col))
            above = d[d.g_val > 0.05]
            med = float(np.median(above[col] / above.g_val)) if len(above) else np.nan
            rows.append(dict(grupo=name, aprendiz=lab, celulas=len(d), classes=nc, captura=pt, cap_lo=lo, cap_hi=hi, mediana_celulas_val_gt_005=med, spearman=sp, sp_lo=slo, sp_hi=shi))
    T = pd.DataFrame(rows)
    store["capture"] = T
    fmt = T.copy()
    for c in ("captura", "cap_lo", "cap_hi", "mediana_celulas_val_gt_005", "spearman", "sp_lo", "sp_hi"):
        fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    L.append(fmt.to_string(index=False))
    L.append("(com menos de 5 classes o IC por conglomerados nao e' informativo: a colecao 'familias' tem 3)")
    return T


def part_floor_margin(C: pd.DataFrame, L: list, store: dict):
    L += ["", "-" * 110, "2. CHAO E SENSIBILIDADE DA MARGEM (aprendiz de desempate aleatorio, 85 celulas)", "-" * 110]
    for thr in (0.05, 0.10):
        low = C[C.g_val <= thr]
        L.append(f"celulas em que a validada ganha no maximo {thr:.2f}: {len(low)}; nelas o desempate aleatorio ganha alem da margem 0,05 em {int(low['rt_beyond_0.05'].sum())}")
    rows = []
    for coll in ("padrao", "familias", "sintetico", "publicado", "tsp"):
        d = C[C.coll == coll]
        rows.append(dict(colecao=coll, celulas=len(d), **{f"alem_{m:.2f}": int(d[f"rt_beyond_{m:.2f}"].sum()) for m in MARGINS}))
    rows.append(dict(colecao="total", celulas=len(C), **{f"alem_{m:.2f}": int(C[f"rt_beyond_{m:.2f}"].sum()) for m in MARGINS}))
    T = pd.DataFrame(rows)
    store["margin"] = T
    L.append(T.to_string(index=False))
    return T


def part_order(L: list, store: dict):
    L += ["", "-" * 110, "3. A ORDEM DAS ACOES NA R||Cmax DE FANJUL-PEYRO (7 tipos x 3 espacos; ganho sobre o aleatorio a iteracoes iguais, em pontos)", "-" * 110]
    q260 = pd.read_csv(R / "entrada260_qperm_cells_fanjul.csv")
    C280 = pd.read_csv(R / "entrada280_tiesall_cells.csv")
    C280 = C280[C280.coll == "padrao"].copy()
    C280["space"] = C280.cell.str.split("|").str[0]
    C280["type"] = C280.cell.str.split("|").str[1]
    rows = []
    for sp, A in (("B5", 5), ("B15", 15), ("B30", 30)):
        d1 = pd.read_csv(R / f"mech_{sp}_entrada258.csv")
        d2 = pd.read_csv(R / f"mech2_{sp}_entrada258.csv")
        ucols = [f"used_{k}" for k in range(A)]

        def gain(d, meth):
            W = d.groupby(["type", "m", "n", "method"]).makespan.mean().unstack()
            return float(-rel(W[meth], W["rand"]).mean())

        def share_idx(d, meth, idx):
            x = d[d.method == meth]
            u = x[ucols].to_numpy(float)
            return float((u[:, idx] / u.sum(axis=1)).mean())

        x = d2[d2.method == "ql_v0"]
        u = x[ucols].to_numpy(float)
        sh_v0 = float((u[np.arange(len(x)), x.val_arm.to_numpy(int)] / u.sum(axis=1)).mean())
        g_val = gain(d2, "val")
        for lab, g, sh in (("padrao (lista original)", gain(d1, "ql"), share_idx(d1, "ql", 0)),
                           ("lista invertida", gain(d1, "ql_rev"), share_idx(d1, "ql_rev", A - 1)),
                           ("validada em primeiro", gain(d2, "ql_v0"), sh_v0)):
            rows.append(dict(espaco=sp, variante=lab, fracao_na_primeira_acao=sh, ganho=g, ganho_validada=g_val, captura=g / g_val))
        qp = q260[q260.space == sp]
        rt = C280[C280.space == sp]
        rows.append(dict(espaco=sp, variante="permutada por execucao", fracao_na_primeira_acao=np.nan, ganho=float(qp.g_qp.mean()), ganho_validada=float(qp.g_val.mean()), captura=float(qp.g_qp.sum() / qp.g_val.sum())))
        rows.append(dict(espaco=sp, variante="desempate aleatorio", fracao_na_primeira_acao=np.nan, ganho=float(rt.g_rt.mean()), ganho_validada=float(rt.g_val.mean()), captura=float(rt.g_rt.sum() / rt.g_val.sum())))
    T = pd.DataFrame(rows)
    store["order"] = T
    L.append(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return T


def part_first_action(L: list, store: dict):
    L += ["", "-" * 110, "4. O Q-LEARNING PADRAO E A PRIMEIRA ACAO NAS CLASSES SINTETICAS (12 classes x 4 espacos; ganho sobre o aleatorio, a iteracoes iguais)", "-" * 110]
    rows = []
    for sp in ("S3", "S5", "S15", "S30"):
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        W = d.groupby(["cls", "rep", "method"]).makespan.mean().unstack()
        for c, x in W.groupby(level=0):
            rows.append(dict(espaco=sp, cls=c, g_arm0=float(-rel(x["arm0"], x["rand"]).mean()), g_ql=float(-rel(x["ql"], x["rand"]).mean()),
                             g_ucb=float(-rel(x["ucb"], x["rand"]).mean()), g_pg=float(-rel(x["pg"], x["rand"]).mean()), g_val=float(-rel(x["val"], x["rand"]).mean())))
    T = pd.DataFrame(rows)
    store["first"] = T
    out = {}
    for k in ("ql", "ucb", "pg"):
        out[k] = dict(spearman=float(ss.spearmanr(T.g_arm0, T[f"g_{k}"]).statistic), pearson=float(np.corrcoef(T.g_arm0, T[f"g_{k}"])[0, 1]), med_abs_dif=float((T.g_arm0 - T[f"g_{k}"]).abs().median()))
    store["first_stats"] = out
    L.append(pd.DataFrame(out).T.to_string(float_format=lambda v: f"{v:.3f}"))
    L.append(f"celulas: {len(T)}; ganho medio da acao 0: {T.g_arm0.mean():.3f}; do Q-Learning: {T.g_ql.mean():.3f}; do UCB1: {T.g_ucb.mean():.3f}; do gradiente de politica: {T.g_pg.mean():.3f}; da validada: {T.g_val.mean():.3f}")
    return T


def part_classes(L: list, store: dict):
    L += ["", "-" * 110, "5. TABELA DE CLASSES SINTETICAS DE P||Cmax (n, m, n/m, otimo provado, lacuna absoluta do aleatorio ao limite, ganho da validada em unidades de makespan, folga com validacao cruzada)", "-" * 110]
    ref = pd.read_csv(R / "exact_reference_synthetic_entrada101.csv")
    P = ref.groupby("instance").agg(n=("n", "first"), m=("m", "first"), proven=("proven", "mean")).reset_index().rename(columns={"instance": "cls"})
    cf = pd.read_csv(R / "entrada209_cforacle_cells.csv")
    cf = cf[(cf.problem == "pcmax") & (cf.set == "syn")]
    rows = []
    for sp in ("S5", "S15"):
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        r = d[d.method == "rand"]
        W = d.groupby(["cls", "rep", "method"]).makespan.mean().unstack()
        for c, x in W.groupby(level=0):
            rr = r[r.cls == c]
            g = float(-rel(x["val"], x["rand"]).mean())
            mk = float(x["rand"].mean())
            hcf = float(cf[(cf.space == sp) & (cf.cell == c)].h_cf.iloc[0])
            rows.append(dict(espaco=sp, cls=c, makespan_medio=mk, lacuna_unidades=float((rr.makespan - rr.lb).mean()), lacuna_pct=float(100 * ((rr.makespan - rr.lb) / rr.lb).mean()),
                             no_limite=float((rr.makespan == rr.lb).mean()), g_val=g, g_val_unidades=g * mk / 100.0, h_cf=hcf))
    T = pd.DataFrame(rows).merge(P, on="cls")
    T["n_sobre_m"] = T.n / T.m
    T = T.sort_values(["espaco", "n_sobre_m", "n"]).reset_index(drop=True)
    store["classes"] = T
    L.append(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    L.append(f"otimo provado: {100 * ref.proven.mean():.1f}% das {len(ref)} instancias; n/m de {T.n_sobre_m.min():.1f} a {T.n_sobre_m.max():.1f}; ganho da validada em unidades de makespan: de {T.g_val_unidades.min():.3f} a {T.g_val_unidades.max():.3f}")
    return T


def part_cf_vs_val(L: list, store: dict):
    L += ["", "-" * 110, "6. A FOLGA COM VALIDACAO CRUZADA NAO E' LIMITE: ganho da validada contra folga cruzada, nas 60 celulas de P||Cmax", "-" * 110]
    cf = pd.read_csv(R / "entrada209_cforacle_cells.csv")
    cf = cf[cf.problem == "pcmax"]
    cls = store["classes"]
    rows = []
    for _, r in cls.iterrows():
        rows.append(dict(conjunto="sintetico", espaco=r.espaco, cell=r.cls, g_val=r.g_val, h_cf=r.h_cf))
    pub = pd.read_csv(R / "entrada192_published_cells.csv")
    pub["cell"] = pub.fam + "|" + pub.grp.astype(str)
    for sset, s209 in (("discovery", "discovery"), ("replication", "heldout")):
        for _, r in pub[pub.set == sset].iterrows():
            x = cf[(cf.set == s209) & (cf.space == r.space) & (cf.cell == r.cell)]
            rows.append(dict(conjunto=s209, espaco=r.space, cell=r.cell, g_val=-float(r.val), h_cf=float(x.h_cf.iloc[0])))  # entrada192 guarda d (negativo = ganho)
    T = pd.DataFrame(rows)
    T["val_maior"] = T.g_val > T.h_cf
    store["cfval"] = T
    L.append(f"celulas: {len(T)}; o ganho da validada supera a folga cruzada em {int(T.val_maior.sum())}; por conjunto: " + "; ".join(f"{k} {int(v.val_maior.sum())} de {len(v)}" for k, v in T.groupby("conjunto")))
    L.append(f"media da folga cruzada {T.h_cf.mean():.3f}; media do ganho da validada {T.g_val.mean():.3f}; folga = max(cruzada, validada): media {np.maximum(T.h_cf, T.g_val).mean():.3f}")
    return T


def main():
    L = ["=" * 110, "REANALISE PARA O CEC: RAZAO DE CAPTURA, CONGLOMERADOS, MARGEM, ORDEM E CLASSES (Entrada 283)", "=" * 110]
    S: dict = {}
    C = load_cells()
    part_capture(C, L, S)
    part_floor_margin(C, L, S)
    O = part_order(L, S)
    F = part_first_action(L, S)
    K = part_classes(L, S)
    V = part_cf_vs_val(L, S)

    cap_t = S["capture"]
    pooled = lambda lab: cap_t[(cap_t.grupo == "todas as 85") & (cap_t.aprendiz == lab)].iloc[0]
    rt, ql, qp = pooled("desempate aleatorio"), pooled("padrao"), pooled("permutada")
    low = C[C.g_val <= 0.05]
    mg = S["margin"].set_index("colecao").loc["total"]
    stdq = O[O.variante == "padrao (lista original)"]
    rev = O[O.variante == "lista invertida"]
    v0 = O[O.variante == "validada em primeiro"]
    fs = S["first_stats"]
    yn = lambda b: "SIM" if b else "NAO"
    p = []
    L += ["", "-" * 110, "PREVISOES", "-" * 110]
    r1 = 0.08 <= rt.captura <= 0.30; p.append(r1)
    L.append(f"(W1) razao de captura agrupada do desempate aleatorio: {rt.captura:.3f} (previsto 0,08 a 0,30): {yn(r1)}")
    r2 = rt.cap_hi < 0.40; p.append(r2)
    L.append(f"(W2) limite superior do IC90 por classes: {rt.cap_hi:.3f} (previsto < 0,40; IC {rt.cap_lo:.3f} a {rt.cap_hi:.3f}): {yn(r2)}")
    r3 = rt.spearman >= 0.60 and rt.sp_lo > 0.30; p.append(r3)
    L.append(f"(W3) Spearman agrupado do desempate aleatorio com a validada: {rt.spearman:.3f}, IC90 {rt.sp_lo:.3f} a {rt.sp_hi:.3f} (previsto >= 0,60 e limite inferior > 0,30): {yn(r3)}")
    n_low = int(low["rt_beyond_0.05"].sum()); r4 = n_low <= 3; p.append(r4)
    L.append(f"(W4) celulas com validada <= 0,05 em que o desempate aleatorio ganha alem da margem: {n_low} de {len(low)} (previsto <= 3): {yn(r4)}")
    r5 = qp.captura < 0; p.append(r5)
    L.append(f"(W5) razao de captura agrupada da lista permutada: {qp.captura:.3f} (previsto < 0): {yn(r5)}")
    r6 = (rt.spearman - ql.spearman) >= 0.15; p.append(r6)
    L.append(f"(W6) Spearman agrupado: padrao {ql.spearman:.3f}, desempate aleatorio {rt.spearman:.3f}; diferenca {rt.spearman - ql.spearman:.3f} (previsto >= 0,15): {yn(r6)}")
    r7 = int(mg["alem_0.02"]) >= 40 and int(mg["alem_0.10"]) <= 25; p.append(r7)
    L.append(f"(W7) desempate aleatorio alem da margem: {int(mg['alem_0.02'])} com 0,02 (previsto >= 40), {int(mg['alem_0.05'])} com 0,05, {int(mg['alem_0.10'])} com 0,10 (previsto <= 25): {yn(r7)}")
    r8 = fs["ql"]["med_abs_dif"] < fs["ucb"]["med_abs_dif"] and fs["ql"]["med_abs_dif"] <= 0.05; p.append(r8)
    L.append(f"(W8) mediana de |ganho do Q-Learning - ganho da acao 0|: {fs['ql']['med_abs_dif']:.3f}; do UCB1: {fs['ucb']['med_abs_dif']:.3f}; do gradiente: {fs['pg']['med_abs_dif']:.3f} (previsto Q < UCB1 e Q <= 0,05): {yn(r8)}")
    r9 = V.val_maior.mean() > 0.5; p.append(r9)
    L.append(f"(W9) ganho da validada > folga cruzada em {int(V.val_maior.sum())} de {len(V)} celulas (previsto > metade): {yn(r9)}")
    r10 = bool(stdq.fracao_na_primeira_acao.between(0.55, 0.75).all() and (rev.fracao_na_primeira_acao >= 0.50).all() and (v0.captura >= 0.75).all()); p.append(r10)
    L.append(f"(W10) fracao do padrao na primeira acao {stdq.fracao_na_primeira_acao.round(3).tolist()}, da lista invertida {rev.fracao_na_primeira_acao.round(3).tolist()}, captura da validada em primeiro {v0.captura.round(3).tolist()} (previsto 0,55-0,75; >= 0,50; >= 0,75): {yn(r10)}")
    L.append(f"previsoes acertadas: {sum(p)} de {len(p)}")

    text = "\n".join(L)
    print(text)
    (R / "entrada283_reanalysis.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada283_cells.csv", index=False)
    cap_t.to_csv(R / "entrada283_capture.csv", index=False)
    S["margin"].to_csv(R / "entrada283_margin.csv", index=False)
    O.to_csv(R / "entrada283_order.csv", index=False)
    F.to_csv(R / "entrada283_first_action.csv", index=False)
    K.to_csv(R / "entrada283_classtable.csv", index=False)
    V.to_csv(R / "entrada283_cf_vs_val.csv", index=False)


if __name__ == "__main__":
    main()
