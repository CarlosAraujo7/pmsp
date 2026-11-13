"""
analysis/report_entrada267_tsp.py
==================================
Entrada 267 (Ciclo 40): o protocolo do artigo num SEGUNDO DOMINIO, fora do escalonamento: a busca local iterada do TSP euclidiano (pmsp/tsp_numba.py; 2-opt + Or-opt; perturbacao escolhida por um seletor a cada iteracao;
experiment/run_cycle40_tsp_entrada267.py). 8 classes (tipo uniforme ou agrupado x n em {200, 500, 1000, 2000}), 12 instancias de ajuste e 50 de teste por classe, T = 1.000, 3 sementes; espacos T5 (cinco tamanhos de uma perturbacao) e T15
(tres perturbacoes x cinco tamanhos); celulas = classe x espaco = 16 (50 instancias cada); unidade = media das 3 sementes por instancia; ganho = -d sobre o aleatorio (pontos; d = 100 (A - B) / B no comprimento do tour);
testes como nas Entradas 180-266 (Wilcoxon pareado bilateral, IC90 bootstrap, equivalencia a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente a zero). Familias (Holm sobre as 8 classes de cada espaco, por metodo): A (metodo
contra o aleatorio), B (a validada contra o aprendiz), Bp (o aprendiz contra a validada), E (ql_perm contra ql). A acao validada de cada classe e' a de menor razao media nas instancias de AJUSTE (results/tsp_choice_entrada267.json). Granularidade
do conhecimento offline (so' T15; 8 classes): G0 (uma acao global), Gn (por n), Gk (por tipo), Gkn (por classe = a validada), LOSO-G0 (acao global escolhida sem os dados do tamanho n testado), LOKO-G0 (sem os dados do tipo testado).
Referencia absoluta: melhor comprimento encontrado por qualquer metodo, semente ou acao na instancia (NAO e' o otimo).

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada267_prereg_hashes.txt; so' as instancias de ajuste foram usadas para escolher a validada; nenhum resultado de teste visto, a nao ser uma instancia de fumaca (n = 100, nao usada))
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(P1) A acao validada ganha do aleatorio alem da margem em pelo menos 12 das 16 celulas. (65%)
(P2) Nenhum dos seis aprendizes online (ql, ucb, pg, alns, ts, qo) bate a validada alem da margem (Bp) em nenhum dos 96 pares aprendiz-celula. (60%)
(P3) O Q-Learning padrao ganha do aleatorio alem da margem em no maximo 3 das 16 celulas. (60%)
(P4) O ql_perm ganha do aleatorio alem da margem em no maximo 1 das 16 celulas. (75%)
(P5) O Q-Learning padrao em T15 concentra pelo menos 20% das escolhas na acao listada primeiro (o aleatorio: 6,7%). (70%)
(P6) O ql_perm difere do ql padrao alem da margem (E, nos dois sentidos: p_holm < 0,05 e nao equivalente) em pelo menos 4 das 16 celulas. (55%)
(P7) Em T15, o ganho medio de G0 e' de pelo menos 50% do de Gkn (a validada por classe), media das 8 classes. (60%)
(P8) Em T15, LOSO-G0 ganha do aleatorio alem da margem em pelo menos 3 dos 4 tamanhos (celula = tamanho, as duas classes agrupadas). (60%)
(P9) O melhor aprendiz online de cada celula recupera no maximo 50% do ganho da validada em pelo menos 10 das 16 celulas (nas celulas em que a validada ganha mais de 0,05). (60%)
(P10) O ganho medio da validada em T15 e' maior que em T5 (o espaco maior tem mais folga). (55%)
Uso: python -m analysis.report_entrada267_tsp
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
ONLINE = ["ql", "ucb", "pg", "alns", "ts", "qo"]
ALL = ONLINE + ["ql_perm"]
KINDS = ["unif", "clust"]
NS = [200, 500, 1000, 2000]
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def main():
    d = read("tsp_test_s*of*_entrada267.csv")
    ch = json.loads((R / "tsp_choice_entrada267.json").read_text())
    tune = read("tsp_tune_s*of*_entrada267.csv")
    tune["kind"] = tune.cls.str.split("|").str[0]
    tune["n"] = tune.cls.str.split("|").str[1].astype(int)
    key = ["inst", "kind", "n"]
    best = d.groupby("inst").length.min().rename("best")
    arms = d[d.method.str.startswith("arm")].copy()
    arms["a"] = arms.method.str[3:].astype(int)
    WA = arms.groupby(key + ["a"]).length.mean().unstack("a")
    WA.columns = [f"arm{c}" for c in WA.columns]
    WA = WA.reset_index()
    assert len(WA) == 400, len(WA)
    L = ["=" * 110, "O PROTOCOLO NUM SEGUNDO DOMINIO: ILS DO TSP (Entrada 267)", "=" * 110]
    cells, TA, TB, TBp, TE = [], [], [], [], []
    W = {}
    for sp in ("T5", "T15"):
        x = d[(d.space == sp) & ~d.method.str.startswith("arm")]
        w = x.groupby(key + ["method"]).length.mean().unstack("method").reset_index()
        w = w.merge(WA, on=key)
        w["cls"] = w.kind + "|" + w.n.astype(str)
        val = ch[sp]
        w["val"] = [float(r[f"arm{val[r.cls]}"]) for _, r in w.iterrows()]
        w = w.merge(best.reset_index(), on="inst")
        W[sp] = w
        for (k, n), g in w.groupby(["kind", "n"]):
            cell = dict(space=sp, kind=k, n=n, g_val=float(-rel(g.val, g.rand).mean()), gap_val=float((100 * (g.val - g.best) / g.best).mean()), gap_rand=float((100 * (g.rand - g.best) / g.best).mean()))
            for m in ALL:
                cell[f"g_{m}"] = float(-rel(g[m], g.rand).mean())
            cells.append(cell)
            c = f"{k}|{n}"
            for m in ALL:
                a = test(g[m], g.rand); a.update(id=f"A-{m}-{c}-{sp}"); TA.append(a)
                b = test(g.val, g[m]); b.update(id=f"B-{m}-{c}-{sp}"); TB.append(b)
                bp = test(g[m], g.val); bp.update(id=f"Bp-{m}-{c}-{sp}"); TBp.append(bp)
            a = test(g.val, g.rand); a.update(id=f"A-val-{c}-{sp}"); TA.append(a)
            e = test(g.ql_perm, g.ql); e.update(id=f"E-ql_perm-{c}-{sp}"); TE.append(e)
    C = pd.DataFrame(cells)

    def fam(rows, prefix_len=2):
        out = []
        tags = sorted({(r["id"].split("-")[1], r["id"].split("-")[-1]) for r in rows})
        for m, sp in tags:
            out.append(family([r for r in rows if r["id"].split("-")[1] == m and r["id"].split("-")[-1] == sp]))
        return pd.concat(out, ignore_index=True)

    FA, FB, FBp, FE = fam(TA), fam(TB), fam(TBp), fam(TE)
    FE["diff_beyond"] = FE.reject & ~FE["equivalente_0.05"]
    mean_val = C.groupby("space").g_val.mean()
    cnt_a = lambda m: int(beyond(FA[FA.id.str.split("-").str[1] == m]).sum())
    # ---- granularidade (T15; 8 classes)
    w = W["T15"]
    t15 = tune
    pick = lambda df, by=None: (int(df.groupby("arm").ratio.mean().idxmin()) if by is None else {k: int(g.groupby("arm").ratio.mean().idxmin()) for k, g in df.groupby(by)})
    g0 = pick(t15)
    gn = pick(t15, "n")
    gk = pick(t15, "kind")
    gkn = pick(t15, "cls")
    loso = {n: pick(t15[t15.n != n]) for n in NS}
    loko = {k: pick(t15[t15.kind != k]) for k in KINDS}
    pol = {"G0": lambda r: g0, "Gn": lambda r: gn[r.n], "Gk": lambda r: gk[r.kind], "Gkn": lambda r: gkn[r.cls], "LOSO_G0": lambda r: loso[r.n], "LOKO_G0": lambda r: loko[r.kind]}
    for nm, f in pol.items():
        w[nm] = [float(r[f"arm{f(r)}"]) for _, r in w.iterrows()]
    TG = []
    gcells = []
    for (k, n), g in w.groupby(["kind", "n"]):
        row = dict(kind=k, n=n)
        for nm in pol:
            row[f"g_{nm}"] = float(-rel(g[nm], g.rand).mean())
        gcells.append(row)
    GC = pd.DataFrame(gcells)
    for nm in pol:
        TG.append(family([dict(test(g[nm], g.rand), id=f"G-{nm}-{k}|{n}") for (k, n), g in w.groupby(["kind", "n"])]))
    FG = pd.concat(TG, ignore_index=True)
    sz = family([dict(test(g["LOSO_G0"], g.rand), id=f"S-LOSO_G0-n{n}") for n, g in w.groupby("n")])
    mg = {nm: float(GC[f"g_{nm}"].mean()) for nm in pol}
    # ---- previsoes
    p1 = int(beyond(FA[FA.id.str.split("-").str[1] == "val"]).sum())
    p2 = int(sum(int(beyond(FBp[FBp.id.str.split("-").str[1] == m]).sum()) for m in ONLINE))
    p3 = cnt_a("ql")
    p4 = cnt_a("ql_perm")
    sh0 = float(d[(d.space == "T15") & (d.method == "ql")].share0.mean())
    p6 = int(FE.diff_beyond.sum())
    p7 = mg["G0"] / mg["Gkn"]
    p8 = int(beyond(sz).sum())
    pos = C[C.g_val > 0.05]
    ratio = pos[[f"g_{m}" for m in ONLINE]].max(axis=1) / pos.g_val
    p9 = int((ratio <= 0.5).sum())
    p10 = bool(mean_val["T15"] > mean_val["T5"])
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- celulas (classe x espaco): ganhos sobre o aleatorio (pontos) e lacuna da validada ao melhor encontrado (%) ---", C.round(3).to_string(index=False), "",
          f"ganho medio da validada: T5 {mean_val['T5']:.3f}, T15 {mean_val['T15']:.3f}; lacuna media da validada ao melhor encontrado (T15): {C[C.space == 'T15'].gap_val.mean():.2f}% (aleatorio {C[C.space == 'T15'].gap_rand.mean():.2f}%)",
          "celulas em que cada metodo ganha do aleatorio alem da margem (de 16): " + "; ".join(f"{m} {cnt_a(m)}" for m in ["val"] + ALL),
          "celulas em que a validada bate o aprendiz (B, confirmada; de 16): " + "; ".join(f"{m} {int(FB[FB.id.str.split('-').str[1] == m].confirmada.sum())}" for m in ALL),
          "celulas em que o aprendiz bate a validada alem da margem (Bp; de 16): " + "; ".join(f"{m} {int(beyond(FBp[FBp.id.str.split('-').str[1] == m]).sum())}" for m in ALL),
          f"fracao das escolhas do Q-Learning padrao na acao listada primeiro (T15): {sh0:.3f}; (T5): {float(d[(d.space == 'T5') & (d.method == 'ql')].share0.mean()):.3f}",
          f"razao de avaliacoes ao aleatorio: ql {float((d[(d.space == 'T15') & (d.method == 'ql')].evals.mean() / d[(d.space == 'T15') & (d.method == 'rand')].evals.mean())):.3f}; ql_perm {float((d[(d.space == 'T15') & (d.method == 'ql_perm')].evals.mean() / d[(d.space == 'T15') & (d.method == 'rand')].evals.mean())):.3f}", "",
          "--- granularidade do conhecimento offline (T15; ganhos medios nas 8 classes) ---", "; ".join(f"{nm} {v:.3f}" for nm, v in mg.items()), GC.round(3).to_string(index=False), "",
          "LOSO-G0 por tamanho:", sz[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), "",
          "--- previsoes ---",
          f"(P1) celulas em que a validada ganha alem da margem: {p1} de 16 (previsto >= 12): {yn(p1 >= 12)}",
          f"(P2) pares aprendiz online-celula em que o aprendiz bate a validada alem da margem: {p2} de 96 (previsto 0): {yn(p2 == 0)}",
          f"(P3) celulas em que o ql padrao ganha alem da margem: {p3} de 16 (previsto <= 3): {yn(p3 <= 3)}",
          f"(P4) celulas em que o ql_perm ganha alem da margem: {p4} de 16 (previsto <= 1): {yn(p4 <= 1)}",
          f"(P5) fracao do ql padrao na acao listada primeiro em T15: {sh0:.3f} (previsto >= 0,20): {yn(sh0 >= 0.20)}",
          f"(P6) celulas em que o ql_perm difere do ql alem da margem: {p6} de 16 (previsto >= 4): {yn(p6 >= 4)}",
          f"(P7) ganho medio G0 / Gkn em T15: {p7:.3f} (previsto >= 0,50): {yn(p7 >= 0.5)}",
          f"(P8) tamanhos em que LOSO-G0 ganha alem da margem: {p8} de 4 (previsto >= 3): {yn(p8 >= 3)}",
          f"(P9) celulas (com validada > 0,05) em que o melhor aprendiz online recupera <= 50% do ganho da validada: {p9} de {len(pos)} (previsto >= 10): {yn(p9 >= 10)}",
          f"(P10) ganho medio da validada T15 {mean_val['T15']:.3f} > T5 {mean_val['T5']:.3f}: {yn(p10)}"]
    pred = [p1 >= 12, p2 == 0, p3 <= 3, p4 <= 1, sh0 >= 0.20, p6 >= 4, p7 >= 0.5, p8 >= 3, p9 >= 10, p10]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada267_tsp.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada267_tsp_cells.csv", index=False)
    GC.to_csv(R / "entrada267_tsp_gran_cells.csv", index=False)
    pd.concat([FA, FB, FBp, FE, FG, sz]).to_csv(R / "entrada267_tsp_tests.csv", index=False)


if __name__ == "__main__":
    main()
