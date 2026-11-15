"""
analysis/report_entrada274_warmtsp.py
======================================
Entrada 274 (Ciclo 45): a REPLICA, NO TSP, do achado da Entrada 271 (o conhecimento offline como ponto de partida do aprendiz online). ILS do TSP da Entrada 267 (8 classes = tipo x n; 50 instancias de teste por classe; T = 1.000; 3 sementes);
Q-Learning otimista com hiperparametros congelados (c = 0,02; alfa = 0,1) e tabela Q inicial Q0[s, a] = c * escore_a vinda das instancias de AJUSTE (experiment/run_cycle45_warmtsp_entrada274.py; nucleo pmsp/tsp_warm.py). Prioris: full (a classe),
loso (o mesmo tipo, os outros tamanhos: no TSP os rankings de acoes das classes quase coincidem, Spearman >= 0,92: priori quase exata) e rev (o ranking da classe INVERTIDO: priori deliberadamente errada); cada uma com aprendizagem (warm_*) e sem
(frozen_*: alfa = 0). Comparadores dos arquivos da Entrada 267: o aleatorio, o qo frio, a validada por classe (val) e a acao global escolhida sem os dados do tamanho testado (LOSO-G0). Unidade = media das 3 sementes por instancia; celulas =
8 classes (50 instancias cada) por espaco (T15: 15 acoes; T5: 5 acoes); ganho = -d sobre o aleatorio (pontos; d = 100 (A - B) / B no comprimento do tour); testes como nas Entradas 180-273 (Wilcoxon pareado bilateral, IC90 bootstrap,
equivalencia a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente a zero). Familias (Holm sobre as 8 classes, por comparacao e espaco): A (metodo contra o aleatorio), E (aquecido contra o qo frio), L (aquecido contra o congelado:
confirmada = a aprendizagem ajuda), Lp (o congelado contra o aquecido: confirmada = a aprendizagem custa). O orcamento e' o de iteracoes (T = 1.000); as avaliacoes de movimentos (CPU) diferem entre as variantes e sao relatadas.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada274_prereg_hashes.txt; as prioris vem so' das instancias de ajuste; so' uma instancia de fumaca (n = 100, nao usada) foi rodada; os resultados da Entrada 267 (qo frio, validada) ja' eram conhecidos)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Hipotese de trabalho: o sinal da contribuicao da aprendizagem durante a execucao segue a qualidade da priori (negativo ou nulo quando a priori e' quase exata; positivo quando e' errada).
(S1) Em T15, o qo_warm_full bate o qo frio alem da margem (E) em pelo menos 5 das 8 classes (a priori ajuda o aprendiz). (65%)
(S2) Em T15, o ganho medio do qo_frozen_loso e' de pelo menos 90% do da LOSO-G0 (sem aprender, a priori quase exata vale a acao estatica). (70%)
(S3) Em T15, o ganho medio do qo_frozen_loso supera o do qo_warm_loso em pelo menos 0,10 ponto (aprender custa quando a priori e' quase exata). (70%)
(S4) Em T15, o qo_frozen_loso bate o qo_warm_loso alem da margem (Lp) em pelo menos 4 das 8 classes. (65%)
(S5) Em T15, o qo_warm_rev bate o qo_frozen_rev alem da margem (L) em pelo menos 7 das 8 classes (a aprendizagem corrige uma priori errada). (85%)
(S6) Em T15, o ganho medio do qo_warm_full supera o do qo_warm_rev em pelo menos 0,20 ponto (a priori errada nao e' corrigida por completo em T = 1.000). (80%)
(S7) Em T15, o ganho medio do qo_warm_rev e' negativo. (65%)
(S8) Em T5, o ganho medio do qo_frozen_loso supera o do qo_warm_loso em pelo menos 0,05 ponto. (65%)
(S9) Em T15, as avaliacoes medias do qo_frozen_loso sao pelo menos 1,2 vez as do qo_warm_loso (o congelado fica na acao pesada: o orcamento de iteracoes nao e' CPU igual). (70%)
Uso: python -m analysis.report_entrada274_warmtsp
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
NS = [200, 500, 1000, 2000]
KINDS = ["unif", "clust"]
PRI = ["full", "loso", "rev"]
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def main():
    d0 = read("tsp_test_s*of*_entrada267.csv")
    dw = read("tspwarm_test_s*of*_entrada274.csv")
    ch = json.loads((R / "tsp_choice_entrada267.json").read_text())
    tune = read("tsp_tune_s*of*_entrada267.csv")
    tune["kind"] = tune.cls.str.split("|").str[0]
    tune["n"] = tune.cls.str.split("|").str[1].astype(int)
    key = ["inst", "kind", "n"]
    arms = d0[d0.method.str.startswith("arm")].copy()
    arms["a"] = arms.method.str[3:].astype(int)
    WA = arms.groupby(key + ["a"]).length.mean().unstack("a")
    WA.columns = [f"arm{c}" for c in WA.columns]
    WA = WA.reset_index()
    assert len(WA) == 400 and dw.inst.nunique() == 400, (len(WA), dw.inst.nunique())
    var = [f"{m}_{p}" for p in PRI for m in ("warm", "frozen")]
    L = ["=" * 110, "A REPLICA NO TSP: O CONHECIMENTO OFFLINE COMO PONTO DE PARTIDA DO APRENDIZ ONLINE (Entrada 274)", "=" * 110]
    cells, TA, TE, TL, TLp = [], [], [], [], []
    ev = {}
    for sp, na in (("T15", 15), ("T5", 5)):
        x0 = d0[(d0.space == sp) & d0.method.isin(["rand", "qo"])]
        w = x0.groupby(key + ["method"]).length.mean().unstack("method").reset_index()
        xw = dw[dw.space == sp]
        w = w.merge(xw.groupby(key + ["method"]).length.mean().unstack("method").reset_index(), on=key).merge(WA, on=key)
        ev[sp] = {m: float(xw[xw.method == m].evals.mean()) for m in var}
        w["cls"] = w.kind + "|" + w.n.astype(str)
        w["val"] = [float(r[f"arm{ch[sp][r.cls]}"]) for _, r in w.iterrows()]
        t = tune[tune.arm < na]
        loso = {n: int(t[t.n != n].groupby("arm").ratio.mean().idxmin()) for n in NS}
        w["LOSO_G0"] = [float(r[f"arm{loso[r.n]}"]) for _, r in w.iterrows()]
        for (k, n), g in w.groupby(["kind", "n"]):
            c = f"{k}|{n}"
            cell = dict(space=sp, kind=k, n=n)
            for m in ["qo", "val", "LOSO_G0"] + var:
                cell[f"g_{m}"] = float(-rel(g[m], g.rand).mean())
            cells.append(cell)
            for m in var:
                a = test(g[m], g.rand); a.update(id=f"A-{m}-{c}-{sp}"); TA.append(a)
            for p in PRI:
                e = test(g[f"warm_{p}"], g.qo); e.update(id=f"E-warm_{p}-{c}-{sp}"); TE.append(e)
                l = test(g[f"warm_{p}"], g[f"frozen_{p}"]); l.update(id=f"L-{p}-{c}-{sp}"); TL.append(l)
                lp = test(g[f"frozen_{p}"], g[f"warm_{p}"]); lp.update(id=f"Lp-{p}-{c}-{sp}"); TLp.append(lp)
    C = pd.DataFrame(cells)

    def fam(rows):
        tags = sorted({(r["id"].split("-")[1], r["id"].split("-")[-1]) for r in rows})
        return pd.concat([family([r for r in rows if r["id"].split("-")[1] == m and r["id"].split("-")[-1] == sp]) for m, sp in tags], ignore_index=True)

    FA, FE, FL, FLp = fam(TA), fam(TE), fam(TL), fam(TLp)
    nb = lambda F_, tag, sp: int(beyond(F_[(F_.id.str.split("-").str[1] == tag) & (F_.id.str.endswith(f"-{sp}"))]).sum())
    mg = {sp: {m: float(C[C.space == sp][f"g_{m}"].mean()) for m in ["qo", "val", "LOSO_G0"] + var} for sp in ("T15", "T5")}
    s1 = nb(FE, "warm_full", "T15")
    s2v = mg["T15"]["frozen_loso"] / mg["T15"]["LOSO_G0"]
    s3d = mg["T15"]["frozen_loso"] - mg["T15"]["warm_loso"]
    s4 = nb(FLp, "loso", "T15")
    s5 = nb(FL, "rev", "T15")
    s6d = mg["T15"]["warm_full"] - mg["T15"]["warm_rev"]
    s7v = mg["T15"]["warm_rev"]
    s8d = mg["T5"]["frozen_loso"] - mg["T5"]["warm_loso"]
    s9v = ev["T15"]["frozen_loso"] / ev["T15"]["warm_loso"]
    yn = lambda b: "SIM" if b else "NAO"
    sh = dw.groupby(["space", "method"]).share_top.mean().unstack("method")
    L += ["--- ganhos medios nas 8 classes (pontos sobre o aleatorio) ---"]
    for sp in ("T15", "T5"):
        L.append(f"{sp}: " + "; ".join(f"{m} {v:.3f}" for m, v in mg[sp].items()))
    L += ["", "--- celulas (classe x espaco) ---", C.round(3).to_string(index=False), "",
          "fracao das escolhas na acao de maior priori (media das instancias):", sh.round(3).to_string(), "",
          "avaliacoes medias por execucao (T15): " + "; ".join(f"{m} {v:,.0f}" for m, v in ev["T15"].items()),
          "classes em que o aquecido bate o qo frio alem da margem (E; T15 / T5): " + "; ".join(f"{p} {nb(FE, f'warm_{p}', 'T15')}/{nb(FE, f'warm_{p}', 'T5')}" for p in PRI),
          "classes em que a aprendizagem ajuda alem da margem (L; T15 / T5): " + "; ".join(f"{p} {nb(FL, p, 'T15')}/{nb(FL, p, 'T5')}" for p in PRI),
          "classes em que a aprendizagem custa alem da margem (Lp; T15 / T5): " + "; ".join(f"{p} {nb(FLp, p, 'T15')}/{nb(FLp, p, 'T5')}" for p in PRI),
          "classes em que a variante ganha do aleatorio alem da margem (A; T15 / T5): " + "; ".join(f"{m} {nb(FA, m, 'T15')}/{nb(FA, m, 'T5')}" for m in var), "",
          "--- previsoes ---",
          f"(S1) classes em que o qo_warm_full bate o qo frio alem da margem (T15): {s1} de 8 (previsto >= 5): {yn(s1 >= 5)}",
          f"(S2) ganho do qo_frozen_loso / ganho da LOSO-G0 (T15): {s2v:.3f} (previsto >= 0,90): {yn(s2v >= 0.9)}",
          f"(S3) ganho do qo_frozen_loso - ganho do qo_warm_loso (T15): {s3d:.3f} (previsto >= 0,10): {yn(s3d >= 0.10)}",
          f"(S4) classes em que o qo_frozen_loso bate o qo_warm_loso alem da margem (T15): {s4} de 8 (previsto >= 4): {yn(s4 >= 4)}",
          f"(S5) classes em que o qo_warm_rev bate o qo_frozen_rev alem da margem (T15): {s5} de 8 (previsto >= 7): {yn(s5 >= 7)}",
          f"(S6) ganho do qo_warm_full - ganho do qo_warm_rev (T15): {s6d:.3f} (previsto >= 0,20): {yn(s6d >= 0.20)}",
          f"(S7) ganho medio do qo_warm_rev (T15): {s7v:.3f} (previsto < 0): {yn(s7v < 0)}",
          f"(S8) ganho do qo_frozen_loso - ganho do qo_warm_loso (T5): {s8d:.3f} (previsto >= 0,05): {yn(s8d >= 0.05)}",
          f"(S9) avaliacoes do qo_frozen_loso / qo_warm_loso (T15): {s9v:.3f} (previsto >= 1,2): {yn(s9v >= 1.2)}"]
    pred = [s1 >= 5, s2v >= 0.9, s3d >= 0.10, s4 >= 4, s5 >= 7, s6d >= 0.20, s7v < 0, s8d >= 0.05, s9v >= 1.2]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada274_warmtsp.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada274_warmtsp_cells.csv", index=False)
    pd.concat([FA, FE, FL, FLp]).to_csv(R / "entrada274_warmtsp_tests.csv", index=False)


if __name__ == "__main__":
    main()
