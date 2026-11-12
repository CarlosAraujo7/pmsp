"""
analysis/report_entrada258L_longT.py
=====================================
Entrada 258 (frente 1, Ciclo 34): o protocolo no benchmark de Fanjul-Peyro e Ruiz com HORIZONTE LONGO, T = 10.000 iteracoes (em vez de 1.000), espaco B30 (30 acoes, com e sem busca local), as mesmas 700 instancias de teste e 280 de ajuste
da Entrada 256; a acao validada foi ESCOLHIDA DE NOVO no horizonte longo (results/frL_choice_B30_T10000_entrada258.json) e os aprendizes mantem os hiperparametros congelados (experiment/run_cycle34_longT_entrada258.py). Pergunta: as
conclusoes valem quando o IG chega mais perto dos melhores valores conhecidos? (a lacuna da validada a T = 1.000 foi de 4,89%). Celulas = os 7 tipos de instancia (100 instancias de teste cada); unidade = media das 3 sementes.
Familias de testes (Holm sobre as 7 celulas, por aprendiz): A (aprendiz contra o aleatorio), B (a validada contra o aprendiz), B' (o aprendiz contra a validada). Margem +-0,05; 'alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada258L_prereg_hashes.txt; so' as instancias de ajuste foram usadas para escolher a validada; nenhum resultado de teste visto)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(L1) A lacuna media da validada ao melhor conhecido nas 700 instancias a T = 10.000 e' de no maximo 3,5% (a T = 1.000: 4,89%). (60%)
(L2) Nenhum dos sete aprendizes bate a validada alem da margem (B') em nenhuma das 49 celulas aprendiz-tipo. (70%)
(L3) A validada ganha do aleatorio alem da margem em pelo menos 6 dos 7 tipos. (80%)
(L4) O Q-Learning padrao ganha do aleatorio alem da margem em no maximo 1 dos 7 tipos. (75%)
(L5) O ganho medio da validada sobre o aleatorio nos 7 tipos a T = 10.000 e' menor que a T = 1.000 (a folga diminui com o horizonte). (65%)
(L6) O melhor aprendiz de cada tipo recupera no maximo 50% do ganho da validada em pelo menos 6 dos 7 tipos. (65%)
Uso: python -m analysis.report_entrada258L_longT
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
LEARN = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr"]
pd.set_option("display.width", 230)


def wide(files):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(files))], ignore_index=True)
    return d, d.groupby(["type", "instance"]).apply(lambda g: g.groupby("method").makespan.mean(), include_groups=False).reset_index()


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def main():
    dL, wL = wide(str(R / "frL_test_B30_T10000_s*of12_entrada258.csv"))
    d1, w1 = wide(str(R / "fr_test_B30_s*of4_entrada256.csv"))
    bks = dL.drop_duplicates(["type", "instance"]).set_index(["type", "instance"]).bks
    wL = wL.merge(bks.reset_index(), on=["type", "instance"])
    w1 = w1.merge(bks.reset_index(), on=["type", "instance"])
    gapL = {k: float((100 * (wL[k] - wL.bks) / wL.bks).mean()) for k in ("rand", "val", "ql")}
    gap1 = {k: float((100 * (w1[k] - w1.bks) / w1.bks).mean()) for k in ("rand", "val", "ql")}
    cells, TA, TB, TC = [], [], [], []
    for ty in sorted(wL.type.unique()):
        x, x1 = wL[wL.type == ty], w1[w1.type == ty]
        g = {k: float(-rel(x[k], x.rand).mean()) for k in ["val"] + LEARN}
        cells.append(dict(type=ty, g_val_T10000=g["val"], g_val_T1000=float(-rel(x1.val, x1.rand).mean()), **{f"g_{k}": g[k] for k in LEARN}, best=max(g[k] for k in LEARN),
                          gap_val_T10000=float((100 * (x.val - x.bks) / x.bks).mean()), gap_val_T1000=float((100 * (x1.val - x1.bks) / x1.bks).mean())))
        for k in LEARN:
            a = test(x[k], x.rand); a.update(id=f"A-{ty}-{k}"); TA.append(a)
            b = test(x.val, x[k]); b.update(id=f"B-{ty}-{k}"); TB.append(b)
            c = test(x[k], x.val); c.update(id=f"B'-{ty}-{k}"); TC.append(c)
    C = pd.DataFrame(cells)
    fam = lambda rows: pd.concat([family([r for r in rows if r["id"].endswith(f"-{k}")]) for k in LEARN], ignore_index=True)
    FA, FB, FC = fam(TA), fam(TB), fam(TC)
    Av = family([dict(test(wL[wL.type == ty].val, wL[wL.type == ty].rand), id=f"V-{ty}") for ty in sorted(wL.type.unique())])
    l1 = gapL["val"] <= 3.5
    n2 = int(beyond(FC).sum())
    l3 = int(beyond(Av).sum())
    l4 = int(beyond(FA[FA.id.str.endswith("-ql")]).sum())
    l5 = C.g_val_T10000.mean() < C.g_val_T1000.mean()
    ratio = (C.best / C.g_val_T10000)
    l6 = int((ratio <= 0.5).sum())
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "BENCHMARK DE FANJUL-PEYRO E RUIZ, HORIZONTE LONGO T = 10.000, ESPACO B30 (Entrada 258)", "=" * 110,
         "--- por tipo: ganhos sobre o aleatorio (pontos) e lacuna da validada ao melhor conhecido (%) ---", C.round(3).to_string(index=False), "",
         f"lacuna media ao melhor conhecido nas 700 instancias: T = 10.000: validada {gapL['val']:.3f}%, aleatorio {gapL['rand']:.3f}%, ql {gapL['ql']:.3f}% | T = 1.000: validada {gap1['val']:.3f}%, aleatorio {gap1['rand']:.3f}%, ql {gap1['ql']:.3f}%",
         f"a validada bate o aprendiz (B, confirmada): {int(FB.confirmada.sum())} de 49; aprendizes que batem a validada alem da margem (B'): {n2} de 49",
         f"celulas em que cada aprendiz ganha do aleatorio alem da margem: " + ", ".join(f"{k} {int(beyond(FA[FA.id.str.endswith('-' + k)]).sum())}" for k in LEARN), "",
         "--- previsoes ---",
         f"(L1) lacuna da validada a T = 10.000: {gapL['val']:.3f}% (previsto <= 3,5%): {yn(l1)}",
         f"(L2) aprendizes que batem a validada alem da margem: {n2} de 49 (previsto 0): {yn(n2 == 0)}",
         f"(L3) tipos em que a validada ganha do aleatorio alem da margem: {l3} de 7 (previsto >= 6): {yn(l3 >= 6)}",
         f"(L4) tipos em que o Q-Learning padrao ganha alem da margem: {l4} de 7 (previsto <= 1): {yn(l4 <= 1)}",
         f"(L5) ganho medio da validada: T = 10.000 {C.g_val_T10000.mean():.3f} contra T = 1.000 {C.g_val_T1000.mean():.3f} (previsto menor): {yn(l5)}",
         f"(L6) tipos em que o melhor aprendiz recupera no maximo 50% do ganho da validada: {l6} de 7 (previsto >= 6): {yn(l6 >= 6)}"]
    pred = [l1, n2 == 0, l3 >= 6, l4 <= 1, l5, l6 >= 6]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada258L_longT.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([FA, FB, FC]).to_csv(R / "entrada258L_longT_tests.csv", index=False)
    C.to_csv(R / "entrada258L_longT_cells.csv", index=False)


if __name__ == "__main__":
    main()
