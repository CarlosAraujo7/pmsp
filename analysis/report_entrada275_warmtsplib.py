"""
analysis/report_entrada275_warmtsplib.py
=========================================
Entrada 275 (Ciclo 46): a replica da Entrada 274 em instancias PUBLICADAS: as 37 instancias da TSPLIB da Entrada 270 (EUC_2D, 200 <= n <= 2103; otimos do arquivo `solutions`). A priori vem so' das instancias SINTETICAS de ajuste (Entrada 267):
syn_n (o tamanho sintetico mais proximo; os dois tipos agrupados; o par de Gn_syn) e syn_0 (global; o par de G0_syn); cada uma com aprendizagem (warm_*) e sem (frozen_*; alfa = 0); Q-Learning otimista com hiperparametros congelados
(experiment/run_cycle46_warmtsplib_entrada275.py). Comparadores dos arquivos da Entrada 270: o aleatorio, o qo frio, Gn_syn e G0_syn (acoes estaticas). Unidade = instancia (media das 3 sementes); 37 instancias; faixas de tamanho como na Entrada 270
(pequena n < 500: 15 instancias; media 500-1200: 10; grande > 1200: 12); ganho = -d sobre o aleatorio (pontos); lacuna = 100 (comprimento - otimo) / otimo. Testes como nas Entradas 180-274 (Wilcoxon pareado bilateral, IC90 bootstrap, equivalencia
a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente a zero). Familias (Holm por espaco): A (variante contra o aleatorio), E (aquecido contra o qo frio), Bp (a variante contra a acao estatica correspondente: confirmada = a variante ganha),
Bs (a acao estatica contra a variante). Por faixa: diferenca media de ganhos aquecido - congelado (descritivo + teste pareado nas instancias da faixa).

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada275_prereg_hashes.txt; as prioris vem so' das instancias sinteticas de ajuste; nenhum resultado da TSPLIB com aprendizes aquecidos visto; os resultados da Entrada 270 (Gn_syn, G0_syn, qo frio) ja' eram conhecidos,
em particular que Gn_syn nao ganha na faixa pequena)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Hipotese de trabalho (a mesma da Entrada 274): a aprendizagem durante a execucao ajuda onde a priori transferida erra (faixa pequena, onde a acao do tamanho sintetico mais proximo nao ganha) e custa onde ela acerta (faixas media e grande).
(M1) Em T15, o qo_warm_synn bate o qo frio alem da margem (E; 37 instancias). (75%)
(M2) Em T15, o ganho medio do qo_frozen_syn0 e' de pelo menos 90% do da G0_syn. (70%)
(M3) Em T15, na faixa pequena, o ganho medio do qo_warm_synn supera o do qo_frozen_synn. (70%)
(M4) Em T15, nas faixas media e grande reunidas (22 instancias), o ganho medio do qo_warm_synn e' MENOR que o do qo_frozen_synn. (80%)
(M5) Em T15, nenhuma das duas variantes aquecidas (warm_synn, warm_syn0) bate a acao estatica correspondente alem da margem (Bp; 0 de 2). (85%)
(M6) Em T15, a Gn_syn bate o qo_warm_synn alem da margem (Bs; 37 instancias). (60%)
(M7) Em T15, na faixa pequena, o ganho medio do qo_warm_synn supera o da Gn_syn. (60%)
Uso: python -m analysis.report_entrada275_warmtsplib
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
from experiment.run_cycle42_tsplib_entrada270 import nearest  # noqa: E402

R = ROOT / "results"
VAR = ["warm_synn", "frozen_synn", "warm_syn0", "frozen_syn0"]
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def main():
    d0 = read("tsplib_test_s*of*_entrada270.csv")
    dw = read("tsplibwarm_test_s*of*_entrada275.csv")
    ch = json.loads((R / "tsplib_choice_entrada270.json").read_text())
    key = ["name", "n", "opt"]
    arms = d0[d0.method.str.startswith("arm")].copy()
    arms["a"] = arms.method.str[3:].astype(int)
    WA = arms.groupby(key + ["a"]).length.mean().unstack("a")
    WA.columns = [f"arm{c}" for c in WA.columns]
    WA = WA.reset_index()
    assert len(WA) == 37 and dw.name.nunique() == 37, (len(WA), dw.name.nunique())
    L = ["=" * 110, "A REPLICA NA TSPLIB: A PRIORI SINTETICA COMO PONTO DE PARTIDA DO APRENDIZ (Entrada 275)", "=" * 110]
    TA, TE, TBp, TBs, rows, brow = [], [], [], [], [], []
    W = {}
    for sp in ("T15", "T5"):
        x0 = d0[(d0.space == sp) & d0.method.isin(["rand", "qo"])]
        w = x0.groupby(key + ["method"]).length.mean().unstack("method").reset_index()
        xw = dw[dw.space == sp]
        w = w.merge(xw.groupby(key + ["method"]).length.mean().unstack("method").reset_index(), on=key).merge(WA, on=key)
        w["band"] = np.where(w.n < 500, "small", np.where(w.n <= 1200, "medium", "large"))
        gn, g0 = ch[sp]["Gn"], ch[sp]["G0"]
        w["Gn_syn"] = [float(r[f"arm{gn[str(nearest(int(r.n)))]}"]) for _, r in w.iterrows()]
        w["G0_syn"] = [float(r[f"arm{g0}"]) for _, r in w.iterrows()]
        W[sp] = w
        row = dict(space=sp, n=len(w), evals_frozen_synn=float(xw[xw.method == "frozen_synn"].evals.mean()), evals_warm_synn=float(xw[xw.method == "warm_synn"].evals.mean()))
        for m in ["qo", "Gn_syn", "G0_syn"] + VAR:
            row[f"g_{m}"] = float(-rel(w[m], w.rand).mean())
            row[f"gap_{m}"] = float((100 * (w[m] - w.opt) / w.opt).mean())
        row["gap_rand"] = float((100 * (w.rand - w.opt) / w.opt).mean())
        rows.append(row)
        for m in VAR:
            a = test(w[m], w.rand); a.update(id=f"A-{m}-{sp}"); TA.append(a)
        for m in ("warm_synn", "warm_syn0"):
            e = test(w[m], w.qo); e.update(id=f"E-{m}-{sp}"); TE.append(e)
        for m, st in (("warm_synn", "Gn_syn"), ("warm_syn0", "G0_syn")):
            b = test(w[m], w[st]); b.update(id=f"Bp-{m}-{sp}"); TBp.append(b)
            c = test(w[st], w[m]); c.update(id=f"Bs-{m}-{sp}"); TBs.append(c)
    C = pd.DataFrame(rows)
    fam = lambda rs: pd.concat([family([r for r in rs if r["id"].endswith(f"-{sp}")]) for sp in ("T15", "T5")], ignore_index=True)
    FA, FE, FBp, FBs = fam(TA), fam(TE), fam(TBp), fam(TBs)
    w15 = W["T15"]
    for b, g in list(w15.groupby("band")) + [("medium+large", w15[w15.band != "small"]), ("all", w15)]:
        brow.append(dict(band=b, n=len(g), **{f"g_{m}": float(-rel(g[m], g.rand).mean()) for m in ["qo", "Gn_syn", "G0_syn"] + VAR},
                         d_synn=float(-rel(g.warm_synn, g.rand).mean() + rel(g.frozen_synn, g.rand).mean()), d_syn0=float(-rel(g.warm_syn0, g.rand).mean() + rel(g.frozen_syn0, g.rand).mean())))
    BR = pd.DataFrame(brow).set_index("band")
    c15 = C[C.space == "T15"].iloc[0]
    inF = lambda F_, i: bool(beyond(F_[F_.id == i]).iloc[0])
    m1 = inF(FE, "E-warm_synn-T15")
    m2v = float(c15.g_frozen_syn0 / c15.g_G0_syn)
    m3d = float(BR.loc["small", "g_warm_synn"] - BR.loc["small", "g_frozen_synn"])
    m4d = float(BR.loc["medium+large", "g_warm_synn"] - BR.loc["medium+large", "g_frozen_synn"])
    m5 = int(inF(FBp, "Bp-warm_synn-T15")) + int(inF(FBp, "Bp-warm_syn0-T15"))
    m6 = inF(FBs, "Bs-warm_synn-T15")
    m7d = float(BR.loc["small", "g_warm_synn"] - BR.loc["small", "g_Gn_syn"])
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- ganhos medios sobre o aleatorio (pontos) e lacunas ao otimo (%) por espaco; 37 instancias ---", C.set_index("space").T.round(3).to_string(), "",
          "--- por faixa de tamanho (T15; ganhos medios; d_* = ganho do aquecido - ganho do congelado) ---", BR.round(3).T.to_string(), "",
          "variantes que ganham do aleatorio alem da margem (A): " + "; ".join(f"{sp}: " + ", ".join(m for m in VAR if inF(FA, f"A-{m}-{sp}")) for sp in ("T15", "T5")),
          "aquecido bate o qo frio alem da margem (E): " + "; ".join(f"{r.id} d={r.mean_d_pct:.3f} p_holm={r.p_holm:.3g}" for _, r in FE.iterrows()),
          "aquecido contra a acao estatica (Bp: o aquecido ganha; Bs: a estatica ganha): " + "; ".join(f"{r.id} d={r.mean_d_pct:.3f} p_holm={r.p_holm:.3g} conf={bool(r.confirmada)}" for _, r in pd.concat([FBp, FBs]).iterrows()), "",
          "--- previsoes ---",
          f"(M1) o qo_warm_synn bate o qo frio alem da margem (T15): {yn(m1)}",
          f"(M2) ganho do qo_frozen_syn0 / ganho da G0_syn (T15): {m2v:.3f} (previsto >= 0,90): {yn(m2v >= 0.9)}",
          f"(M3) faixa pequena: ganho do warm_synn - ganho do frozen_synn = {m3d:.3f} (previsto > 0): {yn(m3d > 0)}",
          f"(M4) faixas media e grande: ganho do warm_synn - ganho do frozen_synn = {m4d:.3f} (previsto < 0): {yn(m4d < 0)}",
          f"(M5) variantes aquecidas que batem a estatica correspondente alem da margem: {m5} de 2 (previsto 0): {yn(m5 == 0)}",
          f"(M6) a Gn_syn bate o qo_warm_synn alem da margem (T15): {yn(m6)}",
          f"(M7) faixa pequena: ganho do warm_synn - ganho da Gn_syn = {m7d:.3f} (previsto > 0): {yn(m7d > 0)}"]
    pred = [m1, m2v >= 0.9, m3d > 0, m4d < 0, m5 == 0, m6, m7d > 0]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada275_warmtsplib.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada275_warmtsplib_cells.csv", index=False)
    BR.reset_index().to_csv(R / "entrada275_warmtsplib_bands.csv", index=False)
    pd.concat([FA, FE, FBp, FBs]).to_csv(R / "entrada275_warmtsplib_tests.csv", index=False)


if __name__ == "__main__":
    main()
