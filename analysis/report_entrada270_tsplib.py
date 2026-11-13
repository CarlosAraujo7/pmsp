"""
analysis/report_entrada270_tsplib.py
=====================================
Entrada 270 (Ciclo 42): o ILS do TSP (Entrada 267) em 37 instancias PUBLICADAS da TSPLIB (EUC_2D, 200 <= n <= 2103; valores otimos do arquivo `solutions` da TSPLIB; experiment/run_cycle42_tsplib_entrada270.py). Nenhuma instancia da TSPLIB e'
usada para escolher acoes: as acoes estaticas sao escolhidas nas instancias SINTETICAS de ajuste da Entrada 267 (Gn_syn: a acao do tamanho sintetico mais proximo; G0_syn: uma acao global); os aprendizes mantem os hiperparametros congelados.
Unidade = instancia (media das 3 sementes); 37 instancias por espaco (T5, T15); ganho = -d sobre o aleatorio (pontos; d = 100 (A - B) / B no comprimento do tour); lacuna = 100 (comprimento - otimo) / otimo. Testes como nas Entradas 180-269
(Wilcoxon pareado bilateral, IC90 bootstrap, equivalencia a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente a zero). Familias (Holm por espaco): A (metodo contra o aleatorio; 9 metodos), Bp (o aprendiz online contra Gn_syn; 6 aprendizes),
E (ql_perm contra ql); por faixa de tamanho (pequena n < 500, media 500-1200, grande > 1200; Holm sobre as tres faixas): Ab (Gn_syn contra o aleatorio, T15).

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada270_prereg_hashes.txt; so' as instancias sinteticas de ajuste foram usadas para escolher as acoes estaticas; nenhum resultado em instancias da TSPLIB visto)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(L1) Gn_syn ganha do aleatorio alem da margem (A, 37 instancias) em T5 e em T15. (75%)
(L2) Nenhum dos seis aprendizes online bate Gn_syn alem da margem (Bp) em nenhum dos 12 testes (6 aprendizes x 2 espacos). (70%)
(L3) Nem o ql padrao nem o ql_perm ganham do aleatorio alem da margem em T5 ou em T15. (70%)
(L4) G0_syn ganha do aleatorio alem da margem em T15. (65%)
(L5) A lacuna media de Gn_syn ao otimo, em T15, e' de no maximo 6%. (65%)
(L6) Em cada espaco, o ganho medio do melhor aprendiz online e' de no maximo 60% do de Gn_syn. (60%)
(L7) Em T15 o ql padrao concentra pelo menos 40% das escolhas na acao listada primeiro. (75%)
(L8) O Thompson (ts) ganha do aleatorio alem da margem em T15. (50%)
(L9) Em T15, Gn_syn ganha do aleatorio alem da margem em pelo menos 2 das 3 faixas de tamanho. (60%)
Uso: python -m analysis.report_entrada270_tsplib
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
ONLINE = ["ql", "ucb", "pg", "alns", "ts", "qo"]
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def main():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "tsplib_test_s*of*_entrada270.csv")))], ignore_index=True)
    ch = json.loads((R / "tsplib_choice_entrada270.json").read_text())
    key = ["name", "n", "opt"]
    arms = d[d.method.str.startswith("arm")].copy()
    arms["a"] = arms.method.str[3:].astype(int)
    WA = arms.groupby(key + ["a"]).length.mean().unstack("a")
    WA.columns = [f"arm{c}" for c in WA.columns]
    WA = WA.reset_index()
    assert len(WA) == 37, len(WA)
    L = ["=" * 110, "O ILS DO TSP EM INSTANCIAS PUBLICADAS DA TSPLIB (Entrada 270)", "=" * 110]
    W, TA, TB, TE, rows, gaps = {}, [], [], [], [], []
    for sp in ("T5", "T15"):
        x = d[(d.space == sp) & ~d.method.str.startswith("arm")]
        w = x.groupby(key + ["method"]).length.mean().unstack("method").reset_index().merge(WA, on=key)
        w["band"] = np.where(w.n < 500, "small", np.where(w.n <= 1200, "medium", "large"))
        gn, g0 = ch[sp]["Gn"], ch[sp]["G0"]
        w["Gn_syn"] = [float(r[f"arm{gn[str(nearest(int(r.n)))]}"]) for _, r in w.iterrows()]
        w["G0_syn"] = float("nan")
        w["G0_syn"] = [float(r[f"arm{g0}"]) for _, r in w.iterrows()]
        W[sp] = w
        meths = ["Gn_syn", "G0_syn"] + ONLINE + ["ql_perm"]
        row = dict(space=sp, n=len(w))
        for m in meths:
            row[f"g_{m}"] = float(-rel(w[m], w.rand).mean())
            row[f"gap_{m}"] = float((100 * (w[m] - w.opt) / w.opt).mean())
            a = test(w[m], w.rand); a.update(id=f"A-{m}-{sp}"); TA.append(a)
        row["gap_rand"] = float((100 * (w.rand - w.opt) / w.opt).mean())
        for m in ONLINE:
            b = test(w[m], w.Gn_syn); b.update(id=f"Bp-{m}-{sp}"); TB.append(b)
        e = test(w.ql_perm, w.ql); e.update(id=f"E-ql_perm-{sp}"); TE.append(e)
        rows.append(row)
    C = pd.DataFrame(rows)
    FA = pd.concat([family([r for r in TA if r["id"].endswith(f"-{sp}")]) for sp in ("T5", "T15")], ignore_index=True)
    FB = pd.concat([family([r for r in TB if r["id"].endswith(f"-{sp}")]) for sp in ("T5", "T15")], ignore_index=True)
    FE = pd.concat([family([r for r in TE if r["id"].endswith(f"-{sp}")]) for sp in ("T5", "T15")], ignore_index=True)
    w15 = W["T15"]
    FBand = family([dict(test(g.Gn_syn, g.rand), id=f"Ab-Gn_syn-{b}") for b, g in w15.groupby("band")])
    bandrows = [dict(band=b, n=len(g), g_Gn_syn=float(-rel(g.Gn_syn, g.rand).mean()), g_G0_syn=float(-rel(g.G0_syn, g.rand).mean()), g_best_online=float(max(-rel(g[m], g.rand).mean() for m in ONLINE)),
                     gap_Gn_syn=float((100 * (g.Gn_syn - g.opt) / g.opt).mean()), gap_rand=float((100 * (g.rand - g.opt) / g.opt).mean())) for b, g in w15.groupby("band")]
    BR = pd.DataFrame(bandrows)
    inA = lambda m, sp: bool(beyond(FA[FA.id == f"A-{m}-{sp}"]).iloc[0])
    l1 = inA("Gn_syn", "T5") and inA("Gn_syn", "T15")
    l2n = int(beyond(FB).sum())
    l3 = not any(inA(m, sp) for m in ("ql", "ql_perm") for sp in ("T5", "T15"))
    l4 = inA("G0_syn", "T15")
    c15 = C[C.space == "T15"].iloc[0]
    l5 = float(c15.gap_Gn_syn) <= 6.0
    l6v = [max(C.loc[C.space == sp, f"g_{m}"].iloc[0] for m in ONLINE) / C.loc[C.space == sp, "g_Gn_syn"].iloc[0] for sp in ("T5", "T15")]
    l6 = all(v <= 0.6 for v in l6v)
    sh0 = float(d[(d.space == "T15") & (d.method == "ql")].share0.mean())
    l8 = inA("ts", "T15")
    l9n = int(beyond(FBand).sum())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- ganhos medios sobre o aleatorio (pontos) e lacunas ao otimo (%) por espaco; 37 instancias ---", C.round(3).T.to_string(), "",
          "--- por faixa de tamanho (T15) ---", BR.round(3).to_string(index=False), "",
          "metodos que ganham do aleatorio alem da margem (A): " + "; ".join(f"{sp}: " + ", ".join(m for m in ["Gn_syn", "G0_syn"] + ONLINE + ["ql_perm"] if inA(m, sp)) for sp in ("T5", "T15")),
          "aprendizes online que batem Gn_syn alem da margem (Bp): " + (", ".join(FB[beyond(FB)].id) if beyond(FB).any() else "nenhum"),
          f"ql_perm difere do ql (E): " + "; ".join(f"{r.id} d={r.mean_d_pct:.3f} p_holm={r.p_holm:.3g}" for _, r in FE.iterrows()),
          f"fracao do ql padrao na acao listada primeiro: T15 {sh0:.3f}; T5 {float(d[(d.space == 'T5') & (d.method == 'ql')].share0.mean()):.3f}", "",
          "--- previsoes ---",
          f"(L1) Gn_syn ganha alem da margem em T5 e T15: {yn(l1)}",
          f"(L2) aprendizes online que batem Gn_syn alem da margem: {l2n} de 12 (previsto 0): {yn(l2n == 0)}",
          f"(L3) ql e ql_perm nao ganham alem da margem em nenhum espaco: {yn(l3)}",
          f"(L4) G0_syn ganha alem da margem em T15: {yn(l4)}",
          f"(L5) lacuna media de Gn_syn ao otimo em T15: {float(c15.gap_Gn_syn):.2f}% (previsto <= 6%): {yn(l5)}",
          f"(L6) razao melhor aprendiz online / Gn_syn: T5 {l6v[0]:.3f}, T15 {l6v[1]:.3f} (previsto <= 0,60): {yn(l6)}",
          f"(L7) fracao do ql na acao listada primeiro em T15: {sh0:.3f} (previsto >= 0,40): {yn(sh0 >= 0.40)}",
          f"(L8) Thompson ganha alem da margem em T15: {yn(l8)}",
          f"(L9) faixas de tamanho em que Gn_syn ganha alem da margem (T15): {l9n} de 3 (previsto >= 2): {yn(l9n >= 2)}"]
    pred = [l1, l2n == 0, l3, l4, l5, l6, sh0 >= 0.40, l8, l9n >= 2]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada270_tsplib.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada270_tsplib_cells.csv", index=False)
    BR.to_csv(R / "entrada270_tsplib_bands.csv", index=False)
    pd.concat([FA, FB, FE, FBand]).to_csv(R / "entrada270_tsplib_tests.csv", index=False)


if __name__ == "__main__":
    main()
