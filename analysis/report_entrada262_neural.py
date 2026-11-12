"""
analysis/report_entrada262_neural.py
=====================================
Entrada 262 (Ciclo 36): a POLITICA NEURAL treinada OFFLINE (estilo DAC; pmsp/ur_numba_neural.py; treino por estrategias evolutivas nas 280 instancias de AJUSTE, experiment/train_neural_entrada259.py; pesos fixos no teste) no benchmark
de Fanjul-Peyro e Ruiz, espacos B15 e B30 (14 celulas tipo x espaco; 700 instancias de teste; T = 1.000; 3 sementes). Comparada com o aleatorio, a validada e os sete aprendizes online da Entrada 256 (mesmas instancias) e com o
controle nn_const (mistura constante de acoes treinada com o mesmo orcamento, sem caracteristicas). Custo de treino: execucoes do IG registradas em results/nn_train_*_entrada259.csv (nao e' cobrado no teste).
Familias de testes (Holm sobre as 14 celulas, por comparacao): A (nn contra o aleatorio), B (a validada contra nn), B' (nn contra a validada), D (nn contra nn_const). Margem +-0,05; 'alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada262_prereg_hashes.txt; a politica foi treinada so' nas instancias de ajuste; nenhum resultado de teste da politica visto)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(N1) A politica neural ganha do aleatorio alem da margem em pelo menos 11 das 14 celulas. (60%)
(N2) Nenhuma celula em que a politica neural bate a validada alem da margem (0 de 14). (60%)
(N3) A politica neural supera o melhor dos sete aprendizes online (ganho maior) em pelo menos 10 das 14 celulas. (65%)
(N4) O ganho medio da politica neural nas 14 celulas e' pelo menos 50% do da validada. (55%)
(N5) A politica neural bate o controle nn_const (D, confirmada e nao equivalente) em pelo menos 5 das 14 celulas (as caracteristicas do estado e da instancia ajudam). (45%)
Uso: python -m analysis.report_entrada262_neural
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


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def main():
    cells, TA, TB, TC, TD = [], [], [], [], []
    for sp in ("B15", "B30"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"fr_test_{sp}_s*of*_entrada256.csv")))], ignore_index=True)
        w = d.groupby(["type", "instance"]).apply(lambda g: g.groupby("method").makespan.mean(), include_groups=False).reset_index()
        q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"nn_test_{sp}_s*of*_entrada262.csv")))], ignore_index=True)
        qm = q.groupby(["type", "instance", "method"]).makespan.mean().unstack().reset_index()
        w = w.merge(qm, on=["type", "instance"])
        for ty in sorted(w.type.unique()):
            x = w[w.type == ty]
            g = {k: float(-rel(x[k], x.rand).mean()) for k in ["val", "nn", "nn_const"] + LEARN}
            gap = lambda c: float((100 * (x[c] - 0).mean()))
            cells.append(dict(space=sp, type=ty, **{f"g_{k}": v for k, v in g.items()}, best_online=max(g[k] for k in LEARN)))
            a = test(x.nn, x.rand); a.update(id=f"A-{ty}-{sp}"); TA.append(a)
            b = test(x.val, x.nn); b.update(id=f"B-{ty}-{sp}"); TB.append(b)
            c = test(x.nn, x.val); c.update(id=f"B'-{ty}-{sp}"); TC.append(c)
            e = test(x.nn, x.nn_const); e.update(id=f"D-{ty}-{sp}"); TD.append(e)
    C = pd.DataFrame(cells)
    FA, FB, FC, FD = family(TA), family(TB), family(TC), family(TD)
    n1 = int(beyond(FA).sum())
    n2 = int(beyond(FC).sum())
    n3 = int((C.g_nn > C.best_online).sum())
    ratio = float(C.g_nn.mean() / C.g_val.mean())
    n5 = int(beyond(FD).sum())
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "POLITICA NEURAL TREINADA OFFLINE NO BENCHMARK DE FANJUL-PEYRO E RUIZ (Entrada 262)", "=" * 110,
         "--- ganhos sobre o aleatorio por celula (pontos) ---", C.round(3).to_string(index=False), "",
         f"ganho medio nas 14 celulas: validada {C.g_val.mean():.3f}; nn {C.g_nn.mean():.3f}; nn_const {C.g_nn_const.mean():.3f}; melhor online por celula (media) {C.best_online.mean():.3f}",
         f"a validada bate nn (B, confirmada): {int(FB.confirmada.sum())} de 14; nn bate a validada alem da margem (B'): {n2}", ""]
    for k_, lab in ((FA, "A: nn contra o aleatorio"), (FD, "D: nn contra nn_const")):
        L += [f"--- {lab} ---", k_[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].to_string(index=False), ""]
    L += ["--- previsoes ---",
          f"(N1) celulas em que nn ganha do aleatorio alem da margem: {n1} de 14 (previsto >= 11): {yn(n1 >= 11)}",
          f"(N2) celulas em que nn bate a validada alem da margem: {n2} de 14 (previsto 0): {yn(n2 == 0)}",
          f"(N3) celulas em que nn supera o melhor aprendiz online: {n3} de 14 (previsto >= 10): {yn(n3 >= 10)}",
          f"(N4) razao do ganho medio nn / validada: {ratio:.3f} (previsto >= 0,5): {yn(ratio >= 0.5)}",
          f"(N5) celulas em que nn bate nn_const alem da margem: {n5} de 14 (previsto >= 5): {yn(n5 >= 5)}"]
    pred = [n1 >= 11, n2 == 0, n3 >= 10, ratio >= 0.5, n5 >= 5]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada262_neural.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([FA, FB, FC, FD]).to_csv(R / "entrada262_neural_tests.csv", index=False)
    C.to_csv(R / "entrada262_neural_cells.csv", index=False)


if __name__ == "__main__":
    main()
