"""
analysis/report_entrada260_qperm.py
====================================
Entrada 260 (Ciclo 35): o Q-LEARNING INVARIANTE A ORDEM DAS ACOES (ql_perm: permutacao aleatoria das acoes a cada execucao) no benchmark de Fanjul-Peyro e Ruiz (Entrada 256: 21 celulas tipo x espaco; 700 instancias de teste)
e nas familias publicadas de R||Cmax (Entrada 254: 6 celulas familia x espaco; 500 instancias por celula). Motivo: Entrada 258 (etapa 2) mostrou que o Q-Learning padrao recupera 81-86% do ganho da validada quando a validada fica no
indice 0, isto e', repete o primeiro indice (55-73% das escolhas) e o seu resultado depende da ordem das acoes. Pergunta: o que sobra do Q-Learning padrao quando a ordem deixa de importar, nas celulas em que ele ganhava
(familias com maquinas correlacionadas: 53-65% da validada) ou perdia (B30)?
Familias de testes (Holm sobre as celulas, por comparacao): A (ql_perm contra o aleatorio), B (a validada contra ql_perm), E (ql_perm contra o ql padrao). Margem +-0,05; 'alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar; hashes em results/entrada260_prereg_hashes.txt; nenhum resultado do ql_perm visto; o ql padrao e a validada vem das Entradas 254/256)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(P1) No benchmark de Fanjul-Peyro e Ruiz, o ganho medio do ql_perm sobre o aleatorio nas 21 celulas fica entre -0,25 e +0,25 pontos. (65%)
(P2) No benchmark de Fanjul-Peyro e Ruiz, o ql_perm ganha do aleatorio alem da margem em no maximo 3 das 21 celulas (o ql padrao: 3). (60%)
(P3) Nas familias publicadas, o ql_perm ganha do aleatorio alem da margem em no maximo 3 das 6 celulas (o ql padrao: 4). (55%)
(P4) Nas familias publicadas, a maior razao ganho do ql_perm / ganho da validada nas 6 celulas e' no maximo 0,45 (o ql padrao: 0,65). (55%)
(P5) A validada bate o ql_perm (B, confirmada) nas 21 celulas de Fanjul-Peyro e Ruiz e em pelo menos 5 das 6 celulas das familias. (80%)
(P6) Nas familias publicadas, o ql_perm difere do ql padrao alem da margem (E, confirmada e nao equivalente) em pelo menos 1 das 6 celulas. (70%)
Uso: python -m analysis.report_entrada260_qperm
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
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def main():
    L = ["=" * 110, "Q-LEARNING INVARIANTE A ORDEM (Entrada 260)", "=" * 110]
    # ------------------------------------------------ Fanjul-Peyro e Ruiz
    cells, TA, TB, TE = [], [], [], []
    for sp in ("B5", "B15", "B30"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"fr_test_{sp}_s*of*_entrada256.csv")))], ignore_index=True)
        w = d.groupby(["type", "instance"]).apply(lambda g: g.groupby("method").makespan.mean(), include_groups=False).reset_index()
        q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"qperm_fanjul_{sp}_s*of*_entrada260.csv")))], ignore_index=True)
        qm = q.groupby(["type", "instance"]).makespan.mean().rename("qp").reset_index()
        w = w.merge(qm, on=["type", "instance"])
        for ty in sorted(w.type.unique()):
            x = w[w.type == ty]
            cells.append(dict(space=sp, type=ty, g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean())))
            a = test(x.qp, x.rand); a.update(id=f"A-{ty}-{sp}"); TA.append(a)
            b = test(x.val, x.qp); b.update(id=f"B-{ty}-{sp}"); TB.append(b)
    CF = pd.DataFrame(cells)
    FA, FB = family(TA), family(TB)
    p1v = float(CF.g_qp.mean())
    p1 = -0.25 <= p1v <= 0.25
    n2 = int(beyond(FA).sum())
    p2 = n2 <= 3
    L += ["--- Fanjul-Peyro e Ruiz: ganhos sobre o aleatorio por celula (pontos) ---", CF.round(4).to_string(index=False), "",
          f"ganho medio nas 21 celulas: ql_perm {p1v:+.4f}; ql padrao {CF.g_ql.mean():+.4f}; validada {CF.g_val.mean():+.4f}",
          f"celulas em que o ql_perm ganha alem da margem: {n2} de 21 (ql padrao: 3)",
          f"celulas em que a validada bate o ql_perm (B, confirmada): {int(FB.confirmada.sum())} de 21", ""]
    # ------------------------------------------------ familias publicadas
    fcells, TA2, TB2, TE2 = [], [], [], []
    for sp in ("S5", "S15"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_ext_test_{sp}_s*of*_entrada254.csv")))], ignore_index=True)
        d = d[d.method.isin(["rand", "ql", "val"])]
        w = d.groupby(["cls", "rep"]).apply(lambda g: g.groupby("method").makespan.mean(), include_groups=False).reset_index()
        q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"qperm_families_{sp}_s*of*_entrada260.csv")))], ignore_index=True)
        qm = q.groupby(["cls", "rep"]).makespan.mean().rename("qp").reset_index()
        w = w.merge(qm, on=["cls", "rep"])
        w["fam"] = w.cls.str.split("|").str[0]
        for fam in ("unc", "cj", "cm"):
            x = w[w.fam == fam]
            fcells.append(dict(space=sp, fam=fam, g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean())))
            a = test(x.qp, x.rand); a.update(id=f"A-{fam}-{sp}"); TA2.append(a)
            b = test(x.val, x.qp); b.update(id=f"B-{fam}-{sp}"); TB2.append(b)
            e = test(x.qp, x.ql); e.update(id=f"E-{fam}-{sp}"); TE2.append(e)
    CF2 = pd.DataFrame(fcells)
    GA, GB, GE = family(TA2), family(TB2), family(TE2)
    n3 = int(beyond(GA).sum())
    p3 = n3 <= 3
    CF2["ratio_qp"] = CF2.g_qp / CF2.g_val
    CF2["ratio_ql"] = CF2.g_ql / CF2.g_val
    p4 = float(CF2.ratio_qp.max()) <= 0.45
    nb_f = int(GB.confirmada.sum())
    p5 = (int(FB.confirmada.sum()) == 21) and (nb_f >= 5)
    ne = int(beyond(GE).sum())
    p6 = ne >= 1
    L += ["--- familias publicadas de R||Cmax: ganhos sobre o aleatorio por celula (pontos) ---", CF2.round(4).to_string(index=False), "",
          f"celulas em que o ql_perm ganha alem da margem: {n3} de 6 (ql padrao: 4)", f"a validada bate o ql_perm (B): {nb_f} de 6; ql_perm difere do ql padrao alem da margem (E): {ne} de 6", "",
          "--- testes A, B, E nas familias ---", pd.concat([GA, GB, GE])[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].to_string(index=False), "",
          "--- previsoes ---",
          f"(P1) ganho medio do ql_perm nas 21 celulas: {p1v:+.3f} (previsto em [-0,25; +0,25]): {'SIM' if p1 else 'NAO'}",
          f"(P2) celulas de Fanjul-Peyro e Ruiz com ganho do ql_perm alem da margem: {n2} (previsto <= 3): {'SIM' if p2 else 'NAO'}",
          f"(P3) celulas das familias com ganho do ql_perm alem da margem: {n3} (previsto <= 3): {'SIM' if p3 else 'NAO'}",
          f"(P4) maior razao ql_perm / validada nas familias: {CF2.ratio_qp.max():.3f} (previsto <= 0,45): {'SIM' if p4 else 'NAO'}",
          f"(P5) a validada bate o ql_perm: {int(FB.confirmada.sum())} de 21 e {nb_f} de 6 (previsto 21 e >= 5): {'SIM' if p5 else 'NAO'}",
          f"(P6) ql_perm difere do ql padrao alem da margem em {ne} das 6 celulas das familias (previsto >= 1): {'SIM' if p6 else 'NAO'}"]
    pred = [p1, p2, p3, p4, p5, p6]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada260_qperm.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([FA, FB, GA, GB, GE]).to_csv(R / "entrada260_qperm_tests.csv", index=False)
    CF.to_csv(R / "entrada260_qperm_cells_fanjul.csv", index=False)
    CF2.to_csv(R / "entrada260_qperm_cells_families.csv", index=False)


if __name__ == "__main__":
    main()
