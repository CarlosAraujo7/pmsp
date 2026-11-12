"""
analysis/report_entrada263_qperm_longT.py
==========================================
Entrada 263 (Ciclo 37): o Q-LEARNING INVARIANTE A ORDEM (ql_perm) no horizonte longo T = 10.000 do benchmark de Fanjul-Peyro e Ruiz, espaco B30, 7 tipos de instancia (100 instancias de teste cada), 3 sementes.
Motivo: na Entrada 258 (horizonte longo, results/entrada258L_longT.txt) o Q-Learning padrao ganhou do aleatorio alem da margem em 2 dos 7 tipos (U1000a1100: 0,07 pontos; U10a100: 0,45), contra a previsao L4 (no maximo 1).
Pergunta: isso e' o travamento no indice 0 (Entradas 258 etapas 1-2 e 260), ou e' aprendizado? Mesmas instancias, mesma validada e mesmo aleatorio da Entrada 258; o ql_perm tem sementes proprias.
Familias de testes (Holm sobre os 7 tipos): A (ql_perm contra o aleatorio), B (a validada contra ql_perm), E (ql_perm contra o ql padrao). Margem +-0,05; 'alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar; hashes em results/entrada263_prereg_hashes.txt; nenhum resultado do ql_perm a T = 10.000 visto; o ql padrao, o aleatorio e a validada vem da Entrada 258)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(Q1) O ql_perm ganha do aleatorio alem da margem (A) em no maximo 1 dos 7 tipos (o ql padrao: 2). (75%)
(Q2) Nos dois tipos em que o ql padrao ganhou alem da margem (U1000a1100 e U10a100), o ganho do ql_perm e' menor que o do ql padrao nos dois. (70%)
(Q3) A razao entre as avaliacoes do ql_perm e as do aleatorio, media dos 7 tipos, fica entre 0,9 e 1,1 (a T = 1.000 foi 0,99-1,01 e a do padrao 0,44-0,46). (80%)
(Q4) A validada bate o ql_perm (B, confirmada) nos 7 tipos. (85%)
(Q5) O ganho medio do ql_perm nos 7 tipos e' de no maximo 0,15 ponto (o ql padrao: 0,057). (70%)
Uso: python -m analysis.report_entrada263_qperm_longT
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
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "frL_test_B30_T10000_s*of12_entrada258.csv")))], ignore_index=True)
    w = d.groupby(["type", "instance"]).apply(lambda g: g.groupby("method").makespan.mean(), include_groups=False).reset_index()
    ev = d.groupby(["type", "instance"]).apply(lambda g: g.groupby("method").evals.mean(), include_groups=False).reset_index()
    q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "qpermL_test_B30_T10000_s*of*_entrada263.csv")))], ignore_index=True)
    qm = q.groupby(["type", "instance"]).agg(qp=("makespan", "mean"), qp_ev=("evals", "mean")).reset_index()
    w = w.merge(qm, on=["type", "instance"])
    ev = ev.merge(qm, on=["type", "instance"])
    cells, TA, TB, TE = [], [], [], []
    for ty in sorted(w.type.unique()):
        x, e = w[w.type == ty], ev[ev.type == ty]
        cells.append(dict(type=ty, g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean()),
                          ev_ql=float((e.ql / e.rand).mean()), ev_qp=float((e.qp_ev / e.rand).mean()), n=len(x)))
        a = test(x.qp, x.rand); a.update(id=f"A-{ty}"); TA.append(a)
        b = test(x.val, x.qp); b.update(id=f"B-{ty}"); TB.append(b)
        c = test(x.qp, x.ql); c.update(id=f"E-{ty}"); TE.append(c)
    C = pd.DataFrame(cells)
    FA, FB, FE = family(TA), family(TB), family(TE)
    q1 = int(beyond(FA).sum())
    two = C[C.type.isin(["U1000a1100", "U10a100"])]
    q2 = bool((two.g_qp < two.g_ql).all())
    evr = float(C.ev_qp.mean())
    q3 = 0.9 <= evr <= 1.1
    q4 = int(FB.confirmada.sum())
    gm = float(C.g_qp.mean())
    q5 = gm <= 0.15
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "Q-LEARNING INVARIANTE A ORDEM, HORIZONTE LONGO T = 10.000, B30 (Entrada 263)", "=" * 110,
         "--- por tipo: ganhos sobre o aleatorio (pontos) e razao de avaliacoes ao aleatorio ---", C.round(3).to_string(index=False), "",
         "--- A: ql_perm contra o aleatorio ---", FA.round(4).to_string(index=False), "",
         "--- E: ql_perm contra o ql padrao ---", FE.round(4).to_string(index=False), "",
         f"o ql_perm ganha do aleatorio alem da margem em {q1} de 7 tipos; a validada bate o ql_perm (B, confirmada) em {q4} de 7; razao media de avaliacoes do ql_perm {evr:.3f}, do ql padrao {C.ev_ql.mean():.3f}", "",
         "--- previsoes ---",
         f"(Q1) tipos em que o ql_perm ganha alem da margem: {q1} de 7 (previsto <= 1): {yn(q1 <= 1)}",
         f"(Q2) ganho do ql_perm menor que o do ql padrao em U1000a1100 e U10a100: {two[['type', 'g_ql', 'g_qp']].round(3).values.tolist()}: {yn(q2)}",
         f"(Q3) razao media de avaliacoes do ql_perm ao aleatorio: {evr:.3f} (previsto 0,9-1,1): {yn(q3)}",
         f"(Q4) tipos em que a validada bate o ql_perm: {q4} de 7 (previsto 7): {yn(q4 == 7)}",
         f"(Q5) ganho medio do ql_perm: {gm:.3f} (previsto <= 0,15): {yn(q5)}"]
    pred = [q1 <= 1, q2, q3, q4 == 7, q5]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada263_qperm_longT.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([FA, FB, FE]).to_csv(R / "entrada263_qperm_longT_tests.csv", index=False)
    C.to_csv(R / "entrada263_qperm_longT_cells.csv", index=False)


if __name__ == "__main__":
    main()
