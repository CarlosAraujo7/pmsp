"""
analysis/report_entrada261_qperm_pcmax.py
==========================================
Entrada 261 (Ciclo 35): o Q-Learning padrao invariante a ordem (ql_perm) na bateria principal de P||Cmax (Entrada 180; 12 classes (n, m) x 100 instancias de teste; espacos S5 e S15; T = 1.000; 5 sementes). Celula = (classe, espaco)
(24); unidade = media das sementes por instancia (100 pares). Comparacoes (Holm sobre as 24 celulas): A (ql_perm contra o aleatorio), B (a validada contra ql_perm). O ql padrao, o aleatorio e a validada vem dos arquivos da Entrada 180.
Margem +-0,05; 'alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar; hashes em results/entrada261_prereg_hashes.txt; nenhum resultado do ql_perm no P||Cmax visto; antes, Entrada 260: no R||Cmax o ql_perm ganha de o aleatorio em 0 de 27 celulas)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(R1) O ql_perm ganha do aleatorio alem da margem em no maximo 1 das 24 celulas. (75%)
(R2) O ganho medio do ql_perm sobre o aleatorio (media das 24 celulas) e' menor ou igual ao do ql padrao em cada um dos dois espacos. (65%)
(R3) A validada bate o ql_perm (B, confirmada) em pelo menos 20 das 24 celulas. (75%)
(R4) A fracao de escolhas do ql_perm na acao de indice original 0 e' perto de 1/A (no maximo 2/A em cada espaco), isto e', a permutacao desfaz o travamento; (o uso nao e' gravado nesta bateria: previsao avaliada pela razao evals ql_perm / evals aleatorio >= 0,8 em ambos os espacos, pois o travamento na acao mais barata reduz as avaliacoes (ql padrao: <= 0,6)). (60%)
Uso: python -m analysis.report_entrada261_qperm_pcmax
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


def main():
    cells, TA, TB = [], [], []
    ratios = {}
    for sp in ("S5", "S15"):
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        d = d[d.method.isin(["rand", "ql", "val"])]
        w = d.groupby(["cls", "rep"]).apply(lambda g: g.groupby("method").makespan.mean(), include_groups=False).reset_index()
        ev = d.groupby("method").evals.mean()
        q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"qperm_pcmax_{sp}_s*of*_entrada261.csv")))], ignore_index=True)
        w = w.merge(q.groupby(["cls", "rep"]).makespan.mean().rename("qp").reset_index(), on=["cls", "rep"])
        ratios[sp] = (float(q.evals.mean() / ev["rand"]), float(ev["ql"] / ev["rand"]))
        for cls in sorted(w.cls.unique()):
            x = w[w.cls == cls]
            cells.append(dict(space=sp, cls=cls, g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean())))
            a = test(x.qp, x.rand); a.update(id=f"A-{cls}-{sp}"); TA.append(a)
            b = test(x.val, x.qp); b.update(id=f"B-{cls}-{sp}"); TB.append(b)
    C = pd.DataFrame(cells)
    FA, FB = family(TA), family(TB)
    nb = int((FA.confirmada & ~FA["equivalente_0.05"]).sum())
    r1 = nb <= 1
    mg = C.groupby("space")[["g_ql", "g_qp", "g_val"]].mean()
    r2 = bool(((mg.g_qp <= mg.g_ql + 1e-12)).all())
    nbb = int(FB.confirmada.sum())
    r3 = nbb >= 20
    r4 = all(ratios[sp][0] >= 0.8 for sp in ratios)
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "Q-LEARNING INVARIANTE A ORDEM NA BATERIA PRINCIPAL DE P||Cmax (Entrada 261)", "=" * 110,
         "--- ganhos sobre o aleatorio por celula (pontos) ---", C.round(4).to_string(index=False), "",
         "--- medias por espaco ---", mg.round(4).to_string(), "",
         f"razao de avaliacoes (ql_perm / aleatorio; ql padrao / aleatorio): S5 {ratios['S5'][0]:.3f}; {ratios['S5'][1]:.3f} | S15 {ratios['S15'][0]:.3f}; {ratios['S15'][1]:.3f}", "",
         "--- previsoes ---",
         f"(R1) celulas com ganho do ql_perm alem da margem: {nb} de 24 (previsto <= 1): {yn(r1)}",
         f"(R2) ganho medio ql_perm <= ql padrao nos dois espacos: S5 {mg.loc['S5', 'g_qp']:+.4f} vs {mg.loc['S5', 'g_ql']:+.4f}; S15 {mg.loc['S15', 'g_qp']:+.4f} vs {mg.loc['S15', 'g_ql']:+.4f}: {yn(r2)}",
         f"(R3) a validada bate o ql_perm em {nbb} de 24 celulas (previsto >= 20): {yn(r3)}",
         f"(R4) razao de avaliacoes do ql_perm >= 0,8 nos dois espacos: {ratios['S5'][0]:.3f} e {ratios['S15'][0]:.3f}: {yn(r4)}"]
    pred = [r1, r2, r3, r4]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada261_qperm_pcmax.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([FA, FB]).to_csv(R / "entrada261_qperm_pcmax_tests.csv", index=False)
    C.to_csv(R / "entrada261_qperm_pcmax_cells.csv", index=False)


if __name__ == "__main__":
    main()
