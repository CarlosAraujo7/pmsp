"""
analysis/report_entrada254_ur_external.py
==========================================
Entrada 254 (Ciclo 30): o protocolo no R||Cmax com as FAMILIAS PUBLICADAS de instancias (experiment/run_cycle30_ur_external_entrada254.py; pmsp/instance_ur_ext.py): nao correlacionada U{10,100} (unc), correlacionada por tarefa (cj) e por maquina (cm), 25 tamanhos por familia (m em {3,5,10,15,20}, n em {30,50,80,100,200}), 20 instancias de teste por classe, espacos S5 e S15, T = 1.000; unidade de analise: a media das sementes por instancia (500 pares por familia e espaco). A acao validada e' a melhor acao fixa por classe em 20 instancias de ajuste; os hiperparametros dos aprendizes sao os congelados das Entradas 180/234. Celula = (familia, espaco) (6); ganho = -d sobre o aleatorio (pontos do makespan); folga enviesada = melhor acao fixa por instancia em 1 semente; folga cf = escolha em 2 sementes e avaliacao em 2 outras (4 sementes por acao).
Familias de testes (Holm sobre as 6 celulas, por aprendiz): A (aprendiz contra o aleatorio), B (a validada contra o aprendiz), B' (o aprendiz contra a validada). Margem de equivalencia +-0,05; 'bate alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada254_prereg_hashes.txt; nenhum resultado de teste foi visto; o ajuste da acao validada so' usa instancias de ajuste)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(X1) O Q-Learning padrao (ql) nao ganha alem da margem sobre o aleatorio em nenhuma das 6 celulas. (80%)
(X2) A validada bate o Q-Learning (B, confirmada) em pelo menos 5 das 6 celulas. (85%)
(X3) Nenhum aprendiz (ql, ucb, pg, alns, qo) bate a validada alem da margem (B') em nenhum dos 30 pares aprendiz-celula. (85%)
(X4) O ganho da validada sobre o aleatorio e' maior que a margem (0,05) em pelo menos 5 das 6 celulas (ha' folga nas familias publicadas). (75%)
(X5) Em todas as celulas em que a validada ganha mais que a margem, o melhor aprendiz (maximo entre os cinco) obtem no maximo 30% do ganho da validada. (60%)
(X6) O ganho da validada na familia nao correlacionada (unc) com quinze acoes (S15) e' pelo menos 0,5 pontos (a mesma ordem de grandeza do gerador original, 2,08). (55%)
Uso: python -m analysis.report_entrada254_ur_external
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
from analysis.report_entrada234_ur import per_instance  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
FAMS, SPACES = ("unc", "cj", "cm"), ("S5", "S15")
LEARN = ["ql", "ucb", "pg", "alns", "qo"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def beats(F):
    return (F.confirmada) & (~F["equivalente_0.05"])


def main():
    T = {k: [] for k in "ABC"}
    cells = []
    for sp in SPACES:
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_ext_test_{sp}_s*of*_entrada254.csv")))], ignore_index=True)
        w = per_instance(d)
        w["fam"] = w.cell.str.split("|").str[0]
        for fam in FAMS:
            x = w[w.fam == fam]
            cells.append(dict(fam=fam, space=sp, n=len(x), h_biased=float(-rel(x.biased, x.rand).mean()), h_cf=float(-rel(x.cf, x.rand).mean()), g_val=float(-rel(x.val, x.rand).mean()),
                              **{f"g_{k}": float(-rel(x[k], x.rand).mean()) for k in LEARN}))
            for k in LEARN:
                a = test(x[k], x.rand); a.update(id=f"A-{fam}-{sp}-{k}"); T["A"].append(a)
                b = test(x.val, x[k]); b.update(id=f"B-{fam}-{sp}-{k}"); T["B"].append(b)
                c = test(x[k], x.val); c.update(id=f"B'-{fam}-{sp}-{k}"); T["C"].append(c)
    F = {}
    for fam_, rows in T.items():
        F[fam_] = pd.concat([family([r for r in rows if r["id"].endswith(f"-{k}")]) for k in LEARN], ignore_index=True)
    C = pd.DataFrame(cells)
    L = ["=" * 110, "R||Cmax NAS FAMILIAS PUBLICADAS (Entrada 254)", "=" * 110,
         "--- ganhos sobre o aleatorio por familia e espaco (pontos; folga enviesada = uma semente, cf = validacao cruzada de sementes) ---", C.to_string(index=False)]
    for k_, lab in (("A", "A: aprendiz contra o aleatorio"), ("B", "B: validada contra o aprendiz"), ("C", "B': aprendiz contra a validada")):
        L += ["", f"--- Familia {lab} (Holm sobre as 6 celulas, por aprendiz) ---", F[k_][COLS].to_string(index=False)]
    x1 = bool(not beats(F["A"][F["A"].id.str.endswith("-ql")]).any())
    x2 = int(F["B"][F["B"].id.str.endswith("-ql")].confirmada.sum())
    x3 = int(beats(F["C"]).sum())
    x4 = int((C.g_val > 0.05).sum())
    pos = C[C.g_val > 0.05]
    ratio = (pos[[f"g_{k}" for k in LEARN]].max(axis=1) / pos.g_val)
    x5 = bool((ratio <= 0.30).all())
    g6 = float(C[(C.fam == "unc") & (C.space == "S15")].g_val.iloc[0])
    yn = lambda b: "SIM" if b else "NAO"
    L += ["", "--- previsoes ---",
          f"(X1) ql nao ganha alem da margem em nenhuma celula: {yn(x1)}",
          f"(X2) a validada bate o ql: {x2} de 6 (previsto >= 5): {yn(x2 >= 5)}",
          f"(X3) aprendizes que batem a validada alem da margem (B'): {x3} de 30 (previsto 0): {yn(x3 == 0)}",
          f"(X4) celulas com ganho da validada > 0,05: {x4} de 6 (previsto >= 5): {yn(x4 >= 5)}",
          f"(X5) razao maxima melhor aprendiz / validada nas celulas com ganho: {ratio.max():.3f} (previsto <= 0,30): {yn(x5)}",
          f"(X6) ganho da validada em unc-S15: {g6:.3f} (previsto >= 0,5): {yn(g6 >= 0.5)}"]
    pred = [x1, x2 >= 5, x3 == 0, x4 >= 5, x5, g6 >= 0.5]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada254_ur_external.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([F["A"], F["B"]]).to_csv(R / "entrada254_ur_external_tests.csv", index=False)
    C.to_csv(R / "entrada254_ur_external_cells.csv", index=False)


if __name__ == "__main__":
    main()
