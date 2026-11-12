"""
analysis/report_entrada256_fanjul.py
=====================================
Entrada 256 (Ciclo 32): o protocolo compilado no BENCHMARK PADRAO de R||Cmax de Fanjul-Peyro e Ruiz (experiment/run_cycle32_fanjul_entrada256.py): 700 instancias de teste (7 tipos x m em {10,...,50} x n em {100,...,1000} x 5), espacos B5, B15 e B30 (grade de d/n ajustada a escala: 0,01 a 0,20; B30 inclui busca local), metodos rand, ql, ucb, pg, alns, qo, iq, fr (hiperparametros congelados) e val (a melhor acao fixa por classe (tipo, n) nas instancias de ajuste), 3 sementes, T = 1.000; unidade: a media das sementes por instancia; celula = (tipo, espaco) (21; 100 instancias por celula). Referencia absoluta: o MELHOR VALOR CONHECIDO (minimo de CPLEX 2 h, Gurobi 300 s e LR 300 s; terceiros) e se algum metodo provou otimalidade. Lacuna = 100 (makespan - melhor conhecido) / melhor conhecido.
Familias de testes (Holm sobre as 21 celulas, por aprendiz): A (o aprendiz contra o aleatorio), B (a validada contra o aprendiz), B' (o aprendiz contra a validada). Margem +-0,05; 'alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada256_prereg_hashes.txt; so' as instancias de AJUSTE foram usadas para escolher a acao validada; nenhum resultado de teste visto)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(Z1) A validada bate o Q-Learning (B, confirmada) em pelo menos 15 das 21 celulas. (65%)
(Z2) Nenhum dos sete aprendizes bate a validada alem da margem (B') em nenhuma das 147 celulas aprendiz-tipo-espaco. (55%)
(Z3) O Q-Learning ganha do aleatorio alem da margem em pelo menos um tipo em cada um dos tres espacos. (65%)
(Z4) A acao validada em B30 tem lacuna media ao melhor conhecido de no maximo 5% nas 700 instancias. (60%)
(Z5) A validada em B30 tem lacuna media menor que a validada em B5 (a busca local ajuda a qualidade absoluta). (85%)
(Z6) O Spearman entre o ganho da validada e o do Q-Learning nas 21 celulas e' >= 0,5. (60%)
(Z7) A validada bate o aleatorio alem da margem (ganho > 0 confirmado e nao equivalente) em pelo menos 18 das 21 celulas (a grade tem folga real). (75%)
Uso: python -m analysis.report_entrada256_fanjul
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
SPACES = ("B5", "B15", "B30")
LEARN = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def beats(F):
    return (F.confirmada) & (~F["equivalente_0.05"])


def main():
    parts = []
    for sp in SPACES:
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"fr_test_{sp}_s*of*_entrada256.csv")))], ignore_index=True)
        w = d.groupby(["space", "type", "instance", "m", "n", "bks"]).apply(lambda g: g.groupby("method").makespan.mean(), include_groups=False).reset_index()
        parts.append(w)
    W = pd.concat(parts, ignore_index=True)
    cells, T = [], {k: [] for k in "ABC"}
    for sp in SPACES:
        for ty in sorted(W.type.unique()):
            x = W[(W.space == sp) & (W.type == ty)]
            gap = lambda c: float((100 * (x[c] - x.bks) / x.bks).mean())
            cells.append(dict(space=sp, type=ty, n=len(x), g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), **{f"g_{k}": float(-rel(x[k], x.rand).mean()) for k in LEARN[1:]},
                              gap_rand=gap("rand"), gap_ql=gap("ql"), gap_val=gap("val")))
            for k in LEARN:
                a = test(x[k], x.rand); a.update(id=f"A-{ty}-{sp}-{k}"); T["A"].append(a)
                b = test(x.val, x[k]); b.update(id=f"B-{ty}-{sp}-{k}"); T["B"].append(b)
                c = test(x[k], x.val); c.update(id=f"B'-{ty}-{sp}-{k}"); T["C"].append(c)
    # Holm sobre as 21 celulas por aprendiz
    F = {}
    for fam, rows in T.items():
        F[fam] = pd.concat([family([r for r in rows if r["id"].endswith(f"-{k}")]) for k in LEARN], ignore_index=True)
    C = pd.DataFrame(cells)
    tot = {sp: float((100 * (W[W.space == sp].val - W[W.space == sp].bks) / W[W.space == sp].bks).mean()) for sp in SPACES}
    totr = {sp: float((100 * (W[W.space == sp].rand - W[W.space == sp].bks) / W[W.space == sp].bks).mean()) for sp in SPACES}
    L = ["=" * 110, "BENCHMARK PADRAO DE R||Cmax DE FANJUL-PEYRO E RUIZ (Entrada 256)", "=" * 110,
         "--- por tipo e espaco (100 instancias por celula): ganhos sobre o aleatorio (pontos) e lacunas ao melhor conhecido (%) ---", C.to_string(index=False),
         "", "--- lacuna media ao melhor conhecido nas 700 instancias (%): validada / aleatorio ---"] + [f"{sp}: validada {tot[sp]:.3f}; aleatorio {totr[sp]:.3f}" for sp in SPACES]
    for k_, lab in (("A", "A: aprendiz contra o aleatorio"), ("B", "B: validada contra o aprendiz"), ("C", "B': aprendiz contra a validada")):
        L += ["", f"--- Familia {lab} (Holm sobre as 21 celulas, por aprendiz) ---", F[k_][COLS].to_string(index=False)]
    z1 = int(F["B"][F["B"].id.str.endswith("-ql")].confirmada.sum())
    z2 = int(beats(F["C"]).sum())
    Aq = F["A"][F["A"].id.str.endswith("-ql")]
    z3 = all(bool(beats(Aq[Aq.id.str.endswith(f"-{sp}-ql")]).any()) for sp in SPACES)
    z4 = tot["B30"] <= 5.0
    z5 = tot["B30"] < tot["B5"]
    rho = float(sstats.spearmanr(C.g_val, C.g_ql).statistic)
    Av = []
    for sp in SPACES:
        for ty in sorted(W.type.unique()):
            x = W[(W.space == sp) & (W.type == ty)]
            t = test(x.val, x.rand)
            Av.append(t)
    Av = family(pd.DataFrame(Av).assign(id=[f"V-{i}" for i in range(len(Av))]).to_dict("records"))
    z7 = int(beats(Av).sum())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["", "--- previsoes ---",
          f"(Z1) a validada bate o ql: {z1} de 21 (previsto >= 15): {yn(z1 >= 15)}",
          f"(Z2) aprendizes que batem a validada alem da margem: {z2} de 147 (previsto 0): {yn(z2 == 0)}",
          f"(Z3) o ql ganha alem da margem em pelo menos um tipo em cada espaco: {yn(z3)}",
          f"(Z4) lacuna da validada em B30: {tot['B30']:.3f}% (previsto <= 5%): {yn(z4)}",
          f"(Z5) lacuna da validada B30 {tot['B30']:.3f}% contra B5 {tot['B5']:.3f}%: {yn(z5)}",
          f"(Z6) Spearman ganho da validada x ganho do ql nas 21 celulas: {rho:+.3f} (previsto >= 0,5): {yn(rho >= 0.5)}",
          f"(Z7) celulas em que a validada bate o aleatorio alem da margem: {z7} de 21 (previsto >= 18): {yn(z7 >= 18)}"]
    pred = [z1 >= 15, z2 == 0, z3, z4, z5, rho >= 0.5, z7 >= 18]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada256_fanjul.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([F["A"], F["B"]]).to_csv(R / "entrada256_fanjul_tests.csv", index=False)
    C.to_csv(R / "entrada256_fanjul_cells.csv", index=False)


if __name__ == "__main__":
    main()
