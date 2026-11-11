"""
analysis/report_entrada255_ext_competitors.py
==============================================
Entrada 255 (Ciclo 31): os concorrentes da Tabela III (o irace como configurador, Entrada 244, e os dois desenhos publicados de AOS, Entrada 242) nas FAMILIAS PUBLICADAS de R||Cmax (Entrada 254; experiment/run_cycle31_irace_ext_entrada255.py e run_cycle31_pubaos_ext_entrada255.py). Mesmas celulas da Entrada 254: (familia, espaco) = (unc, cj, cm) x (S5, S15); unidade: a media das sementes por instancia (500 pares por celula); o aleatorio, a validada e o Q-Learning padrao vem dos arquivos da Entrada 254.
Metodos: cfg (a configuracao estatica que o irace acha: d/n continuo; estrategia tambem no S15), q (o Q-Learning com os quatro hiperparametros ajustados pelo irace), iq e fr (os desenhos publicados). O irace recebe como orcamento o custo de ajuste da acao validada (200 e 600 experimentos por classe).
Familias de testes (Holm sobre as 6 celulas, por metodo): A (o metodo contra o aleatorio), B (a validada contra o metodo), B' (o metodo contra a validada), E (q contra o Q-Learning padrao). Margem +-0,05; 'alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada255_prereg_hashes.txt; nenhum resultado de teste visto)
--------------------------------------------------------------------------------------------------------------------------------
(Y1) A configuracao estatica do irace (cfg) nao bate a acao validada alem da margem (B') em nenhuma das 6 celulas. (70%)
(Y2) O Q-Learning ajustado pelo irace (q) ganha do aleatorio alem da margem em no maximo 2 das 6 celulas. (60%)
(Y3) A validada bate o q (B, confirmada) em pelo menos 5 das 6 celulas. (80%)
(Y4) Nenhum dos dois desenhos publicados (iq, fr) bate a validada alem da margem (B') em nenhuma celula (0 de 12). (85%)
(Y5) O q nao bate o Q-Learning padrao alem da margem (E) nas duas celulas de maquinas correlacionadas (cm-S5, cm-S15). (50%)
(Y6) O desenho tipo FRRMAB (fr) ganha do aleatorio alem da margem em pelo menos uma das duas celulas de maquinas correlacionadas. (40%)
Uso: python -m analysis.report_entrada255_ext_competitors
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
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
FAMS, SPACES = ("unc", "cj", "cm"), ("S5", "S15")
METH = ["cfg", "q", "iq", "fr"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def beats(F):
    return (F.confirmada) & (~F["equivalente_0.05"])


def load(sp):
    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_ext_test_{sp}_s*of*_entrada254.csv")))], ignore_index=True)
    m = m[m.method.isin(["rand", "val", "ql"])]
    base = m.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    ir = pd.read_csv(R / "irace_test_ext_entrada255.csv")
    ir = ir[ir.space == sp]
    k = sp[1:]
    ira = ir.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").rename(columns={f"cfg{k}_b1": "cfg", f"q{k}_b1": "q"})[["cfg", "q"]]
    pa = pd.read_csv(R / f"pubaos_ext_test_{sp}_entrada255.csv").groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")[["iq", "fr"]]
    w = base.join(ira, how="inner").join(pa, how="inner").reset_index()
    w["fam"] = w.cls.str.split("|").str[0]
    return w


def main():
    T = {k: [] for k in "ABCE"}
    cells = []
    for sp in SPACES:
        w = load(sp)
        for fam in FAMS:
            x = w[w.fam == fam]
            cells.append(dict(fam=fam, space=sp, n=len(x), g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), **{f"g_{k}": float(-rel(x[k], x.rand).mean()) for k in METH}))
            for k in METH:
                a = test(x[k], x.rand); a.update(id=f"A-{fam}-{sp}-{k}"); T["A"].append(a)
                b = test(x.val, x[k]); b.update(id=f"B-{fam}-{sp}-{k}"); T["B"].append(b)
                c = test(x[k], x.val); c.update(id=f"B'-{fam}-{sp}-{k}"); T["C"].append(c)
            e = test(x.q, x.ql); e.update(id=f"E-{fam}-{sp}-q"); T["E"].append(e)
    F = {}
    for k_, rows in T.items():
        if k_ == "E":
            F[k_] = family(rows)
        else:
            F[k_] = pd.concat([family([r for r in rows if r["id"].endswith(f"-{k}")]) for k in METH], ignore_index=True)
    C = pd.DataFrame(cells)
    L = ["=" * 110, "CONCORRENTES NAS FAMILIAS PUBLICADAS DE R||Cmax (Entrada 255)", "=" * 110, "--- ganhos sobre o aleatorio (pontos) ---", C.to_string(index=False)]
    for k_, lab in (("A", "A: metodo contra o aleatorio"), ("B", "B: validada contra o metodo"), ("C", "B': metodo contra a validada"), ("E", "E: q contra o Q-Learning padrao (Holm sobre 6)")):
        L += ["", f"--- Familia {lab} ---", F[k_][COLS].to_string(index=False)]
    y1 = int(beats(F["C"][F["C"].id.str.endswith("-cfg")]).sum())
    y2 = int(beats(F["A"][F["A"].id.str.endswith("-q")]).sum())
    y3 = int(F["B"][F["B"].id.str.endswith("-q")].confirmada.sum())
    y4 = int(beats(F["C"][F["C"].id.str.endswith(("-iq", "-fr"))]).sum())
    Ecm = F["E"][F["E"].id.str.contains("-cm-")]
    y5 = bool(not beats(Ecm).any())
    Afr = F["A"][F["A"].id.str.endswith("-fr") & F["A"].id.str.contains("-cm-")]
    y6 = int(beats(Afr).sum())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["", "--- previsoes ---",
          f"(Y1) cfg bate a validada alem da margem: {y1} de 6 (previsto 0): {yn(y1 == 0)}",
          f"(Y2) q ganha do aleatorio alem da margem: {y2} de 6 (previsto <= 2): {yn(y2 <= 2)}",
          f"(Y3) a validada bate o q: {y3} de 6 (previsto >= 5): {yn(y3 >= 5)}",
          f"(Y4) iq ou fr batem a validada alem da margem: {y4} de 12 (previsto 0): {yn(y4 == 0)}",
          f"(Y5) q nao bate o Q padrao alem da margem nas celulas cm: {yn(y5)}",
          f"(Y6) fr ganha do aleatorio alem da margem nas celulas cm: {y6} de 2 (previsto >= 1): {yn(y6 >= 1)}"]
    pred = [y1 == 0, y2 <= 2, y3 >= 5, y4 == 0, y5, y6 >= 1]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada255_ext_competitors.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([F["A"], F["B"]]).to_csv(R / "entrada255_ext_competitors_tests.csv", index=False)
    C.to_csv(R / "entrada255_ext_competitors_cells.csv", index=False)


if __name__ == "__main__":
    main()
