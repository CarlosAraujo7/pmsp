"""
analysis/report_entrada280b_headline.py
=======================================
Entrada 280b (POS HOC, declarada): o ql_rt (Q-Learning com desempate aleatorio) BATE a acao validada alem da margem em alguma das 85 celulas? A Entrada 280 (previsoes S1-S8) testou so' 'a validada bate o ql_rt' (teste B); o resumo do manuscrito afirma
que a validada nunca foi batida alem da margem por um aprendiz online, o que exige o teste na direcao inversa (o ql_rt contra a validada, mesmas celulas, mesmos testes, Holm sobre as celulas de cada colecao). Este script NAO altera o relatorio da Entrada 280 (cujo hash esta
congelado): reutiliza as suas funcoes e intercepta, a cada celula, o quadro de dados para o teste inverso. Nenhuma previsao registrada; os resultados saem em results/entrada280b_headline.txt.
Uso: python -m analysis.report_entrada280b_headline
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analysis.report_entrada280_tiesall as M  # noqa: E402
from analysis.report_entrada180_agents import family  # noqa: E402

R = ROOT / "results"
SIZES = [("padrao", 21), ("familias", 6), ("sintetico", 24), ("publicado", 18), ("tsp", 16)]


def main():
    store = []
    orig = M.cell_tests

    def patched(x, tag, TA, TB):
        orig(x, tag, TA, TB)
        t = M.test(x.rt, x.val)
        t.update(id=f"Bp-{tag}", g_rt=float(-M.rel(x.rt, x.rand).mean()), g_val=float(-M.rel(x.val, x.rand).mean()))
        store.append(t)

    M.cell_tests = patched
    import io
    import contextlib
    real_write = Path.write_text
    real_csv = pd.DataFrame.to_csv
    Path.write_text = lambda self, *a, **k: None          # nao regrava os arquivos da Entrada 280
    pd.DataFrame.to_csv = lambda self, *a, **k: None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            M.main()
    finally:
        Path.write_text = real_write
        pd.DataFrame.to_csv = real_csv
    L = ["=" * 110, "O ql_rt BATE A VALIDADA ALEM DA MARGEM? (Entrada 280b, pos hoc)", "=" * 110,
         "Teste inverso: ql_rt contra a validada por celula (confirmada = o ql_rt e' melhor, Holm sobre as celulas da colecao); 'alem da margem' = confirmada e nao equivalente a zero em +-0,05.", ""]
    i, tot_beat, tot_par = 0, 0, 0
    allrows = []
    for coll, n in SIZES:
        part = store[i:i + n]
        i += n
        F_ = family(part)
        F_["coll"] = coll
        beat = F_.confirmada & ~F_["equivalente_0.05"]
        worse = (F_.reject & (F_.mean_d_pct > 0))
        tot_beat += int(beat.sum())
        L.append(f"{coll}: o ql_rt bate a validada alem da margem em {int(beat.sum())} de {n} celulas; a validada bate o ql_rt (Holm) em {int(worse.sum())}; sem diferenca detectavel em {n - int(beat.sum()) - int(worse.sum())}")
        if beat.any():
            L.append(F_[beat][["id", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "g_rt", "g_val"]].round(4).to_string(index=False))
        allrows.append(F_)
    L += ["", f"total: o ql_rt bate a validada alem da margem em {tot_beat} das 85 celulas"]
    text = "\n".join(L)
    print(text)
    (R / "entrada280b_headline.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat(allrows).to_csv(R / "entrada280b_headline_tests.csv", index=False)


if __name__ == "__main__":
    main()
