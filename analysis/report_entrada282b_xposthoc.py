"""
analysis/report_entrada282b_xposthoc.py
=======================================
Entrada 282b (POS HOC, declarada): nas 22 instancias X publicadas a acao global (G0, escolhida so' no ajuste sintetico) foi a melhor, a por (n, clientes por rota) (Gnr) nao ganhou do aleatorio; os testes registrados da Entrada 282 compararam os aprendizes com a Gnr.
Aqui, sem previsao registrada: cada aprendiz contra a G0, e a lacuna media aos melhores conhecidos por metodo com IC90 bootstrap. Mesmos testes (Wilcoxon pareado, Holm sobre os 5 aprendizes, equivalencia em +-0,05).
Uso: python -m analysis.report_entrada282b_xposthoc
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
from analysis.report_entrada180_agents import family, test  # noqa: E402

R = ROOT / "results"
LEARN = ["rw", "srw", "aucb", "mab_ucb1", "mab_eps"]


def main():
    ch = json.loads((R / "cvrp_choice_entrada282.json").read_text())
    X = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "cvrp_x_s*of*_entrada282.csv")))], ignore_index=True)
    W = X.groupby(["cls", "rep", "method"]).cost.mean().unstack("method").reset_index()
    bks = X.drop_duplicates("cls").set_index("cls").bks
    W["bks"] = W.cls.map(bks)
    W["g0"] = W[f"arm{ch['g0']}"]
    rng = np.random.default_rng(0)
    L = ["=" * 100, "INSTANCIAS X: CADA APRENDIZ CONTRA A ACAO GLOBAL (Entrada 282b, pos hoc)", "=" * 100, ""]
    rows = []
    for m in ["random", "g0"] + LEARN:
        gap = 100 * (W[m] / W.bks - 1)
        idx = rng.integers(0, len(gap), size=(2000, len(gap)))
        lo, hi = np.percentile(gap.to_numpy()[idx].mean(axis=1), [5, 95])
        rows.append(dict(method=m, gap_mean=float(gap.mean()), ci90_lo=float(lo), ci90_hi=float(hi)))
    G = pd.DataFrame(rows)
    L += ["lacuna media (%) aos melhores conhecidos, 22 instancias, IC90 bootstrap:", G.round(3).to_string(index=False), ""]
    t = []
    for m in LEARN:
        a = test(W[m], W.g0); a.update(id=f"XG-{m}"); t.append(a)
    F_ = family(t)
    L += ["aprendiz contra a G0 (confirmada = o aprendiz e' melhor; Holm sobre 5):", F_[["id", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada282b_xposthoc.txt").write_text(text + "\n", encoding="utf-8")
    G.to_csv(R / "entrada282b_xposthoc_gaps.csv", index=False)
    F_.to_csv(R / "entrada282b_xposthoc_tests.csv", index=False)


if __name__ == "__main__":
    main()
