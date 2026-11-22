"""
analysis/predict_entrada289_twin.py (Entrada 289, Ciclo 55)
=============================================================
PREVISOES NUMERICAS DO GEMEO DIGITAL para o hospedeiro de evolucao diferencial, calculadas SO' a partir dos registros do controle aleatorio (results/logs289/f<fun>_rand.npz, estagio base) pelo codigo congelado de
analysis/learnability_dense.py, e gravadas em results/entrada289_twin_predictions.csv ANTES de qualquer execucao dos aprendizes (estagio learn). Para cada funcao: sucessos por acao por execucao (informacao), G_U e G_max do bandido
oblivio e a captura de recompensa prevista (soma das taxas de sucesso; contra o aleatorio e a melhor acao fixa do bandido) para UCB1, PM, AP, ETC e o Q-Learning sem estados.
Uso: python -m analysis.predict_entrada289_twin
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis import learnability_dense as LD  # noqa: E402

R = ROOT / "results"
T, K, NP = 1000, 5, 100


def main():
    rows = []
    for fun in range(1, 25):
        z = np.load(R / "logs289" / f"f{fun:02d}_rand.npz")
        A, Rw = z["A"], z["Rw"].astype(np.float64)
        pr = LD.predict(A, Rw, K, T, n_sim=3000, seed=289 + fun)
        row = dict(fun=fun, runs=len(A), succ_per_arm_run=float(Rw.sum(axis=1).mean() * NP / K), G_uniform=pr["G_uniform"], G_max=pr["G_max"])
        for nm in ("ucb", "pm", "ap", "etc", "ql1"):
            row[f"kappa_{nm}"] = pr[f"kappa_{nm}"]
            row[f"gain_{nm}"] = pr[f"gain_{nm}"]
        rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(R / "entrada289_twin_predictions.csv", index=False)
    print(D.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
