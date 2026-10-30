"""
analysis/report_entrada248_ur_eqcpu.py
=======================================
Entrada 248 (Ciclo 26): CPU IGUAL no R||Cmax. Cada aprendiz (ql, ucb, pg, alns, qo) roda T = 1000 (results/ur_test_*_entrada234.csv); o controle aleatorio roda T = max(50, round(1000 K_aprendiz)) e a acao validada T = max(50, round(1000 K_aprendiz / K_val)) (results/ur_eqcpu_*_entrada248.csv; K calibrados em results/ur_eqcpu_k_entrada248.json). 12 classes x 100 instancias x 2 espacos; unidade de analise: a media das 5 sementes por instancia (1.200 pares por linha).
Familia A: aprendiz contra o controle aleatorio a CPU igual (ganho de CPU igual; 10 testes, Holm sobre 10). Familia B: acao validada contra o aprendiz a CPU igual (10 testes, Holm sobre 10); B' e' o sentido contrario (o aprendiz melhor que a validada; Holm sobre 10). Efeito de decisao = ganho a iteracoes iguais (aprendiz contra aleatorio, T = 1000); efeito de custo = ganho de CPU igual menos o de decisao. Margem de equivalencia +-0,05; 'ganha alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de olhar qualquer resultado de teste e os K calibrados; hashes em results/entrada248_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------------------
(E1) O Q-Learning (ql) nao ganha alem da margem sobre o aleatorio a CPU igual em nenhum dos dois espacos. (90%)
(E2) A CPU igual, no maximo 4 dos 10 pares (aprendiz, espaco) ganham alem da margem sobre o aleatorio. (75%)
(E3) Nenhum aprendiz bate a acao validada a CPU igual alem da margem (B': 0 de 10). (92%)
(E4) A acao validada bate o aprendiz a CPU igual (B, confirmada) em pelo menos 8 dos 10 pares. (85%)
(E5) Em cada espaco, o melhor ganho de aprendiz a CPU igual sobre o aleatorio e' no maximo 25% do ganho da acao validada sobre o aleatorio a iteracoes iguais. (75%)
(E6) O efeito de custo do Q-Learning e' positivo no S15 (ele escolhe acoes mais baratas e o aleatorio perde iteracoes ao igualar a CPU). (60%)
Uso: python -m analysis.report_entrada248_ur_eqcpu
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
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
SPACES = ("S5", "S15")
LEARNERS = ["ql", "ucb", "pg", "alns", "qo"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load(sp):
    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_test_{sp}_s*of*_entrada234.csv")))], ignore_index=True)
    m = m[m.method.isin(["rand", "val"] + LEARNERS)]
    e = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_eqcpu_{sp}_s*of*_entrada248.csv")))], ignore_index=True)
    a = m.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    b = e.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    return a.join(b, how="inner").reset_index()


def beats(F):
    return (F.confirmada) & (~F["equivalente_0.05"])


def main():
    K = json.loads((R / "ur_eqcpu_k_entrada248.json").read_text())
    W = {sp: load(sp) for sp in SPACES}
    TA, TB, TR, split = [], [], [], []
    for sp in SPACES:
        w = W[sp]
        gval = float(-rel(w.val, w.rand).mean())
        for k in LEARNERS:
            a = test(w[k], w[f"rand_eq_{k}"]); a.update(id=f"A-{sp}-{k}"); TA.append(a)
            b = test(w[f"val_eq_{k}"], w[k]); b.update(id=f"B-{sp}-{k}"); TB.append(b)
            r = test(w[k], w[f"val_eq_{k}"]); r.update(id=f"B'-{sp}-{k}"); TR.append(r)
            gcpu, gdec = float(-rel(w[k], w[f"rand_eq_{k}"]).mean()), float(-rel(w[k], w.rand).mean())
            kk = [K[sp][c][k] for c in K[sp]]
            split.append(dict(space=sp, learner=k, n=len(w), K_med=float(np.median(kk)), K_min=float(min(kk)), K_max=float(max(kk)), g_cpu=gcpu, g_dec=gdec, g_cost=gcpu - gdec,
                              g_val_over_learner=float(-rel(w[f"val_eq_{k}"], w[k]).mean()), g_val_iter=gval))
    FA, FB, FR = family(TA), family(TB), family(TR)
    S = pd.DataFrame(split)
    kv = {sp: [K[sp][c]["val"] for c in K[sp]] for sp in SPACES}
    L = ["=" * 110, "CPU IGUAL NO R||Cmax (Entrada 248)", "=" * 110,
         "Custo K de uma execucao (mediana [min-max] sobre as 12 classes), em relacao ao controle aleatorio; acao validada: " +
         "; ".join(f"{sp} {np.median(kv[sp]):.3f} [{min(kv[sp]):.3f}-{max(kv[sp]):.3f}]" for sp in SPACES),
         "", "--- ganho de CPU igual (g_cpu), efeito de decisao (g_dec = ganho a iteracoes iguais) e de custo (g_cost); ganho da validada sobre o aprendiz a CPU igual; ganho da validada sobre o aleatorio a iteracoes iguais ---",
         S.to_string(index=False),
         "", "--- Familia A: aprendiz contra o aleatorio a CPU igual (Holm sobre 10) ---", FA[COLS].to_string(index=False),
         "", "--- Familia B: validada contra o aprendiz a CPU igual (Holm sobre 10) ---", FB[COLS].to_string(index=False),
         "", "--- Familia B': o aprendiz contra a validada a CPU igual (Holm sobre 10) ---", FR[COLS].to_string(index=False)]
    e1 = bool(not beats(FA[FA.id.str.endswith("-ql")]).any())
    e2 = int(beats(FA).sum())
    e3 = int(beats(FR).sum())
    e4 = int(FB.confirmada.sum())
    ratios = {sp: float(S[S.space == sp].g_cpu.max() / S[S.space == sp].g_val_iter.iloc[0]) for sp in SPACES}
    e5 = all(v <= 0.25 for v in ratios.values())
    e6 = bool(float(S[(S.space == "S15") & (S.learner == "ql")].g_cost.iloc[0]) > 0)
    yn = lambda b: "SIM" if b else "NAO"
    L += ["", "--- previsoes ---",
          f"(E1) ql nao ganha alem da margem a CPU igual em nenhum espaco: {yn(e1)}",
          f"(E2) pares que ganham alem da margem sobre o aleatorio a CPU igual: {e2} de 10 (previsto <= 4): {yn(e2 <= 4)}",
          f"(E3) aprendizes que batem a validada alem da margem (B'): {e3} de 10 (previsto 0): {yn(e3 == 0)}",
          f"(E4) a validada bate o aprendiz (B, confirmada): {e4} de 10 (previsto >= 8): {yn(e4 >= 8)}",
          f"(E5) melhor ganho de CPU igual / ganho da validada a iteracoes iguais: S5 {ratios['S5']:.3f}, S15 {ratios['S15']:.3f} (previsto <= 0,25): {yn(e5)}",
          f"(E6) efeito de custo do ql no S15 positivo ({float(S[(S.space == 'S15') & (S.learner == 'ql')].g_cost.iloc[0]):.3f}): {yn(e6)}"]
    pred = [e1, e2 <= 4, e3 == 0, e4 >= 8, e5, e6]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada248_ur_eqcpu.txt").write_text(text + "\n", encoding="utf-8")
    FA.to_csv(R / "entrada248_ur_eqcpu_A.csv", index=False)
    FB.to_csv(R / "entrada248_ur_eqcpu_B.csv", index=False)
    S.to_csv(R / "entrada248_ur_eqcpu_split.csv", index=False)


if __name__ == "__main__":
    main()
