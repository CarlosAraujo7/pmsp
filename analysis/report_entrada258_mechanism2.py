"""
analysis/report_entrada258_mechanism2.py
=========================================
Entrada 258, ETAPA 2 do mecanismo. Resultado da etapa 1 (results/entrada258_mechanism.txt; 3 de 7 previsoes): o Q-Learning padrao poe 55% (B30) a 73% (B5) das escolhas na acao 0 (o aleatorio e os demais aprendizes, 3-20%),
mas inverter a ORDEM das acoes nao o melhora (a inversao faz o Q travar na ultima acao original, outra acao extrema e ruim): o travamento no primeiro indice por desempate e' a explicacao compativel, mas o teste causal da etapa 1
estava mal desenhado (as duas pontas da lista sao ruins). A etapa 2 faz o teste correto: mover a acao VALIDADA da classe para o indice 0 (ql_v0). Se o Q-Learning padrao apenas repete o primeiro indice, ql_v0 deve chegar perto da validada;
se aprende algo, o resultado de ql_v0 deve ficar perto do de ql. Mesmas instancias (teste, idx = 2: 140), 2 sementes (experiment/run_cycle34_mechanism_entrada258.py --stage 2).

PRE-REGISTRO DA ETAPA 2 (congelado ANTES de rodar; hashes em results/entrada258_prereg_hashes2.txt; escrito depois de ver a etapa 1)
------------------------------------------------------------------------------------------------------------------------------------
(S1) Em B30, ql_v0 ganha do aleatorio alem da margem (teste pareado por instancia; Holm nos tres espacos). (70%)
(S2) Em B30, ql_v0 recupera pelo menos 50% do ganho da validada (razao das medias dos ganhos sobre o aleatorio, na mesma amostra). (55%)
(S3) Em B30, ql_v0 usa a acao validada da classe em pelo menos 40% das escolhas. (65%)
(S4) Em B5 e em B15, ql_v0 ganha do aleatorio alem da margem. (60%)
(S5) Em nenhum dos tres espacos ql_v0 bate a validada alem da margem. (85%)
Uso: python -m analysis.report_entrada258_mechanism2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
SPACES = ("B5", "B15", "B30")
pd.set_option("display.width", 220)


def main():
    tests, gains, share = [], {}, {}
    out = []
    for sp in SPACES:
        D = pd.read_csv(R / f"mech2_{sp}_entrada258.csv")
        ucols = [c for c in D.columns if c.startswith("used_") and D[c].notna().all()]
        U = D[ucols].to_numpy(float)
        D["share_val"] = U[np.arange(len(D)), D.val_arm.to_numpy()] / U.sum(axis=1)
        D["gap"] = 100 * (D.makespan - D.bks) / D.bks
        out += [f"--- {sp} ---", D.groupby("method")[["share_val", "gap", "evals"]].mean().round(4).to_string(), ""]
        W = D.groupby(["type", "m", "n", "method"]).makespan.mean().unstack()
        g = {k: float(-rel(W[k], W["rand"]).mean()) for k in ("ql", "ql_v0", "val")}
        gains[sp] = g
        share[sp] = float(D[D.method == "ql_v0"].share_val.mean())
        for k in ("ql", "ql_v0", "val"):
            t = test(W[k], W["rand"]); t.update(id=f"A-{sp}-{k}"); tests.append(t)
        t = test(W["ql_v0"], W["val"]); t.update(id=f"B'-{sp}-ql_v0"); tests.append(t)
    T = pd.DataFrame(tests)
    A = T[T.id.str.startswith("A-")]
    A = pd.concat([family(A[A.id.str.endswith(f"-{k}")].to_dict("records")) for k in ("ql", "ql_v0", "val")], ignore_index=True)
    Bp = family(T[T.id.str.startswith("B'-")].to_dict("records"))
    beyond = lambda F_, i: bool(F_[F_.id == i].confirmada.iloc[0]) and not bool(F_[F_.id == i]["equivalente_0.05"].iloc[0])
    s1 = beyond(A, "A-B30-ql_v0")
    s2 = gains["B30"]["ql_v0"] / gains["B30"]["val"] >= 0.5
    s3 = share["B30"] >= 0.40
    s4 = beyond(A, "A-B5-ql_v0") and beyond(A, "A-B15-ql_v0")
    s5 = not any(beyond(Bp, f"B'-{sp}-ql_v0") for sp in SPACES)
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "MECANISMO, ETAPA 2: A ACAO VALIDADA NO INDICE 0 (Entrada 258)", "=" * 110] + out
    L += ["--- ganhos sobre o aleatorio (pontos) ---", pd.DataFrame(gains).T.round(4).to_string(), "",
          "--- testes: A (metodo contra o aleatorio), B' (ql_v0 contra a validada); mean_d_pct negativo = ganho do primeiro ---",
          A[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].to_string(index=False),
          Bp[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].to_string(index=False), "",
          "--- previsoes ---",
          f"(S1) ql_v0 ganha do aleatorio alem da margem em B30: ganho {gains['B30']['ql_v0']:.3f}: {yn(s1)}",
          f"(S2) fracao do ganho da validada recuperada em B30: {gains['B30']['ql_v0'] / gains['B30']['val']:.3f} (previsto >= 0,5): {yn(s2)}",
          f"(S3) fracao das escolhas do ql_v0 na acao validada em B30: {share['B30']:.3f} (previsto >= 0,40): {yn(s3)}",
          f"(S4) ql_v0 ganha alem da margem em B5 ({gains['B5']['ql_v0']:.3f}) e B15 ({gains['B15']['ql_v0']:.3f}): {yn(s4)}",
          f"(S5) ql_v0 nao bate a validada alem da margem em nenhum espaco: {yn(s5)}"]
    pred = [s1, s2, s3, s4, s5]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada258_mechanism2.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
