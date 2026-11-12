"""
analysis/report_entrada258_mechanism.py
========================================
Entrada 258 (Ciclo 34, frente 3): o MECANISMO do resultado inesperado da Entrada 256 (em B30 o Q-Learning padrao perde para o aleatorio ate -0,94 pontos e avalia 0,72x as solucoes do aleatorio; Spearman
ganho da validada x ganho do Q-Learning = -0,46). Instancias: as de teste com idx = 2 (140), espacos B5, B15, B30, metodos rand, ql, ql_rev (Q-Learning com a ORDEM das acoes invertida), ucb, pg, alns; 2 sementes
(experiment/run_cycle34_mechanism_entrada258.py). Qualidade de uma acao = razao media makespan/melhor conhecido da acao FIXA nas instancias de AJUSTE da classe (results/fr_tune_*), menor e' melhor.
Teste causal do desempate: o Q-Learning padrao inicia a tabela em zero e escolhe o PRIMEIRO maximo; se a perda vem de repetir a acao 0 em estados nao visitados, inverter a ordem das acoes (a acao 0 passa a ser a maior
destruicao com busca local, em B30) deve mudar muito o desempenho; o aleatorio e' invariante a permutacao.

PRE-REGISTRO (congelado ANTES de rodar; hashes em results/entrada258_prereg_hashes.txt; nenhum resultado visto)
----------------------------------------------------------------------------------------------------------------
(M1) Em B30 o Q-Learning padrao (ql) poe pelo menos 20% das escolhas na acao 0 (o aleatorio: 1/30 = 3,3%). (65%)
(M2) Em B30 a fracao de escolhas do ql nas acoes COM busca local (15-29) e' menor que 50% (a do aleatorio). (70%)
(M3) Em B30 a qualidade media ponderada pelo uso do ql e' PIOR (razao maior) que a do aleatorio. (75%)
(M4) Em B30 o ucb, o pg e a roleta ALNS poem cada um no maximo 10% das escolhas na acao 0. (70%)
(M5) Entre as 21 celulas (tipo x espaco), o Spearman entre a fracao do ql na acao 0 e o ganho do ql sobre o aleatorio (Entrada 256) e' <= -0,4. (55%)
(M6) Teste causal: em B30 o ql_rev (ordem invertida) tem makespan menor que o ql por mais que a margem (0,05 pontos, teste pareado por instancia, Holm sobre os tres espacos), e seu ganho sobre o aleatorio e' positivo. (65%)
(M7) Em B5 e B15 a inversao da ordem muda o desempenho do ql em menos de 0,15 pontos em modulo (as acoes extremas diferem menos). (50%)
Uso: python -m analysis.report_entrada258_mechanism
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
SPACES = ("B5", "B15", "B30")
pd.set_option("display.width", 220)


def quality(space):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"fr_tune_{space}_s*of*_entrada256.csv")))], ignore_index=True)
    return d.groupby(["cls", "arm"]).ratio.mean().unstack()


def main():
    out, tests, cellshare = [], [], []
    res = {}
    for sp in SPACES:
        D = pd.read_csv(R / f"mech_{sp}_entrada258.csv")
        ucols = [c for c in D.columns if c.startswith("used_") and D[c].notna().all()]
        A = len(ucols)
        U = D[ucols].to_numpy(float)
        D["share0"] = U[:, 0] / U.sum(axis=1)
        D["share_ls"] = U[:, 15:].sum(axis=1) / U.sum(axis=1) if A == 30 else np.nan
        Q = quality(sp)
        D["cls"] = D["type"] + "|" + D["n"].astype(str)
        D["qual"] = [float((U[i] / U[i].sum() * Q.loc[D.cls.iloc[i]].to_numpy()[:A]).sum()) for i in range(len(D))]
        D["gap"] = 100 * (D.makespan - D.bks) / D.bks
        tab = D.groupby("method")[["share0", "share_ls", "qual", "gap", "evals"]].mean()
        res[sp] = tab
        out += [f"--- {sp}: uso e qualidade por metodo (140 instancias x 2 sementes) ---", tab.round(4).to_string(), ""]
        W = D.groupby(["type", "m", "n", "method"]).makespan.mean().unstack()
        for k in ("ql", "ql_rev"):
            t = test(W[k], W["rand"]); t.update(id=f"V-{sp}-{k}-rand"); tests.append(t)
        t = test(W["ql_rev"], W["ql"]); t.update(id=f"R-{sp}"); tests.append(t)
        for ty, g in D[D.method == "ql"].groupby("type"):
            cellshare.append(dict(space=sp, type=ty, share0=float(g.share0.mean())))
    T = pd.DataFrame(tests)
    # Holm sobre os tres espacos nos testes ql_rev contra ql
    Rv = family(T[T.id.str.startswith("R-")].to_dict("records"))
    Vr = T[T.id.str.startswith("V-")]
    cells = pd.read_csv(R / "entrada256_fanjul_cells.csv")
    CS = pd.DataFrame(cellshare).merge(cells[["space", "type", "g_ql"]], on=["space", "type"])
    rho = float(sstats.spearmanr(CS.share0, CS.g_ql).statistic)
    b30 = res["B30"]
    m1 = b30.loc["ql", "share0"] >= 0.20
    m2 = b30.loc["ql", "share_ls"] < 0.50
    m3 = b30.loc["ql", "qual"] > b30.loc["rand", "qual"]
    m4 = all(b30.loc[k, "share0"] <= 0.10 for k in ("ucb", "pg", "alns"))
    m5 = rho <= -0.4
    r30 = Rv[Rv.id == "R-B30"].iloc[0]
    v30 = Vr[Vr.id == "V-B30-ql_rev-rand"].iloc[0]
    m6 = (r30.mean_d_pct < -0.05) and bool(r30.confirmada) and (v30.mean_d_pct < 0)
    d5 = abs(float(Rv[Rv.id == "R-B5"].mean_d_pct.iloc[0])); d15 = abs(float(Rv[Rv.id == "R-B15"].mean_d_pct.iloc[0]))
    m7 = (d5 < 0.15) and (d15 < 0.15)
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "MECANISMO DA PERDA DO Q-LEARNING EM B30 (Entrada 258)", "=" * 110] + out
    L += ["--- ql_rev contra ql (R-) e contra o aleatorio (V-); mean_d_pct negativo = ganho do primeiro; Holm nos tres espacos para R- ---",
          Rv[["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "leitura"]].to_string(index=False),
          Vr[["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi"]].to_string(index=False), "",
          "--- fracao do ql na acao 0 por celula e ganho do ql (Entrada 256) ---", CS.round(4).to_string(index=False), "",
          "--- previsoes ---",
          f"(M1) fracao do ql na acao 0 em B30: {b30.loc['ql', 'share0']:.3f} (previsto >= 0,20): {yn(m1)}",
          f"(M2) fracao do ql em acoes com busca local em B30: {b30.loc['ql', 'share_ls']:.3f} (previsto < 0,50): {yn(m2)}",
          f"(M3) qualidade ponderada pelo uso em B30: ql {b30.loc['ql', 'qual']:.4f} contra aleatorio {b30.loc['rand', 'qual']:.4f} (previsto ql pior): {yn(m3)}",
          f"(M4) fracao na acao 0 em B30 de ucb/pg/alns: {b30.loc['ucb', 'share0']:.3f}/{b30.loc['pg', 'share0']:.3f}/{b30.loc['alns', 'share0']:.3f} (previsto <= 0,10): {yn(m4)}",
          f"(M5) Spearman fracao na acao 0 x ganho do ql nas 21 celulas: {rho:+.3f} (previsto <= -0,4): {yn(m5)}",
          f"(M6) ql_rev contra ql em B30: {r30.mean_d_pct:+.4f}%, confirmada={bool(r30.confirmada)}; ql_rev contra aleatorio: {v30.mean_d_pct:+.4f}%: {yn(m6)}",
          f"(M7) |ql_rev - ql| em B5 e B15: {d5:.3f} e {d15:.3f} (previsto < 0,15): {yn(m7)}"]
    pred = [m1, m2, m3, m4, m5, m6, m7]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada258_mechanism.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([Rv, Vr]).to_csv(R / "entrada258_mechanism_tests.csv", index=False)


if __name__ == "__main__":
    main()
