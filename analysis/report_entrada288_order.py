"""
analysis/report_entrada288_order.py (Entrada 288, Ciclo 54, parte B)
======================================================================
O TESTE DE INVARIANCIA A ORDEM e a CONTABILIDADE DA LINHA ZERADA. Resposta a E da revisao cega #2 do CEC (Revisores 1 a 4: lema do aprisionamento, teste de permutacao reutilizavel).
Dados: results/order288_*.csv (experiment/run_cycle54b_order_entrada288.py): 12 classes sinteticas de P||Cmax (S5) e 7 tipos do benchmark padrao de R||Cmax (B5); 20 instancias, 12 ordens da lista de acoes (identidade, invertida e 10
aleatorias), 3 sementes, Q-Learning com desempate pelo PRIMEIRO maximo (tie 0, o padrao) e com desempate ALEATORIO (tie 1). Resultado y[i, r] = makespan relativo ao limite inferior (%), media das sementes, da instancia i sob a ordem r.
Teste (pmsp/order_test.py): estatistica n sum_r (media da ordem r depois de centrar cada instancia)^2; p por embaralhamento dos rotulos de ordem DENTRO de cada instancia (2.000); tamanho do efeito = DP entre as ordens do resultado medio (pontos).
Contabilidade: com desempate pelo primeiro maximo, P(escolher o indice 0 no passo t) >= eps_t/K + (1 - eps_t) z_t, z_t = 1 se a linha Q do estado estava zerada; o executor registra a cota media por execucao e a fracao observada.

PRE-REGISTRO (congelado ANTES de rodar o executor; hashes em results/entrada288_prereg_hashes.txt). Probabilidades do autor. Ressalva: a participacao de 55-73% da primeira acao no R||Cmax e a dependencia da ordem do Q-Learning padrao
(lista invertida, validada primeiro) ja' eram conhecidas (Entrada 258); o teste de permutacao e a contabilidade da linha zerada sao novos.
(Q1) Q-Learning padrao (tie 0): o efeito de ordem tem p < 0,05 em pelo menos 80% dos 7 tipos de R||Cmax (B5). (60%)
(Q2) Q-Learning padrao (tie 0): p < 0,05 em pelo menos 50% das 12 classes sinteticas de P||Cmax (S5). (55%)
(Q3) Q-Learning com desempate aleatorio (tie 1): p < 0,05 em no maximo 2 das 19 celulas. (70%)
(Q4) A cota da linha zerada explica pelo menos 75% da participacao do indice 0 no Q-Learning padrao (media das cotas / media das participacoes, sobre todas as execucoes). (65%)
(Q5) Com desempate aleatorio a participacao do indice 0 (todas as execucoes) fica a menos de 0,03 de 1/K = 0,2. (75%)
(Q6) A mediana, sobre as celulas, de DP da ordem (tie 0) / DP da ordem (tie 1) e' de pelo menos 5. (65%)
(Q7) No R||Cmax (B5) a participacao do indice 0 no Q-Learning padrao e' de pelo menos 0,50 (media sobre as celulas). (70%)
Uso: python -m analysis.report_entrada288_order
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pmsp.order_test import order_invariance_test  # noqa: E402

R = ROOT / "results"
pd.set_option("display.width", 230)


def holm(ps, alpha=0.05):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    rej = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def main():
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "order288_*.csv")))], ignore_index=True)
    D["y"] = 100.0 * (D.makespan - D.lb) / D.lb
    rows = []
    for (coll, cell, tie), g in D.groupby(["coll", "cell", "tie"]):
        y = g.groupby(["inst", "perm_id"]).y.mean().unstack("perm_id").to_numpy()
        t = order_invariance_test(y, n_perm=2000, seed=288)
        rows.append(dict(coll=coll, cell=cell, tie=int(tie), p=t["p"], sd_order=t["sd_between"], share_first=float(g.share_first.mean()), share_zero=float(g.share_zero.mean()),
                         bound=float(g.bound.mean()), n_inst=y.shape[0], n_ord=y.shape[1], K=5))
    C = pd.DataFrame(rows)
    for tie in (0, 1):
        m = C.tie == tie
        C.loc[m, "holm"] = holm(C.loc[m, "p"].to_numpy())
    L = ["=" * 118, "TESTE DE INVARIANCIA A ORDEM E CONTABILIDADE DA LINHA ZERADA (Entrada 288)", "=" * 118,
         f"execucoes: {len(D)}; celulas: {C.cell.nunique()} ({(C[C.tie == 0].coll == 'sintetico').sum()} sinteticas, {(C[C.tie == 0].coll == 'padrao').sum()} do benchmark padrao); instancias {int(C.n_inst.min())}, ordens {int(C.n_ord.min())}", ""]
    for tie, nm in ((0, "Q-Learning PADRAO (primeiro maximo)"), (1, "Q-Learning com DESEMPATE ALEATORIO")):
        c = C[C.tie == tie]
        L += [f"--- {nm} ---", c[["coll", "cell", "p", "holm", "sd_order", "share_first", "share_zero", "bound"]].round(4).to_string(index=False), ""]
    pooled = D.groupby("tie").agg(share_first=("share_first", "mean"), share_zero=("share_zero", "mean"), bound=("bound", "mean"))
    L += ["--- agregados (todas as execucoes) ---", pooled.round(4).to_string(), ""]
    c0, c1 = C[C.tie == 0].set_index("cell"), C[C.tie == 1].set_index("cell")
    q1 = float((c0[c0.coll == "padrao"].p < 0.05).mean())
    q2 = float((c0[c0.coll == "sintetico"].p < 0.05).mean())
    q3 = int((c1.p < 0.05).sum())
    q4 = float(pooled.loc[0, "bound"] / pooled.loc[0, "share_first"])
    q5 = float(pooled.loc[1, "share_first"])
    ratio = (c0.sd_order / c1.sd_order.replace(0, np.nan)).dropna()
    q6 = float(ratio.median())
    q7 = float(D[(D.tie == 0) & (D.coll == "padrao")].groupby("cell").share_first.mean().mean())
    info = [f"celulas com Holm (tie 0): {int(c0.holm.sum())} de {len(c0)}; (tie 1): {int(c1.holm.sum())} de {len(c1)}",
            f"DP da ordem mediano (pontos): tie 0 {c0.sd_order.median():.4f}; tie 1 {c1.sd_order.median():.4f}",
            f"participacao do indice 0 (media das celulas): tie 0 sintetico {c0[c0.coll == 'sintetico'].share_first.mean():.3f}, padrao {c0[c0.coll == 'padrao'].share_first.mean():.3f}; tie 1 sintetico {c1[c1.coll == 'sintetico'].share_first.mean():.3f}, padrao {c1[c1.coll == 'padrao'].share_first.mean():.3f}",
            f"fracao das decisoes em linha zerada (tie 0): {pooled.loc[0, 'share_zero']:.3f}; (tie 1): {pooled.loc[1, 'share_zero']:.3f}"]
    L += ["--- informacoes adicionais (nao previstas) ---"] + info + [""]
    yn = lambda b: "SIM" if b else "NAO"
    preds = [("Q1", f"padrao, tie 0: fracao dos 7 tipos com p < 0,05: {q1:.3f} (previsto >= 0,80)", q1 >= 0.8),
             ("Q2", f"sintetico, tie 0: fracao das 12 classes com p < 0,05: {q2:.3f} (previsto >= 0,50)", q2 >= 0.5),
             ("Q3", f"tie 1: celulas com p < 0,05: {q3} de {len(c1)} (previsto <= 2)", q3 <= 2),
             ("Q4", f"cota da linha zerada / participacao do indice 0 (tie 0): {q4:.3f} (previsto >= 0,75)", q4 >= 0.75),
             ("Q5", f"participacao do indice 0 com desempate aleatorio: {q5:.4f} (previsto em [0,17; 0,23])", abs(q5 - 0.2) < 0.03),
             ("Q6", f"mediana de DP(ordem, tie 0) / DP(ordem, tie 1): {q6:.2f} (previsto >= 5)", q6 >= 5),
             ("Q7", f"R||Cmax B5, tie 0: participacao media do indice 0: {q7:.3f} (previsto >= 0,50)", q7 >= 0.5)]
    L += ["--- previsoes ---"] + [f"({k}) {t}: {yn(ok)}" for k, t, ok in preds] + [f"previsoes acertadas: {sum(ok for _, _, ok in preds)} de {len(preds)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada288_order.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada288_cells.csv", index=False)


if __name__ == "__main__":
    main()
