"""
analysis/report_entrada287_learnability.py (Entrada 287, Ciclo 54)
====================================================================
APRENDIBILIDADE: por que o aprendizado online captura so' uma fracao pequena do ganho da acao validada? Resposta a S2 e S6 da segunda revisao cega do CEC (revisao_cega_cec_2026-11-22.md) e as propostas A e B do plano.
Entrada: os registros por decisao do controle aleatorio das 69 celulas do CEC (results/logs287/*.npz, gerados por experiment/run_cycle54_logs_entrada287.py; 100 instancias x 5 sementes por celula, T = 1000; manifesto
com os md5) e as tabelas por instancia das Entradas 280/283 (analysis/cells_instance_tables.py) para os ganhos OBSERVADOS (validada, rt = Q-Learning com desempate aleatorio, UCB1), em pontos do makespan do aleatorio.
O modelo esta em analysis/learnability_model.py (congelado antes de abrir os registros): bandido oblivio de K bracos com recompensa esparsa; indice L = T Delta_cf^2 / (K sigma^2); captura 'ganha-fica' kappa_WS (sem parametro ajustavel);
gemeo digital de UCB1 e de Q-Learning sem estados no bandido por faixas de tempo (T1); folgas de fase e de estado miopes com validacao cruzada de instancias. Conferencia de integridade: soma das recompensas = makespan inicial - final (todas as execucoes).
Conjunto de capturas: celulas com ganho observado da validada >= 0,05 (denominador afastado de zero).

PRE-REGISTRO (congelado ANTES de abrir os registros; hashes em results/entrada287_prereg_hashes.txt). Probabilidades do autor. Ressalva de transparencia: sete celulas-piloto (uma por colecao e espaco, 2 instancias, 1 semente) mostraram taxas
de recompensa positiva de 0,0 a 0,4% das decisoes, entao P1 e P2 sao informadas por elas; nada mais dos registros foi visto (so' a existencia e a contagem dos arquivos). Os ganhos observados sao os das Entradas 280/283.
(P1) A fracao mediana de decisoes com recompensa positiva no controle aleatorio, sobre as 69 celulas, e' de no maximo 0,5%. (85%)
(P2) O numero mediano de recompensas positivas por acao por execucao (T = 1000) e' de no maximo 1,0. (85%)
(P3) O indice L e' menor que 0,1 em pelo menos 90% das 69 celulas. (70%)
(P4) Spearman entre kappa_WS e a captura observada do rt nas celulas com ganho da validada >= 0,05 e' de pelo menos 0,40 E o limite inferior do IC90 por bootstrap de cluster e' acima de zero. (50%)
(P5) A mediana de (captura observada do rt) / kappa_WS nessas celulas (kappa_WS > 0) fica em [0,5; 2,0]. (50%)
(P6) Spearman entre o ganho do UCB1 previsto pelo gemeo e o observado, nas 69 celulas, e' de pelo menos 0,40. (40%)
(P7) O kappa do Q-Learning sem estados no gemeo e' maior ou igual a captura observada do rt em pelo menos 70% das celulas com ganho da validada >= 0,05 (o sem-estados e' um limite superior). (55%)
(P8) O ganho da melhor acao fixa no gemeo (G_max - G_U) e' maior ou igual ao ganho observado da validada em pelo menos 80% das celulas (o bandido oblivio e' otimista). (60%)
(P9) A fracao das celulas com folga de fase positiva (validacao cruzada) e' de no maximo 0,60. (60%)
(P10) A fracao das celulas com folga de estado positiva (validacao cruzada) e' de no maximo 0,60. (55%)
(P11) A media, sobre as 24 celulas sinteticas, da correlacao de Spearman entre os ganhos das acoes fixas previstos pelo gemeo e os observados e' de pelo menos 0,50. (40%)
PORTAO (congelado): o enquadramento PREDITIVO (a captura e' explicada por um modelo limitado pela informacao) e' adotado se P4 for verdadeira E (P6 ou P7 for verdadeira). Caso contrario, os fatos de escassez de informacao (P1 a P3) sao
relatados como a explicacao, sem alegar que o modelo preve as magnitudes.
Uso: python -m analysis.report_entrada287_learnability
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis import learnability_model as LM  # noqa: E402
from analysis.cells_instance_tables import load_cells  # noqa: E402
from analysis.report_entrada180_agents import rel  # noqa: E402

R = ROOT / "results"
LOGS = R / "logs287"
T = 1000
pd.set_option("display.width", 230)


def logfile(cell):
    return LOGS / (cell.replace("|", "_") + ".npz")


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(sstats.spearmanr(a[ok], b[ok]).statistic)


def boot_sp(a, b, cluster, nb=2000, seed=287):
    a, b, cluster = np.asarray(a, float), np.asarray(b, float), np.asarray(cluster)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, cluster = a[ok], b[ok], cluster[ok]
    keys = pd.unique(cluster)
    idx = [np.where(cluster == k)[0] for k in keys]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nb):
        pick = rng.integers(0, len(keys), len(keys))
        ix = np.concatenate([idx[i] for i in pick])
        v = sp(a[ix], b[ix])
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def process_cell(ci, c):
    z = np.load(logfile(c["cell"]), allow_pickle=False)
    A, S, Rw, lb, inst = z["A"], z["S"], z["Rw"].astype(np.float64), z["lb"], z["inst"]
    K = len(z["arm_d"])
    # integridade: soma das recompensas = makespan inicial - final
    assert np.allclose(Rw.sum(axis=1), z["C0"] - z["best"], atol=1e-3), c["cell"]
    rt = LM.normalise(Rw, lb)
    half = (pd.factorize(inst)[0] % 2) == 0
    par = LM.arm_params(A, rt, K)
    tw = LM.twin_predictions(par, T, n_sim=2000, seed=287 + ci)
    L, delta, sigma2 = LM.index_L(A, rt, K, T, half)
    hd = LM.headroom_crossfit(A, rt, S, K, T, half)
    x = c["x"]
    g = lambda m: float(-rel(x[m], x.rand).mean()) if m in x.columns else float("nan")
    row = dict(coll=c["coll"], cell=c["cell"], K=K, runs=len(A), pos_rate=float((rt > 0).mean()), pos_per_run=float((rt > 0).sum() / len(A)),
               pos_per_arm_run=float((rt > 0).sum() / len(A) / K), L=L, delta_cf=delta, sigma2=sigma2, kappa_ws=LM.kappa_win_stay(par, T),
               tw_oracle=tw["G_max"] - tw["G_uniform"], tw_gain_ucb=tw["gain_ucb"], tw_gain_ql=tw["gain_ql"], tw_kappa_ucb=tw["kappa_ucb"], tw_kappa_ql=tw["kappa_ql"],
               H_phase=hd["H_phase"], H_state=hd["H_state"], adv_static=hd["adv_static"], n_state_used=hd["n_state_used"],
               g_val=g("val"), g_rt=g("rt"), g_ucb=g("ucb"))
    row["kap_rt"] = row["g_rt"] / row["g_val"] if row["g_val"] != 0 else float("nan")
    row["kap_ucb"] = row["g_ucb"] / row["g_val"] if row["g_val"] != 0 else float("nan")
    # fidelidade das acoes fixas (so' no sintetico, onde ha' braco fixo por instancia no teste)
    fid = float("nan")
    if c["coll"] == "sintetico":
        real = np.array([float(-rel(x[f"arm{a}"], x.rand).mean()) for a in range(K)])
        fid = sp(tw["G_static"] - tw["G_uniform"], real)
    row["static_fidelity"] = fid
    return row


def main():
    cells = load_cells(include_tsp=False)
    cl = pd.read_csv(R / "entrada283_cells.csv")[["coll", "cell", "cluster"]]
    rows = [process_cell(i, c) for i, c in enumerate(cells)]
    D = pd.DataFrame(rows).merge(cl, on=["coll", "cell"], how="left")
    assert D.cluster.notna().all() and len(D) == 69
    M = D[D.g_val >= 0.05].reset_index(drop=True)
    L = ["=" * 118, "APRENDIBILIDADE: ESCASSEZ DE SINAL, INDICE L, CAPTURA GANHA-FICA E GEMEO DIGITAL (Entrada 287)", "=" * 118,
         f"celulas: {len(D)}; execucoes por celula: {int(D.runs.min())}-{int(D.runs.max())}; acoes: {int(D.K.min())}-{int(D.K.max())}; celulas com ganho da validada >= 0,05: {len(M)}", ""]
    num = ["pos_rate", "pos_per_run", "pos_per_arm_run", "L", "kappa_ws", "tw_oracle", "g_val", "g_rt", "kap_rt", "g_ucb", "tw_gain_ucb", "tw_kappa_ql", "H_phase", "H_state"]
    byc = D.groupby("coll")[num].median().round(5)
    byc.insert(0, "celulas", D.groupby("coll").size())
    L += ["--- medianas por colecao ---", byc.to_string(), "", "--- medianas gerais ---", D[num].median().round(5).to_string(), ""]
    L += ["--- por celula ---", D[["coll", "cell", "K", "pos_rate", "pos_per_run", "L", "kappa_ws", "tw_oracle", "g_val", "g_rt", "kap_rt", "g_ucb", "tw_gain_ucb", "tw_kappa_ql", "H_phase", "H_state"]].round(4).to_string(index=False), ""]
    # ---- estatisticas das previsoes
    k_ws, k_rt = M.kappa_ws.to_numpy(), M.kap_rt.to_numpy()
    s4 = sp(k_ws, k_rt)
    lo4, hi4 = boot_sp(k_ws, k_rt, M.cluster)
    ratio = (M.kap_rt / M.kappa_ws)[M.kappa_ws > 0]
    s5 = float(ratio.median())
    ok6 = D.g_ucb.notna()
    s6 = sp(D.tw_gain_ucb[ok6], D.g_ucb[ok6])
    s7 = float((M.tw_kappa_ql >= M.kap_rt).mean())
    s8 = float((D.tw_oracle >= D.g_val).mean())
    s9 = float((D.H_phase > 0).mean())
    s10 = float((D.H_state > 0).mean())
    s11 = float(D.static_fidelity.mean(skipna=True))
    info = {"Spearman(L, captura rt) em M": sp(M.L, M.kap_rt), "Spearman(L, ganho rt) nas 69": sp(D.L, D.g_rt), "Spearman(kappa_ws, ganho rt) nas 69": sp(D.kappa_ws, D.g_rt),
            "Spearman(tw_oracle, g_val) nas 69": sp(D.tw_oracle, D.g_val), "mediana tw_oracle / g_val": float((D.tw_oracle / D.g_val)[D.g_val >= 0.05].median()),
            "Spearman(tw_kappa_ql, captura rt) em M": sp(M.tw_kappa_ql, M.kap_rt), "mediana kappa_ws em M": float(M.kappa_ws.median()), "mediana captura rt em M": float(M.kap_rt.median()),
            "mediana tw_kappa_ql em M": float(M.tw_kappa_ql.median()), "mediana tw_kappa_ucb em M": float(M.tw_kappa_ucb.median()), "mediana captura ucb observada em M": float(M.kap_ucb.median()),
            "mediana folga de fase / vantagem da estatica": float((D.H_phase / D.adv_static)[D.adv_static > 0].median()), "mediana folga de estado / vantagem da estatica": float((D.H_state / D.adv_static)[D.adv_static > 0].median())}
    L += ["--- informacoes adicionais (nao previstas) ---"] + [f"{k}: {v:+.4f}" for k, v in info.items()] + [""]
    yn = lambda b: "SIM" if b else "NAO"
    preds = [("P1", f"fracao mediana de decisoes com recompensa positiva: {D.pos_rate.median():.5f} (previsto <= 0,005)", D.pos_rate.median() <= 0.005),
             ("P2", f"mediana de recompensas positivas por acao por execucao: {D.pos_per_arm_run.median():.3f} (previsto <= 1,0)", D.pos_per_arm_run.median() <= 1.0),
             ("P3", f"fracao das celulas com L < 0,1: {(D.L < 0.1).mean():.3f} (previsto >= 0,90)", (D.L < 0.1).mean() >= 0.90),
             ("P4", f"Spearman(kappa_ws, captura observada do rt) em {len(M)} celulas: {s4:+.3f}, IC90 por cluster [{lo4:+.3f}; {hi4:+.3f}] (previsto >= 0,40 e limite inferior > 0)", s4 >= 0.40 and lo4 > 0),
             ("P5", f"mediana de captura observada do rt / kappa_ws: {s5:.3f} (previsto em [0,5; 2,0])", 0.5 <= s5 <= 2.0),
             ("P6", f"Spearman(ganho do UCB1 previsto, observado) em {int(ok6.sum())} celulas: {s6:+.3f} (previsto >= 0,40)", s6 >= 0.40),
             ("P7", f"fracao das celulas com kappa do Q-Learning sem estados >= captura observada do rt: {s7:.3f} (previsto >= 0,70)", s7 >= 0.70),
             ("P8", f"fracao das celulas com G_max - G_U do gemeo >= ganho observado da validada: {s8:.3f} (previsto >= 0,80)", s8 >= 0.80),
             ("P9", f"fracao das celulas com folga de fase positiva: {s9:.3f} (previsto <= 0,60)", s9 <= 0.60),
             ("P10", f"fracao das celulas com folga de estado positiva: {s10:.3f} (previsto <= 0,60)", s10 <= 0.60),
             ("P11", f"media da correlacao de Spearman das acoes fixas (gemeo contra observado) nas 24 celulas sinteticas: {s11:+.3f} (previsto >= 0,50)", s11 >= 0.50)]
    d = {k: ok for k, _, ok in preds}
    gate = d["P4"] and (d["P6"] or d["P7"])
    L += ["--- previsoes ---"] + [f"({k}) {t}: {yn(ok)}" for k, t, ok in preds] + [f"previsoes acertadas: {sum(d.values())} de {len(d)}",
          f"PORTAO (P4 e (P6 ou P7)): {yn(gate)} -> {'o enquadramento PREDITIVO e adotado' if gate else 'os fatos de escassez de informacao sao a explicacao; nao se alega predicao das magnitudes'}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada287_learnability.txt").write_text(text + "\n", encoding="utf-8")
    D.to_csv(R / "entrada287_cells.csv", index=False)


if __name__ == "__main__":
    main()
