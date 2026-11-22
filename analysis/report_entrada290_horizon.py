"""
analysis/report_entrada290_horizon.py (Entrada 290, Ciclo 54, parte C)
=======================================================================
DOSE-RESPOSTA DE HORIZONTE no benchmark padrao de R||Cmax (B5; 7 tipos; T = 1000, 4000, 16000; 30 instancias x 2 sementes por tipo e horizonte). Dados: results/horizon290_<tipo>.csv (teste) e horizon290_tune_<tipo>.csv (ajuste da validada
em cada T), gerados por experiment/run_cycle54c_horizon_entrada290.py. Ganho = -100 (m - m_aleatorio) / m_aleatorio (pontos; positivo = melhor), medias das sementes por instancia; captura = ganho do ql_rt / ganho da validada (razao de somas
sobre as instancias do tipo); sucessos por acao por execucao = recompensas positivas do controle aleatorio por execucao / 5 acoes.

PRE-REGISTRO (congelado ANTES de rodar o executor; hashes em results/entrada290_prereg_hashes.txt). Probabilidades do autor. Ressalva: a captura do ql_rt em T = 1000 nos mesmos tipos ja' era conhecida (Entrada 283: 0,16 no padrao); o horizonte longo
com ql_rt nao tinha sido rodado (as Entradas 185, 217 e 258L usaram o Q-Learning padrao).
(H1) A captura do ql_rt em T = 16000 e' maior que em T = 1000 em pelo menos 5 dos 7 tipos. (60%)
(H2) Os sucessos por acao por execucao do controle aleatorio crescem pelo menos 8 vezes de T = 1000 para T = 16000 (mediana dos tipos). (80%)
(H3) A mediana, sobre os tipos, da captura do ql_rt em T = 16000 e' de pelo menos 0,25. (45%)
(H4) A validada em T ainda bate o ql_rt de forma estrita (Holm; IC90 inteiro alem de 0,05) em pelo menos 4 dos 7 tipos em T = 16000. (55%)
(H5) Spearman, nas 21 celulas (tipo, T), entre o log dos sucessos por acao por execucao e a captura do ql_rt, de pelo menos 0,40. (55%)
(H6) A mediana, sobre os tipos, da captura do UCB1 em T = 16000 e' menor que a do ql_rt. (70%)
Uso: python -m analysis.report_entrada290_horizon
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
R = ROOT / "results"
MARGIN = 0.05
pd.set_option("display.width", 230)


def holm_reject(ps, alpha=0.05):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    rej = np.zeros(len(ps), bool)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            rej[i] = True
        else:
            break
    return rej


def paired(d, seed=290):
    d = np.asarray(d, float)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else 1.0
    idx = np.random.default_rng(seed).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return float(d.mean()), float(lo), float(hi), p


def main():
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "horizon290_[!t]*.csv")))], ignore_index=True)
    D = D[~D.method.str.startswith("arm")]
    per = D.groupby(["type", "T", "instance", "method"]).agg(mk=("makespan", "mean"), pos=("pos", "mean"), lb=("lb", "mean")).reset_index()
    W = per.pivot(index=["type", "T", "instance"], columns="method", values="mk")
    rel = lambda a, b: 100.0 * (a - b) / b
    rows, tests = [], []
    for (ty, T), x in W.groupby(level=[0, 1]):
        pos = per[(per["type"] == ty) & (per["T"] == T) & (per.method == "rand")].pos.mean()
        row = dict(type=ty, T=T, n=len(x), pos_per_arm_run=float(pos / 5.0))
        for m in ("val", "ql_rt", "ucb"):
            row[f"g_{m}"] = float(-rel(x[m], x.rand).mean())
        row["cap_rt"] = row["g_ql_rt"] / row["g_val"] if row["g_val"] > 0 else float("nan")
        row["cap_ucb"] = row["g_ucb"] / row["g_val"] if row["g_val"] > 0 else float("nan")
        m_, lo, hi, p = paired(rel(x.val, x.ql_rt).to_numpy())            # negativo = a validada e' melhor
        tests.append(dict(type=ty, T=T, mean=m_, lo=lo, hi=hi, p=p))
        rows.append(row)
    C = pd.DataFrame(rows)
    TS = pd.DataFrame(tests)
    TS["rej"] = holm_reject(TS.p.to_numpy())
    TS["val_beats_rt_strict"] = TS.rej & (TS["mean"] < 0) & (TS.hi < -MARGIN)
    C = C.merge(TS[["type", "T", "val_beats_rt_strict"]], on=["type", "T"])
    L = ["=" * 110, "DOSE-RESPOSTA DE HORIZONTE NO R||Cmax B5 (Entrada 290)", "=" * 110, C.round(4).to_string(index=False), ""]
    byT = C.groupby("T").agg(pos_per_arm_run=("pos_per_arm_run", "median"), g_val=("g_val", "median"), g_ql_rt=("g_ql_rt", "median"), cap_rt=("cap_rt", "median"), cap_ucb=("cap_ucb", "median"),
                             val_bate_rt_estrito=("val_beats_rt_strict", "sum"))
    L += ["--- medianas por horizonte ---", byT.round(4).to_string(), ""]
    piv = C.pivot(index="type", columns="T", values="cap_rt")
    pos = C.pivot(index="type", columns="T", values="pos_per_arm_run")
    h1 = int((piv[16000] > piv[1000]).sum())
    h2 = float((pos[16000] / pos[1000]).median())
    h3 = float(piv[16000].median())
    h4 = int(C[C["T"] == 16000].val_beats_rt_strict.sum())
    ok = C.cap_rt.notna()
    h5 = float(sstats.spearmanr(np.log(C.pos_per_arm_run[ok]), C.cap_rt[ok]).statistic)
    h6a, h6b = float(C[C["T"] == 16000].cap_ucb.median()), float(C[C["T"] == 16000].cap_rt.median())
    yn = lambda b: "SIM" if b else "NAO"
    preds = [("H1", f"tipos com captura do ql_rt maior em T = 16000 que em T = 1000: {h1} de 7 (previsto >= 5)", h1 >= 5),
             ("H2", f"razao mediana dos sucessos por acao por execucao (16000 / 1000): {h2:.1f} (previsto >= 8)", h2 >= 8),
             ("H3", f"mediana da captura do ql_rt em T = 16000: {h3:.3f} (previsto >= 0,25)", h3 >= 0.25),
             ("H4", f"tipos em que a validada bate o ql_rt de forma estrita em T = 16000: {h4} de 7 (previsto >= 4)", h4 >= 4),
             ("H5", f"Spearman(log sucessos por acao por execucao, captura do ql_rt) nas {int(ok.sum())} celulas: {h5:+.3f} (previsto >= 0,40)", h5 >= 0.40),
             ("H6", f"mediana da captura em T = 16000: UCB1 {h6a:.3f} contra ql_rt {h6b:.3f} (previsto UCB1 < ql_rt)", h6a < h6b)]
    L += ["--- previsoes ---"] + [f"({k}) {t}: {yn(o)}" for k, t, o in preds] + [f"previsoes acertadas: {sum(o for _, _, o in preds)} de {len(preds)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada290_horizon.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada290_cells.csv", index=False)


if __name__ == "__main__":
    main()
