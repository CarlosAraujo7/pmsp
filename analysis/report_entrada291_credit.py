"""
analysis/report_entrada291_credit.py (Entrada 291, Ciclo 56)
==============================================================
O DILEMA INFORMACAO-ALINHAMENTO no hospedeiro de evolucao diferencial: os sete aprendizes com o credito por sucesso (Entrada 289; mesmas sementes) e com dois creditos alinhados com o objetivo (log_incumbent, que usa f_opt:
limite superior nao praticavel; rel_incumbent, praticavel). Ganho = log10 do erro do aleatorio menos o do aprendiz, em decadas, media das sementes por instancia, 10 instancias de teste por funcao; validada = a da Entrada 289
(estrategia de menor erro nas instancias 1-5). Captura = ganho do aprendiz / ganho da validada nas funcoes com ganho da validada >= 0,1 decada. Inferencia: a FUNCAO como unidade (postos sinalizados sobre os 24 ganhos), sem o Holm
de 168 comparacoes que a Entrada 289 mostrou nao ter poder com 10 instancias. Alinhamento: a acao de maior credito medio no controle aleatorio (results/logs291) e' a validada?

PRE-REGISTRO (congelado ANTES de rodar o executor; hashes em results/entrada291_prereg_hashes.txt). Probabilidades do autor.
(G1) Com log_incumbent, a mediana (sobre as 24 funcoes) do ganho do ql_rt sobre o aleatorio e' de pelo menos -0,05 decada. (55%)
(G2) Com log_incumbent, pelo menos 3 dos 7 aprendizes tem mediana de ganho >= -0,05. (50%)
(G3) Com log_incumbent, a mediana do ganho e' maior que com o credito por sucesso em pelo menos 5 dos 7 aprendizes. (70%)
(G4) Com log_incumbent, nenhum aprendiz tem mediana de captura >= 0,5 (funcoes com ganho da validada >= 0,1). (75%)
(G5) Com rel_incumbent, a mediana do ganho do ql_rt e' de pelo menos -0,05 decada. (40%)
(G6a) A fracao das funcoes em que a acao de maior credito medio no controle aleatorio e' a validada e' de pelo menos 0,5 com log_incumbent. (60%)
(G6b) O mesmo, com rel_incumbent. (40%)
(G7) Com log_incumbent, a participacao media do UCB1 na acao validada (24 funcoes) passa de 0,25 (com o credito por sucesso foi 0,179). (55%)
Uso: python -m analysis.report_entrada291_credit
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
LEARNERS = ["ql_std", "ql_rt", "ucb", "pg", "pm", "ap", "etc"]
CREDITS = ["success", "log_incumbent", "rel_incumbent"]
pd.set_option("display.width", 220)


def main():
    tune = pd.read_csv(R / "de_tune_entrada289.csv")
    val_arm = {f: int(g.groupby("method").lg.mean().idxmin()[3:]) for f, g in tune.groupby("fun")}
    base = pd.read_csv(R / "de_test_base_entrada289.csv")
    old = pd.read_csv(R / "de_test_learn_entrada289.csv").assign(credit="success")
    new = pd.read_csv(R / "de_credit_learn_entrada291.csv")
    allr = pd.concat([old, new, base.assign(credit="success")], ignore_index=True)
    inst = allr.groupby(["fun", "inst", "credit", "method"]).agg(lg=("lg", "mean"), u0=("u0", "mean"), u1=("u1", "mean"), u2=("u2", "mean"), u3=("u3", "mean"), u4=("u4", "mean")).reset_index()
    rnd = inst[(inst.credit == "success") & (inst.method == "rand")].set_index(["fun", "inst"]).lg
    funs = sorted(rnd.index.get_level_values(0).unique())
    arm_lg = {a: inst[(inst.credit == "success") & (inst.method == f"arm{a}")].set_index(["fun", "inst"]).lg for a in range(5)}
    gval = {f: float(np.mean([rnd.loc[(f, i)] - arm_lg[val_arm[f]].loc[(f, i)] for i in rnd.loc[f].index])) for f in funs}
    head = [f for f in funs if gval[f] >= 0.1]
    rows = []
    for credit in CREDITS:
        for m in LEARNERS:
            x = inst[(inst.credit == credit) & (inst.method == m)].set_index(["fun", "inst"]).lg
            g = np.array([float(np.mean([rnd.loc[(f, i)] - x.loc[(f, i)] for i in rnd.loc[f].index])) for f in funs])
            nz = g[g != 0]
            p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
            cap = np.array([g[funs.index(f)] / gval[f] for f in head])
            uu = inst[(inst.credit == credit) & (inst.method == m)].groupby("fun")[[f"u{k}" for k in range(5)]].mean()
            share_val = float(np.mean([uu.loc[f, f"u{val_arm[f]}"] for f in funs]))
            rows.append(dict(credito=credit, aprendiz=m, mediana_ganho=float(np.median(g)), neg=int((g < 0).sum()), pos=int((g > 0).sum()), p_postos=p, mediana_captura=float(np.median(cap)), na_validada=share_val))
    S = pd.DataFrame(rows)
    L = ["=" * 112, "O DILEMA INFORMACAO-ALINHAMENTO: CREDITO ALINHADO NO HOSPEDEIRO DE DE (Entrada 291)", "=" * 112,
         f"funcoes: {len(funs)}; funcoes com ganho da validada >= 0,1: {len(head)}; execucoes novas: {len(new)}", "", S.round(4).to_string(index=False), ""]
    # alinhamento no controle aleatorio
    al = {}
    for credit in ("log_incumbent", "rel_incumbent"):
        hit = []
        for f in funs:
            z = np.load(R / "logs291" / f"f{f:02d}_{credit}.npz")
            A, Rc = z["A"], z["Rc"].astype(float)
            hit.append(int(np.argmax([Rc[A == k].mean() for k in range(5)])) == val_arm[f])
        al[credit] = float(np.mean(hit))
    ref = pd.read_csv(R / "entrada289c_alignment.csv")
    L += [f"fracao das funcoes em que a acao de maior credito medio no controle aleatorio e' a validada: sucesso {float(ref.coincidem.mean()):.3f}; log_incumbent {al['log_incumbent']:.3f}; rel_incumbent {al['rel_incumbent']:.3f}", ""]
    g1 = S[(S.credito == "log_incumbent") & (S.aprendiz == "ql_rt")].mediana_ganho.iloc[0]
    lg = S[S.credito == "log_incumbent"].set_index("aprendiz")
    su = S[S.credito == "success"].set_index("aprendiz")
    rl = S[S.credito == "rel_incumbent"].set_index("aprendiz")
    g2 = int((lg.mediana_ganho >= -0.05).sum())
    g3 = int((lg.mediana_ganho > su.mediana_ganho).sum())
    g4 = float(lg.mediana_captura.max())
    g5 = float(rl.loc["ql_rt", "mediana_ganho"])
    g7 = float(lg.loc["ucb", "na_validada"])
    yn = lambda b: "SIM" if b else "NAO"
    preds = [("G1", f"log_incumbent: mediana do ganho do ql_rt {g1:+.3f} (previsto >= -0,05)", g1 >= -0.05),
             ("G2", f"log_incumbent: aprendizes com mediana >= -0,05: {g2} de 7 (previsto >= 3)", g2 >= 3),
             ("G3", f"log_incumbent: aprendizes com mediana maior que a do credito por sucesso: {g3} de 7 (previsto >= 5)", g3 >= 5),
             ("G4", f"log_incumbent: maior mediana de captura entre os aprendizes: {g4:.3f} (previsto < 0,5)", g4 < 0.5),
             ("G5", f"rel_incumbent: mediana do ganho do ql_rt {g5:+.3f} (previsto >= -0,05)", g5 >= -0.05),
             ("G6a", f"log_incumbent: fracao das funcoes com a acao de maior credito = validada: {al['log_incumbent']:.3f} (previsto >= 0,5)", al["log_incumbent"] >= 0.5),
             ("G6b", f"rel_incumbent: a mesma fracao: {al['rel_incumbent']:.3f} (previsto >= 0,5)", al["rel_incumbent"] >= 0.5),
             ("G7", f"log_incumbent: participacao media do UCB1 na validada {g7:.3f} (previsto > 0,25; com sucesso 0,179)", g7 > 0.25)]
    L += ["--- previsoes ---"] + [f"({k}) {t}: {yn(o)}" for k, t, o in preds] + [f"previsoes acertadas: {sum(o for _, _, o in preds)} de {len(preds)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada291_credit.txt").write_text(text + "\n", encoding="utf-8")
    S.to_csv(R / "entrada291_summary.csv", index=False)


if __name__ == "__main__":
    main()
