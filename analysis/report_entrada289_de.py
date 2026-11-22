"""
analysis/report_entrada289_de.py (Entrada 289, Ciclo 55)
==========================================================
O PROTOCOLO NO HOSPEDEIRO DE EVOLUCAO DIFERENCIAL (BBOB, D = 10, pool de 5 estrategias, sinal DENSO). Dados: results/de_tune_entrada289.csv (ajuste), results/de_test_base_entrada289.csv (controle aleatorio e acoes fixas),
results/de_test_learn_entrada289.csv (sete aprendizes), results/entrada289_twin_predictions.csv (previsoes do gemeo, congeladas antes do estagio learn).
Ganho = log10 do erro final do aleatorio menos o do metodo (decadas; positivo = melhor), medias das sementes por instancia, 10 instancias de teste por funcao. 'Alem da margem' (regra ESTRITA da Entrada 286): significativo (Holm) e IC90
do ganho inteiro alem de +-0,05 decada. Familias de Holm: (A) validada contra aleatorio (24), (B) aprendizes contra aleatorio (24 x 7 = 168), (C) validada contra ql_rt (24). Captura = ganho do aprendiz / ganho da validada, nas funcoes com ganho da validada >= 0,1.
Captura de RECOMPENSA = (R_aprendiz - R_aleatorio) / (R_melhor acao fixa - R_aleatorio), R = soma das taxas de sucesso por execucao.

PREVISOES (qualitativas D1-D9 congeladas no diario ANTES de qualquer estagio; numericas D10-D12 congeladas ANTES do estagio learn, com a tabela do gemeo e o hash deste arquivo):
(D1) a validada ganha do aleatorio alem da margem em pelo menos 12 das 24 funcoes. (70%)
(D2) pelo menos 6 funcoes tem o aleatorio e as cinco acoes fixas a menos de 0,05 decada umas das outras. (60%)
(D3) sucessos por acao por execucao no controle aleatorio >= 100 em todas as funcoes. (95%)
(D4) mediana da captura do ql_rt nas funcoes com ganho da validada >= 0,1 e' de pelo menos 0,30. (45%)
(D5) em pelo menos 8 das 24 funcoes ao menos um aprendiz ganha do aleatorio alem da margem (Holm sobre 168). (60%)
(D6) entre as funcoes em que a validada ganha do aleatorio alem da margem, a validada bate o ql_rt (estrito) em no maximo metade. (45%)
(D7) a mediana da captura de recompensa do ql_rt excede a da captura de erro nas funcoes com ganho da validada >= 0,1. (65%)
(D8) |ganho(ql_rt) - ganho(ql_std)| <= 0,05 em pelo menos 70% das 24 funcoes. (60%)
(D9) a participacao media da acao 0 no ql_std e' de no maximo 0,35. (70%)
(D10) Spearman, nas funcoes com folga de recompensa (R_melhor fixa - R_aleatorio >= 5% de R_aleatorio), entre a captura de recompensa do UCB1 prevista pelo gemeo e a observada >= 0,50. (45%)
(D11) o mesmo para o PM. (45%)
(D12) o kappa do Q-Learning sem estados do gemeo >= captura de recompensa observada do ql_rt em pelo menos 70% dessas funcoes. (55%)
Uso: python -m analysis.report_entrada289_de
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
MARGIN = 0.05
LEARNERS = ["ql_std", "ql_rt", "ucb", "pg", "pm", "ap", "etc"]
pd.set_option("display.width", 230)


def paired(d, seed=289):
    d = np.asarray(d, float)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else 1.0
    idx = np.random.default_rng(seed).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), p=p)


def holm_reject(ps, alpha=0.05):
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


def family(rows):
    F = pd.DataFrame(rows)
    F["rej"] = holm_reject(F.p.to_numpy())
    F["better"] = F.rej & (F["mean"] > 0) & (F.lo > MARGIN)          # estritamente alem da margem a favor
    F["worse"] = F.rej & (F["mean"] < 0) & (F.hi < -MARGIN)
    F["parity"] = (F.lo > -MARGIN) & (F.hi < MARGIN)
    return F


def main():
    tune = pd.read_csv(R / "de_tune_entrada289.csv")
    base = pd.read_csv(R / "de_test_base_entrada289.csv")
    learn = pd.read_csv(R / "de_test_learn_entrada289.csv")
    tw = pd.read_csv(R / "entrada289_twin_predictions.csv").set_index("fun")
    test = pd.concat([base, learn], ignore_index=True)
    val_arm = {f: int(g.groupby("method").lg.mean().idxmin()[3:]) for f, g in tune.groupby("fun")}
    inst = test.groupby(["fun", "inst", "method"]).agg(lg=("lg", "mean"), rew=("rew", "mean"), u0=("u0", "mean")).reset_index()
    LG = inst.pivot(index=["fun", "inst"], columns="method", values="lg")
    RW = inst.pivot(index=["fun", "inst"], columns="method", values="rew")
    funs = sorted(LG.index.get_level_values(0).unique())
    LG["val"] = [LG.loc[(f, i), f"arm{val_arm[f]}"] for f, i in LG.index]
    RW["val"] = [RW.loc[(f, i), f"arm{val_arm[f]}"] for f, i in RW.index]
    # ---- testes
    FA, FB, FC = [], [], []
    for f in funs:
        x = LG.loc[f]
        a = paired(x.rand - x.val); a.update(fun=f, who="val"); FA.append(a)
        c = paired(x.ql_rt - x.val); c.update(fun=f, who="val_vs_ql_rt"); FC.append(c)           # positivo = a validada e' melhor
        for m in LEARNERS:
            b = paired(x.rand - x[m]); b.update(fun=f, who=m); FB.append(b)
    FA, FB, FC = family(FA), family(FB), family(FC)
    L = ["=" * 118, "O PROTOCOLO NO HOSPEDEIRO DE EVOLUCAO DIFERENCIAL (Entrada 289)", "=" * 118,
         f"funcoes: {len(funs)}; instancias de teste por funcao: {LG.groupby(level=0).size().iloc[0]}; execucoes de teste: {len(test)}; estrategias validadas: {val_arm}", ""]
    # ---- tabela por funcao
    G = pd.DataFrame({f: dict(rand_lg=LG.loc[f].rand.mean(), val=val_arm[f], gain_val=(LG.loc[f].rand - LG.loc[f].val).mean(), **{f"g_{m}": (LG.loc[f].rand - LG.loc[f][m]).mean() for m in LEARNERS},
                              spread=max(LG.loc[f][c].mean() for c in ["rand"] + [f"arm{k}" for k in range(5)]) - min(LG.loc[f][c].mean() for c in ["rand"] + [f"arm{k}" for k in range(5)]),
                              u0_qlstd=RW.loc[f].shape[0] and inst[(inst.fun == f) & (inst.method == "ql_std")].u0.mean(),
                              succ_per_arm_run=tw.loc[f, "succ_per_arm_run"]) for f in funs}).T
    G.index.name = "fun"
    G["val_beats_rand"] = FA.better.to_numpy()
    G["val_beats_qlrt"] = FC.better.to_numpy()
    G["any_learner_beats_rand"] = [bool(FB[FB.fun == f].better.any()) for f in funs]
    L += ["--- por funcao (ganhos em decadas; 'spread' = maior menos menor media de log10 entre aleatorio e acoes fixas) ---", G.round(3).to_string(), ""]
    # ---- reforco
    RWm = RW.groupby(level=0).mean()
    rcap = {}
    for m in LEARNERS:
        den = RWm[[f"arm{k}" for k in range(5)]].max(axis=1) - RWm.rand
        rcap[m] = ((RWm[m] - RWm.rand) / den).where(den > 0)
    RC = pd.DataFrame(rcap)
    head = G[G.gain_val >= 0.1].index
    kap = pd.DataFrame({m: (G[f"g_{m}"] / G.gain_val).where(G.gain_val >= 0.1) for m in LEARNERS})
    L += [f"--- captura de erro (ganho do aprendiz / ganho da validada) nas {len(head)} funcoes com ganho da validada >= 0,1: mediana, quartis e funcoes alem da margem ---"]
    summ = pd.DataFrame({m: dict(mediana_captura=float(kap[m].median()), q25=float(kap[m].quantile(.25)), q75=float(kap[m].quantile(.75)), mediana_captura_recompensa=float(RC.loc[head, m].median()),
                                 alem_da_margem_sobre_aleatorio=int(FB[FB.who == m].better.sum()), pior_alem=int(FB[FB.who == m].worse.sum()), paridade=int(FB[FB.who == m].parity.sum())) for m in LEARNERS}).T
    L += [summ.round(3).to_string(), ""]
    # ---- previsoes
    d1 = int(FA.better.sum())
    d2 = int((G.spread <= MARGIN).sum())
    d3 = float(G.succ_per_arm_run.min())
    d4 = float(kap.ql_rt.median())
    d5 = int(G.any_learner_beats_rand.sum())
    mv = FA.better.to_numpy()
    d6 = float(FC.better.to_numpy()[mv].mean()) if mv.any() else float("nan")
    d7a, d7b = float(RC.loc[head, "ql_rt"].median()), float(kap.ql_rt.median())
    d8 = float(((G.g_ql_rt - G.g_ql_std).abs() <= MARGIN).mean())
    d9 = float(G.u0_qlstd.mean())
    # reward headroom e gemeo
    den = RWm[[f"arm{k}" for k in range(5)]].max(axis=1) - RWm.rand
    hr = (den >= 0.05 * RWm.rand).to_numpy()
    obs = {m: ((RWm[m] - RWm.rand) / den) for m in ("ucb", "pm", "ap", "etc", "ql_rt")}
    sp = lambda a, b: float(sstats.spearmanr(a, b).statistic) if len(a) > 3 else float("nan")
    d10 = sp(tw.loc[funs, "kappa_ucb"].to_numpy()[hr], obs["ucb"].to_numpy()[hr])
    d11 = sp(tw.loc[funs, "kappa_pm"].to_numpy()[hr], obs["pm"].to_numpy()[hr])
    d12 = float((tw.loc[funs, "kappa_ql1"].to_numpy()[hr] >= obs["ql_rt"].to_numpy()[hr]).mean())
    info = [f"funcoes com folga de recompensa (R_melhor fixa - R_aleatorio >= 5% de R_aleatorio): {int(hr.sum())} de 24",
            "Spearman gemeo x observado (captura de recompensa) nessas funcoes: " + "; ".join(f"{nm} {sp(tw.loc[funs, f'kappa_{nm}'].to_numpy()[hr], obs[of].to_numpy()[hr]):+.3f}" for nm, of in (("ucb", "ucb"), ("pm", "pm"), ("ap", "ap"), ("etc", "etc"), ("ql1", "ql_rt"))),
            "mediana do erro absoluto do gemeo (captura de recompensa): " + "; ".join(f"{nm} {float(np.nanmedian(np.abs(tw.loc[funs, f'kappa_{nm}'].to_numpy()[hr] - obs[of].to_numpy()[hr]))):.3f}" for nm, of in (("ucb", "ucb"), ("pm", "pm"), ("ap", "ap"), ("etc", "etc"), ("ql1", "ql_rt"))),
            f"validada alem da margem sobre o aleatorio: {d1} de 24; validada pior que o aleatorio alem da margem: {int(FA.worse.sum())}; validada bate ql_rt (estrito) em {int(FC.better.sum())}, ql_rt bate a validada alem da margem em {int(FC.worse.sum())}",
            f"participacao da acao 0 no ql_std: {d9:.3f}; no ql_rt: {float(inst[inst.method == 'ql_rt'].u0.mean()):.3f}",
            f"Spearman entre captura de recompensa e de erro do ql_rt: {sp(RC.loc[head, 'ql_rt'].to_numpy(), kap.loc[head, 'ql_rt'].to_numpy()):+.3f} (n = {len(head)})"]
    L += ["--- informacoes adicionais (nao previstas) ---"] + info + [""]
    yn = lambda b: "SIM" if b else "NAO"
    preds = [("D1", f"validada alem da margem sobre o aleatorio: {d1} de 24 (previsto >= 12)", d1 >= 12),
             ("D2", f"funcoes com spread <= 0,05: {d2} (previsto >= 6)", d2 >= 6),
             ("D3", f"minimo de sucessos por acao por execucao: {d3:.1f} (previsto >= 100)", d3 >= 100),
             ("D4", f"mediana da captura do ql_rt: {d4:.3f} (previsto >= 0,30; {len(head)} funcoes)", d4 >= 0.30),
             ("D5", f"funcoes com algum aprendiz alem da margem sobre o aleatorio: {d5} (previsto >= 8)", d5 >= 8),
             ("D6", f"fracao das funcoes (validada alem da margem) em que a validada bate o ql_rt: {d6:.3f} (previsto <= 0,50)", d6 <= 0.5),
             ("D7", f"mediana da captura de recompensa {d7a:.3f} contra de erro {d7b:.3f} do ql_rt (previsto recompensa > erro)", d7a > d7b),
             ("D8", f"fracao das funcoes com |ganho(ql_rt) - ganho(ql_std)| <= 0,05: {d8:.3f} (previsto >= 0,70)", d8 >= 0.70),
             ("D9", f"participacao media da acao 0 no ql_std: {d9:.3f} (previsto <= 0,35)", d9 <= 0.35),
             ("D10", f"Spearman gemeo x observado, UCB1: {d10:+.3f} (previsto >= 0,50)", d10 >= 0.50),
             ("D11", f"Spearman gemeo x observado, PM: {d11:+.3f} (previsto >= 0,50)", d11 >= 0.50),
             ("D12", f"fracao das funcoes com kappa do Q-Learning sem estados >= captura de recompensa do ql_rt: {d12:.3f} (previsto >= 0,70)", d12 >= 0.70)]
    L += ["--- previsoes ---"] + [f"({k}) {t}: {yn(ok)}" for k, t, ok in preds] + [f"previsoes acertadas: {sum(ok for _, _, ok in preds)} de {len(preds)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada289_de.txt").write_text(text + "\n", encoding="utf-8")
    G.to_csv(R / "entrada289_cells.csv")
    pd.concat([FA.assign(family="A"), FB.assign(family="B"), FC.assign(family="C")]).to_csv(R / "entrada289_tests.csv", index=False)
    summ.to_csv(R / "entrada289_summary.csv")


if __name__ == "__main__":
    main()
