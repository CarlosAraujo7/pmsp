"""
analysis/report_entrada277_cost.py
==================================
Entrada 277 (Ciclo 47): O PRECO DO CONHECIMENTO OFFLINE. Resposta as revisoes cegas de 15/11/2026 (os dois revisores): 'a acao estatica validada vence todo aprendiz' compara INFORMACAO, nao aprendizagem: a acao e' escolhida com 30 acoes x 10 instancias de ajuste x 2 sementes por classe, o aprendiz tem uma
execucao, e o custo do ajuste nao e' cobrado. Aqui (so' com dados que ja existem; nenhuma execucao nova) (i) o ajuste e' contado em AVALIACOES DE MOVIMENTOS (a mesma medida de CPU do CPU igual), (ii) a CURVA DE APRENDIZAGEM da acao estatica no numero k de instancias de ajuste por classe
(k = 1, 2, 3, 5, 10; 1.000 sorteios de k das 10 instancias de cada uma das 28 classes; G_tn, G_n, G_0 escolhidas como na Entrada 266) e (iii) a comparacao a CPU TOTAL igual: a acao estatica com o custo do ajuste diluido em N instancias da classe contra a ESCOLHA ALEATORIA com o mesmo total
(a T = 10.000, medida; entre 1.000 e 10.000, interpolacao log-linear da lacuna aos melhores valores conhecidos, SINALIZADA como interpolacao).
Dados: fr_tune_B30_s*of4_entrada256.csv (razao makespan/melhor conhecido por classe, acao, instancia e semente de ajuste; a ordem das linhas e' instancia > acao > semente e permite reconstruir as instancias), arms_test_B30_*_entrada266.csv (as 30 acoes nas 700 instancias de teste; avaliacoes),
fr_test_B30_*_entrada256.csv (aleatorio e validada a T = 1.000), frL_test_B30_T10000_*_entrada258.csv (os mesmos a T = 10.000, com os sete aprendizes). Ressalva: o custo do ajuste e' ESTIMADO com as avaliacoes medias por execucao de cada acao na classe no teste (as execucoes de ajuste nao registraram avaliacoes).

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hashes em results/entrada277_prereg_hashes.txt; nenhuma curva k nem comparacao a CPU total foi calculada antes):
(C1) O custo do ajuste da acao validada por classe (30 acoes x 10 instancias x 2 sementes) e' pelo menos 10 vezes o custo de rodar uma vez a acao validada nas 25 instancias de teste da classe. (70%)
(C2) O ponto de cruzamento MEDIDO N_10k (numero de instancias por classe acima do qual o custo total da acao validada, com o ajuste, fica abaixo do custo da escolha aleatoria a 10.000 iteracoes) esta entre 40 e 200. (60%)
(C3) O ganho medio de G_tn com k = 1 instancia de ajuste por classe e' pelo menos 50% do ganho com k = 10. (60%)
(C4) O ganho medio de G_tn nao decresce com k (tolerancia de 0,03 ponto) em k = 1, 2, 3, 5, 10. (80%)
(C5) Com N = 25 instancias de teste por classe (o tamanho do benchmark), nenhum k em {1, 2, 3, 5, 10} da uma acao estatica cuja lacuna aos melhores valores conhecidos seja menor que a da escolha aleatoria com o mesmo custo total. (70%)
(C6) O N* interpolado (k = 10) em que a acao estatica passa a compensar e' pelo menos 100 instancias por classe. (70%)
(C7) Com k = 3, o ganho medio de G_n (uma acao por tamanho, agregando os tipos) e' pelo menos o de G_tn. (50%)
Uso: python -m analysis.report_entrada277_cost
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import rel  # noqa: E402

R = ROOT / "results"
A = 30
KS = [1, 2, 3, 5, 10]
D = 1000
NSIZES = [100, 200, 500, 1000]
KEY = ["type", "instance"]
pd.set_option("display.width", 230)


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def tuning_blocks():
    """razao[cls] = array (10 instancias, A acoes, 2 sementes), reconstruido da ordem de escrita (instancia > acao > semente) de cada fatia."""
    out = {}
    for sh in range(4):
        d = pd.read_csv(R / f"fr_tune_B30_s{sh}of4_entrada256.csv")
        nb = len(d) // (2 * A)
        assert nb * 2 * A == len(d)
        cls = d.cls.to_numpy(dtype=object).reshape(nb, 2 * A)
        assert (cls == cls[:, [0]]).all()
        arm = d.arm.to_numpy().reshape(nb, A, 2)
        assert (arm[:, :, 0] == np.arange(A)).all() and (arm[:, :, 1] == np.arange(A)).all()
        ratio = d.ratio.to_numpy().reshape(nb, A, 2)
        for b in range(nb):
            out.setdefault(cls[b, 0], []).append(ratio[b])
    return {c: np.stack(v) for c, v in out.items()}


def main():
    T = tuning_blocks()
    classes = sorted(T)
    assert len(classes) == 28 and all(T[c].shape[0] == 10 for c in classes), {c: T[c].shape for c in classes}
    ratios = np.stack([T[c] for c in classes])                                           # [28, 10, A, 2]
    sizes = np.array([int(c.split("|")[1]) for c in classes])
    arms = read("arms_test_B30_s*of*_entrada266.csv")
    arms["cls"] = arms["type"] + "|" + arms.n.astype(str)
    fr = read("fr_test_B30_s*of4_entrada256.csv")
    fr["cls"] = fr["type"] + "|" + fr.n.astype(str)
    lg = read("frL_test_B30_T10000_s*of*_entrada258.csv")
    lg["cls"] = lg["type"] + "|" + lg.n.astype(str)
    # ---- matrizes de teste por classe: makespan medio por (instancia, acao) e o aleatorio
    WA = arms.groupby(KEY + ["cls", "arm"]).makespan.mean().unstack("arm")
    bks = arms.drop_duplicates(KEY).set_index(KEY).bks
    rnd = fr[fr.method == "rand"].groupby(KEY).makespan.mean()
    G = np.zeros((28, A))                                                                # ganho (pontos) de cada acao sobre o aleatorio, por classe
    GAP = np.zeros((28, A))                                                              # lacuna (%) aos melhores valores conhecidos
    for ci, c in enumerate(classes):
        w = WA[WA.index.get_level_values("cls") == c].droplevel("cls")
        for a in range(A):
            G[ci, a] = float(-rel(w[a], rnd.loc[w.index]).mean())
            GAP[ci, a] = float(100 * (w[a] / bks.loc[w.index] - 1).mean())
    E = arms.groupby(["cls", "arm"]).evals.mean().unstack("arm").reindex(classes).values  # avaliacoes medias por execucao [28, A]
    gap_rand1 = float(100 * (fr[fr.method == "rand"].groupby(KEY).makespan.mean() / bks - 1).mean())
    gap_val1 = float(100 * (fr[fr.method == "val"].groupby(KEY).makespan.mean() / bks - 1).mean())
    E_rand1 = float(fr[fr.method == "rand"].evals.mean())
    E_val1 = float(fr[fr.method == "val"].evals.mean())
    lg_bks = lg.drop_duplicates(KEY).set_index(KEY).bks
    gap_rand10 = float(100 * (lg[lg.method == "rand"].groupby(KEY).makespan.mean() / lg_bks - 1).mean())
    gap_val10 = float(100 * (lg[lg.method == "val"].groupby(KEY).makespan.mean() / lg_bks - 1).mean())
    E_rand10 = float(lg[lg.method == "rand"].evals.mean())
    learners = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr"]
    gap_l10 = {m: float(100 * (lg[lg.method == m].groupby(KEY).makespan.mean() / lg_bks - 1).mean()) for m in learners}
    ev_l10 = {m: float(lg[lg.method == m].evals.mean()) for m in learners}
    # ---- verificacao: com k = 10 as escolhas reproduzem a Entrada 266 (G_tn 2,24; G_n 2,00; G_0 1,15)
    ma = ratios.mean(axis=(1, 3))
    a_tn = ma.argmin(1)
    a_0 = ma.mean(0).argmin()
    a_n = {n: ma[sizes == n].mean(0).argmin() for n in NSIZES}
    chk = dict(Gtn=G[np.arange(28), a_tn].mean(), G0=G[:, a_0].mean(), Gn=np.mean([G[ci, a_n[sizes[ci]]] for ci in range(28)]))
    L = ["=" * 110, "O PRECO DO CONHECIMENTO OFFLINE: custo do ajuste, curva por numero de instancias de ajuste e CPU total igual (Entrada 277)", "=" * 110,
         f"verificacao (k = 10, todas as instancias de ajuste): G_tn {chk['Gtn']:.3f}, G_n {chk['Gn']:.3f}, G_0 {chk['G0']:.3f} (Entrada 266: 2,24 / 2,00 / 1,15)", ""]
    assert abs(chk["Gtn"] - 2.24) < 0.02 and abs(chk["G0"] - 1.15) < 0.02 and abs(chk["Gn"] - 2.00) < 0.02, chk
    # ---- curva de aprendizagem
    rng = np.random.default_rng(277)
    rows, curves = [], {}
    for k in KS:
        res = []
        for _ in range(D if k < 10 else 1):
            sel = np.stack([rng.permutation(10)[:k] for _ in range(28)])
            sub = np.take_along_axis(ratios, sel[:, :, None, None], axis=1).mean(axis=(1, 3))     # [28, A]
            atn = sub.argmin(1)
            a0 = sub.mean(0).argmin()
            an = {n: sub[sizes == n].mean(0).argmin() for n in NSIZES}
            ci = np.arange(28)
            gtn, g0 = G[ci, atn].mean(), G[:, a0].mean()
            gn = np.mean([G[c, an[sizes[c]]] for c in range(28)])
            gaptn, etn = GAP[ci, atn].mean(), E[ci, atn].mean()
            tune = k * 2 * E.sum(1).mean()                                                  # avaliacoes do ajuste por classe (k instancias x 2 sementes x 30 acoes)
            res.append((gtn, gn, g0, gaptn, etn, tune))
        r = np.array(res)
        curves[k] = r
        rows.append(dict(k=k, runs_tuning_per_class=k * 2 * A, Gtn=r[:, 0].mean(), Gtn_p05=np.percentile(r[:, 0], 5), Gtn_p95=np.percentile(r[:, 0], 95), Gn=r[:, 1].mean(), G0=r[:, 2].mean(),
                         gap_tn=r[:, 3].mean(), evals_per_run_M=r[:, 4].mean() / 1e6, tune_evals_per_class_M=r[:, 5].mean() / 1e6))
    C = pd.DataFrame(rows)
    L += ["--- curva de aprendizagem da acao estatica no numero k de instancias de ajuste por classe (B30; ganho sobre o aleatorio em pontos; media dos 28 classes e de 1.000 sorteios; p05-p95 dos sorteios) ---", C.round(3).to_string(index=False), ""]
    # ---- custo e CPU total igual
    n_test = 25
    e_static10 = curves[10][:, 4].mean()
    tune10 = curves[10][:, 5].mean()
    c1 = tune10 / (n_test * e_static10)
    L += [f"custo do ajuste (k = 10) por classe: {tune10 / 1e6:.0f} M avaliacoes = {c1:.1f} vezes o custo de rodar a validada nas {n_test} instancias de teste da classe ({n_test * e_static10 / 1e6:.0f} M)",
          f"avaliacoes por execucao: aleatorio T=1.000 {E_rand1 / 1e6:.1f} M; validada T=1.000 {E_val1 / 1e6:.1f} M; aleatorio T=10.000 {E_rand10 / 1e6:.1f} M",
          f"lacuna aos melhores valores conhecidos (media das 700 instancias): aleatorio T=1.000 {gap_rand1:.2f}%; validada T=1.000 {gap_val1:.2f}%; aleatorio T=10.000 {gap_rand10:.2f}%; validada T=10.000 {gap_val10:.2f}%",
          "aprendizes a T=10.000 (lacuna %; avaliacoes por execucao em M): " + "; ".join(f"{m} {gap_l10[m]:.2f}% ({ev_l10[m] / 1e6:.0f})" for m in learners), ""]
    tab = []
    for k in KS:
        r = curves[k]
        gs, es, tu = r[:, 3].mean(), r[:, 4].mean(), r[:, 5].mean()
        n10 = tu / (E_rand10 - es)                                                           # cruzamento medido: a CPU total da estatica = a do aleatorio a 10.000
        # lacuna do aleatorio como funcao de T (log-linear entre 1.000 e 10.000)
        slope = (gap_rand1 - gap_rand10)                                                     # por decada de T
        Tstar = 1000.0 * 10 ** ((gap_rand1 - gs) / slope)                                    # T em que a lacuna do aleatorio iguala a da estatica
        nstar = tu / (E_rand1 * Tstar / 1000.0 - es) if E_rand1 * Tstar / 1000.0 > es else float("inf")
        Tp = 1000.0 * (es + tu / n_test) / E_rand1                                           # T' do aleatorio com o mesmo custo total a N = 25
        gap_at = gap_rand1 - slope * np.log10(Tp / 1000.0)
        tab.append(dict(k=k, gap_static=gs, tune_over_test_cost=tu / (n_test * es), N_10k=n10, T_equal_total_N25=Tp, gap_random_at_that_T=gap_at, static_better_at_N25=bool(gs < gap_at), Tstar_interp=Tstar, Nstar_interp=nstar,
                        extrapolated_N25=bool(Tp > 10000)))
    TB = pd.DataFrame(tab)
    L += ["--- CPU total igual (B30): acao estatica com o ajuste diluido contra a escolha aleatoria com o mesmo custo total ---",
          "N_10k = instancias por classe acima das quais o custo total da estatica (ajuste incluido) fica abaixo do da escolha aleatoria a T = 10.000 (cruzamento MEDIDO, lacuna do aleatorio a 10.000 = " + f"{gap_rand10:.2f}%);",
          "T_equal_total_N25 = iteracoes do aleatorio com o mesmo custo total da estatica quando a classe tem N = 25 instancias (o benchmark); gap_random_at_that_T e Nstar_interp usam a interpolacao log-linear da lacuna (hipotese), 'extrapolated' marca T > 10.000.", TB.round(2).to_string(index=False), ""]
    # ---- previsoes
    c2 = float(TB[TB.k == 10].N_10k.iloc[0])
    g1, g10 = float(C[C.k == 1].Gtn.iloc[0]), float(C[C.k == 10].Gtn.iloc[0])
    mono = all(C.Gtn.iloc[i + 1] >= C.Gtn.iloc[i] - 0.03 for i in range(len(C) - 1))
    c5 = not bool(TB.static_better_at_N25.any())
    c6 = float(TB[TB.k == 10].Nstar_interp.iloc[0])
    gn3, gtn3 = float(C[C.k == 3].Gn.iloc[0]), float(C[C.k == 3].Gtn.iloc[0])
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- previsoes ---",
          f"(C1) custo do ajuste / custo de rodar a validada nas 25 instancias de teste: {c1:.1f} (previsto >= 10): {yn(c1 >= 10)}",
          f"(C2) N_10k (k = 10): {c2:.1f} instancias por classe (previsto entre 40 e 200): {yn(40 <= c2 <= 200)}",
          f"(C3) ganho de G_tn com k = 1 / com k = 10: {g1:.3f} / {g10:.3f} = {g1 / g10:.2f} (previsto >= 0,50): {yn(g1 / g10 >= 0.5)}",
          f"(C4) ganho de G_tn nao decresce com k (tolerancia 0,03): {yn(mono)}",
          f"(C5) a N = 25 nenhum k da uma estatica com lacuna menor que a do aleatorio com o mesmo custo total: {yn(c5)}",
          f"(C6) N* interpolado (k = 10): {c6:.0f} instancias por classe (previsto >= 100): {yn(c6 >= 100)}",
          f"(C7) k = 3: ganho de G_n {gn3:.3f} contra G_tn {gtn3:.3f} (previsto G_n >= G_tn): {yn(gn3 >= gtn3)}"]
    pred = [c1 >= 10, 40 <= c2 <= 200, g1 / g10 >= 0.5, mono, c5, c6 >= 100, gn3 >= gtn3]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada277_cost.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada277_cost_curve.csv", index=False)
    TB.to_csv(R / "entrada277_cost_breakeven.csv", index=False)


if __name__ == "__main__":
    main()
