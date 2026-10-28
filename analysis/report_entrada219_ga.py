"""
analysis/report_entrada219_ga.py
==================================
Ciclo 19, frente F: o TERCEIRO HOSPEDEIRO, POPULACIONAL (algoritmo genetico de estado estacionario para P||Cmax com selecao de operadores; pmsp/ga_numba.py; experiment/run_cycle19_ga_entrada219.py, docstring). Diario: Entrada 219 (pre-registro) e 220 (resultados).
Delineamento: 12 classes x 100 instancias de teste com tempos U[1, 10^4]; T = 10.000 avaliacoes de filho; 5 sementes (unidade = media das sementes por instancia); espacos G9 (9 acoes) e G18 (18 acoes); aprendizes ql, ucb, pg, alns, ts; acao validada por classe (30 instancias de ajuste); K calibrado em sequencia para a CPU igual.
Estatistica: a das Entradas 180-218 (d = 100 (A - B)/B; Wilcoxon pareado; IC90 bootstrap; margem +-0,05; Holm sobre 10 = 5 aprendizes x 2 espacos em cada familia): Aiter aprendiz vs aleatorio (iteracoes iguais), Bit validada vs aprendiz (iteracoes iguais), Beq validada a CPU igual vs aprendiz, Acpu (ql, alns, ts) vs aleatorio a CPU igual (Holm sobre 6).
Folga: enviesada (melhor acao com a semente 0) e com validacao cruzada de sementes (escolhe em {0, 1}, avalia em {2, 3}), por instancia; celulas = classe x espaco (12 por espaco).

PRE-REGISTRO (congelado ANTES de rodar o ajuste e o teste; hashes em results/entrada219_prereg_hashes.txt; vi apenas uma execucao de fumaca de UMA instancia de AJUSTE por classe com tempo e makespan de uma semente, que mostrou acoes de efeitos muito diferentes)
------------------------------------------------------------------------------------------------------------
(G1) o hospedeiro populacional tem MUITO mais folga que o IG: a folga cf agrupada e' >= 0,3 pontos em G9 e em G18 (o IG: 0,12-0,17 enviesada).
(G2) pela primeira vez um aprendiz online ganha do aleatorio: pelo menos um dos cinco aprendizes e' confirmado melhor que o aleatorio a iteracoes iguais (Aiter, Holm) em G9 e em G18. Probabilidade a priori do autor: ~65%.
(G3) mesmo assim a acao validada vence todos os aprendizes: Bit confirmada em >= 8 dos 10 pares.  (G4) a CPU igual (com K proprio): Beq confirmada em >= 7 dos 10 pares.
(G5) o melhor aprendiz recupera pelo menos metade do ganho da validada em G9 (ganho do melhor aprendiz >= 0,5 x ganho da validada, agrupado) mas menos da metade em G18.
(G6) a regra da folga vale no hospedeiro populacional: Spearman(folga cf da classe, ganho do Q-Learning sobre o aleatorio) >= +0,5 nas 12 classes de G9 e de G18.
(G7) a folga enviesada supera a cf por >= 10% (relativo) agrupada em G9 e G18.
Leitura P ('o resultado central generaliza para um hospedeiro populacional': validada >= aprendizes) = G3 e G4. Leitura Q ('um aprendiz online paga no hospedeiro populacional') = G2 e G5.
Uso: python -m analysis.report_entrada219_ga
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
LEARNERS = ["ql", "ucb", "pg", "alns", "ts"]
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_test(sp):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ga_test_{sp}_s*of*_entrada219.csv")))], ignore_index=True)
    d["key"] = d.cls + "#" + d.rep.astype(str)
    ms = d[~d.method.str.startswith("arm")].groupby(["key", "method"]).makespan.mean().unstack("method")
    a = d[d.method.str.startswith("arm")].copy()
    a["arm"] = a.method.str[3:].astype(int)
    p = a.pivot_table(index=["key", "arm"], columns="seed_i", values="makespan").reset_index()
    p.columns = ["key", "arm"] + [f"s{c}" for c in p.columns[2:]]
    p["sel"], p["ev"] = (p.s0 + p.s1) / 2, (p.s2 + p.s3) / 2
    g = p.groupby("key")
    cf = p.loc[g.sel.idxmin()].set_index("key").ev.rename("cf")
    biased = g.s0.min().rename("biased")
    w = ms.join(cf).join(biased).reset_index()
    w["klass"] = [k.split("#")[0] for k in w.key]
    return w


def main():
    L = ["=" * 110, "HOSPEDEIRO POPULACIONAL (algoritmo genetico para P||Cmax; Entrada 219)", "=" * 110]
    K = json.loads((R / "ga_cpu_ratio_entrada219.json").read_text())
    W = {sp: load_test(sp) for sp in ("G9", "G18")}
    fam, rows = {}, []
    ri, rb, re_, ra = [], [], [], []
    for sp, w in W.items():
        for k in LEARNERS:
            t = test(w[k], w["rand"]); t.update(space=sp, learner=k, id=f"Aiter-{sp}-{k}"); ri.append(t)
            t = test(w["val"], w[k]); t.update(space=sp, learner=k, id=f"Bit-{sp}-{k}"); rb.append(t)
            t = test(w[f"val_eq_{k}"], w[k]); t.update(space=sp, learner=k, id=f"Beq-{sp}-{k}"); re_.append(t)
            if k in ("ql", "alns", "ts"):
                t = test(w[k], w[f"rand_eq_{k}"]); t.update(space=sp, learner=k, id=f"Acpu-{sp}-{k}"); ra.append(t)
        for klass, x in w.groupby("klass"):
            rows.append(dict(space=sp, cell=klass, n=len(x), h_biased=float(-rel(x.biased, x.rand).mean()), h_cf=float(-rel(x.cf, x.rand).mean()), val=float(-rel(x.val, x.rand).mean()), **{k: float(-rel(x[k], x.rand).mean()) for k in LEARNERS}))
    fam = dict(Aiter=family(ri), Bit=family(rb), Beq=family(re_), Acpu=family(ra))
    for nm, F in fam.items():
        L += [f"\n--- {nm} (Holm sobre {len(F)}) ---", F[COLS].to_string(index=False)]
    C = pd.DataFrame(rows)
    pooled = []
    for sp, w in W.items():
        g = lambda col: float(-rel(w[col], w.rand).mean())
        pooled.append(dict(space=sp, n=len(w), h_biased=g("biased"), h_cf=g("cf"), val=g("val"), **{k: g(k) for k in LEARNERS}))
    Pd = pd.DataFrame(pooled)
    L += ["\n--- ganho sobre o aleatorio, agrupado (%, positivo: melhor; folga cf com validacao cruzada de sementes) ---", Pd.to_string(index=False), "\n--- celulas (classe x espaco) ---", C.to_string(index=False)]
    kq = {sp: [K[sp][c]["ql"] for c in K[sp]] for sp in K}
    yn = lambda b: "SIM" if b else "NAO"
    ai = fam["Aiter"].set_index("id")
    g2 = all(bool(ai.loc[[f"Aiter-{sp}-{k}" for k in LEARNERS], "confirmada"].any()) for sp in ("G9", "G18"))
    g1 = bool((Pd.h_cf >= 0.3).all())
    g3 = int(fam["Bit"].confirmada.sum()) >= 8
    g4 = int(fam["Beq"].confirmada.sum()) >= 7
    best = {sp: float(Pd.set_index("space").loc[sp, LEARNERS].max()) for sp in ("G9", "G18")}
    valg = Pd.set_index("space").val
    g5 = best["G9"] >= 0.5 * valg["G9"] and best["G18"] < 0.5 * valg["G18"]
    rho = {sp: sstats.spearmanr(C[C.space == sp].h_cf, C[C.space == sp].ql) for sp in ("G9", "G18")}
    g6 = all(rho[sp].statistic >= 0.5 for sp in rho)
    g7 = bool(((Pd.h_biased - Pd.h_cf) / Pd.h_biased >= 0.10).all())
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(G1) folga cf agrupada >= 0,3 em G9 e G18: {yn(g1)} (" + ", ".join(f"{v:.3f}" for v in Pd.h_cf) + ")",
          f"(G2) pelo menos um aprendiz confirmado melhor que o aleatorio em G9 e em G18: {yn(g2)} (" + ", ".join(f"{k}: {-ai.loc[k, 'mean_d_pct']:+.3f}{'*' if ai.loc[k, 'confirmada'] else ''}" for k in ai.index) + ")",
          f"(G3) Bit confirmada em >= 8 dos 10 pares: {yn(g3)} ({int(fam['Bit'].confirmada.sum())}/10)", f"(G4) Beq confirmada em >= 7 dos 10: {yn(g4)} ({int(fam['Beq'].confirmada.sum())}/10)",
          f"(G5) melhor aprendiz >= metade do ganho da validada em G9 e < metade em G18: {yn(g5)} (melhor {best['G9']:.3f} contra validada {valg['G9']:.3f}; G18 {best['G18']:.3f} contra {valg['G18']:.3f})", f"(G6) Spearman(folga cf, ganho do Q-Learning) >= +0,5 em G9 e G18: {yn(g6)} (" + ", ".join(f"{sp} {rho[sp].statistic:+.2f} (p = {rho[sp].pvalue:.3f})" for sp in rho) + ")",
          f"(G7) folga enviesada >= cf em >= 10% relativo: {yn(g7)} (" + ", ".join(f"{(a - b) / a:.2f}" for a, b in zip(Pd.h_biased, Pd.h_cf)) + ")", f"P (validada >= aprendizes: G3 e G4): {yn(g3 and g4)}   |   Q (um aprendiz paga: G2 e G5): {yn(g2 and g5)}",
          f"K_CPU do Q-Learning (min, max) G9 {min(kq['G9']):.2f}-{max(kq['G9']):.2f}; G18 {min(kq['G18']):.2f}-{max(kq['G18']):.2f}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada219_ga.txt").write_text(text, encoding="utf-8")
    C.to_csv(R / "entrada219_ga_cells.csv", index=False)
    Pd.to_csv(R / "entrada219_ga_pooled.csv", index=False)
    for nm, F in fam.items():
        F.to_csv(R / f"entrada219_ga_{nm}.csv", index=False)


if __name__ == "__main__":
    main()
