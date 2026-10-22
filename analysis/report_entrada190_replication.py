"""
analysis/report_entrada190_replication.py
===========================================
Ciclo 17, estagio 3: REPLICACAO CONFIRMATORIA, nas instancias publicadas do conjunto held-out (Entrada 90), do que o estagio 2 (Entrada 189) encontrou. Diario: Entrada 191 (pre-registro) e 192 (resultados).
Delineamento: experiment/run_cycle17_replication_entrada190.py (nada e' reajustado: hiperparametros, acoes validadas por (familia, tercil de n), limiares de n e K por grupo sao os do estagio 2).

PRE-REGISTRO (congelado ANTES de rodar; hashes em results/entrada190_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------
ORIGEM DAS HIPOTESES (transparencia): no estagio 2, as previsoes sobre as instancias publicadas erraram em 4 de 9 (R1', R2', Acpu e R4') e uma analise EXPLORATORIA, nao registrada, por (familia, tercil de n) mostrou que o ganho do Q-Learning sobre o aleatorio acompanha a folga do oraculo: na familia C, grupos 1 e 2 (folga 0,20-0,39%), o Q-Learning
recupera ~65-95% do ganho da acao validada; nas familias B (grupo 0) e E (todos os grupos; folga <= 0,17%) ele e' PIOR que o aleatorio; Spearman(folga, ganho do Q-Learning) = +0,53 em 18 celulas. As hipoteses abaixo sao essas observacoes, agora a serem CONFIRMADAS em instancias que nao participaram da exploracao.
Estatistica (a das Entradas 180-189): unidade = media das 5 sementes por instancia; d = 100 (A - B) / B em makespan (negativo: A melhor); Wilcoxon pareado bilateral (empates descartados), IC90 bootstrap (2000); equivalencia a 0 se o IC90 cabe em +-0,05%.
FAMILIA CONFIRMATORIA H (Holm sobre 5; alfa 0,05): H1 Q-Learning vs aleatorio a iteracoes iguais em S15, instancias da familia C nos grupos 1 e 2: confirmada se p_holm < 0,05 e d <= -0,10; H2 o mesmo em S5 com d <= -0,05; H3a e H3b Q-Learning vs aleatorio a iteracoes iguais em S5 e em S15, todas as instancias
da familia E: confirmada se p_holm < 0,05 e d > 0 (PIOR que o aleatorio); H4 roleta tipo ALNS vs aleatorio a iteracoes iguais em S5, todas as instancias: confirmada se p_holm < 0,05 e d <= -0,03.
FAMILIAS DESCRITIVAS (como no estagio 2, Holm sobre 18 cada): Aiter, Acpu, Bval para os nove aprendizes e as duas acoes, em todas as instancias.
Previsoes (12): (1)-(5) H1, H2, H3a, H3b e H4 confirmadas; (6) Spearman entre a folga do oraculo e o ganho do Q-Learning a iteracoes iguais nas 18 celulas (familia x grupo x espaco) >= +0,30; (7) alguma Aiter confirmada e fora da margem (>= 0,05% a iteracoes iguais) nas instancias held-out (a replicacao de R1' do estagio 2: SIM);
(8) Bval confirmada em >= 15 dos 18 pares; (9) d de Aiter do Q-Learning em S15, todas as instancias, em [-0,09; -0,03]; (10) d de Aiter da roleta em S5 em [-0,09; -0,03]; (11) d de Acpu do Q-Learning em S5 em [-0,17; -0,08]; (12) a acao validada vs Q-Learning na familia C, grupos 1 e 2, d em [-0,08; 0,0] em S5 e em S15 (a validada continua pelo menos tao boa, por pouco).
Leitura pre-fixada L: 'um seletor online paga onde a folga e' grande e custa onde e' pequena' = H1, H2 e H3a/H3b confirmadas e a previsao (6) cumprida.
Uso: python -m analysis.report_entrada190_replication
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada189_published import COLS, LEARNERS, per_instance  # noqa: E402

R = ROOT / "results"
EXPECT = int(os.environ.get("C17_EXPECT", "0"))
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def main():
    info = json.loads((R / "pubh_groups_entrada190.json").read_text())
    L = ["=" * 110, "REPLICACAO NAS INSTANCIAS PUBLICADAS HELD-OUT -- Ciclo 17, estagio 3", "=" * 110, f"instancias held-out: {len(info['test'])}; excluidas (soma >= 2^62): {info['excluded']}"]
    W = {}
    ri, ra, rb, cells = [], [], [], []
    for sp in ("S5", "S15"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pubh_{sp}_s*of*_entrada190.csv")))], ignore_index=True)
        w = per_instance(d)
        assert len(w) == (EXPECT or len(info["test"])), (sp, len(w))
        w["fam"] = w.cls.str.split("|").str[0]
        w["grp"] = w.cls.str.split("|").str[1].astype(int)
        W[sp] = w
        for k in LEARNERS:
            t = test(w[k], w["rand"]); t.update(space=sp, learner=k, id=f"Aiter-{sp}-{k}"); ri.append(t)
            t = test(w[k], w[f"rand_eq_{k}"]); t.update(space=sp, learner=k, id=f"Acpu-{sp}-{k}"); ra.append(t)
            t = test(w["val"], w[k]); t.update(space=sp, learner=k, id=f"Bval-{sp}-{k}"); rb.append(t)
        for (f, g), x in w.groupby(["fam", "grp"]):
            cells.append(dict(space=sp, fam=f, grp=g, n=len(x), folga=float(rel(x.oracle, x.rand).mean()), val=float(rel(x.val, x.rand).mean()), ql=float(rel(x.ql, x.rand).mean()), alns=float(rel(x.alns, x.rand).mean()), ts=float(rel(x.ts, x.rand).mean())))
    Ai, Ac, Bv = family(ri), family(ra), family(rb)
    C = pd.DataFrame(cells)
    L += ["\n--- Aiter (Holm sobre 18) ---", Ai[COLS].to_string(index=False), "\n--- Acpu (Holm sobre 18) ---", Ac[COLS].to_string(index=False), "\n--- Bval (Holm sobre 18; negativo: a validada vence) ---", Bv[COLS].to_string(index=False)]
    # ---- familia confirmatoria H
    hrows = []
    sub = lambda sp, m: W[sp][m(W[sp])]
    specs = [("H1", "S15", lambda w: (w.fam == "C") & w.grp.isin([1, 2]), "ql", -0.10, "ate"), ("H2", "S5", lambda w: (w.fam == "C") & w.grp.isin([1, 2]), "ql", -0.05, "ate"),
             ("H3a", "S5", lambda w: w.fam == "E", "ql", 0.0, "pior"), ("H3b", "S15", lambda w: w.fam == "E", "ql", 0.0, "pior"), ("H4", "S5", lambda w: w.fam != "", "alns", -0.03, "ate")]
    for hid, sp, m, k, thr, mode in specs:
        x = sub(sp, m)
        t = test(x[k], x["rand"]); t.update(id=hid, space=sp, learner=k, thr=thr, mode=mode)
        hrows.append(t)
    from analysis.stats import holm_bonferroni
    H = pd.DataFrame(hrows)
    h = holm_bonferroni(H.p_raw.tolist(), alpha=0.05)
    H["p_holm"], H["reject"] = h["p_adjusted"], h["reject"]
    H["confirmada"] = [bool(r.reject and ((r.mean_d_pct <= r.thr) if r["mode"] == "ate" else (r.mean_d_pct > 0))) for _, r in H.iterrows()]
    L += ["\n--- FAMILIA CONFIRMATORIA H (Holm sobre 5) ---", H[["id", "space", "learner", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "thr", "mode", "confirmada"]].to_string(index=False)]
    # ---- celulas
    gain_rho = sstats.spearmanr(-C.folga, -C.ql)      # folga e d sao negativos: a correlacao entre |folga| e o ganho -d
    L += ["\n--- celulas (familia x grupo x espaco): folga do oraculo e d medio contra o aleatorio a iteracoes iguais ---", C.to_string(index=False),
          f"\nSpearman(|folga do oraculo|, ganho do Q-Learning) em {len(C)} celulas: rho = {gain_rho.statistic:+.3f} (p = {gain_rho.pvalue:.3g})"]
    # ---- validada vs Q na familia C, grupos 1 e 2
    vq = {}
    for sp in ("S5", "S15"):
        x = sub(sp, lambda w: (w.fam == "C") & w.grp.isin([1, 2]))
        vq[sp] = float(rel(x.val, x.ql).mean())
    yn = lambda b: "SIM" if b else "NAO"
    Hc = H.set_index("id").confirmada
    n_b = int(Bv.confirmada.sum())
    q15 = float(Ai.loc[Ai.id == "Aiter-S15-ql", "mean_d_pct"].iloc[0])
    a5 = float(Ai.loc[Ai.id == "Aiter-S5-alns", "mean_d_pct"].iloc[0])
    qc5 = float(Ac.loc[Ac.id == "Acpu-S5-ql", "mean_d_pct"].iloc[0])
    r1 = bool((Ai.confirmada & ~Ai["equivalente_0.05"]).any())
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(1) H1 (Q-Learning, S15, C grupos 1-2, d <= -0,10): {yn(Hc['H1'])}", f"(2) H2 (Q-Learning, S5, C grupos 1-2, d <= -0,05): {yn(Hc['H2'])}", f"(3) H3a (Q-Learning pior que o aleatorio na familia E, S5): {yn(Hc['H3a'])}", f"(4) H3b (idem, S15): {yn(Hc['H3b'])}",
          f"(5) H4 (roleta ALNS, S5, d <= -0,03): {yn(Hc['H4'])}", f"(6) Spearman(folga, ganho do Q-Learning) >= +0,30 nas 18 celulas: {yn(gain_rho.statistic >= 0.30)} (rho = {gain_rho.statistic:+.3f}, p = {gain_rho.pvalue:.3g})",
          f"(7) alguma Aiter confirmada e fora da margem nas held-out: {yn(r1)}", f"(8) Bval confirmada em >= 15 dos 18 pares ({n_b}/18): {yn(n_b >= 15)}", f"(9) d de Aiter do Q-Learning em S15 em [-0,09; -0,03]: {yn(-0.09 <= q15 <= -0.03)} ({q15:.4f})",
          f"(10) d de Aiter da roleta em S5 em [-0,09; -0,03]: {yn(-0.09 <= a5 <= -0.03)} ({a5:.4f})", f"(11) d de Acpu do Q-Learning em S5 em [-0,17; -0,08]: {yn(-0.17 <= qc5 <= -0.08)} ({qc5:.4f})",
          f"(12) validada vs Q-Learning na familia C, grupos 1-2, d em [-0,08; 0,0] em S5 e S15: {yn(all(-0.08 <= v <= 0.0 for v in vq.values()))} ({vq})",
          f"L ('um seletor online paga onde a folga e' grande e custa onde e' pequena': H1, H2, H3a, H3b confirmadas e (6) cumprida): {yn(bool(Hc[['H1', 'H2', 'H3a', 'H3b']].all()) and gain_rho.statistic >= 0.30)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada190_replication.txt").write_text(text, encoding="utf-8")
    for name, tb in dict(aiter=Ai, acpu=Ac, bval=Bv, h=H, cells=C).items():
        tb.to_csv(R / f"entrada190_replication_{name}.csv", index=False)


if __name__ == "__main__":
    main()
