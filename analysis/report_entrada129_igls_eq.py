"""
analysis/report_entrada129_igls_eq.py
=======================================
O ganho da busca local no IG hibrido do P||Cmax sobrevive a CPU igual? Diario: Entrada 129.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada129_prereg_hashes.txt)
-------------------------------------------------------------------------------------------
Dados: 12 classes x 100 reps (mesmas instancias/seeds de results.csv e da Entrada 124). igls_random (T = 1000, busca local) de results/igls_p_entrada124.csv;
ig_random_eq = ig_random (sem busca local, d = 0,3 n, 3 estrategias, partida LPT) com T_eq iteracoes (CPU igual; K medido por classe em calibracao sequencial);
ig_random (T = 1000) de results/results.csv. d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados.
Primaria (alpha = 0,05; familia unica, sem ajuste): Q1 igls_random vs ig_random_eq. 'a busca local acrescenta ao IG a CPU igual' = Q1 significativa com d < 0;
'o ganho da Entrada 124 e' efeito de orcamento' = Q1 nao significativa com d < 0 ou d > 0 (em cujo caso se reporta tambem a equivalencia: IC90 dentro de +-0,05%, TOST).
Secundarias (descritivas, nao confirmatorias): Q2 ig_random_eq vs ig_random (T = 1000): efeito de mais iteracoes; por classe; razao de CPU realizada; otimo provado e GAP vs LB certificado.
Uso: python -m analysis.report_entrada129_igls_eq
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import rank_biserial_paired  # noqa: E402

R = ROOT / "results"


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    res = pd.read_csv(R / "results.csv")
    hy = pd.read_csv(R / "igls_p_entrada124.csv")
    eq = pd.read_csv(R / "igls_eq_entrada129.csv")
    ref = pd.read_csv(R / "exact_reference_synthetic_entrada101.csv")[["instance", "rep", "lb_lp"]]
    keep = ["instance", "rep", "method", "makespan"]
    d = pd.concat([res[res.method == "ig_random"][keep], hy[hy.method == "igls_random"][keep], eq[keep]]).drop_duplicates(["instance", "rep", "method"])
    W = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().merge(ref, on=["instance", "rep"], how="left")
    W = W.dropna(subset=["igls_random", "ig_random_eq", "ig_random"])
    assert len(W) == 1200, len(W)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "A BUSCA LOCAL DO IG HIBRIDO SOBREVIVE A CPU IGUAL? -- teste pre-registrado, Entrada 129", "=" * 110, f"instancias pareadas: {len(W)}"]
    q1 = test(W.igls_random, W.ig_random_eq)
    q1["sig"] = q1["p_raw"] < 0.05
    L.append(f"\nQ1 igls_random vs ig_random_eq (CPU igual): d = {q1['mean_d_pct']:+.4f}% IC90 [{q1['ci90_lo']:+.4f}, {q1['ci90_hi']:+.4f}] | melhor/pior/empate {q1['better']}/{q1['worse']}/{q1['tie']} | "
             f"rank-biserial {q1['rank_biserial']:+.3f} | p = {q1['p_raw']:.3g} -> {'SIGNIFICATIVA' if q1['sig'] else 'nao significativa'}")
    for mg in (0.05, 0.10):
        L.append(f"   equivalencia (IC90 dentro de +-{mg:.2f}%): {'SIM' if (q1['ci90_lo'] > -mg and q1['ci90_hi'] < mg) else 'NAO'}")
    leit = "SIM (d < 0)" if (q1["sig"] and q1["mean_d_pct"] < 0) else "NAO"
    L.append(f"'a busca local acrescenta ao IG a CPU igual': {leit}; 'o ganho da Entrada 124 e' efeito de orcamento': {'SIM' if leit == 'NAO' else 'NAO'}")
    q2 = test(W.ig_random_eq, W.ig_random)
    L.append(f"\nQ2 (descritiva) ig_random_eq vs ig_random (T = 1000): d = {q2['mean_d_pct']:+.4f}% IC90 [{q2['ci90_lo']:+.4f}, {q2['ci90_hi']:+.4f}] | {q2['better']}/{q2['worse']}/{q2['tie']} | p = {q2['p_raw']:.3g}")
    q3 = test(W.igls_random, W.ig_random)
    L.append(f"Q0 (reproducao da Entrada 124, P1) igls_random vs ig_random (T = 1000): d = {q3['mean_d_pct']:+.4f}% | {q3['better']}/{q3['worse']}/{q3['tie']}")
    Wl = W.dropna(subset=["lb_lp"])
    L.append(f"\notimo provado (n = {len(Wl)}): ig_random {100 * (Wl.ig_random == Wl.lb_lp).mean():.1f}% | igls_random {100 * (Wl.igls_random == Wl.lb_lp).mean():.1f}% | ig_random_eq {100 * (Wl.ig_random_eq == Wl.lb_lp).mean():.1f}%")
    L.append("GAP medio vs LB certificado: " + ", ".join(f"{k} {100 * ((Wl[k] - Wl.lb_lp) / Wl.lb_lp).mean():.4f}%" for k in ["ig_random", "igls_random", "ig_random_eq"]))
    cpu = pd.concat([hy[hy.method == "igls_random"][["instance", "rep", "method", "cpu_time"]], eq[["instance", "rep", "method", "cpu_time"]]]).pivot(index=["instance", "rep"], columns="method", values="cpu_time")
    L.append(f"CPU medio por execucao (s, execucao paralela): igls_random {cpu.igls_random.mean():.2f} | ig_random_eq {cpu.ig_random_eq.mean():.2f} | razao mediana eq/igls {(cpu.ig_random_eq / cpu.igls_random).median():.2f}")
    sec = []
    for g, sub in W.groupby("instance"):
        t = test(sub.igls_random, sub.ig_random_eq)
        t0 = test(sub.igls_random, sub.ig_random)
        sec.append(dict(instance=g, d_igls_vs_eq=t["mean_d_pct"], d_igls_vs_ig1000=t0["mean_d_pct"], eq_better=t["worse"], igls_better=t["better"]))
    L += ["\n--- por classe (d %; eq_better = vezes em que ig_random_eq e' melhor) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada129_igls_eq.txt").write_text(text, encoding="utf-8")
    pd.DataFrame([dict(id="Q1", comparison="igls_random vs ig_random_eq (CPU igual)", **{k: q1[k] for k in ["n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw"]}),
                  dict(id="Q2", comparison="ig_random_eq vs ig_random (T=1000)", **{k: q2[k] for k in ["n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw"]})]).to_csv(
        R / "entrada129_igls_eq_primary.csv", index=False)


if __name__ == "__main__":
    main()
