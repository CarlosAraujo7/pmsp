"""
analysis/report_entrada105_hyperparam_sensitivity.py
======================================================
WP5: sensibilidade a hiperparametros. Diario: Entrada 105.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada105_prereg_hashes.txt)
-------------------------------------------------------------------------------------------
Dados: 6 classes x 50 repeticoes = 300 instancias (ver experiment/run_hyperparam_sensitivity_entrada105.py), T=1000,
partida LPT. d = 100*(Cmax_A - Cmax_B)/Cmax_B (negativo: A melhor).
Familia confirmatoria (Holm sobre 9, alpha=0,05; Wilcoxon pareado bicaudal pooled; "confirmada" = p_holm<0,05 e d<0):
  efeito da POLITICA com cada configuracao vs o controle aleatorio correspondente
    ig_ql_d10 vs ig_random_d10; ig_ql_d20 vs ig_random_d20; ig_ql_d40 vs ig_random_d40   (mesmo d)
    ig_ql_a05, ig_ql_a30, ig_ql_g50, ig_ql_g99, ig_ql_e990, ig_ql_e999 vs ig_random (d padrao, results.csv)
Regra: o resultado "politica ~ nula" e' ROBUSTO a hiperparametros se NENHUMA das 9 configuracoes mostra beneficio
praticamente relevante do Q-Learning (confirmada E d <= -0,05%); caso contrario, a(s) configuracao(oes) e' (sao)
reportada(s) como excecao e o enunciado do artigo e' qualificado. Equivalencia (IC90 bootstrap dentro de +-0,05%)
e' reportada para cada teste.
Descritivas (sem teste): GAP medio por classe e por d para ig_random e ig_ql (a fracao padrao 0,3 nunca foi ajustada:
mede-se quanto se perde/ganha ao varia-la); sensibilidade do proprio IG-QL a alpha/gamma/decaimento (d vs ig_ql padrao).

Uso: python -m analysis.report_entrada105_hyperparam_sensitivity
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni, rank_biserial_paired  # noqa: E402

MARGIN = 0.05
CLASSES = ["rand_20j_6m", "rand_50j_10m", "rand_75j_15m", "rand_100j_20m", "rand_150j_30m", "rand_300j_20m"]


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()),
                ci90_lo=float(lo), ci90_hi=float(hi), rank_biserial=rank_biserial_paired(x, y), p_raw=p,
                equivalent=bool(lo > -MARGIN and hi < MARGIN))


def main() -> None:
    res = pd.read_csv(ROOT / "results" / "results.csv")
    new = pd.read_csv(ROOT / "results" / "hyperparam_sensitivity_entrada105.csv")
    cols = ["instance", "rep", "method", "makespan", "gap_pct"]
    base = res[res.method.isin(["ig_ql", "ig_random"]) & res.instance.isin(CLASSES) & (res.rep < 50)][cols]
    d = pd.concat([base, new[cols]]).drop_duplicates(["instance", "rep", "method"])
    w = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    g = d.pivot(index=["instance", "rep"], columns="method", values="gap_pct").reset_index()
    w = w.dropna()
    assert len(w) == 300, len(w)
    pairs = [("ig_ql_d10", "ig_random_d10"), ("ig_ql_d20", "ig_random_d20"), ("ig_ql_d40", "ig_random_d40"),
             ("ig_ql_a05", "ig_random"), ("ig_ql_a30", "ig_random"), ("ig_ql_g50", "ig_random"), ("ig_ql_g99", "ig_random"),
             ("ig_ql_e990", "ig_random"), ("ig_ql_e999", "ig_random")]
    rows = []
    for a, b in pairs:
        t = test(w[a], w[b]); t.update(first=a, control=b); rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    P["relevante"] = P.confirmada & (P.mean_d_pct <= -MARGIN)
    pd.set_option("display.width", 230); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "SENSIBILIDADE A HIPERPARAMETROS -- teste pre-registrado, Entrada 105", "=" * 110, f"instancias: {len(w)}",
         "\n--- PRIMARIA: IG-QL (config) vs controle aleatorio correspondente (Holm sobre 9) ---",
         P[["first", "control", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "relevante", "equivalent"]].to_string(index=False)]
    L.append(f"\nREGRA: configuracoes com beneficio praticamente relevante do QL (confirmada e d<=-0,05%): "
             f"{list(P[P.relevante].first) or 'nenhuma'} -> politica ~ nula {'NAO robusta' if P.relevante.any() else 'ROBUSTA a hiperparametros'}")
    L.append(f"equivalentes na margem de 0,05%: {int(P.equivalent.sum())}/9")
    # descritivas
    L.append("\n--- GAP medio (%) vs LB classico por classe (rows) -- IG-random por fracao d ---")
    for lab, cols_ in [("ig_random", ["ig_random_d10", "ig_random_d20", "ig_random", "ig_random_d40"]),
                       ("ig_ql", ["ig_ql_d10", "ig_ql_d20", "ig_ql", "ig_ql_d40"])]:
        T = g.groupby("instance")[cols_].mean().reindex(CLASSES)
        T.columns = ["d=0.1", "d=0.2", "d=0.3 (padrao)", "d=0.4"]
        T.loc["MEDIA"] = T.mean()
        L += [f"\n{lab}", T.to_string()]
    L.append("\n--- sensibilidade do IG-QL a alpha/gamma/decaimento (d relativo medio vs ig_ql padrao; %) ---")
    for k in ["ig_ql_a05", "ig_ql_a30", "ig_ql_g50", "ig_ql_g99", "ig_ql_e990", "ig_ql_e999"]:
        t = test(w[k], w.ig_ql)
        L.append(f"  {k:12s} d={t['mean_d_pct']:+.4f}%  melhor/pior/empate={t['better']}/{t['worse']}/{t['tie']}  p={t['p_raw']:.3g}")
    L.append("\n--- melhor fracao d por classe (IG-random) e perda do padrao 0,3 em relacao a ela (pp de d relativo) ---")
    cols_ = ["ig_random_d10", "ig_random_d20", "ig_random", "ig_random_d40"]
    for cls in CLASSES:
        s = w[w.instance == cls][cols_].mean()
        best = s.idxmin()
        L.append(f"  {cls:15s} melhor={best:14s} perda do padrao = {100 * (s['ig_random'] - s.min()) / s.min():.4f}%")
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada105_hyperparam_sensitivity.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada105_hyperparam_primary.csv", index=False)


if __name__ == "__main__":
    main()
