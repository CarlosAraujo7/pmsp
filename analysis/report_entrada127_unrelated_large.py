"""
analysis/report_entrada127_unrelated_large.py
===============================================
R||Cmax onde o MILP nao alcanca (n=1000, m=50). Diario: Entrada 127.

PRE-REGISTRO (congelado ANTES de executar os testes; hashes em results/entrada127_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: 32 instancias (4 distribuicoes x 8 reps; n=1000, m=50), metodos de results/unrelated_large_entrada127.csv, referencia exata em unrelated_large_exact_entrada127.csv
(LP + MILP-60s). d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal; instancias em que o MILP-60s nao devolve incumbente sao CONTADAS a parte (o IG
vence por W.O.) e ficam fora do teste pareado.
Familia (Holm sobre 2, alpha=0,05; 'confirmada' = p_holm<0,05 e d<0):
  G1  ur_igls_ql (T=1000) vs MILP-60s;   G2  ur_igls_ql_T5000 vs MILP-60s
Leitura pre-fixada: 'o IG bate o MILP onde ele nao alcanca' = G1 confirmada (e G2 como reforco); se G1 nao for confirmada mas G2 sim, 'so' com mais orcamento'; senao 'nao'.
Descritivas: GAP ao melhor limite inferior (max(LP, dual do MILP)); frequencia de incumbente ausente / pior que o construtivo; politica ql vs random (32 instancias, sem teste confirmatorio).
Uso: python -m analysis.report_entrada127_unrelated_large
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

KEY = ["lo", "hi", "n", "m", "rep"]


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    d = pd.read_csv(ROOT / "results" / "unrelated_large_entrada127.csv")
    ex = pd.read_csv(ROOT / "results" / "unrelated_large_exact_entrada127.csv")
    w = d.pivot(index=KEY, columns="method", values="makespan").reset_index().merge(ex, on=KEY)
    assert len(w) == 32 and w[["ur_igls_ql", "ur_igls_ql_T5000", "ur_igls_random", "mct"]].isna().sum().sum() == 0, len(w)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "R||Cmax n=1000, m=50: O IG BATE O MILP ONDE ELE NAO ALCANCA? -- Entrada 127", "=" * 110, f"instancias: {len(w)} | MILP-60s sem incumbente: {int(w.milp60_inc.isna().sum())}"]
    ok = ~w.milp60_inc.isna()
    rows = []
    for pid, lab, a in [("G1", "ur_igls_ql (T=1000) vs MILP-60s", "ur_igls_ql"), ("G2", "ur_igls_ql_T5000 vs MILP-60s", "ur_igls_ql_T5000")]:
        t = test(w.loc[ok, a], w.loc[ok, "milp60_inc"]) if ok.sum() >= 6 else dict(n=int(ok.sum()), better=0, worse=0, tie=0, mean_d_pct=float("nan"), ci90_lo=float("nan"), ci90_hi=float("nan"), rank_biserial=float("nan"), p_raw=float("nan"))
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 2; instancias com incumbente do MILP) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    c = P.set_index("id").confirmada
    L.append(f"\n'o IG bate o MILP onde ele nao alcanca' (G1): {'SIM' if c['G1'] else 'NAO'} | G2: {'SIM' if c['G2'] else 'NAO'}")
    w["lb_best"] = np.maximum(w.lb_lp, w.milp60_dual)
    w["milp_filled"] = w.milp60_inc.fillna(1e18)
    L.append("\n--- GAP medio ao melhor limite inferior e frequencias ---")
    for k in ["mct", "ur_igls_random", "ur_igls_ql", "ur_igls_ql_T5000"]:
        L.append(f"  {k:18s} GAP {100 * ((w[k] - w.lb_best) / w.lb_best).mean():.3f}%")
    L.append(f"  MILP-60s (com incumbente) GAP {100 * ((w.loc[ok, 'milp60_inc'] - w.loc[ok, 'lb_best']) / w.loc[ok, 'lb_best']).mean():.3f}% | otimo provado {100 * w.milp60_proven.mean():.1f}%")
    L.append(f"  IG(T=1000) <= MILP-60s em {100 * (w.ur_igls_ql <= w.milp_filled).mean():.1f}% | IG(T=5000) <= MILP em {100 * (w.ur_igls_ql_T5000 <= w.milp_filled).mean():.1f}% (ausente conta como vitoria do IG)")
    L.append(f"  MILP-60s pior que o construtivo MCT em {int((w.milp60_inc > w.mct).sum())} instancias")
    cpu = d.groupby("method").cpu_time.mean()
    L.append("  CPU medio (s): " + ", ".join(f"{k} {cpu[k]:.1f}" for k in ["mct", "ur_igls_random", "ur_igls_ql", "ur_igls_ql_T5000"]) + f" | MILP-60s {w.milp60_t.mean():.1f}")
    pol = test(w.ur_igls_ql, w.ur_igls_random)
    L.append(f"  politica ql vs random (32 instancias, descritivo): d = {pol['mean_d_pct']:+.3f}% (IC90 {pol['ci90_lo']:+.3f}; {pol['ci90_hi']:+.3f})")
    by = w.groupby(["lo", "hi"]).apply(lambda g: pd.Series(dict(milp_gap=100 * ((g.milp60_inc - g.lb_best) / g.lb_best).mean(), ig_gap=100 * ((g.ur_igls_ql - g.lb_best) / g.lb_best).mean(),
                                                                ig5_gap=100 * ((g.ur_igls_ql_T5000 - g.lb_best) / g.lb_best).mean(), igle=100 * (g.ur_igls_ql <= g.milp_filled).mean())))
    L += ["\nPor distribuicao: GAP ao melhor LB do MILP-60s, do IG (T=1000) e do IG (T=5000); % IG <= MILP", by.to_string()]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada127_unrelated_large.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada127_unrelated_large_primary.csv", index=False)


if __name__ == "__main__":
    main()
