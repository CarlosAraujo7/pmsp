"""
analysis/report_entrada143_lpig_large.py
==========================================
O LP-IG bate o MILP-60s em R||Cmax n = 1000, m = 50? Diario: Entrada 143.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada143_prereg_hashes.txt)
-------------------------------------------------------------------------------------------
Dados: as 32 instancias da Entrada 127 (4 distribuicoes x 8) e o MILP-60s ja' calculado (results/unrelated_large_exact_entrada127.csv); metodos lpig (T = 1000) e lpig_T5000 (d/n = 0,02), de results/lpig_large_entrada143.csv.
d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal sobre as instancias em que o MILP devolveu incumbente (todas, na Entrada 127); bootstrap IC90.
Familia (Holm sobre 2, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0): M1 lpig (T = 1000) vs MILP-60s; M2 lpig_T5000 vs MILP-60s.
Leitura pre-fixada: 'o LP-IG bate o MILP-60s onde o IG+LS-QL perdia' = M1 confirmada (M2 reforco); se so' M2, 'so' com mais orcamento'; senao 'nao'.
Descritivas: GAP ao melhor limite inferior (max(LP, dual do MILP)) de lpig, lpig_T5000, IG+LS-QL (Entrada 127) e MILP; CPU medio; por distribuicao; proporcao de instancias em que o LP-IG iguala/bate o MILP.
Uso: python -m analysis.report_entrada143_lpig_large
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

R = ROOT / "results"
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
    d = pd.read_csv(R / "lpig_large_entrada143.csv")
    old = pd.read_csv(R / "unrelated_large_entrada127.csv")
    ex = pd.read_csv(R / "unrelated_large_exact_entrada127.csv")
    w = d.pivot(index=KEY, columns="method", values="makespan").reset_index().merge(ex, on=KEY)
    o = old.pivot(index=KEY, columns="method", values="makespan").reset_index()[KEY + ["ur_igls_ql", "ur_igls_ql_T5000"]]
    w = w.merge(o, on=KEY)
    cpu = d.groupby("method").cpu_time.mean()
    assert len(w) == 32 and w[["lpig", "lpig_T5000", "milp60_inc"]].isna().sum().sum() == 0, len(w)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "O LP-IG BATE O MILP-60s EM R||Cmax n = 1000, m = 50? -- Entrada 143", "=" * 110, f"instancias: {len(w)}"]
    rows = []
    for pid, lab, a in [("M1", "lpig (T=1000) vs MILP-60s", "lpig"), ("M2", "lpig_T5000 vs MILP-60s", "lpig_T5000")]:
        t = test(w[a], w["milp60_inc"])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 2) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    c = P.set_index("id").confirmada
    leit = "SIM" if c["M1"] else ("so' com mais orcamento" if c["M2"] else "NAO")
    L.append(f"\n'o LP-IG bate o MILP-60s onde o IG+LS-QL perdia': {leit}")
    w["lb_best"] = np.maximum(w.lb_lp, w.milp60_dual)
    gap = lambda k, df=w: 100 * ((df[k] - df.lb_best) / df.lb_best).mean()
    L.append("\nGAP medio ao melhor limite inferior (%): " + ", ".join(f"{lab} {gap(k):.3f}" for lab, k in [("IG+LS-QL T=1000 (E127)", "ur_igls_ql"), ("IG+LS-QL T=5000 (E127)", "ur_igls_ql_T5000"), ("LP-IG T=1000", "lpig"), ("LP-IG T=5000", "lpig_T5000"), ("MILP-60s", "milp60_inc")]))
    L.append(f"CPU medio por execucao (s): LP-IG T=1000 {cpu['lpig']:.1f}; LP-IG T=5000 {cpu['lpig_T5000']:.1f}; MILP {w.milp60_t.mean():.1f}")
    L.append(f"LP-IG <= MILP em {100 * (w.lpig <= w.milp60_inc).mean():.1f}% (T=1000) e {100 * (w.lpig_T5000 <= w.milp60_inc).mean():.1f}% (T=5000); IG+LS-QL T=5000 (E127) em {100 * (w.ur_igls_ql_T5000 <= w.milp60_inc).mean():.1f}%")
    by = w.groupby(["lo", "hi"]).apply(lambda g: pd.Series(dict(milp=gap("milp60_inc", g), lpig=gap("lpig", g), lpig5=gap("lpig_T5000", g), igls_ql5=gap("ur_igls_ql_T5000", g), lpig_le_milp=100 * (g.lpig <= g.milp60_inc).mean())))
    L += ["\nPor distribuicao (GAP % ao melhor LB; % LP-IG <= MILP)", by.to_string()]
    text = "\n".join(L)
    print(text)
    (R / "entrada143_lpig_large.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada143_lpig_large_primary.csv", index=False)


if __name__ == "__main__":
    main()
