"""
analysis/report_entrada133_unrelated_policy.py
================================================
A politica vale em regime nao saturado (R||Cmax 1000 x 50), a CPU igual? Diario: Entrada 133.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada133_prereg_hashes.txt)
-------------------------------------------------------------------------------------------
Dados: 96 instancias novas (4 distribuicoes x 24 reps; n = 1000, m = 50; seed 500000+), metodos de results/unrelated_policy_entrada133.csv. d = 100*(A-B)/B em makespan (negativo: A melhor);
Wilcoxon pareado bicaudal; bootstrap IC90.
Familia (Holm sobre 2, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0): Q1 ur_igls_ql vs ur_igls_random (T = 1000 ambos); Q2 ur_igls_ql vs ur_igls_random_eq (random com T_eq, CPU igual).
Leituras pre-fixadas: 'a politica vale em regime nao saturado' = Q1 e Q2 confirmadas; 'so' via CPU' = Q1 confirmada e Q2 nao; 'nao' = Q1 nao confirmada.
Se Q2 confirmada com d <= -0,10%, a tese e' delimitada (a politica vale onde a busca nao satura).
Descritivas: folga (oraculo por instancia dos 4 braços fixos vs random: d), razao politica/folga com IC90 bootstrap, por distribuicao, CPU medio por metodo, GAP ao LB classico.
Uso: python -m analysis.report_entrada133_unrelated_policy
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
    d = pd.read_csv(R / "unrelated_policy_entrada133.csv")
    w = d.pivot(index=KEY, columns="method", values="makespan").reset_index()
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    assert len(w) == 96 and w.isna().sum().sum() == 0, len(w)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "A POLITICA VALE EM REGIME NAO SATURADO, A CPU IGUAL? (R||Cmax 1000 x 50) -- Entrada 133", "=" * 110, f"instancias: {len(w)}"]
    rows = []
    for pid, lab, a, b in [("Q1", "ur_igls_ql vs ur_igls_random (T = 1000)", "ur_igls_ql", "ur_igls_random"), ("Q2", "ur_igls_ql vs ur_igls_random_eq (CPU igual)", "ur_igls_ql", "ur_igls_random_eq")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 2) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    c = P.set_index("id").confirmada
    leit = "a politica vale em regime nao saturado" if (c["Q1"] and c["Q2"]) else ("so' via CPU (Q1 sim, Q2 nao)" if c["Q1"] else "nao")
    L.append(f"\nLeitura pre-fixada: {leit}" + (f" | delimita a tese (Q2 d <= -0,10%): {'SIM' if (c['Q2'] and P.set_index('id').loc['Q2', 'mean_d_pct'] <= -0.10) else 'NAO'}"))
    fix = w[[f"ur_igls_fix{k}" for k in range(4)]].min(axis=1)
    hd = rel(fix, w.ur_igls_random)
    pol = rel(w.ur_igls_ql, w.ur_igls_random)
    idx = np.random.default_rng(1).integers(0, len(w), size=(2000, len(w)))
    ratio = pol.mean() / hd.mean()
    rb = np.array([pol[i].mean() / hd[i].mean() for i in idx])
    L.append(f"\nFolga (oraculo dos 4 bracos fixos vs random): d = {hd.mean():+.3f}% | politica {pol.mean():+.3f}% | razao politica/folga = {ratio:.2f} (IC90 {np.percentile(rb, 5):.2f}; {np.percentile(rb, 95):.2f})")
    L.append("CPU medio por metodo (s): " + ", ".join(f"{k} {cpu[k].mean():.1f}" for k in ["ur_igls_random", "ur_igls_ql", "ur_igls_random_eq"]) + f" | razao ql/random {cpu.ur_igls_ql.mean() / cpu.ur_igls_random.mean():.2f}; random_eq/ql {cpu.ur_igls_random_eq.mean() / cpu.ur_igls_ql.mean():.2f}")
    lbs = d[d.method == "ur_igls_random"][KEY + ["lb"]]
    wl = w.merge(lbs, on=KEY)
    L.append("GAP medio ao LB classico: " + ", ".join(f"{k} {100 * ((wl[k] - wl.lb) / wl.lb).mean():.3f}%" for k in ["ur_igls_random", "ur_igls_ql", "ur_igls_random_eq"]))
    sec = []
    for (lo, hi), g in w.groupby(["lo", "hi"]):
        t1 = test(g.ur_igls_ql, g.ur_igls_random)
        t2 = test(g.ur_igls_ql, g.ur_igls_random_eq)
        hh = rel(g[[f"ur_igls_fix{k}" for k in range(4)]].min(axis=1), g.ur_igls_random).mean()
        sec.append(dict(dist=f"U[{lo},{hi}]", folga=hh, ql_vs_random=t1["mean_d_pct"], ql_vs_eq=t2["mean_d_pct"], ql_wins_eq=t2["better"], eq_wins=t2["worse"]))
    L += ["\n--- por distribuicao (d %) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada133_unrelated_policy.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada133_unrelated_policy_primary.csv", index=False)


if __name__ == "__main__":
    main()
