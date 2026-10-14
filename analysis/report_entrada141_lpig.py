"""
analysis/report_entrada141_lpig.py
====================================
LP-IG (partida por relaxacao linear) em R||Cmax no benchmark padrao, nas 420 instancias RESERVADAS. Diario: Entrada 141.

PRE-REGISTRO (congelado ANTES de executar o teste; hashes em results/entrada141_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: results/lpig_test_entrada141.csv (420 instancias reservadas idx 7-9 x 5 metodos); melhor conhecido = min(CPLEX 2 h, Gurobi 300 s, LR 300 s) (terceiros).
d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90.
Familia primaria (Holm sobre 3, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  L1  lpig vs ur_igls_random_eq   (o LP-IG vence o IG a partir da construtiva a CPU IGUAL)
  L2  lpig_T5000 vs lpig          (mais orcamento ajuda o LP-IG)
  L3  lpig_ql vs lpig             (a politica ajuda no LP-IG)
Criterios de competitividade (pre-fixados; descritivos, sem teste): o LP-IG com T = 5000 e' 'competitivo' se o GAP medio ao melhor conhecido for <= 1,0% no total e <= 0,5% em n = 1000.
Descritivas: GAP medio ao melhor conhecido por metodo, por n e por tipo; proporcao de instancias em que iguala/bate o melhor conhecido; CPU medio (inclui o LP); GAP dos metodos exatos de terceiros nas mesmas instancias.
Uso: python -m analysis.report_entrada141_lpig
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
KEY = ["type", "instance"]


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
    d = pd.read_csv(R / "lpig_test_entrada141.csv")
    man = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    d["instance"] = d.instance.astype(str)
    w = d.pivot(index=KEY + ["m", "n"], columns="method", values="makespan").reset_index().merge(man[KEY + ["best_known", "proven_opt", "lr", "gurobi", "cplex"]], on=KEY)
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    assert len(w) == 420 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "LP-IG EM R||Cmax NO BENCHMARK PADRAO (420 instancias reservadas) -- Entrada 141", "=" * 110, f"instancias: {len(w)} | referencia provada otima em {100 * w.proven_opt.mean():.1f}%"]
    rows = []
    for pid, lab, a, b in [("L1", "lpig vs ur_igls_random_eq (CPU igual)", "lpig", "ur_igls_random_eq"), ("L2", "lpig_T5000 vs lpig", "lpig_T5000", "lpig"), ("L3", "lpig_ql vs lpig (politica)", "lpig_ql", "lpig")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 3) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    gap = lambda k, df=w: 100 * ((df[k] - df.best_known) / df.best_known).mean()
    ms = ["ur_igls_random", "ur_igls_random_eq", "lpig", "lpig_ql", "lpig_T5000", "gurobi", "lr", "cplex"]
    L.append("\nGAP medio ao melhor conhecido (%): " + ", ".join(f"{k.replace('ur_', '')} {gap(k):.3f}" for k in ms))
    L.append("CPU medio por execucao (s; LP-IG inclui o LP): " + ", ".join(f"{k.replace('ur_', '')} {cpu[k].mean():.2f}" for k in ["ur_igls_random", "ur_igls_random_eq", "lpig", "lpig_ql", "lpig_T5000"]))
    L.append("iguala ou bate o melhor conhecido (%): " + ", ".join(f"{k.replace('ur_', '')} {100 * (w[k] <= w.best_known).mean():.1f}" for k in ["ur_igls_random", "lpig", "lpig_T5000"]) +
             f" | estritamente melhor: lpig_T5000 {100 * (w.lpig_T5000 < w.best_known).mean():.1f}")
    comp_all = gap("lpig_T5000") <= 1.0
    comp_1000 = gap("lpig_T5000", w[w.n == 1000]) <= 0.5
    L.append(f"\nCriterio pre-fixado de competitividade (lpig_T5000: GAP <= 1,0% no total e <= 0,5% em n = 1000): total {gap('lpig_T5000'):.3f}% -> {'SIM' if comp_all else 'NAO'}; "
             f"n = 1000 {gap('lpig_T5000', w[w.n == 1000]):.3f}% -> {'SIM' if comp_1000 else 'NAO'}")
    by_n = w.groupby("n").apply(lambda g: pd.Series({k.replace("ur_", ""): gap(k, g) for k in ["ur_igls_random", "ur_igls_random_eq", "lpig", "lpig_ql", "lpig_T5000", "gurobi", "lr"]}))
    by_t = w.groupby("type").apply(lambda g: pd.Series({k.replace("ur_", ""): gap(k, g) for k in ["ur_igls_random", "ur_igls_random_eq", "lpig", "lpig_ql", "lpig_T5000", "gurobi", "lr"]}))
    L += ["\n--- GAP medio ao melhor conhecido por n (%) ---", by_n.to_string(), "\n--- por tipo ---", by_t.to_string()]
    text = "\n".join(L)
    print(text)
    (R / "entrada141_lpig.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada141_lpig_primary.csv", index=False)


if __name__ == "__main__":
    main()
