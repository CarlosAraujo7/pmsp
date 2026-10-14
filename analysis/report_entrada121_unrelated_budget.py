"""
analysis/report_entrada121_unrelated_budget.py
================================================
R||Cmax: orcamento (T=5000), CPU igual (ILS T=8000) e valor frente ao MILP. Diario: Entrada 121.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada121_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: as 160 instancias (reps 0-7 das 20 celulas) com T=1000 em results/unrelated_entrada118.csv, T=5000 (IG+LS) e T=8000 (ILS) em
results/unrelated_budget_entrada121.csv, e o MILP-60s de results/unrelated_exact_entrada118.csv. d = 100*(A-B)/B em makespan (negativo: A melhor).
Familia confirmatoria (Holm sobre 4, alpha=0,05, Wilcoxon bicaudal; 'confirmada' = p_holm<0,05 e a media no sentido indicado):
  RB1  a politica ENCOLHE com mais orcamento: d_cel(ur_igls_ql vs ur_igls_random) em T=5000 vs T=1000, 20 pares (celulas); confirmada se a diferenca media > 0
       (efeito menos negativo com T maior)
  RB2  o mecanismo sobrevive a CPU igual: ur_igls_random (T=1000) vs ur_ils_random (T=8000), 160 instancias; confirmada se d < 0 (IG+LS melhor). Reporta-se a razao
       de CPU realizada (mediana) para verificar que e' ~1
  RB3  politica em T=5000: ur_igls_ql vs ur_igls_random (160 instancias); 'relevante' se d <= -0,05%
  RB4  valor pratico em T=5000: ur_igls_ql (T=5000) vs MILP-60s (bicaudal); a leitura e' o TAMANHO da diferenca frente a T=1000 (Entrada 118: +1,59%)
Secundarias: politica no ILS em T=8000; ganho do orcamento (T=5000 vs T=1000); por tamanho; GAP ao melhor limite inferior com T=5000; fracao IG <= MILP.
Uso: python -m analysis.report_entrada121_unrelated_budget
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


def inst_test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    a = pd.read_csv(ROOT / "results" / "unrelated_entrada118.csv")
    a = a[a.rep < 8]
    b = pd.read_csv(ROOT / "results" / "unrelated_budget_entrada121.csv")
    ex = pd.read_csv(ROOT / "results" / "unrelated_exact_entrada118.csv")
    mk = pd.concat([a[KEY + ["method", "makespan"]], b[KEY + ["method", "makespan"]]]).pivot(index=KEY, columns="method", values="makespan").reset_index()
    cpu = pd.concat([a[KEY + ["method", "cpu_time"]], b[KEY + ["method", "cpu_time"]]]).pivot(index=KEY, columns="method", values="cpu_time").reset_index()
    w = mk.merge(ex, on=KEY)
    assert len(w) == 160, len(w)
    need = ["ur_igls_random", "ur_igls_ql", "ur_ils_random", "ur_ils_ql", "ur_igls_random_T5000", "ur_igls_ql_T5000", "ur_ils_random_T8000", "ur_ils_ql_T8000"]
    assert w[need].isna().sum().sum() == 0, w[need].isna().sum().to_dict()
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "R||Cmax: ORCAMENTO, CPU IGUAL E VALOR vs MILP -- teste pre-registrado, Entrada 121", "=" * 110, f"instancias: {len(w)}"]
    cells = []
    for c, g in w.groupby(["lo", "hi", "n", "m"]):
        cells.append(dict(cell=c, d1000=rel(g.ur_igls_ql, g.ur_igls_random).mean(), d5000=rel(g.ur_igls_ql_T5000, g.ur_igls_random_T5000).mean()))
    C = pd.DataFrame(cells)
    diff = C.d5000 - C.d1000
    rows = []
    t1 = dict(n=len(C), mean_d_pct=float("nan"), mean_diff=float(diff.mean()), p_raw=float(sstats.wilcoxon(diff).pvalue), id="RB1", comparison="d_cel T=5000 vs T=1000 (20 celulas)")
    rows.append(t1)
    t2 = inst_test(w.ur_igls_random, w.ur_ils_random_T8000)
    t2.update(id="RB2", comparison="ur_igls_random (T=1000) vs ur_ils_random (T=8000)")
    rows.append(t2)
    t3 = inst_test(w.ur_igls_ql_T5000, w.ur_igls_random_T5000)
    t3.update(id="RB3", comparison="ur_igls_ql vs ur_igls_random @T=5000")
    rows.append(t3)
    ok = ~w.milp60_inc.isna()
    t4 = inst_test(w.loc[ok, "ur_igls_ql_T5000"], w.loc[ok, "milp60_inc"])
    t4.update(id="RB4", comparison="ur_igls_ql (T=5000) vs MILP-60s")
    rows.append(t4)
    F = pd.DataFrame(rows)
    h = holm_bonferroni(F.p_raw.fillna(1.0).tolist(), alpha=0.05)
    F["p_holm"], F["reject"] = h["p_adjusted"], h["reject"]
    F["confirmada"] = F.reject & np.where(F.id == "RB1", F.mean_diff > 0, F.mean_d_pct < 0)
    F["relevante"] = F.confirmada & (F.id == "RB3") & (F.mean_d_pct <= -0.05)
    L += ["\n--- PRIMARIA (Holm sobre 4) ---", F[["id", "comparison", "n", "mean_d_pct", "mean_diff", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "relevante"]].to_string(index=False)]
    cpu_ratio = (cpu.merge(w[KEY], on=KEY).pipe(lambda x: x.ur_ils_random_T8000 / x.ur_igls_random)).median()
    L.append(f"\nrazao de CPU realizada ILS(T=8000)/IG+LS(T=1000): mediana {cpu_ratio:.2f}")
    L.append(f"politica por celula: media d(T=1000) = {C.d1000.mean():+.4f}% ; d(T=5000) = {C.d5000.mean():+.4f}% ; celulas em que |d| diminui: {(C.d5000.abs() < C.d1000.abs()).sum()}/20")
    L.append(f"valor frente ao MILP-60s: T=1000 d = {inst_test(w.loc[ok, 'ur_igls_ql'], w.loc[ok, 'milp60_inc'])['mean_d_pct']:+.3f}% -> T=5000 d = {t4['mean_d_pct']:+.3f}%; IG(T=5000) <= MILP em {100 * (w.ur_igls_ql_T5000 <= w.milp60_inc.fillna(1e18)).mean():.1f}% (T=1000: {100 * (w.ur_igls_ql <= w.milp60_inc.fillna(1e18)).mean():.1f}%)")
    w["lb_best"] = np.maximum(w.lb_lp, w.milp60_dual)
    for k in ["ur_igls_ql", "ur_igls_ql_T5000", "ur_ils_ql", "ur_ils_ql_T8000"]:
        L.append(f"  {k:22s} GAP ao melhor LB {100 * ((w[k] - w.lb_best) / w.lb_best).mean():.3f}%")
    L.append(f"  MILP-60s               GAP ao melhor LB {100 * ((w.milp60_inc - w.lb_best) / w.lb_best).mean():.3f}%")
    sec = []
    for lab, x, y in [("ur_ils_ql_T8000 vs ur_ils_random_T8000", w.ur_ils_ql_T8000, w.ur_ils_random_T8000), ("ur_igls_ql_T5000 vs ur_igls_ql (T=1000)", w.ur_igls_ql_T5000, w.ur_igls_ql),
                      ("ur_ils_random_T8000 vs ur_ils_random (T=1000)", w.ur_ils_random_T8000, w.ur_ils_random), ("ur_igls_random_T5000 vs ur_ils_random_T8000", w.ur_igls_random_T5000, w.ur_ils_random_T8000)]:
        t = inst_test(x, y)
        t["comparison"] = lab
        sec.append(t)
    L += ["\n--- SECUNDARIAS ---", pd.DataFrame(sec)[["comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw"]].to_string(index=False)]
    L += ["\n--- por celula: d da politica em T=1000 e T=5000 ---", C.to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada121_unrelated_budget.txt").write_text(text, encoding="utf-8")
    F.to_csv(ROOT / "results" / "entrada121_unrelated_budget_primary.csv", index=False)


if __name__ == "__main__":
    main()
