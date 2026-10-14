"""
analysis/report_entrada149_lpigsr.py
======================================
LP-IG + MILP restrito em R||Cmax no benchmark padrao, 420 instancias RESERVADAS. Diario: Entrada 149.

PRE-REGISTRO (congelado ANTES de executar o teste; hashes em results/entrada149_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: results/lpigsr_test_entrada149.csv (lpig_sr6 e lpig_T8000) e results/lpig_test_entrada141.csv (lpig T = 1000, ur_igls_random e lpig_T5000, ja' rodados nas mesmas instancias); melhor conhecido =
min(CPLEX 2 h, Gurobi 300 s, LR 300 s). d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90.
Familia primaria (Holm sobre 3, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  S1  lpig_sr6 vs lpig_T8000   (o MILP restrito vence mais iteracoes do IG a CPU aproximadamente igual)
  S2  lpig_sr6 vs lpig         (o MILP restrito melhora o LP-IG, T = 1000)
  S3  lpig_sr6 vs lpig_T5000   (idem contra o LP-IG com T = 5000 da Entrada 141)
Criterios de competitividade (pre-fixados; descritivos): GAP medio ao melhor conhecido <= 0,3% no total e <= 0,5% em CADA grupo de n e em CADA tipo de instancia.
Descritivas: GAP por metodo, por n e por tipo; proporcao em que iguala/bate o melhor conhecido (e estritamente melhor); CPU medio (inclui LP e MILP); GAP dos metodos exatos de terceiros; proporcao de instancias em que o MILP
restrito melhora o LP-IG; verificacao de integridade: o makespan do LP-IG do metodo lpig_sr6 antes do MILP nao e' armazenado, mas lpig_sr6 <= lpig (T = 1000, mesmo seed) deve valer em TODAS as instancias.
Uso: python -m analysis.report_entrada149_lpigsr
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
    new = pd.read_csv(R / "lpigsr_test_entrada149.csv")
    old = pd.read_csv(R / "lpig_test_entrada141.csv")
    man = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    for d in (new, old):
        d["instance"] = d.instance.astype(str)
    d = pd.concat([new[KEY + ["m", "n", "method", "makespan", "cpu_time"]], old[KEY + ["m", "n", "method", "makespan", "cpu_time"]]])
    w = d.pivot(index=KEY + ["m", "n"], columns="method", values="makespan").reset_index().merge(man[KEY + ["best_known", "proven_opt", "lr", "gurobi", "cplex"]], on=KEY)
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    need = ["lpig_sr6", "lpig_T8000", "lpig", "lpig_T5000", "ur_igls_random"]
    assert len(w) == 420 and w[need].isna().sum().sum() == 0, (len(w), w[need].isna().sum().to_dict())
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    viol = int((w.lpig_sr6 > w.lpig).sum())
    L = ["=" * 110, "LP-IG + MILP RESTRITO EM R||Cmax NO BENCHMARK PADRAO (420 instancias reservadas) -- Entrada 149", "=" * 110,
         f"instancias: {len(w)} | referencia provada otima em {100 * w.proven_opt.mean():.1f}% | violacoes de integridade (lpig_sr6 > lpig): {viol} (esperado 0)"]
    rows = []
    for pid, lab, a, b in [("S1", "lpig_sr6 vs lpig_T8000 (CPU ~ igual)", "lpig_sr6", "lpig_T8000"), ("S2", "lpig_sr6 vs lpig (T = 1000)", "lpig_sr6", "lpig"), ("S3", "lpig_sr6 vs lpig_T5000", "lpig_sr6", "lpig_T5000")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 3) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    gap = lambda k, df=w: 100 * ((df[k] - df.best_known) / df.best_known).mean()
    ms = ["ur_igls_random", "lpig", "lpig_T5000", "lpig_T8000", "lpig_sr6", "gurobi", "lr", "cplex"]
    L.append("\nGAP medio ao melhor conhecido (%): " + ", ".join(f"{k.replace('ur_', '')} {gap(k):.3f}" for k in ms))
    L.append("CPU medio por execucao (s; inclui LP e MILP): " + ", ".join(f"{k.replace('ur_', '')} {cpu[k].mean():.1f}" for k in ["ur_igls_random", "lpig", "lpig_T5000", "lpig_T8000", "lpig_sr6"]))
    L.append("iguala ou bate o melhor conhecido (%): " + ", ".join(f"{k} {100 * (w[k] <= w.best_known).mean():.1f}" for k in ["ur_igls_random", "lpig", "lpig_T5000", "lpig_sr6"]) +
             f" | estritamente melhor: lpig_sr6 {100 * (w.lpig_sr6 < w.best_known).mean():.1f}%")
    L.append(f"o MILP restrito melhora o LP-IG em {100 * (w.lpig_sr6 < w.lpig).mean():.1f}% das instancias (media {rel(w.lpig_sr6, w.lpig).mean():+.3f}%)")
    by_n = w.groupby("n").apply(lambda g: pd.Series({k.replace("ur_", ""): gap(k, g) for k in ["ur_igls_random", "lpig", "lpig_T8000", "lpig_sr6", "gurobi", "lr"]}))
    by_t = w.groupby("type").apply(lambda g: pd.Series({k.replace("ur_", ""): gap(k, g) for k in ["ur_igls_random", "lpig", "lpig_T8000", "lpig_sr6", "gurobi", "lr"]}))
    L += ["\n--- GAP medio ao melhor conhecido por n (%) ---", by_n.to_string(), "\n--- por tipo ---", by_t.to_string()]
    tot = gap("lpig_sr6")
    ok_n = bool((by_n["lpig_sr6"] <= 0.5).all())
    ok_t = bool((by_t["lpig_sr6"] <= 0.5).all())
    L.append(f"\nCriterio pre-fixado de competitividade (lpig_sr6): total {tot:.3f}% <= 0,3%: {'SIM' if tot <= 0.3 else 'NAO'}; todos os n <= 0,5%: {'SIM' if ok_n else 'NAO'} (max {by_n['lpig_sr6'].max():.3f}%); "
             f"todos os tipos <= 0,5%: {'SIM' if ok_t else 'NAO'} (max {by_t['lpig_sr6'].max():.3f}%)")
    text = "\n".join(L)
    print(text)
    (R / "entrada149_lpigsr.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada149_lpigsr_primary.csv", index=False)


if __name__ == "__main__":
    main()
