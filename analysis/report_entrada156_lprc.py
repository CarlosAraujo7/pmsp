"""
analysis/report_entrada156_lprc.py
====================================
MILP restrito com candidatas por custo reduzido do LP (LP-RC) em R||Cmax no benchmark padrao, 700 instancias de teste. Diario: Entrada 156.

PRE-REGISTRO (congelado ANTES de executar o teste; hashes em results/entrada156_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: results/lprc_test_entrada156.csv (lpig, lpig_sr6, lpig_rc6, lpig_T8000; 700 instancias idx 2-6). Melhor conhecido = min(CPLEX 2 h, Gurobi 300 s, LR 300 s). d = 100*(A-B)/B em makespan (negativo: A melhor);
Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90. lpig_sr6 e lpig_rc6 partem da MESMA solucao do IG (pareamento por construcao).
Familia primaria (Holm sobre 3, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  R1  lpig_rc6 vs lpig_sr6    (candidatas por custo reduzido do LP vs por p_ij; mesmo IG, mesmo MILP restrito)
  R2  lpig_rc6 vs lpig_T8000  (vs mais iteracoes do LP-IG a CPU aproximadamente igual)
  R3  lpig_rc6 vs lpig        (vs o LP-IG sem o MILP)
Criterios de competitividade (pre-fixados; descritivos): GAP medio ao melhor conhecido do lpig_rc6 <= 0,25% no total e <= 0,5% em CADA tipo e em CADA n.
Descritivas: GAP por metodo, por n e por tipo; iguala/bate/estritamente melhor que o melhor conhecido; CPU medio; integridade (lpig_rc6 <= lpig e lpig_sr6 <= lpig em todas as instancias).
Uso: python -m analysis.report_entrada156_lprc
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
    d = pd.read_csv(R / "lprc_test_entrada156.csv")
    d["instance"] = d.instance.astype(str)
    man = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    w = d.pivot(index=KEY + ["m", "n"], columns="method", values="makespan").reset_index().merge(man[KEY + ["best_known", "proven_opt", "lr", "gurobi", "cplex"]], on=KEY)
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    need = ["lpig", "lpig_sr6", "lpig_rc6", "lpig_T8000"]
    assert len(w) == 700 and w[need].isna().sum().sum() == 0, (len(w), w[need].isna().sum().to_dict())
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    viol = int((w.lpig_rc6 > w.lpig).sum() + (w.lpig_sr6 > w.lpig).sum())
    L = ["=" * 110, "MILP RESTRITO COM CANDIDATAS POR CUSTO REDUZIDO DO LP (LP-RC) -- Entrada 156", "=" * 110, f"instancias de teste: {len(w)} | referencia provada otima em {100 * w.proven_opt.mean():.1f}% | violacoes de integridade: {viol} (esperado 0)"]
    rows = []
    for pid, lab, a, b in [("R1", "lpig_rc6 vs lpig_sr6 (custo reduzido vs p_ij)", "lpig_rc6", "lpig_sr6"), ("R2", "lpig_rc6 vs lpig_T8000 (CPU ~ igual)", "lpig_rc6", "lpig_T8000"), ("R3", "lpig_rc6 vs lpig", "lpig_rc6", "lpig")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 3) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    gap = lambda k, df=w: 100 * ((df[k] - df.best_known) / df.best_known).mean()
    ms = ["lpig", "lpig_T8000", "lpig_sr6", "lpig_rc6", "gurobi", "lr", "cplex"]
    L.append("\nGAP medio ao melhor conhecido (%): " + ", ".join(f"{k} {gap(k):.3f}" for k in ms))
    L.append("CPU medio por execucao (s; inclui LP e MILP): " + ", ".join(f"{k} {cpu[k].mean():.1f}" for k in ["lpig", "lpig_T8000", "lpig_sr6", "lpig_rc6"]))
    L.append("iguala ou bate o melhor conhecido (%): " + ", ".join(f"{k} {100 * (w[k] <= w.best_known).mean():.1f}" for k in ["lpig", "lpig_sr6", "lpig_rc6"]) +
             f" | estritamente melhor: lpig_rc6 {100 * (w.lpig_rc6 < w.best_known).mean():.1f}%")
    by_n = w.groupby("n").apply(lambda g: pd.Series({k: gap(k, g) for k in ["lpig", "lpig_sr6", "lpig_rc6", "gurobi", "lr"]}))
    by_t = w.groupby("type").apply(lambda g: pd.Series({k: gap(k, g) for k in ["lpig", "lpig_sr6", "lpig_rc6", "gurobi", "lr"]}))
    L += ["\n--- GAP medio ao melhor conhecido por n (%) ---", by_n.to_string(), "\n--- por tipo ---", by_t.to_string()]
    tot = gap("lpig_rc6")
    L.append(f"\nCriterio pre-fixado (lpig_rc6): total {tot:.3f}% <= 0,25%: {'SIM' if tot <= 0.25 else 'NAO'}; todos os tipos <= 0,5%: {'SIM' if bool((by_t['lpig_rc6'] <= 0.5).all()) else 'NAO'} (max {by_t['lpig_rc6'].max():.3f}%); "
             f"todos os n <= 0,5%: {'SIM' if bool((by_n['lpig_rc6'] <= 0.5).all()) else 'NAO'} (max {by_n['lpig_rc6'].max():.3f}%)")
    text = "\n".join(L)
    print(text)
    (R / "entrada156_lprc.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada156_lprc_primary.csv", index=False)


if __name__ == "__main__":
    main()
