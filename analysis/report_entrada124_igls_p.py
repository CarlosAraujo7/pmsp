"""
analysis/report_entrada124_igls_p.py
======================================
P||Cmax: IG hibrido (destruir/reconstruir + busca local) vs IG vs ILS. Diario: Entrada 124.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada124_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: 12 classes x 100 reps; igls_random e igls_ql (results/igls_p_entrada124.csv) pareados com ig_random, ig_ql e ils_random, ils_ql de results.csv (interseccao:
as classes originais tem so' 30 reps de ils_*, (20,4) so' 30 de ig_ql). d = 100*(A-B)/B (negativo: A melhor); Wilcoxon bicaudal pooled, nulos descartados.
Familia (Holm sobre 3, alpha=0,05; 'confirmada' = p_holm<0,05 e d<0):
  P1  igls_random vs ig_random    (a busca local acrescenta algo ao IG no P||Cmax?)
  P2  igls_random vs ils_random   (o hibrido supera o ILS so' de busca local?)
  P3  igls_ql vs igls_random      (politica no hibrido)
Leitura pre-fixada: 'no P||Cmax o ingrediente ativo e' destruir-reconstruir' = P1 NAO confirmada com |d|<0,05% (a busca local nao acrescenta) E P2 confirmada (o hibrido
supera o ILS); 'a busca local acrescenta' = P1 confirmada com d <= -0,05%. No R||Cmax (Entrada 118) a combinacao e' o ingrediente ativo; esta leitura diz se o P||Cmax difere.
Secundarias: por classe; igls_ql vs ils_ql; CPU relativa.
Uso: python -m analysis.report_entrada124_igls_p
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
    new = pd.read_csv(ROOT / "results" / "igls_p_entrada124.csv")
    res = pd.read_csv(ROOT / "results" / "results.csv")
    keep = ["instance", "rep", "method", "makespan"]
    d = pd.concat([new[keep], res[res.method.isin(["ig_random", "ig_ql", "ils_random", "ils_ql"])][keep]]).drop_duplicates(["instance", "rep", "method"])
    w = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    cpu = new.pivot(index=["instance", "rep"], columns="method", values="cpu_time")
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "P||Cmax: IG HIBRIDO (destruir/reconstruir + busca local) -- teste pre-registrado, Entrada 124", "=" * 110]
    rows = []
    for pid, lab, a, b in [("P1", "igls_random vs ig_random", "igls_random", "ig_random"), ("P2", "igls_random vs ils_random", "igls_random", "ils_random"),
                           ("P3", "igls_ql vs igls_random", "igls_ql", "igls_random")]:
        ok = ~(w[a].isna() | w[b].isna())
        t = test(w.loc[ok, a], w.loc[ok, b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 3) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    r = P.set_index("id")
    active = (not r.loc["P1", "confirmada"] or r.loc["P1", "mean_d_pct"] > -0.05) and r.loc["P2", "confirmada"]
    adds = r.loc["P1", "confirmada"] and r.loc["P1", "mean_d_pct"] <= -0.05
    L.append(f"\n'no P||Cmax o ingrediente ativo e' destruir-reconstruir' (P1 sem efeito relevante E P2 confirmada): {'SIM' if active else 'NAO'}")
    L.append(f"'a busca local acrescenta ao IG' (P1 confirmada com d <= -0,05%): {'SIM' if adds else 'NAO'}")
    sec = []
    for lab, a, b in [("igls_ql vs ils_ql", "igls_ql", "ils_ql"), ("ig_random vs ils_random", "ig_random", "ils_random"), ("igls_ql vs ig_ql", "igls_ql", "ig_ql")]:
        ok = ~(w[a].isna() | w[b].isna())
        t = test(w.loc[ok, a], w.loc[ok, b])
        t["comparison"] = lab
        sec.append(t)
    L += ["\n--- secundarias ---", pd.DataFrame(sec)[["comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw"]].to_string(index=False)]
    rows = []
    for cls, g in w.groupby("instance"):
        ok = ~(g.igls_random.isna() | g.ig_random.isna())
        ok2 = ~(g.igls_random.isna() | g.ils_random.isna())
        rows.append(dict(instance=cls, d_ls_effect=rel(g.loc[ok, "igls_random"], g.loc[ok, "ig_random"]).mean(), d_vs_ils=rel(g.loc[ok2, "igls_random"], g.loc[ok2, "ils_random"]).mean(),
                         d_policy=rel(g.igls_ql, g.igls_random).mean()))
    L += ["\n--- por classe (d %): efeito da busca local sobre o IG, hibrido vs ILS, politica no hibrido ---", pd.DataFrame(rows).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada124_igls_p.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada124_igls_p_primary.csv", index=False)


if __name__ == "__main__":
    main()
