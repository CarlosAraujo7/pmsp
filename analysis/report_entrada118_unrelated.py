"""
analysis/report_entrada118_unrelated.py
=========================================
O protocolo mecanismo-vs-politica em R||Cmax. Diario: Entrada 118.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada118_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: 400 instancias (4 distribuicoes x 5 tamanhos x 20 reps) para os 12 metodos de results/unrelated_entrada118.csv; 160 delas (reps 0-7 de cada celula)
com LP e MILP (5 s e 60 s) em results/unrelated_exact_entrada118.csv. d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled,
diferencas nulas descartadas.
Familia confirmatoria (Holm sobre 4, alpha=0,05; 'confirmada' = p_holm<0,05 e d<0):
  M1  ur_igls_random vs ur_ils_random (mecanismo: o IG com busca local supera o ILS tambem em R||Cmax?)  [400 instancias]
  M2  ur_igls_ql vs ur_igls_random    (politica no IG com busca local)                               [400]
  M3  ur_ils_ql vs ur_ils_random      (politica no ILS)                                               [400]
  M4  ur_igls_ql vs MILP-60s          (valor pratico: o IG, poucos s, bate o incumbente do MILP em 60 s?) [160 menos os sem incumbente]
(d/n por tamanho escolhido em instancias de AJUSTE separadas; limiares do estado calibrados em instancias separadas; ver o runner)
Leituras pre-fixadas:
  * 'mecanismo importa' = M1 confirmada com |d| >= 0,3% (ordem de grandeza do P||Cmax);
  * 'politica ~ nula (como no P||Cmax)' = M2 e M3 com IC90 (bootstrap sobre instancias, 2000 reamostragens, seed 0) dentro de +-0,10%; 'estritamente' se +-0,05%;
    caso contrario o resultado e' reportado como diferenca do P||Cmax;
  * 'valor pratico' = M4 confirmada (d<0); instancias em que o MILP-60s nao devolve incumbente sao contadas a parte (IG vence por W.O.) e fora do teste pareado.
Secundarias (descritivas): por distribuicao x tamanho; folga das 4 estrategias (melhor fixa na amostra vs aleatoria; oraculo por instancia); ur_igls_ucb vs
ur_igls_random; politica no IG puro (ur_ig_ql vs ur_ig_random); efeito da busca local (ur_igls_random vs ur_ig_random); MILP-5s vs IG; IG vs mct; fracao com otimo provado;
GAP medio ao melhor limite inferior (max(LP, dual do MILP-60s)); contagem IG <= MILP-60s.
Uso: python -m analysis.report_entrada118_unrelated
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
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo),
                ci90_hi=float(hi), rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    d = pd.read_csv(ROOT / "results" / "unrelated_entrada118.csv")
    ex = pd.read_csv(ROOT / "results" / "unrelated_exact_entrada118.csv")
    w = d.pivot(index=KEY, columns="method", values="makespan").reset_index()
    lbc = d.drop_duplicates(KEY)[KEY + ["lb"]]
    w = w.merge(lbc, on=KEY)
    assert len(w) == 400 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    W = w.merge(ex, on=KEY, how="inner")
    assert len(W) == 160, len(W)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "R||Cmax: MECANISMO vs POLITICA -- teste pre-registrado, Entrada 118", "=" * 110, f"instancias: {len(w)} | com referencia exata: {len(W)}"]
    rows = []
    for pid, lab, a, b, D in [("M1", "ur_igls_random vs ur_ils_random", "ur_igls_random", "ur_ils_random", w), ("M2", "ur_igls_ql vs ur_igls_random", "ur_igls_ql", "ur_igls_random", w),
                              ("M3", "ur_ils_ql vs ur_ils_random", "ur_ils_ql", "ur_ils_random", w), ("M4", "ur_igls_ql vs MILP-60s", "ur_igls_ql", "milp60_inc", W)]:
        ok = ~(pd.isna(D[a]) | pd.isna(D[b]))
        t = test(D.loc[ok, a], D.loc[ok, b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 4) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    m1 = P[P.id == "M1"].iloc[0]
    L.append(f"\n'mecanismo importa' (M1 confirmada e |d|>=0,3%): {'SIM' if (m1.confirmada and abs(m1.mean_d_pct) >= 0.3) else 'NAO'}")
    for pid in ["M2", "M3"]:
        r = P[P.id == pid].iloc[0]
        L.append(f"politica {pid}: d = {r.mean_d_pct:+.4f}% IC90 [{r.ci90_lo:+.4f}, {r.ci90_hi:+.4f}] -> dentro de +-0,10%: {'SIM' if (r.ci90_lo > -0.10 and r.ci90_hi < 0.10) else 'NAO'}; dentro de +-0,05%: {'SIM' if (r.ci90_lo > -0.05 and r.ci90_hi < 0.05) else 'NAO'}")
    m4 = P[P.id == "M4"].iloc[0]
    L.append(f"'valor pratico' (M4 confirmada, IG < MILP-60s): {'SIM' if m4.confirmada else 'NAO'}")
    # secundarias
    sec = []
    for lab, a, b, D in [("ur_igls_ucb vs ur_igls_random", "ur_igls_ucb", "ur_igls_random", w), ("ur_ig_ql vs ur_ig_random (IG puro)", "ur_ig_ql", "ur_ig_random", w),
                         ("ur_igls_random vs ur_ig_random (efeito da busca local)", "ur_igls_random", "ur_ig_random", w), ("ur_igls_random vs mct", "ur_igls_random", "mct", w),
                         ("ur_ils_random vs mct", "ur_ils_random", "mct", w), ("ur_igls_ql vs MILP-5s", "ur_igls_ql", "milp5_inc", W),
                         ("ur_igls_random vs MILP-60s", "ur_igls_random", "milp60_inc", W), ("ur_ils_ql vs MILP-60s", "ur_ils_ql", "milp60_inc", W)]:
        ok = ~(pd.isna(D[a]) | pd.isna(D[b]))
        t = test(D.loc[ok, a], D.loc[ok, b])
        t["comparison"] = lab
        sec.append(t)
    L += ["\n--- SECUNDARIAS (sem Holm; instancias sem incumbente do MILP sao descartadas do pareamento) ---",
          pd.DataFrame(sec)[["comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw"]].to_string(index=False)]
    fix = ["ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3"]
    hr = []
    for (lo, hi, n, m), g in w.groupby(["lo", "hi", "n", "m"]):
        best = min(fix, key=lambda k: g[k].mean())
        hr.append(dict(dist=f"U[{lo},{hi}]", size=f"{n}x{m}", best=best[-1], d_best_vs_random=rel(g[best], g.ur_igls_random).mean(), d_oracle=rel(g[fix].min(axis=1), g.ur_igls_random).mean(),
                       d_ql=rel(g.ur_igls_ql, g.ur_igls_random).mean(), d_ils_vs_igls=rel(g.ur_ils_random, g.ur_igls_random).mean(), gap_igls_ql=100 * ((g.ur_igls_ql - g.lb) / g.lb).mean()))
    H = pd.DataFrame(hr)
    L += ["\n--- por celula: melhor estrategia fixa, folga (melhor fixa e oraculo vs aleatoria), efeito QL, ILS vs IG (d % medio), GAP do ig_ql vs LB classico ---", H.to_string(index=False)]
    L.append(f"\nfolga agregada: melhor fixa por celula (otimista) d = {H.d_best_vs_random.mean():+.4f}%; oraculo por instancia d = {H.d_oracle.mean():+.4f}%")
    W = W.copy()
    W["lb_best"] = np.maximum(W.lb_lp, W.milp60_dual)
    L.append("\n--- referencia exata (160 instancias) ---")
    L.append(f"MILP-60s: otimo provado em {100 * W.milp60_proven.mean():.1f}% | MILP-5s: {100 * W.milp5_proven.mean():.1f}% | incumbente ausente em 60s: {int(W.milp60_inc.isna().sum())}")
    for k in ["ur_igls_ql", "ur_igls_random", "ur_ig_ql", "ur_ils_ql", "ur_ils_random", "mct"]:
        L.append(f"  {k:14s} GAP ao melhor LB: {100 * ((W[k] - W.lb_best) / W.lb_best).mean():.3f}%  | makespan == melhor LB (otimo provado): {100 * (W[k] == W.lb_best).mean():.1f}%")
    L.append(f"  MILP-60s       GAP ao melhor LB: {100 * ((W.milp60_inc - W.lb_best) / W.lb_best).mean():.3f}%  | MILP-5s: {100 * ((W.milp5_inc - W.lb_best) / W.lb_best).mean():.3f}%")
    L.append(f"  IG+LS-QL <= MILP-60s em {100 * (W.ur_igls_ql <= W.milp60_inc.fillna(1e18)).mean():.1f}% das instancias; < em {100 * (W.ur_igls_ql < W.milp60_inc.fillna(1e18)).mean():.1f}% (incumbente ausente = vitoria do IG)")
    by = W.groupby(["lo", "hi", "n", "m"]).apply(lambda g: pd.Series(dict(milp60_proven=100 * g.milp60_proven.mean(), gap_milp60=100 * ((g.milp60_inc - g.lb_best) / g.lb_best).mean(),
                                                                        gap_igql=100 * ((g.ur_igls_ql - g.lb_best) / g.lb_best).mean(), igql_le_milp=100 * (g.ur_igls_ql <= g.milp60_inc.fillna(1e18)).mean())))
    L += ["\nPor celula (reps 0-7): % otimo provado pelo MILP-60s, GAP do MILP-60s e do IG-QL ao melhor LB, % IG-QL <= MILP-60s", by.to_string()]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada118_unrelated.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada118_unrelated_primary.csv", index=False)
    H.to_csv(ROOT / "results" / "entrada118_unrelated_by_cell.csv", index=False)


if __name__ == "__main__":
    main()
