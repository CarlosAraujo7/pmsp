"""
analysis/report_entrada121_budget_law.py
==========================================
A politica aprendida so' vale com orcamento VINCULANTE? (P||Cmax com T in {100, 250, 1000}). Diario: Entrada 121.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada121_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: 12 classes x 100 reps (mesmas instancias de results.csv). T=100 e 250: results/budget_law_entrada121.csv; T=1000: ig_random (results.csv),
ig_ql_cal (state_recalibration_entrada98.csv) e ig_fix0..2 (signal_headroom_entrada104.csv). Politica = ig_ql_cal (estado informativo; primaria) e
ig_ql (estado original; secundaria). d = 100*(A-B)/B em makespan, negativo = A melhor; politica_d(T) = d de ig_ql_cal vs ig_random; folga_d(T) = d do ORACULO
por instancia min(ig_fix0, ig_fix1, ig_fix2) vs ig_random (cota superior do que um seletor de estrategias ganharia).
Familia confirmatoria (Holm sobre 4, alpha=0,05, Wilcoxon pareado bicaudal; 'confirmada' = p_holm<0,05 e a media no sentido indicado):
  B1  T=100 : politica_d < 0   (ig_ql_cal vs ig_random; 1.200 instancias; 'relevante' se d <= -0,05%)
  B2  T=250 : politica_d < 0   (idem)
  B3  politica_d(100) vs politica_d(1000): d por CLASSE (12 pares); a politica vale MAIS (mais negativa) com orcamento apertado
  B4  folga_d(100) vs folga_d(1000): d por CLASSE (12 pares); a folga e' MAIOR (mais negativa) com orcamento apertado
Leitura pre-fixada: 'orcamento vinculante torna a politica relevante' = B1 confirmada e relevante E B3 confirmada. Se B1 nao for confirmada, o nulo do P||Cmax
nao se deve a saturacao do orcamento e a diferenca para o R||Cmax (Entrada 118) precisa de outra explicacao.
Secundarias: igual para ig_ql (original); GAP medio e folga/politica por classe e T; fracao da folga capturada.
Uso: python -m analysis.report_entrada121_budget_law
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


def inst_test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def pair_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    p = float(sstats.wilcoxon(d).pvalue) if (d != 0).sum() >= 5 else float("nan")
    return dict(n=len(d), mean_diff=float(d.mean()), better=int((d < 0).sum()), worse=int((d > 0).sum()), p_raw=p)


def main() -> None:
    new = pd.read_csv(ROOT / "results" / "budget_law_entrada121.csv")
    res = pd.read_csv(ROOT / "results" / "results.csv")
    rec = pd.read_csv(ROOT / "results" / "state_recalibration_entrada98.csv")
    sig = pd.read_csv(ROOT / "results" / "signal_headroom_entrada104.csv")
    keep = ["instance", "rep", "method", "makespan"]
    t1000 = pd.concat([res[res.method.isin(["ig_random", "ig_ql"])][keep], rec[rec.method == "ig_ql_cal"][keep], sig[sig.method.isin(["ig_fix0", "ig_fix1", "ig_fix2"])][keep]])
    t1000 = t1000.assign(T=1000).drop_duplicates(["instance", "rep", "method"])
    allr = pd.concat([new[["instance", "rep", "method", "T", "makespan"]], t1000[["instance", "rep", "method", "T", "makespan"]]])
    W = {T: g.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index() for T, g in allr.groupby("T")}
    for T in (100, 250):
        assert len(W[T]) == 1200 and W[T].isna().sum().sum() == 0, (T, len(W[T]), W[T].isna().sum().to_dict())
    w1000 = W[1000].dropna(subset=["ig_random", "ig_ql_cal", "ig_fix0", "ig_fix1", "ig_fix2"])
    assert len(w1000) == 1200, len(w1000)
    W[1000] = w1000
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "LEI DO ORCAMENTO: politica e folga vs T (P||Cmax) -- teste pre-registrado, Entrada 121", "=" * 110]

    def class_d(T, pol="ig_ql_cal"):
        w = W[T]
        out_pol, out_hr = {}, {}
        for cls, g in w.groupby("instance"):
            ok = ~g[pol].isna()
            out_pol[cls] = rel(g.loc[ok, pol], g.loc[ok, "ig_random"]).mean()
            out_hr[cls] = rel(g[["ig_fix0", "ig_fix1", "ig_fix2"]].min(axis=1), g.ig_random).mean()
        return pd.Series(out_pol), pd.Series(out_hr)

    P = {}
    for T in (100, 250, 1000):
        w = W[T]
        ok = ~w.ig_ql_cal.isna()
        P[T] = inst_test(w.loc[ok, "ig_ql_cal"], w.loc[ok, "ig_random"])
    rows = []
    for pid, T in [("B1", 100), ("B2", 250)]:
        t = dict(P[T])
        t.update(id=pid, comparison=f"ig_ql_cal vs ig_random @T={T}")
        rows.append(t)
    cp = {T: class_d(T) for T in (100, 250, 1000)}
    t3 = pair_test(cp[100][0].sort_index(), cp[1000][0].sort_index())
    t3.update(id="B3", comparison="politica_d por classe: T=100 vs T=1000", p_raw=t3["p_raw"])
    t4 = pair_test(cp[100][1].sort_index(), cp[1000][1].sort_index())
    t4.update(id="B4", comparison="folga_d por classe: T=100 vs T=1000")
    rows += [t3, t4]
    F = pd.DataFrame(rows)
    h = holm_bonferroni(F.p_raw.fillna(1.0).tolist(), alpha=0.05)
    F["p_holm"], F["reject"] = h["p_adjusted"], h["reject"]
    F["confirmada"] = F.reject & np.where(F.id.isin(["B1", "B2"]), F.mean_d_pct < 0, F.mean_diff < 0)
    F["relevante"] = F.confirmada & F.id.isin(["B1", "B2"]) & (F.mean_d_pct <= -0.05)
    L += ["\n--- PRIMARIA (Holm sobre 4) ---", F[["id", "comparison", "n", "mean_d_pct", "mean_diff", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "relevante"]].to_string(index=False)]
    b1 = F[F.id == "B1"].iloc[0]
    b3 = F[F.id == "B3"].iloc[0]
    L.append(f"\n'orcamento vinculante torna a politica relevante' (B1 confirmada e relevante E B3 confirmada): {'SIM' if (b1.relevante and b3.confirmada) else 'NAO'}")
    L.append("\n--- politica_d (ig_ql_cal vs ig_random, pooled) e folga (oraculo das 3 estrategias) por T ---")
    for T in (100, 250, 1000):
        w = W[T]
        orc = rel(w[["ig_fix0", "ig_fix1", "ig_fix2"]].min(axis=1), w.ig_random)
        ok = ~w.ig_ql.isna() if "ig_ql" in w else None
        s = f"T={T:4d}: politica ig_ql_cal d = {P[T]['mean_d_pct']:+.4f}% IC90 [{P[T]['ci90_lo']:+.4f}, {P[T]['ci90_hi']:+.4f}] | folga oraculo d = {orc.mean():+.4f}%"
        if "ig_ql" in w:
            okq = ~w.ig_ql.isna()
            tq = inst_test(w.loc[okq, "ig_ql"], w.loc[okq, "ig_random"])
            s += f" | politica ig_ql (original) d = {tq['mean_d_pct']:+.4f}% IC90 [{tq['ci90_lo']:+.4f}, {tq['ci90_hi']:+.4f}] p={tq['p_raw']:.3g}"
        L.append(s)
    tab = pd.DataFrame({f"pol_T{T}": cp[T][0] for T in (100, 250, 1000)}).join(pd.DataFrame({f"folga_T{T}": cp[T][1] for T in (100, 250, 1000)}))
    L += ["\n--- por classe: politica_d e folga_d (%) por T ---", tab.to_string()]
    g = {T: {m_: 100 * ((W[T][m_] / 1).mean()) for m_ in ["ig_random", "ig_ql_cal"]} for T in (100, 250, 1000)}
    L.append("\nmakespan medio ig_random por T: " + ", ".join(f"T={T}: {W[T].ig_random.mean():.2f}" for T in (100, 250, 1000)))
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada121_budget_law.txt").write_text(text, encoding="utf-8")
    F.to_csv(ROOT / "results" / "entrada121_budget_law_primary.csv", index=False)
    tab.to_csv(ROOT / "results" / "entrada121_budget_law_by_class.csv")


if __name__ == "__main__":
    main()
