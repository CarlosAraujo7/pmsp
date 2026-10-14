"""
analysis/report_entrada127_igcvls.py
======================================
A busca local melhora a receita (d ajustado + partida forte)? Diario: Entrada 127.

PRE-REGISTRO (congelado ANTES de executar os testes; hashes em results/entrada127_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: (S) 12 classes x 100 reps (interseccao com results.csv: 1.130 para as comparacoes com a melhor construtiva); (X') 1.500 instancias externas (Entrada 117/121).
igcv_seed (Entrada 121) = IG, destruicao `Random`, d/n validado, partida = melhor construtiva, T=1000; igcvls_seed = igcv_seed + descida (inter-troca, realocacao; ate 2 passos)
apos cada reconstrucao, T=1000; igcv_seed_eq = igcv_seed com T_eq = round(1000 K) (K = razao de CPU medida por classe em calibracao sequencial; ver o runner).
d = 100*(A-B)/B (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados.
Familia (Holm sobre 5, alpha=0,05; 'confirmada' = p_holm<0,05 e d<0):
  F1  igcvls_seed vs igcv_seed (S);   F2  igcvls_seed vs igcv_seed_eq (S; CPU igual);   F3  igcvls_seed vs igcv_seed (X');
  F4  igcvls_seed vs melhor construtiva (S);   F5  idem (X').
Leituras: 'a busca local melhora a receita' = F1 e F3 confirmadas; 'a vantagem sobrevive a CPU igual' = F2 confirmada.
Secundarias: otimo provado e GAP vs LB certificado (S); razao de CPU realizada (mediana por classe); % em que a busca melhora a melhor construtiva; por classe/familia.
Uso: python -m analysis.report_entrada127_igcvls
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

EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"


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
    res = pd.read_csv(ROOT / "results" / "results.csv")
    s102 = pd.read_csv(ROOT / "results" / "seeded_baselines_entrada102.csv")
    cv = pd.read_csv(ROOT / "results" / "igcv_seed_entrada121.csv")
    nl = pd.read_csv(ROOT / "results" / "igcvls_seed_entrada127.csv")
    ref = pd.read_csv(ROOT / "results" / "exact_reference_synthetic_entrada101.csv")[["instance", "rep", "lb_lp"]]
    keep = ["instance", "rep", "method", "makespan"]
    d = pd.concat([res[res.method.isin(["lpt", "multifit"])][keep], s102[keep], cv[keep], nl[keep]]).drop_duplicates(["instance", "rep", "method"])
    S = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().merge(ref, on=["instance", "rep"], how="left")
    S["bc"] = S[["lpt", "multifit", "ldm"]].min(axis=1, skipna=False)
    cpu = pd.concat([cv[["instance", "rep", "method", "cpu_time"]], nl[["instance", "rep", "method", "cpu_time"]]]).pivot(index=["instance", "rep"], columns="method", values="cpu_time")
    man = pd.read_csv(EXT / "manifest_ilsseed_entrada117.csv")
    old = pd.read_csv(EXT / "results_external_symmetry2025.csv").drop_duplicates(["member", "method"])
    new = pd.read_csv(EXT / "results_external_seeded_entrada102.csv").drop_duplicates(["member", "method"])
    cvx = pd.read_csv(EXT / "results_external_igcv_seed_entrada121.csv").drop_duplicates(["member", "method"])
    nlx = pd.read_csv(EXT / "results_external_igcvls_seed_entrada127.csv").drop_duplicates(["member", "method"])
    k2 = ["member", "method", "makespan"]
    dx = pd.concat([old[old.method.isin(["lpt", "multifit"])][k2], new[k2], cvx[k2], nlx[k2]])
    dx = dx[dx.member.isin(man.member)]
    X = dx.pivot(index="member", columns="method", values="makespan").reset_index().merge(man[["member", "family"]], on="member")
    X["bc"] = X[["lpt", "multifit", "ldm"]].min(axis=1, skipna=False)
    S1 = S.dropna(subset=["igcvls_seed", "igcv_seed", "igcv_seed_eq"])
    S2 = S1.dropna(subset=["bc"])
    X1 = X.dropna(subset=["igcvls_seed", "igcv_seed", "bc"])
    assert len(S1) == 1200 and len(S2) == 1130 and len(X1) == 1500, (len(S1), len(S2), len(X1))
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "A BUSCA LOCAL MELHORA A RECEITA? -- teste pre-registrado, Entrada 127", "=" * 110, f"S: {len(S1)} (com melhor construtiva: {len(S2)}) | X': {len(X1)}"]
    rows = []
    for pid, lab, a, b, D in [("F1", "igcvls_seed vs igcv_seed (S)", "igcvls_seed", "igcv_seed", S1), ("F2", "igcvls_seed vs igcv_seed_eq (S, CPU igual)", "igcvls_seed", "igcv_seed_eq", S1),
                              ("F3", "igcvls_seed vs igcv_seed (X')", "igcvls_seed", "igcv_seed", X1), ("F4", "igcvls_seed vs melhor construtiva (S)", "igcvls_seed", "bc", S2),
                              ("F5", "igcvls_seed vs melhor construtiva (X')", "igcvls_seed", "bc", X1)]:
        t = test(D[a], D[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 5) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    c = P.set_index("id").confirmada
    L.append(f"\n'a busca local melhora a receita' (F1 e F3): {'SIM' if (c['F1'] and c['F3']) else 'NAO'} | 'a vantagem sobrevive a CPU igual' (F2): {'SIM' if c['F2'] else 'NAO'}")
    Sl = S2.dropna(subset=["lb_lp"])
    L.append(f"S: otimo provado: igcvls_seed {100 * (Sl.igcvls_seed == Sl.lb_lp).mean():.1f}% | igcv_seed {100 * (Sl.igcv_seed == Sl.lb_lp).mean():.1f}% | igcv_seed_eq {100 * (Sl.igcv_seed_eq == Sl.lb_lp).mean():.1f}% | melhor construtiva {100 * (Sl.bc == Sl.lb_lp).mean():.1f}%")
    L.append("S: GAP medio vs LB certificado: " + ", ".join(f"{k} {100 * ((Sl[k] - Sl.lb_lp) / Sl.lb_lp).mean():.4f}%" for k in ["igcvls_seed", "igcv_seed", "igcv_seed_eq", "bc"]))
    L.append(f"S: busca melhora a melhor construtiva: igcvls_seed {100 * (S2.igcvls_seed < S2.bc).mean():.1f}% | igcv_seed {100 * (S2.igcv_seed < S2.bc).mean():.1f}% ; X': {100 * (X1.igcvls_seed < X1.bc).mean():.1f}% | {100 * (X1.igcv_seed < X1.bc).mean():.1f}%")
    r = (cpu.igcvls_seed / cpu.igcv_seed).dropna()
    L.append(f"razao de CPU realizada igcvls_seed/igcv_seed (execucao paralela): mediana {r.median():.2f}; igcv_seed_eq/igcvls_seed: {(cpu.igcv_seed_eq / cpu.igcvls_seed).median():.2f}")
    sec = []
    for g, sub in S1.groupby("instance"):
        t = test(sub.igcvls_seed, sub.igcv_seed)
        t2 = test(sub.igcvls_seed, sub.igcv_seed_eq)
        sec.append(dict(instance=g, d_vs_igcv=t["mean_d_pct"], d_vs_eq=t2["mean_d_pct"]))
    for g, sub in X1.groupby("family"):
        t = test(sub.igcvls_seed, sub.igcv_seed)
        sec.append(dict(instance=g, d_vs_igcv=t["mean_d_pct"], d_vs_eq=float("nan")))
    L += ["\n--- por classe / familia (d %) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada127_igcvls.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada127_igcvls_primary.csv", index=False)


if __name__ == "__main__":
    main()
