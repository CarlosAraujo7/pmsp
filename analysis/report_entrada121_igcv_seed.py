"""
analysis/report_entrada121_igcv_seed.py
=========================================
IG com d AJUSTADO por validacao + partida pela melhor construtiva (`igcv_seed`) vs MULTIFIT, LDM e IG-QL+. Diario: Entrada 121.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada121_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: (S) 12 classes x 100 reps (interseccao com results.csv: 1.130); (X') 1.500 instancias externas (B 400, C 400, E 400, F 300) do manifest_ilsseed_entrada117.csv, d/n por
familia escolhido na CALIBRACAO nao vista da Entrada 115. igcv_seed = IG, destruicao `Random`, d/n validado, partida = min(LPT, MULTIFIT, LDM), T=1000.
d = 100*(A-B)/B (negativo: A melhor); Wilcoxon pareado bicaudal pooled; diferencas nulas descartadas.
Familia confirmatoria (Holm sobre 8 = 4 comparacoes x 2 conjuntos, alpha=0,05; 'confirmada' = p_holm<0,05 e d<0):
  E1  igcv_seed vs ig_ql_seed            (o ajuste de d acrescenta algo ao IG+ padrao d=0,3n com selecao QL de estrategias?)
  E2  igcv_seed vs MULTIFIT
  E3  igcv_seed vs LDM
  E4  igcv_seed vs melhor construtiva    (d<=0 por construcao; testa se a busca agrega valor)
Leitura pre-fixada: 'o ajuste de d melhora o IG+' = E1 confirmada nos dois conjuntos; 'o IG ajustado supera as construtivas' = E2, E3 e E4 confirmadas nos dois.
Secundarias: por classe/familia; % de instancias em que a busca melhora a melhor construtiva (igcv_seed vs ig_ql_seed); otimalidade provada (so' S, vs LB arc-flow);
GAP ao LB certificado.
Uso: python -m analysis.report_entrada121_igcv_seed
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


def load_s():
    res = pd.read_csv(ROOT / "results" / "results.csv")
    s102 = pd.read_csv(ROOT / "results" / "seeded_baselines_entrada102.csv")
    cv = pd.read_csv(ROOT / "results" / "igcv_seed_entrada121.csv")
    ref = pd.read_csv(ROOT / "results" / "exact_reference_synthetic_entrada101.csv")[["instance", "rep", "lb_lp"]]
    keep = ["instance", "rep", "method", "makespan"]
    d = pd.concat([res[res.method.isin(["lpt", "multifit"])][keep], s102[keep], cv[keep]]).drop_duplicates(["instance", "rep", "method"])
    w = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().merge(ref, on=["instance", "rep"], how="left")
    w["bc"] = w[["lpt", "multifit", "ldm"]].min(axis=1, skipna=False)
    return w.rename(columns={"instance": "group"})


def load_x():
    man = pd.read_csv(EXT / "manifest_ilsseed_entrada117.csv")
    old = pd.read_csv(EXT / "results_external_symmetry2025.csv").drop_duplicates(["member", "method"])
    new = pd.read_csv(EXT / "results_external_seeded_entrada102.csv").drop_duplicates(["member", "method"])
    cv = pd.read_csv(EXT / "results_external_igcv_seed_entrada121.csv").drop_duplicates(["member", "method"])
    keep = ["member", "method", "makespan"]
    d = pd.concat([old[old.method.isin(["lpt", "multifit"])][keep], new[keep], cv[keep]])
    d = d[d.member.isin(man.member)]
    w = d.pivot(index="member", columns="method", values="makespan").reset_index().merge(man[["member", "family"]], on="member")
    w["bc"] = w[["lpt", "multifit", "ldm"]].min(axis=1, skipna=False)
    return w.rename(columns={"family": "group"})


def main() -> None:
    S, X = load_s(), load_x()
    S = S.dropna(subset=["igcv_seed", "ig_ql_seed", "multifit", "ldm", "bc"])
    X = X.dropna(subset=["igcv_seed", "ig_ql_seed", "multifit", "ldm", "bc"])
    assert len(S) == 1130 and len(X) == 1500, (len(S), len(X))
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "IG AJUSTADO + SEMEADO (igcv_seed) -- teste pre-registrado, Entrada 121", "=" * 110, f"S: {len(S)} | X': {len(X)} {X.groupby('group').size().to_dict()}"]
    rows = []
    for name, D in [("S", S), ("X", X)]:
        for pid, lab, b in [("E1", "igcv_seed vs ig_ql_seed", "ig_ql_seed"), ("E2", "igcv_seed vs multifit", "multifit"), ("E3", "igcv_seed vs ldm", "ldm"), ("E4", "igcv_seed vs best constructive", "bc")]:
            t = test(D.igcv_seed, D[b])
            t.update(set=name, id=pid, comparison=lab)
            rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 8) ---", P[["set", "id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    e1 = bool(P[P.id == "E1"].confirmada.all())
    e234 = bool(P[P.id.isin(["E2", "E3", "E4"])].confirmada.all())
    L.append(f"\n'o ajuste de d melhora o IG+' (E1 confirmada nos dois conjuntos): {'SIM' if e1 else 'NAO'}")
    L.append(f"'o IG ajustado supera as construtivas' (E2, E3, E4 confirmadas nos dois): {'SIM' if e234 else 'NAO'}")
    sec = []
    for name, D in [("S", S), ("X", X)]:
        for g, sub in D.groupby("group"):
            for lab, b in [("E1", "ig_ql_seed"), ("E2", "multifit"), ("E3", "ldm")]:
                t = test(sub.igcv_seed, sub[b])
                t.update(set=name, group=g, id=lab)
                sec.append(t)
    Sx = pd.DataFrame(sec)
    L += ["\n--- secundaria por classe (S) / familia (X') ---", Sx[["set", "group", "id", "n", "better", "worse", "tie", "mean_d_pct", "p_raw"]].to_string(index=False)]
    for name, D in [("S", S), ("X", X)]:
        L.append(f"{name}: busca melhora a melhor construtiva: igcv_seed {100 * (D.igcv_seed < D.bc).mean():.1f}% | ig_ql_seed {100 * (D.ig_ql_seed < D.bc).mean():.1f}% (n={len(D)})")
    Sl = S.dropna(subset=["lb_lp"])
    L.append(f"S: otimo provado (== LB arc-flow): igcv_seed {100 * (Sl.igcv_seed == Sl.lb_lp).mean():.1f}% | ig_ql_seed {100 * (Sl.ig_ql_seed == Sl.lb_lp).mean():.1f}% | multifit {100 * (Sl.multifit == Sl.lb_lp).mean():.1f}% | ldm {100 * (Sl.ldm == Sl.lb_lp).mean():.1f}% | melhor construtiva {100 * (Sl.bc == Sl.lb_lp).mean():.1f}%")
    L.append("S: GAP medio vs LB certificado: " + ", ".join(f"{k} {100 * ((Sl[k] - Sl.lb_lp) / Sl.lb_lp).mean():.4f}%" for k in ["igcv_seed", "ig_ql_seed", "multifit", "ldm", "bc"]))
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada121_igcv_seed.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada121_igcv_seed_primary.csv", index=False)


if __name__ == "__main__":
    main()
