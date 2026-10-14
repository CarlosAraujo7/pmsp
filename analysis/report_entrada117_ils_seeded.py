"""
analysis/report_entrada117_ils_seeded.py
==========================================
IG+ vs ILS+ (ambos partindo da melhor construtiva). Diario: Entrada 117.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada117_prereg_hashes.txt)
------------------------------------------------------------------------------------------
Dados: (S) sintetico, 12 classes x 100 reps (results/ils_seeded_entrada117.csv + seeded_baselines_entrada102.csv + results.csv);
(X') 1.500 instancias externas (manifest_ilsseed_entrada117.csv; B 400, C 400, E 400, F 300) com ig_ql_seed/ig_random_seed (Entrada 102), lpt/multifit
(bateria externa) e ldm (Entrada 102). d = 100*(A-B)/B (negativo: A melhor); Wilcoxon pareado bicaudal pooled, diferencas nulas descartadas.
Familia confirmatoria (Holm sobre 6 = 3 comparacoes x 2 conjuntos, alpha=0,05; 'confirmada' = p_holm<0,05 e d no sentido indicado):
  I1  ils_ql_seed vs ils_random_seed   previsao: efeito da politica no ILS com partida forte e' DESPREZIVEL (|d|<0,05%), sem direcao fixa
  I2  ig_ql_seed  vs ils_ql_seed       (o mecanismo IG e' melhor que o ILS com partida forte igual?) bicaudal; previsao: IG melhor em S (tempos grossos) e
                                       dependente da familia em X'
  I3  ils_ql_seed vs min(LPT,MULTIFIT,LDM)   previsao: d<=0 por construcao (o ILS nunca piora a semente); testa se ele agrega valor
Descritivas: por classe (S) e por familia (X'); % em que a busca melhora a melhor construtiva (IG+ e ILS+); razoes de CPU (execucao paralela, indicativas).
Uso: python -m analysis.report_entrada117_ils_seeded
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
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo),
                ci90_hi=float(hi), rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def load_s():
    res = pd.read_csv(ROOT / "results" / "results.csv")
    s102 = pd.read_csv(ROOT / "results" / "seeded_baselines_entrada102.csv")
    s117 = pd.read_csv(ROOT / "results" / "ils_seeded_entrada117.csv")
    keep = ["instance", "rep", "method", "makespan"]
    d = pd.concat([res[res.method.isin(["lpt", "multifit"])][keep], s102[keep], s117[keep]]).drop_duplicates(["instance", "rep", "method"])
    w = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    w["bc"] = w[["lpt", "multifit", "ldm"]].min(axis=1, skipna=False)
    return w.rename(columns={"instance": "group"})


def load_x():
    man = pd.read_csv(EXT / "manifest_ilsseed_entrada117.csv")
    old = pd.read_csv(EXT / "results_external_symmetry2025.csv").drop_duplicates(["member", "method"])
    new = pd.read_csv(EXT / "results_external_seeded_entrada102.csv").drop_duplicates(["member", "method"])
    ils = pd.read_csv(EXT / "results_external_ils_seeded_entrada117.csv").drop_duplicates(["member", "method"])
    keep = ["member", "method", "makespan"]
    d = pd.concat([old[old.method.isin(["lpt", "multifit"])][keep], new[keep], ils[keep]])
    d = d[d.member.isin(man.member)]
    w = d.pivot(index="member", columns="method", values="makespan").reset_index().merge(man[["member", "family"]], on="member")
    w["bc"] = w[["lpt", "multifit", "ldm"]].min(axis=1, skipna=False)
    return w.rename(columns={"family": "group"})


def main() -> None:
    S, X = load_s(), load_x()
    need = ["ils_ql_seed", "ils_random_seed", "ig_ql_seed"]
    S = S.dropna(subset=need)
    X = X.dropna(subset=need + ["bc"])
    assert len(S) == 1200 and len(X) == 1500, (len(S), len(X))
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "IG+ vs ILS+ (partida forte nos dois) -- teste pre-registrado, Entrada 117", "=" * 110, f"S: {len(S)} instancias | X': {len(X)} {X.groupby('group').size().to_dict()}"]
    rows = []
    for name, D in [("S", S), ("X", X)]:
        Dp = D.dropna(subset=["bc"])
        for pid, lab, a, b, DD in [("I1", "ils_ql_seed vs ils_random_seed", "ils_ql_seed", "ils_random_seed", D), ("I2", "ig_ql_seed vs ils_ql_seed", "ig_ql_seed", "ils_ql_seed", D),
                                   ("I3", "ils_ql_seed vs best constructive", "ils_ql_seed", "bc", Dp)]:
            t = test(DD[a], DD[b])
            t.update(set=name, id=pid, comparison=lab)
            rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 6; 'confirmada' = p_holm<0,05 e d<0; para I1 e I2 o sinal e' informativo, ver 'reject') ---",
          P[["set", "id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "reject", "confirmada"]].to_string(index=False)]
    sec = []
    for name, D in [("S", S), ("X", X)]:
        for g, sub in D.groupby("group"):
            for lab, a, b in [("I1", "ils_ql_seed", "ils_random_seed"), ("I2", "ig_ql_seed", "ils_ql_seed")]:
                t = test(sub[a], sub[b])
                t.update(set=name, group=g, id=lab)
                sec.append(t)
    Sx = pd.DataFrame(sec)
    L += ["\n--- secundaria por classe (S) / familia (X') ---", Sx[["set", "group", "id", "n", "better", "worse", "tie", "mean_d_pct", "p_raw"]].to_string(index=False)]
    for name, D in [("S", S), ("X", X)]:
        Dp = D.dropna(subset=["bc"])
        L.append(f"{name}: busca melhora a melhor construtiva: IG+ {100 * (Dp.ig_ql_seed < Dp.bc).mean():.1f}% | ILS+ {100 * (Dp.ils_ql_seed < Dp.bc).mean():.1f}% (n={len(Dp)})")
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada117_ils_seeded.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada117_ils_seeded_primary.csv", index=False)


if __name__ == "__main__":
    main()
