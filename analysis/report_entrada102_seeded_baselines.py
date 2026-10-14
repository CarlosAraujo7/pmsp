"""
analysis/report_entrada102_seeded_baselines.py
================================================
Baselines mais fortes (LDM; IG partindo da melhor construtiva). Diario: Entrada 102.

PRE-REGISTRO (congelado ANTES de executar ldm / ig_ql_seed / ig_random_seed; hashes md5 em
results/entrada102_prereg_hashes.txt)
-------------------------------------------------------------------------------------------
Dados: (S) sintetico: 12 classes x 100 repeticoes (rand_20j_4m: 30, ver EMENDA 1) = 1.130 instancias (results.csv +
results/seeded_baselines_entrada102.csv); (X) externo: as 6.780 instancias do manifest.csv
(results_external_symmetry2025.csv + results_external_seeded_entrada102.csv, 1 linha por
(member, metodo); 52 pares duplicados NAO sao removidos -- sao instancias distintas do
manifest). Orcamento T=1000, seeds pareados (mesmo seed em ig_ql, ig_ql_seed e ig_random_seed).

Estatistica: diferenca relativa de makespan por instancia, d = 100*(Cmax_A - Cmax_B)/Cmax_B
(%; d<0 => A melhor). Teste: Wilcoxon signed-rank pareado bicaudal, pooled sobre o conjunto
(S ou X), diferencas nulas descartadas.

Familia confirmatoria (4 comparacoes x 2 conjuntos = 8 testes; Holm, alpha=0,05; "confirmada" =
p_holm<0,05 E d_medio<0 no sentido previsto):
  P1  ig_ql_seed  vs ig_random_seed   (efeito da POLITICA Q-Learning com partida forte; previsao:
                                       efeito minusculo, |d_medio| < 0,05%, mesmo se significativo)
  P2  ig_ql_seed  vs ig_ql            (partida forte ajuda o metodo proposto? previsao: d<0)
  P3  ig_ql_seed  vs min(LPT,MULTIFIT,LDM)  (o IG agrega valor sobre a melhor construtiva? previsao: d<0;
                                       por construcao d<=0, o teste mede se a busca melhora a partida)
  P4  ig_ql       vs ldm              (o IG-QL original supera a construtiva LDM omitida? previsao: d<0)
Margem de equivalencia pratica: |d_medio| < 0,05% (reportada junto de cada teste, sem alterar o
veredito formal). Magnitude absoluta (pp de gap) e contagens melhor/pior/empate sempre reportadas.

Secundarias (descritivas; Holm dentro de cada bloco): os mesmos testes por classe (S) e por familia
(X); fracao de instancias com solucao PROVADAMENTE otima (makespan == LB arc-flow certificado,
so' em S, via results/exact_reference_synthetic_entrada101.csv) para ldm, multifit, ig_ql, ig_ql_seed;
razoes de CPU; gap medio de cada construtiva (lpt, multifit, ldm).

Veredito pre-registrado (apresentado no artigo qualquer que seja): reporta-se quantas das 8 hipoteses
sao confirmadas; P4 ou P2 nao confirmadas NAO sao omitidas -- sao resultado.

Uso:  python -m analysis.report_entrada102_seeded_baselines [--smoke]
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
MARGIN = 0.05


def load_synthetic() -> pd.DataFrame:
    res = pd.read_csv(ROOT / "results" / "results.csv")
    new = pd.read_csv(ROOT / "results" / "seeded_baselines_entrada102.csv")
    keep = ["instance", "rep", "method", "n", "m", "lb", "makespan", "gap_pct", "cpu_time"]
    d = pd.concat([res[res.method.isin(["lpt", "multifit", "ig_ql"])][keep], new[keep]])
    d = d.drop_duplicates(["instance", "rep", "method"])
    w = d.pivot(index=["instance", "rep"], columns="method", values="makespan")
    cpu = d.pivot(index=["instance", "rep"], columns="method", values="cpu_time").add_prefix("cpu_")
    info = d.drop_duplicates(["instance", "rep"]).set_index(["instance", "rep"])[["n", "m", "lb"]]
    out = info.join(w).join(cpu).reset_index().rename(columns={"instance": "group"})
    ref = pd.read_csv(ROOT / "results" / "exact_reference_synthetic_entrada101.csv")[["instance", "rep", "lb_lp"]]
    return out.merge(ref.rename(columns={"instance": "group"}), on=["group", "rep"], how="left")


def load_external() -> pd.DataFrame:
    man = pd.read_csv(EXT / "manifest.csv")
    old = pd.read_csv(EXT / "results_external_symmetry2025.csv").drop_duplicates(["member", "method"])
    new = pd.read_csv(EXT / "results_external_seeded_entrada102.csv").drop_duplicates(["member", "method"])
    keep = ["member", "family", "method", "n", "m", "lb", "makespan", "gap_pct", "cpu_time"]
    d = pd.concat([old[old.method.isin(["lpt", "multifit", "ig_ql"])][keep], new[keep]])
    d = d[d.member.isin(man.member)]
    w = d.pivot(index="member", columns="method", values="makespan")
    cpu = d.pivot(index="member", columns="method", values="cpu_time").add_prefix("cpu_")
    info = d.drop_duplicates("member").set_index("member")[["family", "n", "m", "lb"]]
    return info.join(w).join(cpu).reset_index().rename(columns={"family": "group"})


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    gap_pp = float(np.mean(np.asarray(x, float) - np.asarray(y, float) ))
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()),
               mean_d_pct=float(d.mean()), mean_diff_makespan=gap_pp, rank_biserial=rank_biserial_paired(x, y),
               p_raw=p, equivalent=bool(abs(d.mean()) < MARGIN))


def comparisons(df: pd.DataFrame):
    best_c = df[["lpt", "multifit", "ldm"]].min(axis=1)
    return [("P1", "ig_ql_seed vs ig_random_seed", df.ig_ql_seed, df.ig_random_seed),
            ("P2", "ig_ql_seed vs ig_ql", df.ig_ql_seed, df.ig_ql),
            ("P3", "ig_ql_seed vs best constructive", df.ig_ql_seed, best_c),
            ("P4", "ig_ql vs ldm", df.ig_ql, df.ldm)]


def main() -> None:
    S, X = load_synthetic(), load_external()
    # EMENDA 1 (antes de ver qualquer resultado): rand_20j_4m tem so' 30 repeticoes de lpt/multifit/ig_ql
    # em results.csv (convencao do projeto); usa-se a interseccao -> |S| = 1.130, nao 1.200.
    S = S.dropna(subset=["lpt", "multifit", "ig_ql", "ldm", "ig_ql_seed", "ig_random_seed"]).reset_index(drop=True)
    assert len(S) == 1130 and len(X) == 6780, (len(S), len(X))
    for name, df in [("S", S), ("X", X)]:
        miss = df[["lpt", "multifit", "ldm", "ig_ql", "ig_ql_seed", "ig_random_seed"]].isna().sum()
        assert miss.sum() == 0, f"{name}: dados faltando\n{miss}"
    L = ["=" * 110, "BASELINES MAIS FORTES (LDM; IG com partida forte) -- teste pre-registrado, Entrada 102", "=" * 110]
    rows = []
    for name, df in [("S", S), ("X", X)]:
        for pid, label, a, b in comparisons(df):
            t = test(a, b); t.update(set=name, id=pid, comparison=label); rows.append(t)
    prim = pd.DataFrame(rows)
    h = holm_bonferroni(prim.p_raw.tolist(), alpha=0.05)
    prim["p_holm"], prim["reject"] = h["p_adjusted"], h["reject"]
    prim["confirmada"] = prim.reject & (prim.mean_d_pct < 0)
    pd.set_option("display.width", 230); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    cols = ["set", "id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "rank_biserial", "p_raw", "p_holm",
            "confirmada", "equivalent"]
    L += ["\n--- PRIMARIA (Holm sobre 8; d_medio<0 => primeiro metodo melhor; rank_biserial>0 => primeiro PIOR) ---",
          prim[cols].to_string(index=False)]
    L.append(f"\nCONFIRMADAS: {int(prim.confirmada.sum())}/8 -> " + ", ".join(
        f"{r.set}-{r.id}" for r in prim.itertuples() if r.confirmada))
    L.append("NAO confirmadas: " + (", ".join(f"{r.set}-{r.id}" for r in prim.itertuples() if not r.confirmada) or "nenhuma"))

    sec = []
    for name, df in [("S", S), ("X", X)]:
        for g, sub in df.groupby("group"):
            for pid, label, a, b in comparisons(sub):
                t = test(a, b); t.update(set=name, group=g, id=pid); sec.append(t)
    sec = pd.DataFrame(sec)
    for name in ["S", "X"]:
        idx = sec.index[sec.set == name]
        hh = holm_bonferroni(sec.loc[idx, "p_raw"].fillna(1.0).tolist(), alpha=0.05)
        sec.loc[idx, "p_holm"] = hh["p_adjusted"]
    L += ["\n--- SECUNDARIA por classe (S) / familia (X); Holm dentro de cada conjunto ---",
          sec[["set", "group", "id", "n", "better", "worse", "tie", "mean_d_pct", "p_raw", "p_holm"]].to_string(index=False)]

    L.append("\n--- gap medio (%) em relacao ao LB classico e CPU (s) ---")
    for name, df in [("S", S), ("X", X)]:
        for k in ["lpt", "multifit", "ldm", "ig_ql", "ig_random_seed", "ig_ql_seed"]:
            L.append(f"  {name} {k:15s} gap={100 * ((df[k] - df.lb) / df.lb).mean():.4f}  cpu medio={df['cpu_' + k].mean():.4f}")
    L.append("\n--- provadamente otimo (makespan == LB arc-flow certificado), conjunto S ---")
    for k in ["lpt", "multifit", "ldm", "ig_ql", "ig_random_seed", "ig_ql_seed"]:
        L.append(f"  {k:15s} {100 * (S[k] == S.lb_lp).mean():.1f}%   GAP vs LB certificado = {100 * ((S[k] - S.lb_lp) / S.lb_lp).mean():.4f}")
    L.append(f"  melhor entre {{lpt,multifit,ldm}}: {100 * (S[['lpt', 'multifit', 'ldm']].min(axis=1) == S.lb_lp).mean():.1f}%")
    L.append("\n--- ldm vs multifit, vs lpt (d relativo medio %, melhor/pior/empate; conjuntos S e X) ---")
    for name, df in [("S", S), ("X", X)]:
        for ref in ["lpt", "multifit"]:
            t = test(df.ldm, df[ref])
            L.append(f"  {name} ldm vs {ref:8s}: d={t['mean_d_pct']:.4f}% melhor={t['better']} pior={t['worse']} empate={t['tie']} p={t['p_raw']:.3g}")
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada102_seeded_baselines.txt").write_text(text, encoding="utf-8")
    prim.to_csv(ROOT / "results" / "entrada102_seeded_primary.csv", index=False)
    sec.to_csv(ROOT / "results" / "entrada102_seeded_secondary.csv", index=False)


if __name__ == "__main__":
    main()
