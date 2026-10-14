"""
analysis/report_entrada108_new_distributions.py
=================================================
WP6: generalidade fora de U[1,100] e regiao dificil n/m = 2-3. Diario: Entrada 108.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada108_prereg_hashes.txt)
------------------------------------------------------------------------------------------
Dados: 6 distribuicoes x 5 tamanhos x 30 repeticoes = 900 instancias NOVAS (ver o runner). d = 100*(Cmax_A-Cmax_B)/Cmax_B.

H1 (equivalencia da politica; sem p-valor): IC90 percentil (2000 reamostragens, seed 0) do d medio pooled
    dentro de (-0,05; +0,05) para (a) ig_ql vs ig_random e (b) ig_ql_seed vs ig_random_seed.
    CONFIRMADA se (a) e (b) ambos equivalentes; por distribuicao: descritiva.
Familia confirmatoria (Holm sobre 4, alpha=0,05, Wilcoxon pareado bicaudal, "confirmada" = p_holm<0,05 e d no sentido previsto):
  H2a  despacho a priori (dup_ratio > 0,1 -> ig_ql, senao ils_ql; limiar fixo da Entrada 61, NUNCA retunado) vs ig_ql sempre (d<0)
  H2b  despacho a priori vs ils_ql sempre (d<0)
  H3a  entre instancias com dup_ratio > 0,1: ig_random vs ils_random, previsto d<0 (IG melhor com granularidade grossa)
  H3b  entre instancias com dup_ratio <= 0,1: ig_random vs ils_random, previsto d>0 (ILS melhor com granularidade fina)
    (H3b "confirmada" exige d>0). Se um subconjunto tiver < 20 instancias, o teste e' reportado como INCONCLUSIVO
    e entra na familia com p=1.
Descritivas: provavelmente otimo (makespan == LB arc-flow certificado) por metodo e por distribuicao/tamanho; ig_ql_seed vs
melhor das construtivas; GAP por celula; regiao dificil (n/m <= 3).

Uso: python -m analysis.report_entrada108_new_distributions
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

MARGIN = 0.05
THR = 0.1


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()),
                ci90_lo=float(lo), ci90_hi=float(hi), rank_biserial=rank_biserial_paired(x, y) if len(d) else float("nan"),
                p_raw=p, equivalent=bool(lo > -MARGIN and hi < MARGIN))


def main() -> None:
    d = pd.read_csv(ROOT / "results" / "new_distributions_entrada108.csv")
    ex = pd.read_csv(ROOT / "results" / "new_distributions_entrada108_exact.csv")
    key = ["dist", "n", "m", "rep"]
    w = d.pivot(index=key, columns="method", values="makespan").reset_index()
    info = d.drop_duplicates(key)[key + ["dup_ratio", "lb"]]
    w = w.merge(info, on=key).merge(ex[key + ["lb_lp", "proven"]], on=key)
    assert len(w) == 900 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    w["chose_ig"] = w.dup_ratio > THR
    w["dispatch"] = np.where(w.chose_ig, w.ig_ql, w.ils_ql)
    pd.set_option("display.width", 230); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "GENERALIDADE: NOVAS DISTRIBUICOES E REGIAO DIFICIL -- teste pre-registrado, Entrada 108", "=" * 110,
         f"instancias: {len(w)} | dup_ratio>0,1 em {100 * w.chose_ig.mean():.1f}%"]
    # H1
    h1 = []
    for lab, a, b in [("(a) ig_ql vs ig_random", w.ig_ql, w.ig_random), ("(b) ig_ql_seed vs ig_random_seed", w.ig_ql_seed, w.ig_random_seed)]:
        t = test(a, b); t.update(test=lab); h1.append(t)
    H1 = pd.DataFrame(h1)
    L += ["\n--- H1: equivalencia da politica (IC90 dentro de +-0,05%) ---",
          H1[["test", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "equivalent"]].to_string(index=False),
          f"H1 {'CONFIRMADA' if H1.equivalent.all() else 'NAO confirmada'}"]
    rows = []
    for dist, g in w.groupby("dist"):
        for lab, a, b in [("ig_ql v ig_random", g.ig_ql, g.ig_random), ("seed: ql v random", g.ig_ql_seed, g.ig_random_seed)]:
            t = test(a, b); t.update(dist=dist, test=lab); rows.append(t)
    L += ["\n(por distribuicao, descritiva)", pd.DataFrame(rows)[["dist", "test", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "equivalent"]].to_string(index=False)]
    # familia
    fam = []
    def add(pid, lab, a, b, expect_neg):
        if len(a) < 20:
            fam.append(dict(id=pid, comparison=lab, n=len(a), better=0, worse=0, tie=0, mean_d_pct=float("nan"), ci90_lo=float("nan"),
                            ci90_hi=float("nan"), rank_biserial=float("nan"), p_raw=1.0, expect_neg=expect_neg, inconclusive=True))
        else:
            t = test(a, b); t.update(id=pid, comparison=lab, expect_neg=expect_neg, inconclusive=False); fam.append(t)
    add("H2a", "dispatch vs ig_ql sempre", w.dispatch, w.ig_ql, True)
    add("H2b", "dispatch vs ils_ql sempre", w.dispatch, w.ils_ql, True)
    hi, lo = w[w.chose_ig], w[~w.chose_ig]
    add("H3a", "ig_random vs ils_random | dup>0,1", hi.ig_random, hi.ils_random, True)
    add("H3b", "ig_random vs ils_random | dup<=0,1", lo.ig_random, lo.ils_random, False)
    F = pd.DataFrame(fam)
    h = holm_bonferroni(F.p_raw.fillna(1.0).tolist(), alpha=0.05)
    F["p_holm"], F["reject"] = h["p_adjusted"], h["reject"]
    F["confirmada"] = F.reject & np.where(F.expect_neg, F.mean_d_pct < 0, F.mean_d_pct > 0) & ~F.inconclusive
    L += ["\n--- FAMILIA CONFIRMATORIA (Holm sobre 4) ---",
          F[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "rank_biserial", "p_raw", "p_holm", "confirmada", "inconclusive"]].to_string(index=False),
          f"CONFIRMADAS: {int(F.confirmada.sum())}/4 -> {list(F[F.confirmada].id) or 'nenhuma'}"]
    sec = []
    for dist, g in w.groupby("dist"):
        t1 = test(g.dispatch, g.ig_ql); t2 = test(g.dispatch, g.ils_ql); t3 = test(g.ig_ql, g.ils_ql)
        sec.append(dict(dist=dist, dup_gt=100 * g.chose_ig.mean(), d_dispatch_vs_ig=t1["mean_d_pct"], d_dispatch_vs_ils=t2["mean_d_pct"],
                        d_ig_vs_ils=t3["mean_d_pct"], ig_wins=t3["better"], ils_wins=t3["worse"]))
    L += ["\n--- por distribuicao: % com dup_ratio>0,1; d medio do despacho vs fixas; ig_ql vs ils_ql ---", pd.DataFrame(sec).to_string(index=False)]
    # descritivas: otimalidade provada e GAP por celula
    ms = ["lpt", "multifit", "ldm", "ig_random", "ig_ql", "ils_random", "ils_ql", "ig_ql_seed"]
    L.append("\n--- provadamente otimo (%), por distribuicao (linhas) ---")
    T = pd.DataFrame({k: w.groupby("dist").apply(lambda g, k=k: 100 * (g[k] == g.lb_lp).mean()) for k in ms})
    T["LB prova opt (UB=melhor)"] = w.groupby("dist").proven.mean() * 100
    L.append(T.round(1).to_string())
    L.append("\n--- GAP medio (%) vs LB certificado, por distribuicao ---")
    G = pd.DataFrame({k: w.groupby("dist").apply(lambda g, k=k: 100 * ((g[k] - g.lb_lp) / g.lb_lp).mean()) for k in ms})
    L.append(G.round(3).to_string())
    L.append("\n--- regiao dificil (n/m <= 3: (25,10) e (60,20)) vs resto: provadamente otimo (%) ---")
    w["hard"] = (w.n / w.m) <= 3
    H = pd.DataFrame({k: w.groupby("hard").apply(lambda g, k=k: 100 * (g[k] == g.lb_lp).mean()) for k in ms})
    H["LB prova opt"] = w.groupby("hard").proven.mean() * 100
    L.append(H.round(1).to_string())
    t = test(w.ig_ql_seed, w[["lpt", "multifit", "ldm"]].min(axis=1))
    L.append(f"\nig_ql_seed vs melhor construtiva: d={t['mean_d_pct']:.4f}% melhor/pior/empate={t['better']}/{t['worse']}/{t['tie']}")
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada108_new_distributions.txt").write_text(text, encoding="utf-8")
    F.to_csv(ROOT / "results" / "entrada108_new_distributions_primary.csv", index=False)
    H1.to_csv(ROOT / "results" / "entrada108_h1_equivalence.csv", index=False)


if __name__ == "__main__":
    main()
