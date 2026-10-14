"""
analysis/report_entrada94_dispatch_confirmatory.py
====================================================
Teste CONFIRMATORIO do despacho a priori `apriori_select` (dup_ratio > 0,1
-> IG, senao -> ILS; pmsp/apriori_select.py, limiar fixado desde a Entrada
61, nunca retunado) em instancias NUNCA vistas: o holdout de 4.160 instancias
externas (B, C, E) da Entrada 90/91. Diario: Entrada 94.

PRE-REGISTRO (congelado em 04/10/2026, ANTES de executar `ils_ql` no holdout)
-----------------------------------------------------------------------------
Dados: ig_ql ja executado no holdout (results_holdout_entrada90.csv);
ils_ql a executar (results_holdout_dispatch_entrada94.csv) com o mesmo
manifest, seeds (inst_id 1000+k) e orcamento (T=1000). O resultado do
despacho em cada instancia e' o do especialista escolhido (como nas Entradas
61-63): gap_dispatch = gap(ig_ql) se dup_ratio > 0,1, senao gap(ils_ql).
dup_ratio e' calculado por pmsp.apriori_select.dup_ratio sobre o vetor de
tempos da instancia; LIMIAR = 0,1, SEM retunagem.

Hipotese primaria (superioridade do despacho sobre AMBAS as politicas
fixas), pooled sobre todas as instancias do holdout:
  H1a: gap(dispatch) < gap(ig_ql sempre);  H1b: gap(dispatch) < gap(ils_ql sempre).
  Teste: Wilcoxon signed-rank pareado bicaudal sobre as diferencas por
  instancia, diferencas nulas descartadas; Holm sobre os 2 testes, alpha=0,05;
  "confirmada" = p_holm < 0,05 E diferenca media < 0.
  Veredito: ambas confirmadas -> DESPACHO CONFIRMADO; so' uma -> PARCIAL;
  nenhuma -> NAO CONFIRMADO (o despacho vai ao suplemento como resultado
  preliminar negativo/inconclusivo; nenhuma alegacao de superioridade).
Secundarias (descritivas, sem entrar no veredito): os mesmos testes por
familia (B, C, E; Holm sobre 6); acuracia da escolha (instancias em que o
despacho escolheu o framework de menor gap, empates a parte); fracao do gap
fixo-melhor -> oraculo fechada: (min(media_fixas) - media_dispatch) /
(min(media_fixas) - media_oraculo), oraculo = min(ig_ql, ils_ql) por
instancia. Magnitude absoluta (pp de gap) reportada sempre.

Uso:
    python -m analysis.report_entrada94_dispatch_confirmatory
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_benchmarks" / "pcmax_symmetry2025"))

from analysis.stats import holm_bonferroni, rank_biserial_paired  # noqa: E402
from pmsp.apriori_select import dup_ratio, DEFAULT_DUP_RATIO_THRESHOLD  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

THRESHOLD = 0.1
assert DEFAULT_DUP_RATIO_THRESHOLD == THRESHOLD, "limiar do codigo difere do pre-registrado"

EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
MANIFEST = EXT / "manifest_holdout_entrada90.csv"
IG_RES = EXT / "results_holdout_entrada90.csv"
ILS_RES = EXT / "results_holdout_dispatch_entrada94.csv"
DUP_CACHE = ROOT / "results" / "entrada94_holdout_dup_ratio.csv"


def holdout_dup_ratios() -> pd.DataFrame:
    if DUP_CACHE.exists():
        return pd.read_csv(DUP_CACHE)
    from loader import load_many
    man = pd.read_csv(MANIFEST)
    rows = []
    for fam, g in man.groupby("family"):
        data = load_many(fam, g.member.tolist())
        for member in g.member:
            m, n, p = data[member]
            rows.append(dict(member=member, dup_ratio=dup_ratio(PMSPInstance(p=p, m=m, name="x"))))
    out = pd.DataFrame(rows)
    out.to_csv(DUP_CACHE, index=False)
    return out


def test(x, y):
    d = np.asarray(x) - np.asarray(y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()),
                mean_diff_pp=float(d.mean()), rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    man = pd.read_csv(MANIFEST)
    ig = pd.read_csv(IG_RES).query("method == 'ig_ql'").drop_duplicates("member")[["member", "family", "gap_pct"]]
    ils = pd.read_csv(ILS_RES).query("method == 'ils_ql'").drop_duplicates("member")[["member", "gap_pct"]]
    df = (ig.rename(columns={"gap_pct": "gap_ig"}).merge(ils.rename(columns={"gap_pct": "gap_ils"}), on="member")
            .merge(holdout_dup_ratios(), on="member"))
    assert len(df) == len(man), f"cobertura incompleta: {len(df)} de {len(man)}"
    df["chose_ig"] = df.dup_ratio > THRESHOLD
    df["gap_dispatch"] = np.where(df.chose_ig, df.gap_ig, df.gap_ils)
    df["gap_oracle"] = np.minimum(df.gap_ig, df.gap_ils)
    L = ["=" * 100, "TESTE CONFIRMATORIO DO DESPACHO a priori (limiar 0,1 fixo) -- holdout", "=" * 100,
         f"instancias: {len(df)} {df.groupby('family').size().to_dict()} | escolheu IG em {100 * df.chose_ig.mean():.1f}%"]

    def block(sub, label):
        r = []
        for name, col in [("ig_ql sempre", "gap_ig"), ("ils_ql sempre", "gap_ils")]:
            t = test(sub.gap_dispatch, sub[col]); t.update(group=label, vs=name); r.append(t)
        return r

    prim = pd.DataFrame(block(df, "POOLED"))
    h = holm_bonferroni(prim.p_raw.tolist(), alpha=0.05)
    prim["p_holm"], prim["reject"] = h["p_adjusted"], h["reject"]
    prim["confirmada"] = prim.reject & (prim.mean_diff_pp < 0)
    sec = pd.DataFrame([x for f, g in df.groupby("family") for x in block(g, f)])
    h2 = holm_bonferroni(sec.p_raw.tolist(), alpha=0.05)
    sec["p_holm"], sec["reject"] = h2["p_adjusted"], h2["reject"]
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    cols = ["group", "vs", "n", "better", "worse", "tie", "mean_diff_pp", "rank_biserial", "p_raw", "p_holm", "reject"]
    L += ["\n--- PRIMARIA (pooled; Holm sobre 2; mean_diff<0 => despacho melhor; rank_biserial>0 => despacho PIOR) ---",
          prim[cols + ["confirmada"]].to_string(index=False),
          "\n--- SECUNDARIA por familia (Holm sobre 6; descritiva) ---", sec[cols].to_string(index=False)]
    means = {c: df[c].mean() for c in ["gap_ig", "gap_ils", "gap_dispatch", "gap_oracle"]}
    best_fixed = min(means["gap_ig"], means["gap_ils"])
    closed = (best_fixed - means["gap_dispatch"]) / (best_fixed - means["gap_oracle"]) if best_fixed > means["gap_oracle"] else float("nan")
    L.append(f"\nmedias de gap (pp): {', '.join(f'{k}={v:.4f}' for k, v in means.items())}")
    L.append(f"fracao do gap (melhor fixa -> oraculo) fechada pelo despacho: {closed:.3f}")
    decisive = df[df.gap_ig != df.gap_ils]
    acc = ((decisive.gap_ig < decisive.gap_ils) == decisive.chose_ig).mean()
    L.append(f"acuracia da escolha (so' instancias com gap_ig != gap_ils, n={len(decisive)}): {100 * acc:.1f}%")
    for fam, g in df.groupby("family"):
        d = g[g.gap_ig != g.gap_ils]
        L.append(f"  {fam}: acuracia {100 * ((d.gap_ig < d.gap_ils) == d.chose_ig).mean():.1f}% (n={len(d)}), "
                 f"gap ig={g.gap_ig.mean():.4f} ils={g.gap_ils.mean():.4f} dispatch={g.gap_dispatch.mean():.4f}")
    k = int(prim.confirmada.sum())
    verdict = "DESPACHO CONFIRMADO" if k == 2 else "PARCIAL" if k == 1 else "NAO CONFIRMADO"
    L += ["\n" + "=" * 100, f"VEREDITO pre-registrado: {k}/2 confirmadas -> {verdict}", "=" * 100]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada94_dispatch_confirmatory.txt").write_text(text, encoding="utf-8")
    pd.concat([prim, sec]).to_csv(ROOT / "results" / "entrada94_dispatch_confirmatory.csv", index=False)


if __name__ == "__main__":
    main()
