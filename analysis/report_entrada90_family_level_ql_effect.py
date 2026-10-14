"""
analysis/report_entrada90_family_level_ql_effect.py
=====================================================
Teste a nivel de FAMILIA da contribuicao do Q-Learning dentro do IG
(ig_ql vs. ablacoes IG), sobre a bateria externa Symmetry 2025.

POR QUE ESTE SCRIPT EXISTE (diario_pesquisa.md, Entrada 90)
------------------------------------------------------------
O relatorio por subpasta (report_entrada58_external_validation.py)
roda dezenas de testes de Wilcoxon com poucas instancias cada e aplica
Holm sobre todos -- desenho de baixo poder para um efeito pequeno. Uma
exploracao ad hoc (04/10/2026) da amostra de 6.780 instancias, agregando
por familia, sugeriu que ig_ql tem gap ~2-3% (relativo) MENOR que
ig_random nas familias B, C e E, e ligeiramente PIOR em F. Como essa
exploracao foi feita DEPOIS de ver os resultados, ela e' tratada como
EXPLORATORIA (gera hipotese, nao a confirma). Este script implementa:

  (a) modo `exploratory` -- a mesma analise sobre a amostra de 6.780,
      rotulada EXPLORATORIA em toda a saida;
  (b) modo `holdout` -- a analise CONFIRMATORIA, sobre instancias do
      benchmark que NUNCA foram executadas (manifest_holdout_entrada90.csv,
      geradas por build_holdout_sample.py), com a regra de decisao
      abaixo CONGELADA antes de qualquer execucao do holdout.

PRE-REGISTRO (congelado em 04/10/2026, antes de rodar o holdout)
-----------------------------------------------------------------
Unidade de analise: a instancia. Desfecho: gap_pct (relativo ao LB
analitico do projeto). Pareamento: mesma instancia e mesmo seed
(inst_id) para todos os metodos. Teste: Wilcoxon signed-rank bicaudal
sobre as diferencas pareadas gap(ig_ql) - gap(comparador), diferencas
nulas descartadas (scipy zero_method='wilcox'). Effect size:
rank-biserial pareado (analysis.stats.rank_biserial_paired; sinal:
x=ig_ql, y=comparador => r>0 significa ig_ql PIOR). IC 95%: bootstrap
BCa da diferenca media (analysis.stats.bootstrap_ci_mean_diff).

  Familia PRIMARIA P (3 testes, Holm, alpha=0,05):
      H1_f : ig_ql vs. ig_random, f em {B, C, E}.
      "Confirmada em f"  <=>  p_holm < 0,05  E  mean(gap_ql - gap_rand) < 0.
  Veredito global (apenas sobre P):
      >= 2 de 3 confirmadas  -> "efeito do Q-Learning REPLICADO (pequeno)";
      exatamente 1           -> "INCONCLUSIVO";
      0                      -> "NAO replicado: a exploracao foi falso
                                positivo / maldicao do vencedor, e o
                                manuscrito mantem 'nenhuma contribuicao
                                defensavel do Q-Learning'".
  Familia SECUNDARIA S (9 testes, Holm separado, alpha=0,05):
      ig_ql vs. {ig_ucb, ig_ucb_ctx, ig_ql_flat}, f em {B, C, E}.
      Reportada, mas NAO entra no veredito.
  Familia F: o benchmark nao tem instancia F alem das 780 ja usadas (F
      foi 100% amostrada), portanto F NAO tem teste confirmatorio --
      so' o relato exploratorio. Qualquer alegacao sobre F no texto deve
      dizer isso.
  Significancia pratica: reportar sempre a mudanca relativa (%) E a
      diferenca absoluta em pontos percentuais de gap; nenhuma
      alegacao de "melhoria" pode omitir a magnitude absoluta.

Uso:
    python -m analysis.report_entrada90_family_level_ql_effect --mode exploratory
    python -m analysis.report_entrada90_family_level_ql_effect --mode holdout
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats import holm_bonferroni, rank_biserial_paired, bootstrap_ci_mean_diff  # noqa: E402

EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
HOLDOUT_RESULTS = EXT / "results_holdout_entrada90.csv"
HOLDOUT_MANIFEST = EXT / "manifest_holdout_entrada90.csv"

PRIMARY_COMPARATOR = "ig_random"
SECONDARY_COMPARATORS = ["ig_ucb", "ig_ucb_ctx", "ig_ql_flat"]
CONFIRMATORY_FAMILIES = ["B", "C", "E"]
ALPHA = 0.05


def load_exploratory() -> pd.DataFrame:
    from analysis.report_entrada58_external_validation import load_clean_external_results
    return load_clean_external_results()


def load_holdout() -> pd.DataFrame:
    df = pd.read_csv(HOLDOUT_RESULTS)
    man = pd.read_csv(HOLDOUT_MANIFEST)
    df = df[df["member"].isin(set(man["member"]))]
    df = df.drop_duplicates(["member", "method"])
    return df


def paired_test(df: pd.DataFrame, fam: str, comparator: str, boot_seed: int = 90) -> dict:
    sub = df[(df["family"] == fam) & (df["method"].isin(["ig_ql", comparator]))]
    w = sub.pivot_table(index="member", columns="method", values="gap_pct", aggfunc="first").dropna()
    if len(w) == 0:
        return dict(family=fam, comparator=comparator, n=0)
    x, y = w["ig_ql"].to_numpy(), w[comparator].to_numpy()
    d = x - y
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    ci = bootstrap_ci_mean_diff(x, y, seed=boot_seed)
    return dict(
        family=fam, comparator=comparator, n=len(w),
        ql_better=int((d < 0).sum()), ql_worse=int((d > 0).sum()), tie=int((d == 0).sum()),
        mean_gap_ql=float(x.mean()), mean_gap_cmp=float(y.mean()),
        mean_diff_pp=float(d.mean()), ci_low_pp=ci["ci_low"], ci_high_pp=ci["ci_high"],
        rel_change_pct=100.0 * float(d.mean()) / float(y.mean()) if y.mean() else float("nan"),
        rank_biserial=rank_biserial_paired(x, y),  # >0 => ig_ql PIOR
        p_raw=p,
    )


def add_holm(rows: list[dict], label: str) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    t["holm_family"] = label
    valid = t["p_raw"].notna()
    h = holm_bonferroni(t.loc[valid, "p_raw"].tolist(), alpha=ALPHA)
    t["p_holm"] = np.nan
    t["reject_holm"] = False
    t.loc[valid, "p_holm"] = h["p_adjusted"]
    t.loc[valid, "reject_holm"] = h["reject"]
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exploratory", "holdout"], required=True)
    args = ap.parse_args()

    df = load_exploratory() if args.mode == "exploratory" else load_holdout()
    tag = "EXPLORATORIA (amostra de 6.780; hipotese gerada depois de ver os dados)" \
        if args.mode == "exploratory" else "CONFIRMATORIA (holdout, regra pre-registrada)"
    lines = ["=" * 100, f"ANALISE {tag}", "=" * 100,
             f"instancias: {df['member'].nunique()}  | familias: "
             f"{ {f: int(g['member'].nunique()) for f, g in df.groupby('family')} }"]

    prim = add_holm([paired_test(df, f, PRIMARY_COMPARATOR) for f in CONFIRMATORY_FAMILIES], "P (primaria)")
    sec = add_holm([paired_test(df, f, c) for c in SECONDARY_COMPARATORS for f in CONFIRMATORY_FAMILIES],
                   "S (secundaria)")
    parts = [prim, sec]
    if args.mode == "exploratory":
        # F so' existe no modo exploratorio (nao ha instancias F nao vistas)
        fam_f = add_holm([paired_test(df, "F", c) for c in [PRIMARY_COMPARATOR] + SECONDARY_COMPARATORS],
                         "F (exploratoria, sem holdout possivel)")
        parts.append(fam_f)

    cols = ["family", "comparator", "n", "ql_better", "ql_worse", "tie", "mean_gap_ql", "mean_gap_cmp",
            "mean_diff_pp", "ci_low_pp", "ci_high_pp", "rel_change_pct", "rank_biserial", "p_raw", "p_holm",
            "reject_holm"]
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    for t in parts:
        lines.append(f"\n--- familia de Holm: {t['holm_family'].iloc[0]}  (rank_biserial > 0 => ig_ql PIOR) ---")
        lines.append(t[cols].to_string(index=False))

    # veredito pre-registrado (so' faz sentido no holdout)
    confirmed = prim[(prim["reject_holm"]) & (prim["mean_diff_pp"] < 0)]["family"].tolist()
    k = len(confirmed)
    verdict = ("REPLICADO (pequeno)" if k >= 2 else "INCONCLUSIVO" if k == 1 else "NAO REPLICADO")
    lines.append("\n" + "=" * 100)
    lines.append(f"VEREDITO sobre a familia primaria P: confirmadas em {confirmed} ({k}/3) -> {verdict}")
    if args.mode == "exploratory":
        lines.append("(modo exploratorio: este 'veredito' NAO tem valor confirmatorio -- so' reproduz a exploracao)")
    lines.append("=" * 100)

    text = "\n".join(lines)
    print(text)
    out_txt = ROOT / "results" / f"entrada90_family_level_{args.mode}.txt"
    out_csv = ROOT / "results" / f"entrada90_family_level_{args.mode}.csv"
    out_txt.write_text(text, encoding="utf-8")
    pd.concat(parts, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nSalvo em {out_txt} e {out_csv}")


if __name__ == "__main__":
    main()
