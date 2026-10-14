"""
analysis/report_entrada90_gap_to_iaf.py
=========================================
Gap em relacao a SOLUCAO DE REFERENCIA (iAF, metodo exato -- Ostojic et al.
2025) nas instancias externas das familias C e F que ja foram executadas
(amostra de manifest.csv), em vez do gap relativo ao LB classico fraco
(gap_pct). Entrada 90, item 4. Depende de iaf_reference.csv (gerado por
external_benchmarks/pcmax_symmetry2025/join_iaf_reference.py, join
verificado por evidencia independente).

    gap_opt = 100 * (makespan - Z) / Z

Z e' o valor da solucao do iAF; em 127 das 9.530 linhas a planilha marca
Gap > 0 (otimalidade nao provada): Z e' entao so' um limite superior e
makespan < Z e' possivel. O relatorio mostra os resultados com todas as
linhas e SO' com as de otimalidade provada.

Saidas: (1) tabela por familia x metodo (gap_opt medio, % de instancias com
makespan == Z, gap_pct-vs-LB medio e folga media do LB = (Z-LB)/Z); (2)
ig_ql vs multifit/sa/bco/ig_random: vitorias/derrotas/empates em makespan
(descritivo -- os testes formais continuam em report_entrada58/90).

Uso: python -m analysis.report_entrada90_gap_to_iaf
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")

ref = pd.read_csv(EXT / "iaf_reference.csv")
man = pd.read_csv(EXT / "manifest.csv")
res = pd.read_csv(EXT / "results_external_symmetry2025.csv").drop_duplicates(["member", "method"])
res = res[res.member.isin(set(man.member))]
df = res.merge(ref[["member", "Z", "gap_sheet", "LB_ours"]], on="member", how="inner")
df["gap_opt"] = 100.0 * (df.makespan - df.Z) / df.Z
df["is_opt"] = df.makespan <= df.Z
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


emit(f"instancias com referencia iAF na amostra: {df.member.nunique()} "
     f"({df.groupby('family').member.nunique().to_dict()}); "
     f"com otimalidade NAO provada (Gap>0): {df[df.gap_sheet > 0].member.nunique()}")
emit(f"folga media do LB classico = (Z-LB)/Z: " +
     str({f: round(100 * ((g.Z - g.LB_ours) / g.Z).mean(), 3) for f, g in df.drop_duplicates('member').groupby('family')}) + " %")
order = ["lpt", "multifit", "sa_li_perkins", "bco", "ils_random", "ils_ql", "ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql_flat", "ig_ql"]
for label, sub in [("TODAS as linhas com referencia", df), ("SO' otimalidade provada (Gap=0)", df[df.gap_sheet == 0])]:
    emit("\n" + "=" * 100)
    emit(label)
    emit("=" * 100)
    t = sub.groupby(["family", "method"]).agg(n=("member", "nunique"), gap_opt=("gap_opt", "mean"),
                                              gap_LB=("gap_pct", "mean"), pct_opt=("is_opt", "mean"),
                                              cpu=("cpu_time", "mean")).reset_index()
    t["pct_opt"] = 100 * t.pct_opt
    t["m"] = pd.Categorical(t.method, order, ordered=True)
    emit(t.sort_values(["family", "m"]).drop(columns="m").to_string(index=False))

emit("\n" + "=" * 100)
emit("ig_ql vs outro, em makespan, por familia (so' otimalidade provada): melhor / pior / empate")
emit("=" * 100)
sub = df[df.gap_sheet == 0]
w = sub.pivot_table(index=["family", "member"], columns="method", values="makespan")
for other in ["multifit", "sa_li_perkins", "bco", "ig_random", "ig_ucb", "ils_ql"]:
    for fam, g in w.groupby(level=0):
        d = g["ig_ql"] - g[other]
        emit(f"  {fam} ig_ql vs {other:14s}: melhor {int((d < 0).sum()):5d}  pior {int((d > 0).sum()):5d}  empate {int((d == 0).sum()):5d}")
(ROOT / "results" / "entrada90_gap_to_iaf_report.txt").write_text("\n".join(lines), encoding="utf-8")
