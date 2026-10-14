"""
analysis/report_entrada98_granularity_factorial.py
====================================================
Analise do item 3 (fatorial por granularidade; dados de
experiment/run_granularity_factorial_entrada98.py).

PRE-REGISTRO (congelado em 05/10/2026, antes da execucao)
----------------------------------------------------------
Hipotese: o mecanismo mais forte (IG vs ILS) e' decidido pela GRANULARIDADE dos
tempos (faixa H relativa a n, medida por dup_ratio), nao por n/m nem por uma
"variancia" relativa (o CV e' o mesmo para todo H).
Desfecho por instancia: D = gap(ig_ql) - gap(ils_ql) (D<0 => IG melhor).
 (P1) Para cada nivel de n/m (3 testes, Holm, alpha 0,05): correlacao de Spearman
      entre log10(H) e D sobre as 150 instancias do nivel; "efeito da granularidade
      confirmado" se rho>0 (ILS ganha relativamente quando H cresce) com p_holm<0,05.
 (P2) Descritivas: tabela por celula (dup_ratio medio, CV medio, gap medio dos 4
      metodos, Wilcoxon pareado ig_ql vs ils_ql com Holm sobre as 15 celulas);
      correlacao de Spearman entre n/m e D dentro de cada H (5 testes, Holm) -- se a
      hipotese de granularidade esta certa, estas devem ser pequenas/nao
      consistentes em sinal quando comparadas as de (P1).
 (P3) CV: Kruskal-Wallis do CV entre os 5 valores de H deve ser nao significativo
      (confirma que a "variancia relativa" nao varia).
Veredito: GRANULARIDADE CONFIRMADA se (P1) confirmada em >=2 de 3 niveis de n/m;
INCONCLUSIVO se em 1; NAO CONFIRMADA se em 0.
Uso: python -m analysis.report_entrada98_granularity_factorial
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
df = pd.read_csv(ROOT / "results" / "granularity_factorial_entrada98.csv")
w = df.pivot_table(index=["n", "m", "H", "k", "dup_ratio", "cv"], columns="method", values="gap_pct").reset_index()
w["D"] = w.ig_ql - w.ils_ql
w["D_rand"] = w.ig_random - w.ils_random
w["logH"] = np.log10(w.H)
L = []


def emit(s=""):
    print(s)
    L.append(s)


emit("=" * 100 + "\n(P2) TABELA POR CELULA (n=150; N=30 por celula; D<0 => IG melhor)\n" + "=" * 100)
cells = []
for (m, H), g in w.groupby(["m", "H"]):
    d = g.D.to_numpy(); nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 8 else float("nan")
    cells.append(dict(n_over_m=150 // m, H=H, dup_ratio=g.dup_ratio.mean(), cv=g.cv.mean(), ig_ql=g.ig_ql.mean(),
                      ils_ql=g.ils_ql.mean(), ig_random=g.ig_random.mean(), ils_random=g.ils_random.mean(),
                      D_mean=g.D.mean(), ig_better=int((d < 0).sum()), ils_better=int((d > 0).sum()), p_raw=p))
t = pd.DataFrame(cells)
h = holm_bonferroni(t.p_raw.fillna(1.0).tolist(), alpha=0.05)
t["p_holm"], t["reject"] = h["p_adjusted"], h["reject"]
emit(t.to_string(index=False))

emit("\n" + "=" * 100 + "\n(P1) Spearman(log10 H, D) por nivel de n/m (Holm sobre 3)\n" + "=" * 100)
rows = []
for m, g in w.groupby("m"):
    rho, p = sstats.spearmanr(g.logH, g.D)
    rows.append(dict(n_over_m=150 // m, n=len(g), rho=rho, p_raw=p))
p1 = pd.DataFrame(rows)
hh = holm_bonferroni(p1.p_raw.tolist(), alpha=0.05)
p1["p_holm"], p1["reject"] = hh["p_adjusted"], hh["reject"]
p1["confirmada"] = p1.reject & (p1.rho > 0)
emit(p1.to_string(index=False))
k = int(p1.confirmada.sum())
verdict = "GRANULARIDADE CONFIRMADA" if k >= 2 else "INCONCLUSIVO" if k == 1 else "NAO CONFIRMADA"
emit(f"VEREDITO pre-registrado: {k}/3 niveis confirmados -> {verdict}")

emit("\n" + "=" * 100 + "\n(P2b) Spearman(n/m, D) dentro de cada H (Holm sobre 5; descritivo)\n" + "=" * 100)
rows = []
for H, g in w.groupby("H"):
    rho, p = sstats.spearmanr(g.m.map(lambda x: 150 / x), g.D)
    rows.append(dict(H=H, n=len(g), rho=rho, p_raw=p))
p2 = pd.DataFrame(rows)
h2 = holm_bonferroni(p2.p_raw.tolist(), alpha=0.05)
p2["p_holm"], p2["reject"] = h2["p_adjusted"], h2["reject"]
emit(p2.to_string(index=False))

emit("\n" + "=" * 100 + "\n(P3) CV por H (Kruskal-Wallis) e dup_ratio por H\n" + "=" * 100)
groups = [g.cv.to_numpy() for _, g in w.groupby("H")]
kw = sstats.kruskal(*groups)
emit(f"Kruskal-Wallis do CV entre os 5 valores de H: H={kw.statistic:.3f}, p={kw.pvalue:.3g}")
emit(w.groupby("H").agg(cv_mean=("cv", "mean"), dup_ratio_mean=("dup_ratio", "mean")).to_string())
emit("\nControle sem aprendizado (ig_random - ils_random), media por H e n/m:")
emit(w.pivot_table(index="H", columns="m", values="D_rand", aggfunc="mean").to_string())
(ROOT / "results" / "entrada98_granularity_factorial_report.txt").write_text("\n".join(L), encoding="utf-8")
