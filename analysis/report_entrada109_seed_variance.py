"""
analysis/report_entrada109_seed_variance.py
=============================================
WP7: variancia entre seeds do algoritmo. Diario: Entrada 109.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada109_prereg_hashes.txt)
------------------------------------------------------------------------------------------
Dados: 3 classes ((30,8),(75,15),(150,30)) x 20 instancias x 10 seeds x {ig_ql, ig_random, ils_ql, ils_random}; T=1000.
Medida por instancia = media do makespan sobre os 10 seeds. d = 100*(A-B)/B (negativo: A melhor).
S1 (politica sobre medias por seed): IC90 percentil (2000 reamostragens, seed 0) do d medio de ig_ql vs ig_random, pooled sobre as
   60 instancias, dentro de (-0,05; +0,05) => equivalente (CONFIRMA que o resultado de uma execucao por instancia nao esconde
   um efeito da politica).
S2 (mecanismo sobre medias por seed): por classe, o SINAL do d medio de ig_random vs ils_random (e de ig_ql vs ils_ql) sobre as
   medias por seed coincide com o sinal do mesmo contraste na execucao unica de results.csv (mesmas 20 instancias). Reporta-se
   a concordancia (n de 6 contrastes concordantes) e o p (Wilcoxon pareado sobre as 20 medias por instancia, sem correcao).
Descritivas: decomposicao de variancia (ICC) do GAP por classe e metodo; CV intra-instancia do makespan (%); fracao de seeds em
que o IG vence o ILS por instancia.
Uso: python -m analysis.report_entrada109_seed_variance
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MARGIN = 0.05


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def icc(df, col):
    g = df.groupby("rep")[col]
    k = g.size().mean()
    msb = g.mean().var(ddof=1) * k
    msw = g.var(ddof=1).mean()
    return float((msb - msw) / (msb + (k - 1) * msw))


def main() -> None:
    d = pd.read_csv(ROOT / "results" / "seed_variance_entrada109.csv")
    assert len(d) == 3 * 20 * 10 * 4, len(d)
    one = pd.read_csv(ROOT / "results" / "results.csv")
    one = one[one.method.isin(["ig_ql", "ig_random", "ils_ql", "ils_random"]) & (one.rep < 20)]
    L = ["=" * 110, "VARIANCIA ENTRE SEEDS -- teste pre-registrado, Entrada 109", "=" * 110]
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    # decomposicao
    rows = []
    for (cls, mt), g in d.groupby(["instance", "method"]):
        within_cv = (g.groupby("rep").makespan.std(ddof=1) / g.groupby("rep").makespan.mean()).mean() * 100
        rows.append(dict(instance=cls, method=mt, icc_gap=icc(g, "gap_pct"), within_cv_pct=within_cv,
                         mean_gap=g.gap_pct.mean(), within_sd_gap_pp=g.groupby("rep").gap_pct.std(ddof=1).mean()))
    L += ["\n--- decomposicao de variancia do GAP (ICC) e CV intra-instancia ---", pd.DataFrame(rows).to_string(index=False)]
    m = d.groupby(["instance", "method", "rep"]).makespan.mean().unstack("method").reset_index()
    def tost(a, b):
        x = rel(a, b)
        idx = np.random.default_rng(0).integers(0, len(x), size=(2000, len(x)))
        lo, hi = np.percentile(x[idx].mean(axis=1), [5, 95])
        nz = x[x != 0]
        p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
        return float(x.mean()), float(lo), float(hi), p, bool(lo > -MARGIN and hi < MARGIN)
    mean, lo, hi, p, eq = tost(m.ig_ql, m.ig_random)
    L.append(f"\n--- S1: ig_ql vs ig_random, medias por seed, 60 instancias: d = {mean:+.4f}% IC90 [{lo:+.4f}, {hi:+.4f}] p={p:.3g} "
             f"=> {'EQUIVALENTE (S1 CONFIRMADA)' if eq else 'NAO equivalente (S1 NAO confirmada)'}")
    L.append("\n--- S2: sinal do contraste IG vs ILS: medias por seed (10 seeds) vs execucao unica (results.csv) ---")
    w1 = one.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    agree = 0
    for cls, g in m.groupby("instance"):
        g1 = w1[w1.instance == cls]
        for a, b in [("ig_random", "ils_random"), ("ig_ql", "ils_ql")]:
            dm, _, _, pm, _ = tost(g[a], g[b])
            d1 = rel(g1[a], g1[b]).mean()
            same = np.sign(dm) == np.sign(d1)
            agree += int(same)
            L.append(f"  {cls:14s} {a} vs {b}: seeds-medias d={dm:+.3f}% (p={pm:.3g}) | execucao unica d={d1:+.3f}% | sinal {'igual' if same else 'DIFERENTE'}")
    L.append(f"concordancia de sinal: {agree}/6 -> S2 {'CONFIRMADA' if agree == 6 else 'NAO confirmada integralmente'}")
    L.append("\n--- fracao de seeds em que ig_random bate ils_random (por classe; media sobre instancias) ---")
    pv = d.pivot_table(index=["instance", "rep", "seed"], columns="method", values="makespan").reset_index()
    pv["ig_wins"] = pv.ig_random < pv.ils_random
    pv["ils_wins"] = pv.ig_random > pv.ils_random
    L.append(pv.groupby("instance")[["ig_wins", "ils_wins"]].mean().to_string())
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada109_seed_variance.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
