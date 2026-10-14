"""
analysis/report_entrada117_seed_variance_all.py
=================================================
Variancia entre seeds nas 12 classes (Entrada 109 + 117). Diario: Entrada 117.

PRE-REGISTRO (congelado ANTES de executar as 9 classes novas; hashes em results/entrada117_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Dados: 12 classes x 20 instancias x 10 seeds x {ig_ql, ig_random, ils_ql, ils_random} (3 classes da Entrada 109 + 9 novas); medida por
instancia = media do makespan sobre os 10 seeds (240 instancias). d = 100*(A-B)/B (negativo: A melhor); IC90 percentil (2000 reamostragens
das INSTANCIAS, seed 0).
S1' (politica, medias por seed): ig_ql vs ig_random pooled. Reporta d, IC90 e Wilcoxon (bicaudal). Leituras pre-fixadas:
    'efeito real' = d<0 e p<0,05; 'estritamente equivalente' = IC90 dentro de +-0,05%; 'desprezivel frente ao mecanismo' = IC90 dentro de +-0,10%
    (um terco do menor efeito de mecanismo esperado, 0,3%).
S2' (mecanismo): por classe, o SINAL do contraste IG vs ILS (random e ql => 24 contrastes) sobre medias por seed coincide com o da execucao unica
    de results.csv (mesmas 20 instancias)? Criterio: >= 22 de 24 concordantes.
S3' (mecanismo, pooled): ig_random vs ils_random e ig_ql vs ils_ql sobre medias por seed: d e IC90 (descritivo).
S4' (por classe): d da politica por classe com Wilcoxon (20 instancias) e Holm sobre 12 (quantas classes detectam o efeito).
Descritivas: ICC e SD intra-instancia do GAP por classe/metodo.
Uso: python -m analysis.report_entrada117_seed_variance_all
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


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def boot(d, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(d), size=(2000, len(d)))
    return np.percentile(d[idx].mean(axis=1), [5, 95])


def icc(g, col):
    k = g.groupby("rep")[col].size().mean()
    msb = g.groupby("rep")[col].mean().var(ddof=1) * k
    msw = g.groupby("rep")[col].var(ddof=1).mean()
    return float((msb - msw) / (msb + (k - 1) * msw))


def main() -> None:
    sv = pd.concat([pd.read_csv(ROOT / "results" / "seed_variance_entrada109.csv"), pd.read_csv(ROOT / "results" / "seed_variance_all_entrada117.csv")])
    assert len(sv) == 12 * 20 * 10 * 4, len(sv)
    m = sv.groupby(["instance", "method", "rep"]).makespan.mean().unstack("method").reset_index()
    one = pd.read_csv(ROOT / "results" / "results.csv")
    one = one[one.method.isin(["ig_ql", "ig_random", "ils_ql", "ils_random"]) & (one.rep < 20)]
    w1 = one.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "VARIANCIA ENTRE SEEDS NAS 12 CLASSES -- teste pre-registrado, Entrada 117", "=" * 110, f"instancias (medias por seed): {len(m)}"]
    d = rel(m.ig_ql, m.ig_random)
    lo, hi = boot(d)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue)
    L.append(f"\nS1' politica (ig_ql vs ig_random, medias por seed, 240 instancias): d = {d.mean():+.4f}% IC90 [{lo:+.4f}, {hi:+.4f}] p = {p:.3g}")
    L.append(f"  efeito real (d<0, p<0,05): {'SIM' if (d.mean() < 0 and p < 0.05) else 'NAO'} | estritamente equivalente (IC90 em +-0,05%): "
             f"{'SIM' if (lo > -0.05 and hi < 0.05) else 'NAO'} | desprezivel frente ao mecanismo (IC90 em +-0,10%): {'SIM' if (lo > -0.10 and hi < 0.10) else 'NAO'}")
    agree = tot = 0
    rows = []
    for cls, g in m.groupby("instance"):
        g1 = w1[w1.instance == cls]
        for a, b in [("ig_random", "ils_random"), ("ig_ql", "ils_ql")]:
            dm = rel(g[a], g[b]).mean()
            d1 = rel(g1[a], g1[b]).mean()
            ok = (np.sign(dm) == np.sign(d1)) or (dm == 0 and d1 == 0)
            agree += int(ok)
            tot += 1
            rows.append(dict(instance=cls, pair=f"{a} vs {b}", d_seedavg=dm, d_single=d1, same_sign=ok))
    R = pd.DataFrame(rows)
    L.append(f"\nS2' concordancia de sinal IG vs ILS (medias por seed vs execucao unica): {agree}/{tot} -> {'CONFIRMADA' if agree >= 22 else 'NAO confirmada'} (criterio >= 22/24)")
    L.append(R.to_string(index=False))
    for a, b in [("ig_random", "ils_random"), ("ig_ql", "ils_ql")]:
        dd = rel(m[a], m[b])
        lo2, hi2 = boot(dd)
        L.append(f"S3' {a} vs {b}: d = {dd.mean():+.3f}% IC90 [{lo2:+.3f}, {hi2:+.3f}]; IG melhor em {(dd < 0).sum()} de {len(dd)} instancias (media por seed)")
    s4 = []
    for cls, g in m.groupby("instance"):
        dd = rel(g.ig_ql, g.ig_random)
        nzz = dd[dd != 0]
        pp = float(sstats.wilcoxon(nzz).pvalue) if len(nzz) >= 10 else 1.0
        s4.append(dict(instance=cls, d_policy=float(dd.mean()), better=int((dd < 0).sum()), worse=int((dd > 0).sum()), p_raw=pp))
    S4 = pd.DataFrame(s4)
    S4["p_holm"] = holm_bonferroni(S4.p_raw.tolist(), alpha=0.05)["p_adjusted"]
    L += ["\nS4' politica por classe (Holm sobre 12):", S4.to_string(index=False), f"classes com efeito detectado (p_holm<0,05): {int((S4.p_holm < 0.05).sum())}/12"]
    rows = []
    for (cls, mt), g in sv.groupby(["instance", "method"]):
        rows.append(dict(instance=cls, method=mt, icc=icc(g, "gap_pct"), sd_within_gap=g.groupby("rep").gap_pct.std(ddof=1).mean(), mean_gap=g.gap_pct.mean()))
    D = pd.DataFrame(rows)
    L += ["\nICC e SD intra-instancia do GAP (resumo por metodo):", D.groupby("method")[["icc", "sd_within_gap"]].agg(["min", "median", "max"]).to_string()]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada117_seed_variance_all.txt").write_text(text, encoding="utf-8")
    S4.to_csv(ROOT / "results" / "entrada117_policy_by_class.csv", index=False)


if __name__ == "__main__":
    main()
