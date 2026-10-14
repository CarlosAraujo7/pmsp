"""
analysis/report_entrada110_equal_cpu.py
=========================================
WP3b: IG vs ILS em CPU igual. Diario: Entrada 110.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada110_prereg_hashes.txt)
------------------------------------------------------------------------------------------
Dados: 2 distribuicoes (u1_10000 fina; u1_100 grossa) x 5 tamanhos x 30 repeticoes = 300 instancias; ILS (T=1000) vindo de
results/new_distributions_entrada108.csv (mesmas instancias); IG com T_IG = round(1000*K(n)) (K congelado em
experiment/run_equal_cpu_entrada110.py). d = 100*(IG - ILS)/ILS (negativo: IG melhor).
Familia confirmatoria (Holm sobre 4, Wilcoxon pareado BICAUDAL, pooled por distribuicao):
  E1 u1_10000: ig_ql_eqt vs ils_ql       E2 u1_10000: ig_random_eqt vs ils_random
  E3 u1_100:   ig_ql_eqt vs ils_ql       E4 u1_100:   ig_random_eqt vs ils_random
Leitura pre-especificada: "a vantagem do ILS sobre o IG em granularidade fina PERSISTE em CPU igual" se E1 e E2 tiverem d>0
e p_holm<0,05; "INVERTE" se E1 e E2 tiverem d<0 com p_holm<0,05; caso contrario "DESAPARECE / nao conclusivo".
Descritivas: d por tamanho; razao de CPU realizada (cpu_IG/cpu_ILS; tempos de execucao paralela, indicativos).
Uso: python -m analysis.report_entrada110_equal_cpu
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


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def main() -> None:
    nd = pd.read_csv(ROOT / "results" / "new_distributions_entrada108.csv")
    eq = pd.read_csv(ROOT / "results" / "equal_cpu_entrada110.csv")
    key = ["dist", "n", "m", "rep"]
    sub = nd[nd.method.isin(["ils_ql", "ils_random"]) & nd.dist.isin(["u1_10000", "u1_100"])]
    ils = sub.pivot(index=key, columns="method", values="makespan").reset_index()
    ils_cpu = sub.pivot(index=key, columns="method", values="cpu_time").add_prefix("cpu_").reset_index()
    ig = eq.pivot(index=key, columns="method", values="makespan").reset_index()
    ig_cpu = eq.pivot(index=key, columns="method", values="cpu_time").add_prefix("cpu_").reset_index()
    w = ils.merge(ig, on=key).merge(ils_cpu, on=key).merge(ig_cpu, on=key)
    assert len(w) == 300, len(w)
    rows = []
    for pid, dist, a, b in [("E1", "u1_10000", "ig_ql_eqt", "ils_ql"), ("E2", "u1_10000", "ig_random_eqt", "ils_random"),
                            ("E3", "u1_100", "ig_ql_eqt", "ils_ql"), ("E4", "u1_100", "ig_random_eqt", "ils_random")]:
        g = w[w.dist == dist]
        d = rel(g[a], g[b])
        nz = d[d != 0]
        p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else 1.0
        rows.append(dict(id=pid, dist=dist, comparison=f"{a} vs {b}", n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()),
                         tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), rank_biserial=rank_biserial_paired(g[a], g[b]), p_raw=p))
    F = pd.DataFrame(rows)
    h = holm_bonferroni(F.p_raw.tolist(), alpha=0.05)
    F["p_holm"], F["reject"] = h["p_adjusted"], h["reject"]
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "IG vs ILS EM CPU IGUAL -- teste pre-registrado, Entrada 110", "=" * 110, F.to_string(index=False)]
    e12 = F[F.id.isin(["E1", "E2"])]
    if (e12.reject & (e12.mean_d_pct > 0)).all():
        verdict = "PERSISTE (ILS melhor em CPU igual em granularidade fina)"
    elif (e12.reject & (e12.mean_d_pct < 0)).all():
        verdict = "INVERTE (IG melhor em CPU igual em granularidade fina)"
    else:
        verdict = "DESAPARECE / nao conclusivo (sem vantagem significativa do ILS em CPU igual)"
    L.append(f"\nVEREDITO pre-registrado (E1, E2): a vantagem do ILS {verdict}")
    L.append("\n--- por distribuicao e tamanho: d medio (IG_eqt vs ILS), razao CPU realizada IG/ILS ---")
    for (dist, n), g in w.groupby(["dist", "n"]):
        L.append(f"  {dist:9s} n={n:3d}: ql d={rel(g.ig_ql_eqt, g.ils_ql).mean():+.3f}% (cpu IG/ILS={(g.cpu_ig_ql_eqt / g.cpu_ils_ql).median():.2f}) | "
                 f"random d={rel(g.ig_random_eqt, g.ils_random).mean():+.3f}% (cpu={(g.cpu_ig_random_eqt / g.cpu_ils_random).median():.2f})")
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada110_equal_cpu.txt").write_text(text, encoding="utf-8")
    F.to_csv(ROOT / "results" / "entrada110_equal_cpu_primary.csv", index=False)


if __name__ == "__main__":
    main()
