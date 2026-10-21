"""
analysis/report_entrada162_igls_fast.py
=========================================
Sensibilidade da conclusao 'a CPU igual' a implementacao da busca local (Ciclo 13, frente D). Diario: Entrada 162 (pre-registro) e 164 (resultados).

PRE-REGISTRO (congelado ANTES de executar a fase --run; hashes em results/entrada162_prereg_hashes.txt)
--------------------------------------------------------------------------------------------------------
Contexto: a busca local de P||Cmax (inter-troca + reinsercao) estava em lacos Python; pmsp/operators_fast.py a vetoriza com equivalencia exata (tests/test_operators_fast.py, 188 testes; verificacao de execucao: --verify).
Dados: cenario H (Entrada 129): igls_random (results/igls_p_entrada124.csv; identico com a busca local rapida) vs ig_random_eq_fast (results/igls_eq_fast_entrada162.csv; T_eq = max(1000, round(1000 K_fast)), K_fast = razao sequencial de CPU
com/sem busca local RAPIDA); cenario R (Entrada 127): igcvls_seed (results/igcvls_seed_entrada127.csv) vs igcv_seed_eq_fast (results/igcv_eq_fast_entrada162.csv). 12 classes x 100 reps cada (as mesmas instancias/seeds).
d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90.
Familia primaria (Holm sobre 2, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0, isto e', a busca local ganha a CPU igual):
  D1  cenario H: igls_random vs ig_random_eq_fast
  D2  cenario R: igcvls_seed vs igcv_seed_eq_fast
Leituras pre-fixadas: fracao de orcamento do ganho da busca local no cenario H, B = 1 - |d(D1)| / |d(igls_random vs ig_random, T = 1000)| (o denominador e' o ganho a iteracoes iguais, -0,150% na Entrada 124).
'O achado a CPU igual (93% do ganho era orcamento) e' robusto a implementacao' = B >= 0,5 com a busca local rapida. 'A busca local passa a pagar a CPU igual' = D1 confirmada. Descritivas: K_fast vs K da Entrada 129/127 por classe, razao
de CPU realizada, por classe, otimo provado (cenario H).
Previsoes registradas antes de rodar: K_fast em [1,2; 8] no cenario H (contra 2,1 a 52,4 antes); D1 confirmada com d em [-0,13; -0,04]%; B entre 0,15 e 0,7 (central ~0,3), isto e', o achado NAO e' robusto (B < 0,5);
D2 nao confirmada (|d| < 0,05%).
Uso: python -m analysis.report_entrada162_igls_fast
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni, rank_biserial_paired  # noqa: E402

R = ROOT / "results"


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    res = pd.read_csv(R / "results.csv")
    hy = pd.read_csv(R / "igls_p_entrada124.csv")
    eqh = pd.read_csv(R / "igls_eq_fast_entrada162.csv")
    rc = pd.read_csv(R / "igcvls_seed_entrada127.csv")
    eqr = pd.read_csv(R / "igcv_eq_fast_entrada162.csv")
    ver = pd.read_csv(R / "igls_fast_verify_entrada162.csv")
    ratio = json.loads((R / "igls_fast_cpu_ratio_entrada162.json").read_text())
    k129 = json.loads((R / "igls_eq_cpu_ratio_entrada129.json").read_text())
    k127 = json.loads((R / "igcvls_cpu_ratio_entrada127.json").read_text())
    keep = ["instance", "rep", "method", "makespan"]
    H = pd.concat([res[res.method == "ig_random"][keep], hy[hy.method == "igls_random"][keep], eqh[keep]]).drop_duplicates(["instance", "rep", "method"])
    H = H.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().dropna(subset=["igls_random", "ig_random_eq_fast", "ig_random"])
    Rr = pd.concat([rc[rc.method == "igcvls_seed"][keep], eqr[keep]]).drop_duplicates(["instance", "rep", "method"])
    Rr = Rr.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().dropna(subset=["igcvls_seed", "igcv_seed_eq_fast"])
    assert len(H) == 1200 and len(Rr) == 1200, (len(H), len(Rr))
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "SENSIBILIDADE DA CONCLUSAO 'A CPU IGUAL' A IMPLEMENTACAO DA BUSCA LOCAL -- Entrada 162/164", "=" * 110,
         f"equivalencia a nivel de execucao (--verify): {int(ver.equal.sum())}/{len(ver)} makespans identicos aos armazenados"]
    rows = []
    for pid, lab, a, b, df in [("D1", "H: igls_random vs ig_random_eq_fast (CPU igual)", "igls_random", "ig_random_eq_fast", H), ("D2", "R: igcvls_seed vs igcv_seed_eq_fast (CPU igual)", "igcvls_seed", "igcv_seed_eq_fast", Rr)]:
        t = test(df[a], df[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    P["equivalente_0.05"] = (P.ci90_lo > -0.05) & (P.ci90_hi < 0.05)
    L += ["\n--- PRIMARIA (Holm sobre 2; confirmada = a busca local ganha a CPU igual) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05"]].to_string(index=False)]
    q0 = test(H.igls_random, H.ig_random)
    b = 1 - abs(P.set_index("id").loc["D1", "mean_d_pct"]) / abs(q0["mean_d_pct"])
    L.append(f"\nGanho da busca local a iteracoes iguais (T = 1000): d = {q0['mean_d_pct']:+.4f}% ({q0['better']}/{q0['worse']}/{q0['tie']})")
    L.append(f"Fracao de orcamento B = 1 - |d_CPU igual| / |d_iteracoes iguais| = {b:.2f} (Entrada 131, busca local lenta: 0,93)")
    L.append(f"Leituras pre-fixadas: 'o achado a CPU igual e' robusto a implementacao' (B >= 0,5): {'SIM' if b >= 0.5 else 'NAO'} | 'a busca local passa a pagar a CPU igual' (D1 confirmada): "
             f"{'SIM' if P.set_index('id').loc['D1', 'confirmada'] else 'NAO'}")
    kk = pd.DataFrame([dict(classe=c, K_H_antes=k129[c]["K"], K_H_rapida=ratio["H"][c]["K"], K_R_antes=k127[c]["K"], K_R_rapida=ratio["R"][c]["K"]) for c in k129])
    L += ["\n--- razoes de CPU (com/sem busca local), antes (lacos Python) e depois (vetorizada) ---", kk.to_string(index=False),
          f"K medio geometrico: H antes {np.exp(np.log(kk.K_H_antes).mean()):.2f}, H rapida {np.exp(np.log(kk.K_H_rapida).mean()):.2f}; R antes {np.exp(np.log(kk.K_R_antes).mean()):.2f}, R rapida {np.exp(np.log(kk.K_R_rapida).mean()):.2f}"]
    sec = []
    for c, g in H.assign(cl=H.instance).groupby("cl"):
        sec.append(dict(classe=c, d_H_CPU_igual=rel(g.igls_random, g.ig_random_eq_fast).mean(), d_H_iter_igual=rel(g.igls_random, g.ig_random).mean(),
                        d_R_CPU_igual=rel(Rr[Rr.instance == c].igcvls_seed, Rr[Rr.instance == c].igcv_seed_eq_fast).mean()))
    L += ["\n--- por classe (d %) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada162_igls_fast.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada162_igls_fast_primary.csv", index=False)


if __name__ == "__main__":
    main()
