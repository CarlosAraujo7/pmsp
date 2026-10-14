"""
analysis/report_entrada124_unrelated_law.py
=============================================
R||Cmax em instancias NOVAS: confirmacao da lei da folga e ablacao estado x TD. Diario: Entrada 124.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada124_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: results/unrelated_law_entrada124.csv: 20 celulas x 20 reps = 400 instancias NOVAS (seed 200000+), 9 metodos. d = 100*(A-B)/B em makespan (negativo: A melhor).
folga_cel = d medio do ORACULO por instancia min(fix0..fix3) vs ur_igls_random; politica_cel = d medio de ur_igls_ql vs ur_igls_random.
Familia confirmatoria (Holm sobre 6, alpha=0,05; 'confirmada' = p_holm<0,05 e o sentido indicado):
  L1  Spearman(folga_cel, politica_cel) entre as 20 celulas > 0 (a politica acompanha a folga); 'forte' se rho >= 0,5
  L2  ur_igls_ql vs ur_igls_random (instancia, Wilcoxon bicaudal; replicacao do M2 da Entrada 118), d < 0
  L3  ur_igls_ql vs ur_igls_ql_flat    (condicionar no ESTADO ajuda?)                d < 0
  L4  ur_igls_ql vs ur_igls_ucb_ctx    (TD-learning supera um bandit por estado?)    d < 0
  L5  ur_igls_ql_flat vs ur_igls_random (TD sem estado ja' ajuda?)                   d < 0
  L6  ur_igls_ucb vs ur_igls_random    (bandit sem estado ajuda?)                    d < 0
Leituras pre-fixadas: 'a lei da folga se confirma' = L1 confirmada com rho >= 0,5; 'o ganho vem do estado' = L3 confirmada e L5 nao; 'o ganho vem do TD sem estado' = L5 confirmada e L3 nao;
'os dois' = L3 e L5 confirmadas; 'nenhum isolado' = nenhuma. Razao agregada politica/folga = mean(d_ql)/mean(d_oracle) com IC90 (bootstrap sobre instancias).
Secundarias: por celula; ucb_ctx vs random; razao por distribuicao.
Uso: python -m analysis.report_entrada124_unrelated_law
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

KEY = ["lo", "hi", "n", "m", "rep"]
FIX = ["ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3"]


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def inst_test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    d = pd.read_csv(ROOT / "results" / "unrelated_law_entrada124.csv")
    w = d.pivot(index=KEY, columns="method", values="makespan").reset_index()
    assert len(w) == 400 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "R||Cmax NOVAS: LEI DA FOLGA E ABLACAO ESTADO x TD -- teste pre-registrado, Entrada 124", "=" * 110, f"instancias: {len(w)}"]
    cells = []
    for (lo, hi, n, m), g in w.groupby(["lo", "hi", "n", "m"]):
        cells.append(dict(dist=f"U[{lo},{hi}]", size=f"{n}x{m}", folga=rel(g[FIX].min(axis=1), g.ur_igls_random).mean(), politica=rel(g.ur_igls_ql, g.ur_igls_random).mean(),
                          flat=rel(g.ur_igls_ql_flat, g.ur_igls_random).mean(), ucb=rel(g.ur_igls_ucb, g.ur_igls_random).mean(), ucb_ctx=rel(g.ur_igls_ucb_ctx, g.ur_igls_random).mean()))
    C = pd.DataFrame(cells)
    rho, prho = sstats.spearmanr(C.folga, C.politica)
    rows = [dict(id="L1", comparison="Spearman(folga, politica) entre 20 celulas", n=len(C), mean_d_pct=float(rho), p_raw=float(prho / 2 if rho > 0 else 1 - prho / 2))]
    for pid, lab, a, b in [("L2", "ur_igls_ql vs ur_igls_random", "ur_igls_ql", "ur_igls_random"), ("L3", "ur_igls_ql vs ur_igls_ql_flat", "ur_igls_ql", "ur_igls_ql_flat"),
                           ("L4", "ur_igls_ql vs ur_igls_ucb_ctx", "ur_igls_ql", "ur_igls_ucb_ctx"), ("L5", "ur_igls_ql_flat vs ur_igls_random", "ur_igls_ql_flat", "ur_igls_random"),
                           ("L6", "ur_igls_ucb vs ur_igls_random", "ur_igls_ucb", "ur_igls_random")]:
        t = inst_test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    F = pd.DataFrame(rows)
    h = holm_bonferroni(F.p_raw.fillna(1.0).tolist(), alpha=0.05)
    F["p_holm"], F["reject"] = h["p_adjusted"], h["reject"]
    F["confirmada"] = F.reject & np.where(F.id == "L1", F.mean_d_pct > 0, F.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 6; L1: mean_d_pct = rho de Spearman, p unilateral) ---", F[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    l1 = F[F.id == "L1"].iloc[0]
    L.append(f"\n'a lei da folga se confirma' (L1 confirmada com rho >= 0,5): {'SIM' if (l1.confirmada and l1.mean_d_pct >= 0.5) else 'NAO'} (rho = {rho:.3f})")
    c = F.set_index("id").confirmada
    verdict = ("os dois (estado e TD)" if (c["L3"] and c["L5"]) else "o estado" if (c["L3"] and not c["L5"]) else "o TD sem estado" if (c["L5"] and not c["L3"]) else "nenhum isolado")
    L.append(f"origem do ganho da politica (L3, L5): {verdict}")
    num = rel(w.ur_igls_ql, w.ur_igls_random)
    den = rel(w[FIX].min(axis=1), w.ur_igls_random)
    idx = np.random.default_rng(0).integers(0, len(w), size=(2000, len(w)))
    ratios = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    L.append(f"razao agregada politica/folga = {num.mean() / den.mean():.3f} IC90 [{np.percentile(ratios, 5):.3f}, {np.percentile(ratios, 95):.3f}]; folga agregada (oraculo) d = {den.mean():+.4f}%; politica d = {num.mean():+.4f}%")
    sec = []
    for dist, g in w.groupby(["lo", "hi"]):
        a = rel(g.ur_igls_ql, g.ur_igls_random).mean()
        b = rel(g[FIX].min(axis=1), g.ur_igls_random).mean()
        sec.append(dict(dist=f"U[{dist[0]},{dist[1]}]", politica=a, folga=b, razao=a / b if abs(b) > 1e-9 else np.nan))
    L += ["\n--- por distribuicao ---", pd.DataFrame(sec).to_string(index=False), "\n--- por celula (d % vs ur_igls_random): folga (oraculo), ql, ql_flat, ucb, ucb_ctx ---", C.to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada124_unrelated_law.txt").write_text(text, encoding="utf-8")
    F.to_csv(ROOT / "results" / "entrada124_unrelated_law_primary.csv", index=False)
    C.to_csv(ROOT / "results" / "entrada124_unrelated_law_by_cell.csv", index=False)


if __name__ == "__main__":
    main()
