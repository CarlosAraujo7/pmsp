"""
analysis/report_entrada121_dsel_budget.py
===========================================
A politica captura a folga do espaco de acoes d com mais orcamento ou com pre-treino cruzado? Diario: Entrada 121.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada121_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: 6 classes ((30,8),(50,10),(75,15),(100,20),(150,30),(300,20)) x 50 reps = 300 instancias (reps 0-49 de results.csv). T=5000: dsel_budget_entrada121.csv
(braços d fixos, igd_random, igd_ql_cal, igd_ucb) e AJUSTE do melhor d fixo ao mesmo T em dsel_budget_tuning_entrada121.csv (cvfixed_5000 = d de menor makespan medio
no ajuste, avaliado no teste). T=1000: igd_random e d fixos de dsel_test_entrada114.csv (cvfixed_1000 = escolha do ajuste da Entrada 114) e igd_ql_ws_cal (pre-treinado
por classe em 50 instancias independentes) de dsel_budget_entrada121.csv. d = 100*(A-B)/B (negativo: A melhor); Wilcoxon pareado bicaudal pooled, nulos descartados.
Familia confirmatoria (Holm sobre 4, alpha=0,05; 'confirmada' = p_holm<0,05 e d<0; 'relevante' se d <= -0,05%):
  D1  igd_ql_cal vs igd_random @T=5000       (o aprendizado ONLINE captura folga com 5x o orcamento?)
  D2  cvfixed_5000 vs igd_random @T=5000     (ainda ha' folga em d com T=5000?)
  D3  igd_ql_ws_cal vs igd_random @T=1000    (o aprendizado CRUZADO entre instancias captura folga?)
  D4  igd_ql_ws_cal vs cvfixed_1000 @T=1000  (bicaudal: o pre-treino alcanca o ajuste fixo por validacao?)
Leituras pre-fixadas: 'orcamento maior resolve' = D1 confirmada e relevante; 'pre-treino resolve' = D3 confirmada e relevante e D4 sem diferenca confirmada a favor do
ajuste fixo (p_holm >= 0,05 ou d <= 0). Se nenhuma for satisfeita, o ganho de ajustar d vem de AJUSTE OFFLINE (validacao) e nao de aprendizado online/cruzado do tipo testado.
Secundarias: igd_ucb vs igd_random @T=5000; fracao da folga capturada (random -> cvfixed); por classe.
Uso: python -m analysis.report_entrada121_dsel_budget
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

FIX = ["igd_fix05", "igd_fix10", "igd_fix20", "igd_fix30", "igd_fix40"]


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
    big = pd.read_csv(ROOT / "results" / "dsel_budget_entrada121.csv")
    tune5 = pd.read_csv(ROOT / "results" / "dsel_budget_tuning_entrada121.csv")
    t14 = pd.read_csv(ROOT / "results" / "dsel_test_entrada114.csv")
    tune1 = pd.read_csv(ROOT / "results" / "dsel_tuning_entrada114.csv")
    classes = sorted(big.instance.unique())
    b5 = big[big["T"] == 5000].pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    ws = big[(big["T"] == 1000) & (big.method == "igd_ql_ws_cal")][["instance", "rep", "makespan"]].rename(columns={"makespan": "igd_ql_ws_cal"})
    b1 = t14[t14.instance.isin(classes) & (t14.rep < 50)].pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().merge(ws, on=["instance", "rep"])
    assert len(b5) == 300 and len(b1) == 300 and b5.isna().sum().sum() == 0 and b1[["igd_random", "igd_ql_ws_cal"]].isna().sum().sum() == 0, (len(b5), len(b1))
    ch5 = {c: min(FIX, key=lambda k: g[g.method == k].makespan.mean()) for c, g in tune5.groupby("instance")}
    ch1 = {c: min(FIX, key=lambda k: g[g.method == k].makespan.mean()) for c, g in tune1[tune1.instance.isin(classes)].groupby("instance")}
    b5["cv5000"] = [r[ch5[r["instance"]]] for _, r in b5.iterrows()]
    b1["cv1000"] = [r[ch1[r["instance"]]] for _, r in b1.iterrows()]
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "ORCAMENTO MAIOR E PRE-TREINO NAS ACOES d -- teste pre-registrado, Entrada 121", "=" * 110, f"instancias: {len(b5)}",
         "d escolhido (ajuste) T=5000: " + ", ".join(f"{c.split('_')[1][:-1]}x{c.split('_')[2][:-1]}:{k[-2:]}" for c, k in sorted(ch5.items())),
         "d escolhido (ajuste) T=1000: " + ", ".join(f"{c.split('_')[1][:-1]}x{c.split('_')[2][:-1]}:{k[-2:]}" for c, k in sorted(ch1.items()))]
    rows = []
    for pid, lab, x, y in [("D1", "igd_ql_cal vs igd_random @T=5000", b5.igd_ql_cal, b5.igd_random), ("D2", "cvfixed_5000 vs igd_random @T=5000", b5.cv5000, b5.igd_random),
                           ("D3", "igd_ql_ws_cal vs igd_random @T=1000", b1.igd_ql_ws_cal, b1.igd_random), ("D4", "igd_ql_ws_cal vs cvfixed_1000 @T=1000", b1.igd_ql_ws_cal, b1.cv1000)]:
        t = test(x, y)
        t.update(id=pid, comparison=lab)
        rows.append(t)
    F = pd.DataFrame(rows)
    h = holm_bonferroni(F.p_raw.fillna(1.0).tolist(), alpha=0.05)
    F["p_holm"], F["reject"] = h["p_adjusted"], h["reject"]
    F["confirmada"] = F.reject & (F.mean_d_pct < 0)
    F["relevante"] = F.confirmada & (F.mean_d_pct <= -0.05)
    L += ["\n--- PRIMARIA (Holm sobre 4) ---", F[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "relevante"]].to_string(index=False)]
    r = F.set_index("id")
    L.append(f"\n'orcamento maior resolve' (D1 confirmada e relevante): {'SIM' if r.loc['D1','relevante'] else 'NAO'}")
    d4_ok = (not r.loc["D4", "reject"]) or (r.loc["D4", "mean_d_pct"] <= 0)
    L.append(f"'pre-treino resolve' (D3 confirmada e relevante E D4 sem vantagem confirmada do ajuste fixo): {'SIM' if (r.loc['D3','relevante'] and d4_ok) else 'NAO'}")
    sec = []
    for lab, x, y in [("igd_ucb vs igd_random @T=5000", b5.igd_ucb, b5.igd_random), ("igd_ql_cal vs cvfixed_5000 @T=5000", b5.igd_ql_cal, b5.cv5000)]:
        t = test(x, y)
        t["comparison"] = lab
        sec.append(t)
    L += ["\n--- SECUNDARIAS ---", pd.DataFrame(sec)[["comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw"]].to_string(index=False)]
    rows = []
    for c in classes:
        g5, g1 = b5[b5.instance == c], b1[b1.instance == c]
        rows.append(dict(instance=c, d5_ql=rel(g5.igd_ql_cal, g5.igd_random).mean(), d5_cv=rel(g5.cv5000, g5.igd_random).mean(), d1_ws=rel(g1.igd_ql_ws_cal, g1.igd_random).mean(), d1_cv=rel(g1.cv1000, g1.igd_random).mean()))
    C = pd.DataFrame(rows)
    L += ["\n--- por classe (d % vs igd_random): QL online T=5000, cvfixed T=5000, QL pre-treinado T=1000, cvfixed T=1000 ---", C.to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada121_dsel_budget.txt").write_text(text, encoding="utf-8")
    F.to_csv(ROOT / "results" / "entrada121_dsel_budget_primary.csv", index=False)


if __name__ == "__main__":
    main()
