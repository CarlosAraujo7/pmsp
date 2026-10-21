"""
analysis/report_entrada168_bigtrain.py
========================================
Mais dados de treino offline (280 -> 700 instancias) melhoram o seletor instancia-condicional? Ciclo 13, frente G. Diario: Entrada 168 (pre-registro) e 169 (resultados).

PRE-REGISTRO (congelado ANTES de rodar os bracos da reserva e de treinar; hashes em results/entrada168_prereg_hashes.txt)
-------------------------------------------------------------------------------------------------------------------------
Treino: tune (280) + reserva (420; bracos fixos novos em results/reserve_arms_entrada168.csv) = 700 instancias, disjuntas das 700 de teste. Selecionadores treinados so' com o treino (`run_reserve_arms_entrada168.py --train`; saida results/bigtrain_choice_entrada168.json):
val_tn700 (melhor braco por (tipo, n) no treino de 700), val_knn700 (kNN k = 7 ponderado, 8 features da matriz sem o tipo) e val_mlp700 (MLP 16x16, alvos padronizados por braco), os mesmos modelos das Entradas 159/163. Teste: as 700 instancias de
results/dpolicy_test_entrada145.csv (bracos fixos e sorteio). d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90.
Familia primaria (Holm sobre 4, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  H1  val_knn700 vs val_tn700 ; H2  val_mlp700 vs val_tn700 ; H3  val_tn700 vs val_tn (validacao por celula com 280 -> 700) ; H4  val_tn700 vs rand_d
Leitura pre-fixada: 'mais dados de treino fazem a selecao instancia-condicional superar a validacao por celula' = H1 ou H2 confirmada. Descritivas: curva de aprendizado (kNN e MLP com 280 vs 700; os de 280 sao os das Entradas 159 e 163) contra a validacao por celula.
Previsoes registradas antes de rodar: H1 e H2 NAO confirmadas, com d >= 0 (o acrescimo de dados diminui a diferenca em relacao a +0,10% das Entradas 159/163, mas nao a inverte; d em [0; +0,08]); H3 nao confirmada (|d| < 0,03%); H4 confirmada (d ~ -0,2%).
Uso: python -m analysis.report_entrada168_bigtrain
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
F = [0.01, 0.02, 0.05, 0.10, 0.20]
KEY = ["type", "instance"]


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    d = pd.read_csv(R / "dpolicy_test_entrada145.csv", dtype={"instance": str})
    w = d.pivot(index=KEY + ["m", "n"], columns="method", values="makespan").reset_index()
    assert len(w) == 700 and w.isna().sum().sum() == 0
    ch = json.loads((R / "bigtrain_choice_entrada168.json").read_text())
    c159 = json.loads((R / "ctx_choice_entrada159.json").read_text())
    c163 = json.loads((R / "neural_choice_entrada162.json").read_text())
    tune = pd.read_csv(R / "lpig_tune_d_entrada141.csv", dtype={"instance": str})
    tune["frac"] = tune.method.str.split("@").str[1].astype(float)
    tune["ratio"] = tune.makespan / tune.best_known
    val_tn = {(t, int(n)): F.index(float(g.groupby("frac").ratio.mean().idxmin())) for (t, n), g in tune.groupby(["type", "n"])}
    arm = lambda r, k: r[f"fix_{k}"]
    w["val_tn"] = [arm(r, val_tn[(r["type"], int(r["n"]))]) for _, r in w.iterrows()]
    w["val_tn700"] = [arm(r, ch["val_tn700"][f"{r['type']}|{int(r['n'])}"]) for _, r in w.iterrows()]
    w["val_knn700"] = [arm(r, ch["knn"][r["type"] + "|" + r["instance"]]) for _, r in w.iterrows()]
    w["val_mlp700"] = [arm(r, ch["mlp"][r["type"] + "|" + r["instance"]]) for _, r in w.iterrows()]
    w["val_knn280"] = [arm(r, c159["pred"][r["type"] + "|" + r["instance"]]) for _, r in w.iterrows()]
    w["val_mlp280"] = [arm(r, c163["pred"][r["type"] + "|" + r["instance"]]) for _, r in w.iterrows()]
    w["oracle"] = w[[f"fix_{k}" for k in range(5)]].min(axis=1)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "MAIS DADOS DE TREINO OFFLINE (280 -> 700): SELETOR INSTANCIA-CONDICIONAL vs VALIDACAO POR CELULA -- Entrada 168/169", "=" * 110, f"instancias de teste: {len(w)} | treino: {ch['n_train']}"]
    rows = []
    for pid, lab, a, b in [("H1", "val_knn700 vs val_tn700", "val_knn700", "val_tn700"), ("H2", "val_mlp700 vs val_tn700", "val_mlp700", "val_tn700"), ("H3", "val_tn700 vs val_tn (280)", "val_tn700", "val_tn"),
                           ("H4", "val_tn700 vs rand_d", "val_tn700", "rand_d")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 4) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    c = P.set_index("id")
    L.append(f"\nLeitura pre-fixada: 'mais dados de treino fazem a selecao instancia-condicional superar a validacao por celula' (H1 ou H2 confirmada): {'SIM' if (c.loc['H1', 'confirmada'] or c.loc['H2', 'confirmada']) else 'NAO'}")
    L.append("\nCurva de aprendizado (d % contra a validacao por celula com o MESMO treino):")
    for lab, a, b in [("kNN 280", "val_knn280", "val_tn"), ("kNN 700", "val_knn700", "val_tn700"), ("MLP 280", "val_mlp280", "val_tn"), ("MLP 700", "val_mlp700", "val_tn700")]:
        t = test(w[a], w[b])
        L.append(f"  {lab}: d = {t['mean_d_pct']:+.3f}% [IC90 {t['ci90_lo']:+.3f}, {t['ci90_hi']:+.3f}] ({t['better']}/{t['worse']}/{t['tie']})")
    g = lambda k: rel(w[k], w.rand_d).mean()
    L.append("Ganho sobre rand_d (d %): " + ", ".join(f"{k} {g(k):+.3f}" for k in ["val_tn", "val_tn700", "val_knn280", "val_knn700", "val_mlp280", "val_mlp700", "oracle"]))
    text = "\n".join(L)
    print(text)
    (R / "entrada168_bigtrain.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada168_bigtrain_primary.csv", index=False)


if __name__ == "__main__":
    main()
