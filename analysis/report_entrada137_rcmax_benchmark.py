"""
analysis/report_entrada137_rcmax_benchmark.py
===============================================
Validacao EXTERNA em R||Cmax no benchmark padrao de Fanjul-Peyro e Ruiz (700 instancias de teste). Diario: Entrada 137.

PRE-REGISTRO (congelado ANTES de executar o teste; hashes em results/entrada137_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: results/rcmax_test_entrada137.csv (700 instancias de teste = 7 tipos x 5 (m) x 4 (n) x 5 replicacoes, idx 2-6 de cada celula), melhor valor conhecido = min(LR, Gurobi, CPLEX).
d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled sobre as 700 instancias; nulos descartados; bootstrap IC90.
Familia primaria (Holm sobre 5, alpha = 0,05; 'confirmada' = p_holm < 0,05 e o sinal pre-fixado):
  B1  mecanismo a CPU igual: ur_igls_random vs ur_ils_random_eq   (confirmada se d < 0: o IG+LS vence o ILS a CPU igual; replica M1/RB2)
  B2a politica a T igual:     ur_igls_ql vs ur_igls_random        (confirmada se d < 0)
  B2b politica a CPU igual:   ur_igls_ql vs ur_igls_random_eq     (confirmada se d < 0)
  B3  lei da folga: Spearman entre a folga da celula (tipo x n; 28 celulas; folga = d medio do oraculo por instancia dos 4 bracos fixos vs random) e o ganho da politica da celula
      (d medio de ur_igls_ql vs ur_igls_random); p unilateral (rho > 0); 'confirmada' = p_holm < 0,05 e rho >= 0,5
  B4  validacao vs politica:  braco validado por tipo (results/rcmax_choice_arm_entrada137.json) vs ur_igls_ql   (confirmada se d < 0: a validacao vence o Q-Learning)
Leituras pre-fixadas: 'a decomposicao mecanismo x politica se replica no benchmark padrao' = B1 confirmada e B2a, B2b nao confirmadas ou |d| < 0,10%;
'o aprendizado ajuda' = B2a e B2b confirmadas com d <= -0,10%; B4 nao confirmada com IC90 dentro de +-0,10% = 'equivalentes'.
Descritivas (nao confirmatorias): GAP ao melhor conhecido por metodo, por n e por tipo; proporcao de instancias em que o IG+LS-QL iguala ou bate o melhor conhecido; proporcao de
referencias provadas otimas; politica por tipo (d, W/L/T); CPU medio por metodo; B1 tambem a T igual.
Uso: python -m analysis.report_entrada137_rcmax_benchmark
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
    d = pd.read_csv(R / "rcmax_test_entrada137.csv")
    man = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    d["instance"] = d.instance.astype(str)
    w = d.pivot(index=KEY + ["m", "n"], columns="method", values="makespan").reset_index().merge(man[KEY + ["best_known", "proven_opt"]], on=KEY)
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    assert len(w) == 700 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    choice = json.loads((R / "rcmax_choice_arm_entrada137.json").read_text())
    w["valid"] = [r[choice[r["type"]]["best"]] for _, r in w.iterrows()]
    fix = w[[f"ur_igls_fix{k}" for k in range(4)]].min(axis=1)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "VALIDACAO EXTERNA EM R||Cmax (benchmark de Fanjul-Peyro e Ruiz) -- Entrada 137", "=" * 110,
         f"instancias de teste: {len(w)} | referencia provada otima em {100 * w.proven_opt.mean():.1f}%",
         "braco validado por tipo: " + ", ".join(f"{k} -> {v['best'].replace('ur_igls_', '')}" for k, v in choice.items())]
    rows = []
    for pid, lab, a, b in [("B1", "ur_igls_random vs ur_ils_random_eq (CPU igual)", "ur_igls_random", "ur_ils_random_eq"), ("B2a", "ur_igls_ql vs ur_igls_random (T igual)", "ur_igls_ql", "ur_igls_random"),
                           ("B2b", "ur_igls_ql vs ur_igls_random_eq (CPU igual)", "ur_igls_ql", "ur_igls_random_eq"), ("B4", "validada vs ur_igls_ql", "valid", "ur_igls_ql")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    w["cell"] = w["type"] + "|" + w["n"].astype(str)
    cell = pd.DataFrame({"hd": rel(fix, w.ur_igls_random), "pol": rel(w.ur_igls_ql, w.ur_igls_random), "cell": w.cell}).groupby("cell").mean()
    rho, p_two = sstats.spearmanr(cell.hd, cell.pol)
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    rows.append(dict(id="B3", comparison=f"Spearman(folga, politica) entre {len(cell)} celulas (tipo x n)", n=len(cell), better=np.nan, worse=np.nan, tie=np.nan, mean_d_pct=float(rho), ci90_lo=np.nan,
                     ci90_hi=np.nan, rank_biserial=np.nan, p_raw=float(p_one)))
    P = pd.DataFrame(rows).set_index("id").loc[["B1", "B2a", "B2b", "B3", "B4"]].reset_index()
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = [bool(r.reject and ((r.mean_d_pct >= 0.5) if r.id == "B3" else (r.mean_d_pct < 0))) for r in P.itertuples()]
    L += ["\n--- PRIMARIA (Holm sobre 5; B3: coluna mean_d_pct = rho de Spearman, p unilateral) ---",
          P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    c = P.set_index("id")
    repl = bool(c.loc["B1", "confirmada"] and all((not c.loc[k, "confirmada"]) or abs(c.loc[k, "mean_d_pct"]) < 0.10 for k in ("B2a", "B2b")))
    helps = bool(c.loc["B2a", "confirmada"] and c.loc["B2b", "confirmada"] and c.loc["B2a", "mean_d_pct"] <= -0.10 and c.loc["B2b", "mean_d_pct"] <= -0.10)
    b4 = c.loc["B4"]
    eqv = (not b4.confirmada) and b4.ci90_lo > -0.10 and b4.ci90_hi < 0.10
    L.append(f"\nLeituras pre-fixadas: 'a decomposicao se replica no benchmark padrao': {'SIM' if repl else 'NAO'} | 'o aprendizado ajuda': {'SIM' if helps else 'NAO'} | "
             f"B4 validada vs Q-Learning: {'validacao vence' if b4.confirmada else ('equivalentes' if eqv else 'inconclusivo')}")
    L.append(f"B1 a T igual (secundaria): d = {test(w.ur_igls_random, w.ur_ils_random)['mean_d_pct']:+.3f}%")
    L.append("\nCPU medio por execucao (s): " + ", ".join(f"{k.replace('ur_', '')} {cpu[k].mean():.1f}" for k in ["ur_igls_random", "ur_igls_ql", "ur_igls_random_eq", "ur_ils_random", "ur_ils_random_eq"]))
    L.append("GAP medio ao melhor conhecido (%): " + ", ".join(f"{k.replace('ur_', '')} {100 * ((w[k] - w.best_known) / w.best_known).mean():.3f}" for k in ["mct", "ur_ils_random_eq", "ur_igls_random", "ur_igls_ql", "ur_igls_random_eq", "valid"]))
    L.append(f"IG+LS-QL <= melhor conhecido em {100 * (w.ur_igls_ql <= w.best_known).mean():.1f}% das instancias (< em {100 * (w.ur_igls_ql < w.best_known).mean():.1f}%); entre as referencias provadas otimas: "
             f"{100 * (w[w.proven_opt == 1].ur_igls_ql <= w[w.proven_opt == 1].best_known).mean():.1f}%")
    by_n = w.groupby("n").apply(lambda g: pd.Series({k: 100 * ((g[k] - g.best_known) / g.best_known).mean() for k in ["ur_igls_random", "ur_igls_ql", "ur_ils_random_eq"]}))
    by_t = w.groupby("type").apply(lambda g: pd.Series({k: 100 * ((g[k] - g.best_known) / g.best_known).mean() for k in ["ur_igls_random", "ur_igls_ql", "ur_ils_random_eq"]}))
    L += ["\n--- GAP ao melhor conhecido por n ---", by_n.to_string(), "\n--- GAP ao melhor conhecido por tipo ---", by_t.to_string()]
    sec = []
    for typ, g in w.groupby("type"):
        t1, t2 = test(g.ur_igls_ql, g.ur_igls_random), test(g.ur_igls_ql, g.ur_igls_random_eq)
        sec.append(dict(type=typ, folga=rel(g[[f"ur_igls_fix{k}" for k in range(4)]].min(axis=1), g.ur_igls_random).mean(), ql_vs_random=t1["mean_d_pct"], ql_better=t1["better"], ql_worse=t1["worse"],
                        ql_vs_eq=t2["mean_d_pct"], valid_vs_ql=rel(g.valid, g.ur_igls_ql).mean(), mecanismo_vs_ils_eq=rel(g.ur_igls_random, g.ur_ils_random_eq).mean(), escolha=choice[typ]["best"].replace("ur_igls_", "")))
    L += ["\n--- por tipo (d %; descritivo) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada137_rcmax_benchmark.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada137_rcmax_benchmark_primary.csv", index=False)
    cell.to_csv(R / "entrada137_rcmax_benchmark_cells.csv")


if __name__ == "__main__":
    main()
