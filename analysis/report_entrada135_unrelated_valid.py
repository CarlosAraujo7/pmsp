"""
analysis/report_entrada135_unrelated_valid.py
===============================================
Validacao offline vs Q-Learning no R||Cmax 1000 x 50. Diario: Entrada 135.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada135_prereg_hashes.txt)
-------------------------------------------------------------------------------------------
Dados: 96 instancias NOVAS (4 distribuicoes x 24; seed 600000+), n = 1000, m = 50, T = 1000, d/n = 0,02. 'validada' = candidata escolhida por distribuicao no ajuste
(results/valid_choice_entrada135.json; 5 candidatas: random, fix0-3) e avaliada com os resultados do proprio braco no teste. d = 100(A-B)/B; Wilcoxon pareado bicaudal; bootstrap IC90.
Familia (Holm sobre 3, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0): V1 validada vs ur_igls_ql; V2 validada vs ur_igls_random; V3 ur_igls_ql vs ur_igls_random (replicacao de Q1 da Entrada 133).
Leituras pre-fixadas: 'a validacao offline vence o Q-Learning no R||Cmax grande' = V1 confirmada (d < 0); 'equivalentes' = V1 nao confirmada e IC90 de V1 dentro de +-0,10%;
'o Q-Learning vence a validacao' = V1 significativa com d > 0. Descritivas: por distribuicao (escolha, d), CPU medio (validada nao paga o custo do estado), GAP ao LB classico.
Uso: python -m analysis.report_entrada135_unrelated_valid
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
KEY = ["lo", "hi", "n", "m", "rep"]


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
    d = pd.read_csv(R / "unrelated_valid_entrada135.csv")
    choice = json.loads((R / "valid_choice_entrada135.json").read_text())
    w = d.pivot(index=KEY, columns="method", values="makespan").reset_index()
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time").reset_index()
    assert len(w) == 96 and w.isna().sum().sum() == 0, len(w)
    w["valid"] = [r[choice[f"{int(r.lo)}_{int(r.hi)}"]["best"]] for _, r in w.iterrows()]
    cpu["valid"] = [r[choice[f"{int(r.lo)}_{int(r.hi)}"]["best"]] for _, r in cpu.iterrows()]
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "VALIDACAO OFFLINE vs Q-LEARNING (R||Cmax 1000 x 50) -- Entrada 135", "=" * 110, f"instancias: {len(w)}",
         "escolha por distribuicao: " + ", ".join(f"U[{k.replace('_', ',')}] -> {v['best'].replace('ur_igls_', '')}" for k, v in choice.items())]
    rows = []
    for pid, lab, a, b in [("V1", "validada vs ur_igls_ql", "valid", "ur_igls_ql"), ("V2", "validada vs ur_igls_random", "valid", "ur_igls_random"), ("V3", "ur_igls_ql vs ur_igls_random", "ur_igls_ql", "ur_igls_random")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 3) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    v1 = P.set_index("id").loc["V1"]
    if v1.confirmada:
        leit = "a validacao offline VENCE o Q-Learning no R||Cmax grande"
    elif v1.reject and v1.mean_d_pct > 0:
        leit = "o Q-Learning VENCE a validacao"
    elif v1.ci90_lo > -0.10 and v1.ci90_hi < 0.10:
        leit = "equivalentes (IC90 de V1 dentro de +-0,10%)"
    else:
        leit = "inconclusivo (V1 nao confirmada e IC90 fora de +-0,10%)"
    L.append(f"\nLeitura pre-fixada: {leit}")
    L.append("CPU medio por execucao (s): " + ", ".join(f"{k} {cpu[k].mean():.1f}" for k in ["valid", "ur_igls_ql", "ur_igls_random"]))
    lbs = d[d.method == "ur_igls_random"][KEY + ["lb"]]
    wl = w.merge(lbs, on=KEY)
    L.append("GAP medio ao LB classico: " + ", ".join(f"{k} {100 * ((wl[k] - wl.lb) / wl.lb).mean():.3f}%" for k in ["valid", "ur_igls_ql", "ur_igls_random"]))
    sec = []
    for (lo, hi), g in w.groupby(["lo", "hi"]):
        sec.append(dict(dist=f"U[{lo},{hi}]", escolha=choice[f"{lo}_{hi}"]["best"].replace("ur_igls_", ""), valid_vs_random=rel(g.valid, g.ur_igls_random).mean(), ql_vs_random=rel(g.ur_igls_ql, g.ur_igls_random).mean(),
                        valid_vs_ql=rel(g.valid, g.ur_igls_ql).mean()))
    L += ["\n--- por distribuicao (d %) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada135_unrelated_valid.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada135_unrelated_valid_primary.csv", index=False)


if __name__ == "__main__":
    main()
