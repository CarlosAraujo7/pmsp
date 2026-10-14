"""
analysis/report_entrada152_pfsp.py
====================================
Terceiro problema: flowshop de permutacao (Fm|prmu|Cmax), IG de Ruiz e Stutzle com selecao do tamanho de destruicao. Diario: Entrada 152.

PRE-REGISTRO (congelado ANTES de executar o teste; hashes em results/entrada152_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: results/pfsp_test_entrada152.csv (120 instancias de teste: 4 tamanhos x 30; 10 metodos: ig_fix_0..4, ig_rand, ig_ql, ig_ucb, ils_eq, ig_nols_eq). d = 100*(A-B)/B em makespan (negativo: A melhor);
Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90. val = o braco fixo escolhido por tamanho no ajuste (results/pfsp_choice_entrada152.json).
Familia primaria (Holm sobre 5, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  P1  mecanismo a CPU igual:       ig_rand vs ils_eq          (o IG com busca local vence o ILS com a mesma busca local)
  P2  busca local a CPU igual:     ig_rand vs ig_nols_eq      (a busca local compensa seu custo: IG com LS vs IG sem LS com T_nols iteracoes)
  P3  politica online (T igual):   ig_ql vs ig_rand
  P4  validacao vs online:         val vs ig_ql               (a escolha offline de d vence o Q-Learning online)
  P5  folga de d:                  val vs ig_rand             (escolher d offline captura folga sobre sortea-lo)
Leituras pre-fixadas: 'a decomposicao mecanismo x politica se replica no flowshop' = P1 confirmada E P3 nao confirmada (ou |d| < 0,10%); 'a busca local compensa a CPU igual' = P2 confirmada;
'a validacao offline vence a politica online' = P4 confirmada.
Descritivas: razao de captura (ganho de ig_ql sobre ig_rand / ganho de val sobre ig_rand), oraculo por instancia dos 5 bracos fixos, ig_ucb, RPD ao melhor makespan encontrado (entre todos os metodos) por metodo e por tamanho,
GAP ao limite inferior de Taillard, CPU medio e razoes de CPU realizadas.
Uso: python -m analysis.report_entrada152_pfsp
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
KEY = ["n", "m", "rep"]


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
    d = pd.read_csv(R / "pfsp_test_entrada152.csv")
    ch = json.loads((R / "pfsp_choice_entrada152.json").read_text())
    w = d.pivot(index=KEY, columns="method", values="makespan").reset_index()
    lb = d.drop_duplicates(KEY).set_index(KEY).lb
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    assert len(w) == 120 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    w["val"] = [r[f"ig_fix_{ch['d_val'][f'{int(r.n)}x{int(r.m)}']['action']}"] for _, r in w.iterrows()]
    fx = [f"ig_fix_{k}" for k in range(5)]
    w["oracle"] = w[fx].min(axis=1)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "TERCEIRO PROBLEMA: FLOWSHOP DE PERMUTACAO (IG de Ruiz e Stutzle) -- Entrada 152", "=" * 110, f"instancias de teste: {len(w)} | d validado por tamanho: " +
         ", ".join(f"{k} -> {v['d']}" for k, v in ch["d_val"].items())]
    rows = []
    for pid, lab, a, b in [("P1", "ig_rand vs ils_eq (mecanismo, CPU igual)", "ig_rand", "ils_eq"), ("P2", "ig_rand vs ig_nols_eq (busca local, CPU igual)", "ig_rand", "ig_nols_eq"),
                           ("P3", "ig_ql vs ig_rand (politica online)", "ig_ql", "ig_rand"), ("P4", "val vs ig_ql (validacao vs online)", "val", "ig_ql"), ("P5", "val vs ig_rand (folga de d)", "val", "ig_rand")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 5) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    c = P.set_index("id")
    repl = bool(c.loc["P1", "confirmada"] and ((not c.loc["P3", "confirmada"]) or abs(c.loc["P3", "mean_d_pct"]) < 0.10))
    L.append(f"\nLeituras pre-fixadas: 'a decomposicao se replica no flowshop' (P1 e nao P3): {'SIM' if repl else 'NAO'} | 'a busca local compensa a CPU igual' (P2): {'SIM' if c.loc['P2', 'confirmada'] else 'NAO'} | "
             f"'a validacao offline vence a politica online' (P4): {'SIM' if c.loc['P4', 'confirmada'] else 'NAO'}")
    gain_val = rel(w.val, w.ig_rand).mean()
    L.append("\nGanho sobre ig_rand (d %): " + ", ".join(f"{k} {rel(w[k], w.ig_rand).mean():+.3f}" for k in ["val", "ig_ql", "ig_ucb", "oracle", "ils_eq", "ig_nols_eq"]) +
             f" | razao de captura do ig_ql: {rel(w.ig_ql, w.ig_rand).mean() / gain_val:.2f}" if gain_val != 0 else "")
    best = w[["ig_rand", "ig_ql", "ig_ucb", "ils_eq", "ig_nols_eq"] + fx].min(axis=1)
    rpd = lambda k, df=w, b=best: float((100 * (df[k] - b.loc[df.index]) / b.loc[df.index]).mean())
    L.append("RPD medio ao melhor makespan encontrado (%): " + ", ".join(f"{k} {rpd(k):.3f}" for k in ["ig_rand", "ig_ql", "ig_ucb", "val", "oracle", "ils_eq", "ig_nols_eq"]))
    gap = lambda k: float((100 * (w.set_index(KEY)[k] - lb) / lb).mean())
    L.append("GAP medio ao limite inferior (%): " + ", ".join(f"{k} {gap(k):.2f}" for k in ["ig_rand", "ig_ql", "ils_eq", "ig_nols_eq"]))
    L.append("CPU medio por execucao (s): " + ", ".join(f"{k} {cpu[k].mean():.1f}" for k in ["ig_rand", "ig_ql", "ils_eq", "ig_nols_eq"]) + f" | razao ils_eq/ig_rand {cpu.ils_eq.mean() / cpu.ig_rand.mean():.2f}; ig_nols_eq/ig_rand {cpu.ig_nols_eq.mean() / cpu.ig_rand.mean():.2f}")
    sec = []
    for (n, m), g in w.groupby(["n", "m"]):
        sec.append(dict(size=f"{int(n)}x{int(m)}", d_val=ch["d_val"][f"{int(n)}x{int(m)}"]["d"], val_vs_rand=rel(g.val, g.ig_rand).mean(), ql_vs_rand=rel(g.ig_ql, g.ig_rand).mean(), oracle_vs_rand=rel(g.oracle, g.ig_rand).mean(),
                        ig_vs_ils=rel(g.ig_rand, g.ils_eq).mean(), ls_vs_nols=rel(g.ig_rand, g.ig_nols_eq).mean()))
    L += ["\n--- por tamanho (d %) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada152_pfsp.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada152_pfsp_primary.csv", index=False)


if __name__ == "__main__":
    main()
