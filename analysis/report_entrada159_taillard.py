"""
analysis/report_entrada159_taillard.py
========================================
Replicacao do estudo do flowshop nas instancias CLASSICAS de Taillard (Ciclo 12, frente A). Diario: Entrada 159 (pre-registro) e 160 (resultados).

PRE-REGISTRO (congelado ANTES de executar o teste; hashes em results/entrada159_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: results/pfsp_taillard_entrada159.csv (40 instancias: ta041-050, ta051-060, ta071-080, ta081-090; 10 metodos da Entrada 152, T = 300, parametros congelados da Entrada 152, ver
experiment/run_pfsp_taillard_entrada159.py). d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90.
val = o braco fixo escolhido por tamanho no ajuste GERADO da Entrada 152 (results/pfsp_choice_entrada152.json), aplicado sem reajuste.
Familia primaria (a MESMA da Entrada 152; Holm sobre 5, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  P1  mecanismo a CPU igual:       ig_rand vs ils_eq
  P2  busca local a CPU igual:     ig_rand vs ig_nols_eq
  P3  politica online (T igual):   ig_ql vs ig_rand
  P4  validacao vs online:         val vs ig_ql
  P5  folga de d:                  val vs ig_rand
Leitura pre-fixada: 'o achado do flowshop da Entrada 153 se replica nas instancias classicas' significa P1 NAO confirmada (IG e ILS equivalentes a CPU igual), P2 confirmada (a busca local compensa),
P3 NAO confirmada (a politica nao melhora o sorteio) e sinal de P4 favoravel a validacao (d < 0), com ou sem confirmacao (N = 40: menos poder que os 120 da Entrada 152; so' o sinal e a ordem de grandeza de P4 sao exigidos).
Previsoes registradas antes de rodar: P1 nao confirmada (|d| < 0,10%); P2 confirmada (d ~ -0,2%); P3 nao confirmada com d >= 0; P4 com d ~ -0,1 a -0,2% (confirmada com ~50% de chance, N = 40); P5 nao confirmada (|d| < 0,05%).
Descritivas: razao de captura, oraculo por instancia dos 5 bracos fixos, ig_ucb, RPD ao melhor makespan encontrado por metodo e por tamanho, GAP ao limite inferior de Taillard, CPU medio e razoes de CPU realizadas.
Uso: python -m analysis.report_entrada159_taillard
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
    d = pd.read_csv(R / "pfsp_taillard_entrada159.csv")
    ch = json.loads((R / "pfsp_choice_entrada152.json").read_text())
    w = d.pivot(index=KEY, columns="method", values="makespan").reset_index()
    lb = d.drop_duplicates(KEY).set_index(KEY).lb
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    assert len(w) == 40 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    w["val"] = [r[f"ig_fix_{ch['d_val'][f'{int(r.n)}x{int(r.m)}']['action']}"] for _, r in w.iterrows()]
    fx = [f"ig_fix_{k}" for k in range(5)]
    w["oracle"] = w[fx].min(axis=1)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "FLOWSHOP DE PERMUTACAO EM INSTANCIAS CLASSICAS DE TAILLARD (replicacao da Entrada 152) -- Entrada 159/160", "=" * 110, f"instancias: {len(w)} | d validado por tamanho (ajuste gerado): " +
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
    repl = bool((not c.loc["P1", "confirmada"]) and c.loc["P2", "confirmada"] and (not c.loc["P3", "confirmada"]) and c.loc["P4", "mean_d_pct"] < 0)
    L.append(f"\nLeitura pre-fixada: 'o achado do flowshop (Entrada 153) se replica nas instancias classicas' (P1 nao, P2 sim, P3 nao, sinal de P4 favoravel a validacao): {'SIM' if repl else 'NAO'}")
    gain_val = rel(w.val, w.ig_rand).mean()
    L.append("\nGanho sobre ig_rand (d %): " + ", ".join(f"{k} {rel(w[k], w.ig_rand).mean():+.3f}" for k in ["val", "ig_ql", "ig_ucb", "oracle", "ils_eq", "ig_nols_eq"]) +
             (f" | razao de captura do ig_ql: {rel(w.ig_ql, w.ig_rand).mean() / gain_val:.2f}" if gain_val != 0 else ""))
    best = w[["ig_rand", "ig_ql", "ig_ucb", "ils_eq", "ig_nols_eq"] + fx].min(axis=1)
    rpd = lambda k, df=w, b=best: float((100 * (df[k] - b.loc[df.index]) / b.loc[df.index]).mean())
    L.append("RPD medio ao melhor makespan encontrado (%): " + ", ".join(f"{k} {rpd(k):.3f}" for k in ["ig_rand", "ig_ql", "ig_ucb", "val", "oracle", "ils_eq", "ig_nols_eq"]))
    gap = lambda k: float((100 * (w.set_index(KEY)[k] - lb) / lb).mean())
    L.append("GAP medio ao limite inferior de Taillard (%): " + ", ".join(f"{k} {gap(k):.2f}" for k in ["ig_rand", "ig_ql", "ils_eq", "ig_nols_eq"]))
    L.append("CPU medio por execucao (s): " + ", ".join(f"{k} {cpu[k].mean():.1f}" for k in ["ig_rand", "ig_ql", "ils_eq", "ig_nols_eq"]) + f" | razao ils_eq/ig_rand {cpu.ils_eq.mean() / cpu.ig_rand.mean():.2f}; ig_nols_eq/ig_rand {cpu.ig_nols_eq.mean() / cpu.ig_rand.mean():.2f}")
    sec = []
    for (n, m), g in w.groupby(["n", "m"]):
        sec.append(dict(size=f"{int(n)}x{int(m)}", d_val=ch["d_val"][f"{int(n)}x{int(m)}"]["d"], val_vs_rand=rel(g.val, g.ig_rand).mean(), ql_vs_rand=rel(g.ig_ql, g.ig_rand).mean(), oracle_vs_rand=rel(g.oracle, g.ig_rand).mean(),
                        ig_vs_ils=rel(g.ig_rand, g.ils_eq).mean(), ls_vs_nols=rel(g.ig_rand, g.ig_nols_eq).mean()))
    L += ["\n--- por tamanho (d %) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada159_taillard.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada159_taillard_primary.csv", index=False)


if __name__ == "__main__":
    main()
