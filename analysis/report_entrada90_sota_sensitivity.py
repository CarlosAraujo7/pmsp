"""
analysis/report_entrada90_sota_sensitivity.py
===============================================
Relatorio da analise de sensibilidade dos baselines SOTA (Entrada 90, item
3). Le results/results.csv (bateria oficial) e
results/sota_sensitivity_entrada90.csv (experiment/run_sota_sensitivity_
entrada90.py) e imprime/salva:

  A. Checagem de DETERMINISMO: makespan do ig_ql re-executado
     (ig_ql_cpu) == makespan armazenado em results.csv, par a par.
  B. gap medio por classe: ig_ql, ig_random, multifit, sa (original),
     sa_noearly, bco (regra do projeto), bco_tm (tempo igual), bco_I100.
  C. cpu_time medio NA MESMA SESSAO (ig_ql_cpu, bco_tm, bco_I100,
     sa_noearly) -- razoes entre metodos, nao valores absolutos.
  D. ig_ql vs cada variante (Wilcoxon pareado, Holm sobre a familia
     inteira de variantes x classes, effect size rank-biserial, IC 95%).
  E. ig_random como referencia contra os SOTA ORIGINAIS (multifit, sa,
     bco) -- fecha a ressalva da Discussao do manuscrito de que a
     comparacao SOTA so' foi calculada com ig_ql como referencia.

Convencao de sinal: mean_diff = mean(ref) - mean(other); direction =
ref_better significa que a REFERENCIA tem gap MENOR.

Uso: python -m analysis.report_entrada90_sota_sensitivity
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import pairwise_wilcoxon_family  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")

main = pd.read_csv(ROOT / "results" / "results.csv")
sens = pd.read_csv(ROOT / "results" / "sota_sensitivity_entrada90.csv")
cls_order = main.groupby("instance").n.first().sort_values().index.tolist()
out = []


def emit(s=""):
    print(s)
    out.append(s)


# ---- A. determinismo -------------------------------------------------------
st = main[main.method == "ig_ql"][["instance", "rep", "makespan"]].rename(columns={"makespan": "stored"})
rr = sens[sens.method == "ig_ql_cpu"][["instance", "rep", "makespan"]].rename(columns={"makespan": "rerun"})
mm = st.merge(rr, on=["instance", "rep"])
emit("=" * 100)
emit(f"A. DETERMINISMO: {len(mm)} pares ig_ql (armazenado vs re-execucao); makespans DIFERENTES: "
     f"{int((mm.stored != mm.rerun).sum())}")
emit("=" * 100)

# ---- B/C. tabelas por classe ----------------------------------------------
sens_g = sens.pivot_table(index="instance", columns="method", values="gap_pct")
main_g = main[main.method.isin(["ig_ql", "ig_random", "multifit", "sa_li_perkins", "bco", "lpt"])] \
    .pivot_table(index="instance", columns="method", values="gap_pct")
tabB = pd.concat([main_g, sens_g.drop(columns=["ig_ql_cpu"])], axis=1).loc[cls_order]
tabB = tabB[["lpt", "ig_ql", "ig_random", "multifit", "sa_li_perkins", "sa_noearly", "bco", "bco_tm", "bco_I100"]]
emit("\nB. gap medio (%) por classe (todas as replicas disponiveis de cada metodo)")
emit(tabB.to_string())
emit("\n   I do BCO usado: regra do projeto=max(1,1000//(5n)); bco_tm = calibrado em tempo igual; bco_I100 = publicado")
I_tab = sens[sens.method.isin(["bco_tm"])].groupby("instance").I_used.first().reindex(cls_order)
emit("   I_tm por classe: " + ", ".join(f"{c.replace('rand_', '')}={int(v)}" for c, v in I_tab.items() if v == v))

cpu = sens.pivot_table(index="instance", columns="method", values="cpu_time").loc[cls_order]
emit("\nC. cpu_time medio (s), MESMA sessao/maquina (ratios importam, valores absolutos nao)")
emit(cpu.to_string())
emit("   razao vs ig_ql_cpu:")
emit((cpu.div(cpu["ig_ql_cpu"], axis=0)).drop(columns=["ig_ql_cpu"]).to_string())

# ---- D. testes pareados -----------------------------------------------------
both = pd.concat([main[["instance", "method", "rep", "n", "m", "gap_pct"]],
                  sens[["instance", "method", "rep", "n", "m", "gap_pct"]]], ignore_index=True)
variants = ["sa_noearly", "bco_tm", "bco_I100"]
D = pairwise_wilcoxon_family(both, "ig_ql", variants, family_name="D: ig_ql vs variantes de sensibilidade")
emit("\n" + "=" * 100)
emit("D. ig_ql vs variantes (Holm sobre %d testes; mean_diff<0 => ig_ql melhor)" % D.p_raw.notna().sum())
emit("=" * 100)
cols = ["other", "n", "m", "n_pairs", "mean_ref", "mean_other", "mean_diff", "ci_low", "ci_high",
        "rank_biserial", "p_raw", "p_holm", "reject_holm", "direction"]
emit(D[cols].to_string(index=False))
for v in variants:
    s = D[D.other == v]
    emit(f"   {v:11s}: ig_ql melhor (Holm-sig) em {int(((s.direction == 'ref_better') & s.reject_holm).sum())}/12 classes; "
         f"pior (Holm-sig) em {int(((s.direction == 'ref_worse') & s.reject_holm).sum())}/12")

# ---- E. ig_random como referencia ------------------------------------------
E = pairwise_wilcoxon_family(main[["instance", "method", "rep", "n", "m", "gap_pct"]], "ig_random",
                             ["multifit", "sa_li_perkins", "bco"], family_name="E: ig_random vs SOTA originais")
emit("\n" + "=" * 100)
emit("E. ig_random (sem aprendizado) como referencia vs SOTA ORIGINAIS (Holm sobre %d testes)" % E.p_raw.notna().sum())
emit("=" * 100)
emit(E[cols].to_string(index=False))
for v in ["multifit", "sa_li_perkins", "bco"]:
    s = E[E.other == v]
    emit(f"   {v:14s}: ig_random melhor (Holm-sig) em {int(((s.direction == 'ref_better') & s.reject_holm).sum())}/12; "
         f"pior em {int(((s.direction == 'ref_worse') & s.reject_holm).sum())}/12")

(ROOT / "results" / "entrada90_sota_sensitivity_report.txt").write_text("\n".join(out), encoding="utf-8")
pd.concat([D, E], ignore_index=True).to_csv(ROOT / "results" / "entrada90_sota_sensitivity_tests.csv", index=False)
print("\nSalvo em results/entrada90_sota_sensitivity_report.txt e ..._tests.csv")
