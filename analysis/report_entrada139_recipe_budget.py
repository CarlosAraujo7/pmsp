"""
analysis/report_entrada139_recipe_budget.py
=============================================
Curva qualidade x orcamento da receita sem aprendizado (H3 da auditoria do Ciclo 4). Diario: Entrada 139.

PRE-REGISTRO (descritivo; congelado ANTES de executar; hashes em results/entrada139_prereg_hashes.txt)
-------------------------------------------------------------------------------------------------------
Dados: 6 classes x 100 reps; T in {250, 1000 (igcv_seed da Entrada 121), 4000, 16000, 64000}; LB certificado (lb_lp de exact_reference_synthetic_entrada101.csv).
Verificacoes de integridade (devem valer por construcao): (i) monotonia: makespan(T maior) <= makespan(T menor) em TODAS as instancias; (ii) o ponto T = 1000 reproduz igcv_seed.
Descritivas por classe e T: GAP medio vs LB certificado, % de otimos provados (makespan == lb_lp), % de instancias melhoradas vs T = 1000, CPU medio por execucao (paralelo; indicativo).
Sem hipotese confirmatoria (a monotonia e' estrutural); a pergunta e' ONDE satura: reporta-se o menor T em que o GAP medio da classe fica dentro de 0,005 ponto do GAP de T = 64000.
Uso: python -m analysis.report_entrada139_recipe_budget
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
TS = [250, 1000, 4000, 16000, 64000]


def main() -> None:
    b = pd.read_csv(R / "recipe_budget_entrada139.csv")
    cv = pd.read_csv(R / "igcv_seed_entrada121.csv")
    ref = pd.read_csv(R / "exact_reference_synthetic_entrada101.csv")[["instance", "rep", "lb_lp"]]
    classes = sorted(b.instance.unique(), key=lambda s: (int(s.split("_")[1][:-1]), int(s.split("_")[2][:-1])))
    cv = cv[cv.instance.isin(classes)].assign(T=1000, method="igcv_seed_T1000")[["instance", "method", "rep", "makespan", "cpu_time", "T"]]
    d = pd.concat([b[["instance", "method", "rep", "makespan", "cpu_time", "T"]], cv]).drop_duplicates(["instance", "rep", "T"])
    W = d.pivot(index=["instance", "rep"], columns="T", values="makespan").reset_index().merge(ref, on=["instance", "rep"], how="left")
    cpu = d.pivot_table(index="instance", columns="T", values="cpu_time", aggfunc="mean")
    assert len(W) == 600 and W[TS].isna().sum().sum() == 0, (len(W), W[TS].isna().sum().to_dict())
    viol = sum(int((W[TS[i + 1]] > W[TS[i]]).sum()) for i in range(len(TS) - 1))
    L = ["=" * 110, "CURVA QUALIDADE x ORCAMENTO DA RECEITA SEM APRENDIZADO -- Entrada 139 (descritivo)", "=" * 110, f"instancias: {len(W)} (6 classes x 100) | violacoes de monotonia: {viol} (esperado 0)"]
    rows = []
    for inst, g in W.groupby("instance"):
        row = dict(classe=inst.replace("rand_", "").replace("j_", "x").replace("m", ""))
        gaps = {T: 100 * ((g[T] - g.lb_lp) / g.lb_lp).mean() for T in TS}
        for T in TS:
            row[f"GAP@{T}"] = gaps[T]
        for T in TS:
            row[f"opt%@{T}"] = 100 * (g[T] == g.lb_lp).mean()
        row["melhora_64000_vs_1000_%"] = 100 * (g[64000] < g[1000]).mean()
        sat = [T for T in TS if gaps[T] - gaps[64000] <= 0.005]
        row["T_satura"] = min(sat) if sat else np.nan
        for T in TS:
            row[f"cpu@{T}"] = cpu.loc[inst, T]
        rows.append(row)
    D = pd.DataFrame(rows)
    pool = {"classe": "pooled"}
    gp = W.assign(**{f"g{T}": 100 * ((W[T] - W.lb_lp) / W.lb_lp) for T in TS})
    for T in TS:
        pool[f"GAP@{T}"] = gp[f"g{T}"].mean()
        pool[f"opt%@{T}"] = 100 * (W[T] == W.lb_lp).mean()
    pool["melhora_64000_vs_1000_%"] = 100 * (W[64000] < W[1000]).mean()
    D = pd.concat([D, pd.DataFrame([pool])], ignore_index=True)
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.4g}")
    L += ["\n--- por classe (GAP vs LB certificado em %, otimo provado em %, CPU em s) ---", D.to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada139_recipe_budget.txt").write_text(text, encoding="utf-8")
    D.to_csv(R / "entrada139_recipe_budget_table.csv", index=False)


if __name__ == "__main__":
    main()
