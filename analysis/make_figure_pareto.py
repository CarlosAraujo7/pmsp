"""
analysis/make_figure_pareto.py (Entrada 107)
Figura Pareto qualidade x tempo (paper/fig_pareto.pdf) e tabela de razoes de CPU (results/entrada107_cpu_ratios.txt).
  - Qualidade: GAP medio (%) em relacao ao LB arc-flow certificado, sobre as repeticoes completas (results.csv,
    seeded_baselines_entrada102.csv, sota_sensitivity_entrada90.csv para bco_I100).
  - Tempo: mediana de CPU (s) na MESMA sessao, sequencial (results/timing_entrada107.csv; 5 reps por classe).
Tres paineis por tamanho: n <= 40 (4 classes), 50 <= n <= 100 (4 classes), n >= 150 (4 classes). Cada ponto = metodo, media
(GAP) e media geometrica (CPU) das 4 classes do painel. A fronteira de Pareto (menor GAP para cada tempo) e' tracejada.
Uso: python -m analysis.make_figure_pareto
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABEL = {"lpt": "LPT", "multifit": "MultiFit", "ldm": "LDM", "sa_li_perkins": "SA", "bco": "BCO", "bco_I100": "BCO (I=100)",
         "ils_random": "ILS-Random", "ils_ql": "ILS-QL", "ig_random": "IG-Random", "ig_ql": "IG-QL", "ig_ql_seed": "IG-QL$^{+}$"}
MARK = {"lpt": "s", "multifit": "s", "ldm": "s", "sa_li_perkins": "D", "bco": "D", "bco_I100": "D",
        "ils_random": "^", "ils_ql": "^", "ig_random": "o", "ig_ql": "o", "ig_ql_seed": "*"}
COLOR = {"lpt": "#7f7f7f", "multifit": "#1f77b4", "ldm": "#17becf", "sa_li_perkins": "#8c564b", "bco": "#bcbd22", "bco_I100": "#e377c2",
         "ils_random": "#ff9896", "ils_ql": "#d62728", "ig_random": "#98df8a", "ig_ql": "#2ca02c", "ig_ql_seed": "#000000"}
PANELS = [("$n \\leq 40$", ["rand_20j_4m", "rand_20j_6m", "rand_30j_8m", "rand_40j_5m"]),
          ("$50 \\leq n \\leq 100$", ["rand_50j_10m", "rand_60j_20m", "rand_75j_15m", "rand_100j_20m"]),
          ("$n \\geq 150$", ["rand_150j_15m", "rand_150j_30m", "rand_250j_25m", "rand_300j_20m"])]


def main() -> None:
    res = pd.read_csv(ROOT / "results" / "results.csv")
    seeded = pd.read_csv(ROOT / "results" / "seeded_baselines_entrada102.csv")
    sota = pd.read_csv(ROOT / "results" / "sota_sensitivity_entrada90.csv")
    ref = pd.read_csv(ROOT / "results" / "exact_reference_synthetic_entrada101.csv")[["instance", "rep", "lb_lp"]]
    keep = ["instance", "rep", "method", "makespan"]
    d = pd.concat([res[keep], seeded[keep], sota[sota.method == "bco_I100"][keep]])
    d = d[d.method.isin(LABEL)].drop_duplicates(["instance", "rep", "method"]).merge(ref, on=["instance", "rep"])
    d["gap_af"] = 100 * (d.makespan - d.lb_lp) / d.lb_lp
    gap = d.groupby(["instance", "method"]).gap_af.mean().unstack("method")
    t = pd.read_csv(ROOT / "results" / "timing_entrada107.csv")
    cpu = t.groupby(["instance", "method"]).cpu_time.median().unstack("method")
    fast = ROOT / "results" / "timing_ils_fast_entrada174.csv"
    if fast.exists():
        # Entrada 174: o ILS foi remedido com os operadores vetorizados (resultados identicos, 6,2 vezes mais rapido) numa segunda sessao que reexecutou ig_random e ig_ql_seed como pontes;
        # os tempos novos do ILS sao expressos na escala da sessao antiga pela razao (CPU antigo / CPU novo) das duas pontes, media geometrica por classe.
        tf = pd.read_csv(fast).groupby(["instance", "method"]).cpu_time.median().unstack("method")
        scale = np.exp(0.5 * (np.log(cpu["ig_random"] / tf["ig_random_s2"]) + np.log(cpu["ig_ql_seed"] / tf["ig_ql_seed_s2"])))
        cpu["ils_random"] = tf["ils_random_fast"] * scale
        cpu["ils_ql"] = tf["ils_ql_fast"] * scale
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=False)
    out = []
    for ax, (title, classes) in zip(axes, PANELS):
        pts = {}
        for k in LABEL:
            g = gap.loc[classes, k].mean()
            c = float(np.exp(np.log(cpu.loc[classes, k]).mean()))
            pts[k] = (c, g)
            ax.scatter(c, g, marker=MARK[k], s=70 if k != "ig_ql_seed" else 150, color=COLOR[k], edgecolor="k", linewidth=0.6, zorder=3,
                       label=LABEL[k] if ax is axes[0] else None)
            out.append((title, k, c, g))
        order = sorted(pts.items(), key=lambda kv: kv[1][0])
        front, best = [], np.inf
        for k, (c, g) in order:
            if g < best:
                front.append((c, g)); best = g
        ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="k", linestyle="--", linewidth=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("CPU time per run (s, log scale)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean GAP vs. certified LB (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(ROOT / "paper" / "fig_pareto.pdf")
    fig.savefig(ROOT / "paper" / "fig_pareto.png", dpi=150)
    # razoes de CPU
    L = ["RAZOES DE CPU (mediana por classe, mesma sessao, sequencial; 5 reps)", "classe        ILS-QL/IG-QL  ILS-Rand/IG-Rand  BCO(I=100)/IG-QL  MultiFit/IG-QL  IG-QL+/IG-QL"]
    order = sorted(cpu.index, key=lambda s: int(s.split("_")[1].rstrip("j")))
    for c in order:
        L.append(f"{c:14s}{cpu.loc[c, 'ils_ql'] / cpu.loc[c, 'ig_ql']:12.2f}{cpu.loc[c, 'ils_random'] / cpu.loc[c, 'ig_random']:18.2f}"
                 f"{cpu.loc[c, 'bco_I100'] / cpu.loc[c, 'ig_ql']:18.2f}{cpu.loc[c, 'multifit'] / cpu.loc[c, 'ig_ql']:16.4f}{cpu.loc[c, 'ig_ql_seed'] / cpu.loc[c, 'ig_ql']:13.2f}")
    r = (cpu.ils_ql / cpu.ig_ql)
    L.append(f"\nILS-QL/IG-QL: min {r.min():.2f}x (n=20), max {r.max():.2f}x (n=300); mediana {r.median():.2f}x")
    r2 = (cpu.ils_random / cpu.ig_random)
    L.append(f"ILS-Random/IG-Random: min {r2.min():.2f}x, max {r2.max():.2f}x")
    r3 = cpu.bco_I100 / cpu.ig_ql
    L.append(f"BCO(I=100)/IG-QL: min {r3.min():.2f}x, max {r3.max():.2f}x")
    L.append(f"IG-QL/IG-Random (sobrecusto do QL): mediana {(cpu.ig_ql / cpu.ig_random).median():.2f}x")
    pd.DataFrame(out, columns=["panel", "method", "cpu_s_geomean", "gap_af_mean"]).to_csv(ROOT / "results" / "entrada107_pareto_points.csv", index=False)
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada107_cpu_ratios.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
