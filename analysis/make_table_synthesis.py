"""
analysis/make_table_synthesis.py (Entrada 154)
Sintese transversal do estudo: em cada ambiente (problema x tipo de acao), a folga (oraculo por instancia de acoes FIXAS vs a mistura/selecao aleatoria, a mesma T), o ganho da escolha VALIDADA offline,
o ganho do Q-Learning online e a razao Q-Learning/validada. Tudo recalculado dos CSV brutos (d = 100 (A - B)/B em makespan; negativo = A melhor; B = selecao aleatoria com o mesmo T).
Saidas: results/synthesis_headroom_table.csv, paper/f35_synthesis.tex. Uso: python -m analysis.make_table_synthesis
O oraculo por instancia inclui o ruido de selecao (o melhor de varias execucoes): e' um limite superior da folga, nao atingivel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
OUT = ROOT / "paper"


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def row(name, n_inst, n_act, base, oracle, val, ql, cpu=None, note=""):
    o, v, q = rel(oracle, base).mean(), (rel(val, base).mean() if val is not None else np.nan), rel(ql, base).mean()
    return dict(setting=name, n=int(n_inst), actions=n_act, oracle=float(o), validated=float(v), ql=float(q), ql_over_validated=(float(q / v) if (val is not None and v < -0.005) else np.nan), cpu_ratio=cpu, note=note)


rows = []

# S1: P||Cmax, estrategias de destruicao (3 acoes)
res = pd.read_csv(R / "results.csv")
sh = pd.read_csv(R / "signal_headroom_entrada104.csv")
b = res[res.method.isin(["ig_random", "ig_ql"])].pivot_table(index=["instance", "rep"], columns="method", values="makespan").reset_index()
f = sh[sh.method.isin(["ig_fix0", "ig_fix1", "ig_fix2"])].pivot_table(index=["instance", "rep"], columns="method", values="makespan").reset_index()
w = b.merge(f, on=["instance", "rep"]).dropna()
w["oracle"] = w[["ig_fix0", "ig_fix1", "ig_fix2"]].min(axis=1)
best_cls = {c: min(["ig_fix0", "ig_fix1", "ig_fix2"], key=lambda k: w[w.instance == c][k].mean()) for c in w.instance.unique()}
w["val"] = [r[best_cls[r.instance]] for _, r in w.iterrows()]
rows.append(row(r"$\mathrm{P}||C_{\max}$, destruction strategy", len(w), 3, w.ig_random, w.oracle, w.val, w.ig_ql, note="validated: best per class, chosen on the test set (optimistic)"))

# S2: P||Cmax, tamanho de destruicao (5 acoes)
sys.path.insert(0, str(ROOT / "experiment"))
from experiment.run_dsel_entrada114 import FIX, FRACS  # noqa: E402
from experiment.run_igcvls_entrada127 import d_choice  # noqa: E402
dt = pd.read_csv(R / "dsel_test_entrada114.csv")
w = dt.pivot_table(index=["instance", "rep"], columns="method", values="makespan").reset_index()
chd = d_choice()
w["val"] = [r[FIX[FRACS.index(chd[r.instance])]] for _, r in w.iterrows()]
w["oracle"] = w[FIX].min(axis=1)
rows.append(row(r"$\mathrm{P}||C_{\max}$, destruction size", len(w), 5, w.igd_random, w.oracle, w.val, w.igd_ql_cal, note="validated on 30 separate instances per class"))

# S3: R||Cmax sintetico, estrategias (4 acoes; 400 instancias novas)
dl = pd.read_csv(R / "unrelated_law_entrada124.csv")
w = dl.pivot_table(index=["lo", "hi", "n", "m", "rep"], columns="method", values="makespan").reset_index()
fx = [f"ur_igls_fix{k}" for k in range(4)]
w["oracle"] = w[fx].min(axis=1)
cpu3 = dl.pivot_table(index=["lo", "hi", "n", "m", "rep"], columns="method", values="cpu_time")
rows.append(row(r"$\mathrm{R}||C_{\max}$ synthetic, destruction strategy", len(w), 4, w.ur_igls_random, w.oracle, None, w.ur_igls_ql, cpu=float(cpu3.ur_igls_ql.mean() / cpu3.ur_igls_random.mean())))

# S4: R||Cmax 1000 x 50, estrategias (5 candidatas incl. mistura; 96 instancias novas)
dv = pd.read_csv(R / "unrelated_valid_entrada135.csv")
ch4 = json.loads((R / "valid_choice_entrada135.json").read_text())
w = dv.pivot_table(index=["lo", "hi", "n", "m", "rep"], columns="method", values="makespan").reset_index()
cpu4 = dv.pivot_table(index=["lo", "hi", "n", "m", "rep"], columns="method", values="cpu_time")
w["val"] = [r[ch4[f"{int(r.lo)}_{int(r.hi)}"]["best"]] for _, r in w.iterrows()]
w["oracle"] = w[fx].min(axis=1)
rows.append(row(r"$\mathrm{R}||C_{\max}$ $1000{\times}50$, destruction strategy", len(w), 4, w.ur_igls_random, w.oracle, w.val, w.ur_igls_ql, cpu=float(cpu4.ur_igls_ql.mean() / cpu4.ur_igls_random.mean()),
                 note="validated among the four strategies and the mixture"))

# S5: R||Cmax benchmark padrao, estrategias (700 instancias)
d5 = pd.read_csv(R / "rcmax_test_entrada137.csv")
d5["instance"] = d5.instance.astype(str)
ch5 = json.loads((R / "rcmax_choice_arm_entrada137.json").read_text())
w = d5.pivot_table(index=["type", "instance"], columns="method", values="makespan").reset_index()
cpu5 = d5.pivot_table(index=["type", "instance"], columns="method", values="cpu_time")
w["val"] = [r[ch5[r["type"]]["best"]] for _, r in w.iterrows()]
w["oracle"] = w[fx].min(axis=1)
rows.append(row(r"$\mathrm{R}||C_{\max}$ standard benchmark, destruction strategy", len(w), 4, w.ur_igls_random, w.oracle, w.val, w.ur_igls_ql, cpu=float(cpu5.ur_igls_ql.mean() / cpu5.ur_igls_random.mean())))

# S6: R||Cmax benchmark padrao, tamanho de destruicao (LP-IG; 700 instancias)
d6 = pd.read_csv(R / "dpolicy_test_entrada145.csv")
d6["instance"] = d6.instance.astype(str)
w = d6.pivot_table(index=["type", "instance", "n"], columns="method", values="makespan").reset_index()
cpu6 = d6.pivot_table(index=["type", "instance"], columns="method", values="cpu_time")
F = [0.01, 0.02, 0.05, 0.10, 0.20]
tune = pd.read_csv(R / "lpig_tune_d_entrada141.csv")
tune["frac"] = tune.method.str.split("@").str[1].astype(float)
tune["ratio"] = tune.makespan / tune.best_known
val_tn = {(t, int(n)): F.index(float(g.groupby("frac").ratio.mean().idxmin())) for (t, n), g in tune.groupby(["type", "n"])}
w["val"] = [r[f"fix_{val_tn[(r['type'], int(r['n']))]}"] for _, r in w.iterrows()]
w["oracle"] = w[[f"fix_{k}" for k in range(5)]].min(axis=1)
rows.append(row(r"$\mathrm{R}||C_{\max}$ standard benchmark, destruction size", len(w), 5, w.rand_d, w.oracle, w.val, w.ql_d, cpu=float(cpu6.ql_d.mean() / cpu6.rand_d.mean()), note="validated per type and job count"))

# S7: PFSP, tamanho de destruicao (120 instancias)
p7 = R / "pfsp_test_entrada152.csv"
if p7.exists():
    d7 = pd.read_csv(p7)
    ch7 = json.loads((R / "pfsp_choice_entrada152.json").read_text())
    w = d7.pivot_table(index=["n", "m", "rep"], columns="method", values="makespan").reset_index()
    cpu7 = d7.pivot_table(index=["n", "m", "rep"], columns="method", values="cpu_time")
    w["val"] = [r[f"ig_fix_{ch7['d_val'][f'{int(r.n)}x{int(r.m)}']['action']}"] for _, r in w.iterrows()]
    w["oracle"] = w[[f"ig_fix_{k}" for k in range(5)]].min(axis=1)
    rows.append(row(r"Permutation flowshop, destruction size", len(w), 5, w.ig_rand, w.oracle, w.val, w.ig_ql, cpu=float(cpu7.ig_ql.mean() / cpu7.ig_rand.mean()), note="validated per size"))

T = pd.DataFrame(rows)
T.to_csv(R / "synthesis_headroom_table.csv", index=False)
pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.3f}")
print(T[["setting", "n", "actions", "oracle", "validated", "ql", "ql_over_validated", "cpu_ratio"]].to_string(index=False))


def fmt(v, plus=True):
    return "--" if v != v else (f"{v:+.3f}" if plus else f"{v:.2f}")


L = [r"% Auto-generated by analysis/make_table_synthesis.py (Entrada 154)", r"\begin{table}[t]", r"\centering",
     r"\caption{Headroom and policy gain across the environments of the study (mean $\bar d$ in \% of the makespan relative to random selection with the same budget; negative: better). "
     r"Oracle: best of the fixed actions per instance (an upper bound that includes selection noise); validated: the action chosen offline on separate instances; Q-Learning: the online agent (tabular, $T=1000$ except the flowshop, $T=300$). "
     r"The last column is the CPU time of Q-Learning relative to random selection.}", r"\label{tab:f35_synthesis}", r"{\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
     r"Environment & $N$ & Actions & Oracle & Validated & Q-Learning & CPU ratio \\", r"\midrule"]
for r in T.itertuples():
    L.append(f"{r.setting} & {r.n:,} & {r.actions} & {fmt(r.oracle)} & {fmt(r.validated)} & {fmt(r.ql)} & {('--' if r.cpu_ratio is None or r.cpu_ratio != r.cpu_ratio else f'{r.cpu_ratio:.2f}')} \\\\".replace(f"{r.n:,}", f"{r.n:,}".replace(",", "{,}")))
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
(OUT / "f35_synthesis.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote f35_synthesis.tex")
