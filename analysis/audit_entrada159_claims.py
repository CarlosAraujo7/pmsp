"""
analysis/audit_entrada159_claims.py (Entrada 159/160)
Auditoria numerica das alegacoes do manuscrito que vem do Ciclo 12: replicacao em instancias classicas de Taillard (frente A), auditoria de literatura (frente B) e aprendizes de outra classe (frente C, se os arquivos existirem).
Recalcula dos CSV e compara com o valor literal do texto (paper/short/*.tex e s_*.tex).
Uso: python -m analysis.audit_entrada159_claims
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
rows = []


def chk(label, value, expected, tol):
    rows.append((label, float(value), float(expected), tol, abs(float(value) - float(expected)) <= tol))


# ---- frente A: Taillard
P = pd.read_csv(R / "entrada159_taillard_primary.csv").set_index("id")
d = pd.read_csv(R / "pfsp_taillard_entrada159.csv")
chk("A: n instancias (40)", P.loc["P1", "n"], 40, 0)
chk("A P1 d (-0,006)", P.loc["P1", "mean_d_pct"], -0.006, 0.0006)
chk("A P1 p_holm (0,74)", P.loc["P1", "p_holm"], 0.74, 0.006)
chk("A P2 d (-0,28)", P.loc["P2", "mean_d_pct"], -0.28, 0.0006)
chk("A P2 vitorias (26)", P.loc["P2", "better"], 26, 0)
chk("A P2 derrotas (7)", P.loc["P2", "worse"], 7, 0)
chk("A P2 p_holm (7e-4)", P.loc["P2", "p_holm"] * 1e4, 7.2, 0.6)
chk("A P3 d (+0,065)", P.loc["P3", "mean_d_pct"], 0.065, 0.0006)
chk("A P3 IC lo (-0,02)", P.loc["P3", "ci90_lo"], -0.02, 0.005)
chk("A P3 IC hi (+0,16)", P.loc["P3", "ci90_hi"], 0.16, 0.005)
chk("A P4 d (-0,17)", P.loc["P4", "mean_d_pct"], -0.17, 0.005)
chk("A P4 IC lo (-0,29)", P.loc["P4", "ci90_lo"], -0.29, 0.005)
chk("A P4 IC hi (-0,05)", P.loc["P4", "ci90_hi"], -0.05, 0.005)
chk("A P4 p_holm (0,064)", P.loc["P4", "p_holm"], 0.064, 0.0006)
chk("A P5 d (-0,11)", P.loc["P5", "mean_d_pct"], -0.11, 0.005)
chk("A P5 p_holm (0,12)", P.loc["P5", "p_holm"], 0.12, 0.005)
cpu = d.pivot(index=["n", "m", "rep"], columns="method", values="cpu_time")
chk("A razao de CPU ILS/IG (1,06)", cpu.ils_eq.mean() / cpu.ig_rand.mean(), 1.06, 0.006)
chk("A razao de CPU IG sem LS/IG (1,09)", cpu.ig_nols_eq.mean() / cpu.ig_rand.mean(), 1.09, 0.006)

# ---- frente B: auditoria de literatura
L = pd.read_csv(R / "litaudit_entrada159.csv", dtype=str, keep_default_na=False)
inc = L[L.included == "Y"]
chk("B n incluidos (12)", len(inc), 12, 0)
chk("B controle aleatorio Y (5)", (inc.random_control == "Y").sum(), 5, 0)
chk("B ablacao Y (8)", (inc.ablation_policy == "Y").sum(), 8, 0)
chk("B CPU (2)", (inc.budget_basis == "CPU").sum(), 2, 0)
chk("B offline Y (2)", (inc.offline_tuned_static == "Y").sum(), 2, 0)
chk("B split Y (3)", (inc.tune_test_split == "Y").sum(), 3, 0)
chk("B stat Y (6)", (inc.stat_test == "Y").sum(), 6, 0)
chk("B multiplicidade Y (1)", (inc.multiplicity == "Y").sum(), 1, 0)
chk("B claim Y+MIXED (12)", inc.claim_policy_helps.isin(["Y", "MIXED"]).sum(), 12, 0)

# ---- frente C (se existir)
cp = R / "entrada159_ctx_primary.csv"
if cp.exists():
    C = pd.read_csv(cp).set_index("id")
    ch = __import__("json").loads((R / "ctx_choice_entrada159.json").read_text())
    chk("C n instancias (700)", C.loc["C1", "n"], 700, 0)
    chk("C modelo offline escolhido e' knn", int(ch["model"] == "knn"), 1, 0)
    chk("C3 d linucb vs aleatorio (+0,039 de ganho)", -C.loc["C3", "mean_d_pct"], 0.039, 0.0006)
    chk("C3 IC lo (-0,092)", C.loc["C3", "ci90_lo"], -0.092, 0.0006)
    chk("C3 IC hi (+0,013)", C.loc["C3", "ci90_hi"], 0.013, 0.0006)
    chk("C3 p_holm (0,69)", C.loc["C3", "p_holm"], 0.69, 0.006)
    chk("C4 d (0,159 a favor da validacao)", -C.loc["C4", "mean_d_pct"], 0.159, 0.0006)
    chk("C4 p_holm (9e-15)", C.loc["C4", "p_holm"] * 1e15, 8.7, 0.6)
    chk("C2 ganho do seletor offline (0,100)", -C.loc["C2", "mean_d_pct"], 0.100, 0.0006)
    chk("C2 p_holm (5e-4)", C.loc["C2", "p_holm"] * 1e4, 4.7, 0.6)
    chk("C1 d (+0,105)", C.loc["C1", "mean_d_pct"], 0.105, 0.0006)
    chk("C1 p_holm (3e-9)", C.loc["C1", "p_holm"] * 1e9, 3.4, 0.6)
    chk("C1 vitorias (38)", C.loc["C1", "better"], 38, 0)
    chk("C1 derrotas (116)", C.loc["C1", "worse"], 116, 0)
    t = (R / "entrada159_ctx.txt").read_text(encoding="utf-8")
    import re
    chk("C captura do linucb (0,19)", float(re.search(r"linucb_d (-?\d+\.\d+), ql_d", t.split("Razao de captura")[1]).group(1)), 0.19, 0.006)
    chk("C captura do seletor offline (0,50)", float(re.search(r"val_ctx (-?\d+\.\d+), linucb_d", t.split("Razao de captura")[1]).group(1)), 0.50, 0.006)
    chk("C Spearman do seletor offline (0,64)", float(re.search(r"ganho de val_ctx\) em 28 celulas: rho = (-?\d+\.\d+)", t).group(1)), 0.64, 0.006)
    chk("C razao de CPU linucb/aleatorio (1,03)", float(re.search(r"razao linucb_d/rand_d (\d+\.\d+)", t).group(1)), 1.03, 0.006)

bad = [r for r in rows if not r[4]]
for lab, v, e, t, ok in rows:
    print(f"{lab:46s} {v:12.4f} {e:12.4f} {t:7.4f} {ok}")
print(f"\n{len(rows) - len(bad)}/{len(rows)} OK; divergentes: {[r[0] for r in bad]}")
sys.exit(1 if bad else 0)
