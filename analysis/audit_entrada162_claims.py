"""
analysis/audit_entrada162_claims.py (Entrada 162-166)
Auditoria numerica das alegacoes do manuscrito que vem do Ciclo 13: sensibilidade da conclusao 'a CPU igual' a implementacao (busca local vetorizada, D1/D2), o ILS eficiente (D3) e os aprendizes neurais (E, se existirem os arquivos).
Recalcula dos CSV e compara com o valor literal do texto (paper/short/*.tex, tabelas e suplemento).
Uso: python -m analysis.audit_entrada162_claims
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
rows = []


def chk(label, value, expected, tol):
    rows.append((label, float(value), float(expected), tol, abs(float(value) - float(expected)) <= tol))


# ---- D1/D2: busca local vetorizada
P = pd.read_csv(R / "entrada162_igls_fast_primary.csv").set_index("id")
ver = pd.read_csv(R / "igls_fast_verify_entrada162.csv")
chk("D equivalencia a nivel de execucao (120/120)", ver.equal.sum(), 120, 0)
chk("D1 d (-0,105)", P.loc["D1", "mean_d_pct"], -0.105, 0.0006)
chk("D1 IC lo (-0,115)", P.loc["D1", "ci90_lo"], -0.115, 0.0006)
chk("D1 IC hi (-0,095)", P.loc["D1", "ci90_hi"], -0.095, 0.0006)
chk("D1 vitorias (330)", P.loc["D1", "better"], 330, 0)
chk("D1 derrotas (18)", P.loc["D1", "worse"], 18, 0)
chk("D2 d (-0,008)", P.loc["D2", "mean_d_pct"], -0.008, 0.0006)
chk("D2 IC lo (-0,016)", P.loc["D2", "ci90_lo"], -0.016, 0.0006)
chk("D2 IC hi (-0,001)", P.loc["D2", "ci90_hi"], -0.001, 0.0006)
chk("D2 vitorias (56)", P.loc["D2", "better"], 56, 0)
chk("D2 derrotas (38)", P.loc["D2", "worse"], 38, 0)
chk("D2 p_holm (0,063)", P.loc["D2", "p_holm"], 0.063, 0.0006)
k162 = json.loads((R / "igls_fast_cpu_ratio_entrada162.json").read_text())
k129 = json.loads((R / "igls_eq_cpu_ratio_entrada129.json").read_text())
k127 = json.loads((R / "igcvls_cpu_ratio_entrada127.json").read_text())
gm = lambda d: float(np.exp(np.mean([np.log(v["K"]) for v in d.values()])))
chk("K_H rapida min (1,5)", min(v["K"] for v in k162["H"].values()), 1.5, 0.05)
chk("K_H rapida max (2,1)", max(v["K"] for v in k162["H"].values()), 2.1, 0.05)
chk("K_H antes min (2,1)", min(v["K"] for v in k129.values()), 2.1, 0.05)
chk("K_H antes max (52,4)", max(v["K"] for v in k129.values()), 52.4, 0.05)
chk("K_H media geometrica antes (9,6)", gm(k129), 9.6, 0.05)
chk("K_H media geometrica rapida (1,8)", gm(k162["H"]), 1.8, 0.05)
chk("K_R rapida min (1,2)", min(v["K"] for v in k162["R"].values()), 1.2, 0.05)
chk("K_R rapida max (2,4)", max(v["K"] for v in k162["R"].values()), 2.4, 0.05)
chk("K_R antes min (2,3)", min(v["K"] for v in k127.values()), 2.3, 0.05)
chk("K_R antes max (82,9)", max(v["K"] for v in k127.values()), 82.9, 0.05)
txt = (R / "entrada162_igls_fast.txt").read_text(encoding="utf-8")
chk("D fracao de orcamento B (0,30)", float(re.search(r"B = 1 - .* = (\d\.\d+)", txt).group(1)), 0.30, 0.006)

# ---- D3: ILS eficiente
E = pd.read_csv(R / "entrada162_ils_fast_primary.csv").set_index("id")
v3 = pd.read_csv(R / "ils_fast_verify_entrada162.csv")
chk("D3 equivalencia ILS (36/36)", v3.equal.sum(), 36, 0)
chk("D3 razao de CPU padrao/rapido (6,2)", v3.cpu_standard.sum() / v3.cpu_fast.sum(), 6.2, 0.05)
chk("D3 E1 d (-0,46)", E.loc["E1", "mean_d_pct"], -0.46, 0.006)
chk("D3 E1 vitorias (557)", E.loc["E1", "better"], 557, 0)
chk("D3 E1 derrotas (87)", E.loc["E1", "worse"], 87, 0)
chk("D3 E2 d (-0,50)", E.loc["E2", "mean_d_pct"], -0.50, 0.006)
chk("D3 E1 p_holm < 1e-80", int(E.loc["E1", "p_holm"] < 1e-80), 1, 0)
chk("D3 E2 p_holm < 1e-80", int(E.loc["E2", "p_holm"] < 1e-80), 1, 0)
k3 = json.loads((R / "ils_fast_cpu_ratio_entrada162.json").read_text())
chk("D3 K_ils min (0,4)", min(v["K_ils"] for v in k3.values()), 0.4, 0.05)
chk("D3 K_ils max (1,9)", max(v["K_ils"] for v in k3.values()), 1.9, 0.05)

# ---- E: neurais
N = pd.read_csv(R / "entrada162_neural_primary.csv").set_index("id")
chn = json.loads((R / "nq_choice_entrada162.json").read_text())
chk("E n instancias (700)", N.loc["N1", "n"], 700, 0)
chk("E lr escolhido (0,005)", chn["lr"], 0.005, 0)
chk("E N1 ganho do Q neural (0,005)", -N.loc["N1", "mean_d_pct"], 0.005, 0.0006)
chk("E N1 IC lo (-0,061)", N.loc["N1", "ci90_lo"], -0.061, 0.0006)
chk("E N1 IC hi (+0,046)", N.loc["N1", "ci90_hi"], 0.046, 0.0006)
chk("E N1 p_holm (0,17)", N.loc["N1", "p_holm"], 0.17, 0.006)
chk("E N2 d (0,192 a favor da validacao)", -N.loc["N2", "mean_d_pct"], 0.192, 0.0006)
chk("E N2 p_holm (6e-18)", N.loc["N2", "p_holm"] * 1e18, 6.5, 0.6)
chk("E N4 ganho do MLP offline (0,099)", -N.loc["N4", "mean_d_pct"], 0.099, 0.0006)
chk("E N4 IC lo (-0,162)", N.loc["N4", "ci90_lo"], -0.162, 0.0006)
chk("E N4 IC hi (-0,036)", N.loc["N4", "ci90_hi"], -0.036, 0.0006)
chk("E N4 p_holm (0,17)", N.loc["N4", "p_holm"], 0.17, 0.006)
chk("E N3 d (+0,104)", N.loc["N3", "mean_d_pct"], 0.104, 0.0006)
chk("E N3 p_holm (8e-15)", N.loc["N3", "p_holm"] * 1e15, 7.7, 0.6)
chk("E N3 vitorias (49)", N.loc["N3", "better"], 49, 0)
chk("E N3 derrotas (192)", N.loc["N3", "worse"], 192, 0)
tn = (R / "entrada162_neural.txt").read_text(encoding="utf-8")
chk("E captura do Q neural (0,03)", float(re.search(r"nq_d (-?\d+\.\d+), ql_d", tn.split("Razao de captura")[1]).group(1)), 0.03, 0.006)
chk("E razao de CPU nq_d/rand_d (1,5)", float(re.search(r"razao nq_d/rand_d (\d+\.\d+)", tn).group(1)), 1.5, 0.05)

# ---- F: aprendizes online nos outros ambientes (Entrada 167)
O = pd.read_csv(R / "entrada167_online_primary.csv").set_index("id")
chk("F P-G1 d (+0,003)", O.loc["P-G1", "mean_d_pct"], 0.003, 0.0006)
chk("F P-G2 d (+0,006)", O.loc["P-G2", "mean_d_pct"], 0.006, 0.0006)
chk("F P-G1/G2 p_holm (0,82)", O.loc["P-G1", "p_holm"], 0.82, 0.006)
chk("F P-G3 d (0,09 a favor da validacao)", -O.loc["P-G3", "mean_d_pct"], 0.09, 0.0006 + 0.0008)
chk("F P-G4 d (0,09 a favor da validacao)", -O.loc["P-G4", "mean_d_pct"], 0.09, 0.0006 + 0.005)
chk("F P-G3 p_holm < 1e-30", int(O.loc["P-G3", "p_holm"] < 1e-30), 1, 0)
chk("F P-G4 p_holm < 1e-30", int(O.loc["P-G4", "p_holm"] < 1e-30), 1, 0)
chk("F F-G1 d (+0,027)", O.loc["F-G1", "mean_d_pct"], 0.027, 0.0006)
chk("F F-G2 d (+0,099)", O.loc["F-G2", "mean_d_pct"], 0.099, 0.0006)
chk("F F-G2 p_holm (0,027)", O.loc["F-G2", "p_holm"], 0.027, 0.0006)
chk("F F-G4 d (0,14 a favor da validacao)", -O.loc["F-G4", "mean_d_pct"], 0.14, 0.0016)
chk("F F-G4 p_holm (4e-4)", O.loc["F-G4", "p_holm"] * 1e4, 3.9, 0.6)
chk("F F-G3 d (0,07 a favor da validacao)", -O.loc["F-G3", "mean_d_pct"], 0.07, 0.001)
chk("F F-G3 p_holm (0,15)", O.loc["F-G3", "p_holm"], 0.15, 0.006)
chk("F n instancias P (1200)", O.loc["P-G1", "n"], 1200, 0)
chk("F n instancias F (160)", O.loc["F-G1", "n"], 160, 0)

# ---- G: mais dados de treino offline (Entrada 168)
G = pd.read_csv(R / "entrada168_bigtrain_primary.csv").set_index("id")
gt = (R / "entrada168_bigtrain.txt").read_text(encoding="utf-8")
chk("G H3 d (+0,014)", G.loc["H3", "mean_d_pct"], 0.014, 0.0006)
chk("G H1 d (+0,025)", G.loc["H1", "mean_d_pct"], 0.025, 0.0006)
chk("G H1 IC lo (-0,015)", G.loc["H1", "ci90_lo"], -0.015, 0.0006)
chk("G H1 IC hi (+0,068)", G.loc["H1", "ci90_hi"], 0.068, 0.0006)
chk("G H2 d (-0,011)", G.loc["H2", "mean_d_pct"], -0.011, 0.0006)
chk("G H2 IC lo (-0,046)", G.loc["H2", "ci90_lo"], -0.046, 0.0006)
chk("G H2 IC hi (+0,019)", G.loc["H2", "ci90_hi"], 0.019, 0.0006)
chk("G H2 melhor (36)", G.loc["H2", "better"], 36, 0)
chk("G H2 pior (160)", G.loc["H2", "worse"], 160, 0)
chk("G H2 p_holm (1e-8)", G.loc["H2", "p_holm"] * 1e8, 1.1, 0.1)
chk("G H2 rank-biserial (+0,76)", G.loc["H2", "rank_biserial"], 0.76, 0.006)
chk("G ganho MLP700 sobre aleatorio (0,198)", -float(re.search(r"val_mlp700 (-?\d+\.\d+)", gt).group(1)), 0.198, 0.0006)
chk("G ganho val_tn700 sobre aleatorio (0,187)", -float(re.search(r"val_tn700 (-?\d+\.\d+)", gt).group(1)), 0.187, 0.0006)
chk("G kNN 280 d (+0,105)", float(re.search(r"kNN 280: d = ([+-]\d+\.\d+)", gt).group(1)), 0.105, 0.0006)
chk("G MLP 280 d (+0,104)", float(re.search(r"MLP 280: d = ([+-]\d+\.\d+)", gt).group(1)), 0.104, 0.0006)

bad = [r for r in rows if not r[4]]
for lab, v, e, t, ok in rows:
    print(f"{lab:62s} {v:12.4f} {e:12.4f} {t:7.4f} {ok}")
print(f"\n{len(rows) - len(bad)}/{len(rows)} OK; divergentes: {[r[0] for r in bad]}")
sys.exit(1 if bad else 0)
