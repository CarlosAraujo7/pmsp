"""
analysis/report_entrada213_equalcpu.py
========================================
Ciclo 19, frente C: o SELETOR OFFLINE e o Q-LEARNING AQUECIDO a CPU IGUAL, cada um com o seu proprio K (experiment/run_cycle19_equalcpu_entrada213.py; docstring). Diario: Entrada 213 (pre-registro) e 214 (resultados). Instancias publicadas (2.550 descoberta, 3.950 held-out), S5 e S15, 5 sementes.
Comparacoes (a quantidade de iteracoes do braco A e' T_A = round(1000 K_B / K_A); o braco B roda T = 1000): sel_eq_ql vs ql, sel_eq_alns vs alns, sel_eq_wq vs wq, sel_eq_val vs val; wq_eq_ql vs ql, wq_eq_alns vs alns, wq_eq_val vs val. Holm sobre 4 (S5/S15 x descoberta/held-out) em cada familia.

PRE-REGISTRO (congelado ANTES da calibracao dos K e do teste; hashes em results/entrada213_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Familias: Cql sel_eq_ql vs ql; Cal sel_eq_alns vs alns; Cwq sel_eq_wq vs wq; Cvl sel_eq_val vs val; Dql wq_eq_ql vs ql; Dal wq_eq_alns vs alns; Dvl wq_eq_val vs val.
(C1) o Q aquecido custa no maximo o que custa o Q frio em >= 14 das 18 celulas (K_wq <= K_ql: ele tambem escolhe acoes baratas).
(C2) o seletor vence o Q-Learning frio a CPU igual: Cql confirmada nos quatro pares.  (C3) o seletor vence a roleta ALNS a CPU igual: Cal confirmada em >= 3 dos 4.  (C4) o seletor vence a validada a CPU igual (cada um com o seu K): Cvl confirmada em >= 3 dos 4, com ganho medio >= 0,01 pontos nas confirmadas.
(C5) o seletor vence o Q aquecido a CPU igual: ganho medio do seletor > 0 em >= 3 dos 4 (Cwq).  (C6) a validada ainda vence o Q aquecido a CPU igual: ganho medio da validada > 0 em >= 3 dos 4 (Dvl, d medio de wq_eq_val vs val < 0 significa que o aquecido vence: queremos d > 0).
Leitura E ('a vantagem do seletor offline sobrevive a CPU igual contra todo aprendiz online') = C2 e C3 e C5.
Uso: python -m analysis.report_entrada213_equalcpu
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada200_selector import SETS, load  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_e(sp, cj):
    w = load(sp, cj)
    q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"warmq_{cj}_{sp}_s*of*_entrada211.csv")))], ignore_index=True)
    e = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"eqcpu_{cj}_{sp}_s*of*_entrada213.csv")))], ignore_index=True)
    m1 = q.groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    m2 = e.groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    t2 = e.groupby(["cls", "method"]).T.mean().unstack("method").add_prefix("T_").reset_index()
    w = w.merge(m1, on="cls").merge(m2, on="cls").merge(t2, on="cls")
    assert len(w) == len(load(sp, cj))
    return w


def main():
    L = ["=" * 110, "SELETOR OFFLINE E Q AQUECIDO A CPU IGUAL (Entrada 213)", "=" * 110]
    K = json.loads((R / "eqcpu_k_entrada213.json").read_text())
    Kb = json.loads((R / "pub_cpu_ratio_entrada189.json").read_text())
    rows = []
    for sp in ("S5", "S15"):
        for key, v in K[sp].items():
            rows.append(dict(space=sp, group=key, K_sel=v["sel"], K_wq=v["wq"], K_ql=Kb[sp][key]["ql"], K_alns=Kb[sp][key]["alns"]))
    KD = pd.DataFrame(rows)
    L += ["\n--- razoes de CPU K (acao do seletor / aleatorio; Q aquecido / aleatorio; Q frio e roleta da Entrada 189) ---", KD.to_string(index=False)]
    W = {(sp, cj): load_e(sp, cj) for sp in ("S5", "S15") for cj in SETS}
    spec = {"Cql": ("sel_eq_ql", "ql"), "Cal": ("sel_eq_alns", "alns"), "Cwq": ("sel_eq_wq", "wq"), "Cvl": ("sel_eq_val", "val"), "Dql": ("wq_eq_ql", "ql"), "Dal": ("wq_eq_alns", "alns"), "Dvl": ("wq_eq_val", "val")}
    fams = {k: [] for k in spec}
    for (sp, cj), w in W.items():
        for fid, (a, b) in spec.items():
            t = test(w[a], w[b]); t.update(id=f"{fid}-{sp}-{cj}"); fams[fid].append(t)
    F = {k: family(v) for k, v in fams.items()}
    for k, X in F.items():
        a, b = spec[k]
        L += [f"\n--- {k}: {a} contra {b} (Holm sobre {len(X)}) ---", X[COLS].to_string(index=False)]
    tm = pd.DataFrame([{**dict(space=sp, set=cj), **{c: float(w[c].mean()) for c in w.columns if c.startswith("T_")}} for (sp, cj), w in W.items()])
    L += ["\n--- iteracoes medias dos bracos de CPU igual ---", tm.to_string(index=False)]
    c1 = int((KD.K_wq <= KD.K_ql).sum())
    c2 = bool(F["Cql"].confirmada.all())
    c3 = int(F["Cal"].confirmada.sum()) >= 3
    cv = F["Cvl"]
    c4 = int(cv.confirmada.sum()) >= 3 and bool((-cv[cv.confirmada].mean_d_pct >= 0.01).all())
    c5 = int((F["Cwq"].mean_d_pct < 0).sum()) >= 3
    c6 = int((F["Dvl"].mean_d_pct > 0).sum()) >= 3
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(C1) K_wq <= K_ql em >= 14 das 18 celulas: {yn(c1 >= 14)} ({c1}/18)", f"(C2) Cql confirmada nos quatro pares: {yn(c2)} (ganho medio: " + ", ".join(f"{-v:+.3f}" for v in F['Cql'].mean_d_pct) + ")",
          f"(C3) Cal confirmada em >= 3 dos 4: {yn(c3)} (" + ", ".join(f"{-v:+.3f}" for v in F['Cal'].mean_d_pct) + f"; confirmadas {int(F['Cal'].confirmada.sum())})", f"(C4) Cvl confirmada em >= 3 dos 4 com ganho >= 0,01: {yn(c4)} (" + ", ".join(f"{-v:+.3f}" for v in cv.mean_d_pct) + f"; confirmadas {int(cv.confirmada.sum())})",
          f"(C5) seletor vence o Q aquecido a CPU igual em >= 3 dos 4: {yn(c5)} (" + ", ".join(f"{-v:+.3f}" for v in F['Cwq'].mean_d_pct) + ")", f"(C6) validada vence o Q aquecido a CPU igual em >= 3 dos 4: {yn(c6)} (d medio de wq_eq_val vs val: " + ", ".join(f"{v:+.3f}" for v in F['Dvl'].mean_d_pct) + ")",
          f"E (a vantagem do seletor sobrevive a CPU igual contra todo aprendiz online) = C2 e C3 e C5: {yn(c2 and c3 and c5)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada213_equalcpu.txt").write_text(text, encoding="utf-8")
    KD.to_csv(R / "entrada213_equalcpu_k.csv", index=False)
    for k, X in F.items():
        X.to_csv(R / f"entrada213_equalcpu_{k}.csv", index=False)


if __name__ == "__main__":
    main()
