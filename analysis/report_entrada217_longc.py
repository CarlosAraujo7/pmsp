"""
analysis/report_entrada217_longc.py
=====================================
Ciclo 19, frente E: HORIZONTE LONGO onde ha' folga (familia C publicada, held-out, tercis 1 e 2, S5; 340 de 1.020 instancias; T = 1000, 5.000 e 20.000). Diario: Entrada 217 (pre-registro) e 218 (resultados). Runner: experiment/run_cycle19_longc_entrada217.py. T = 1000 vem dos arquivos existentes (5 sementes; Entradas 190, 200 e 211) restritos a estas instancias;
T = 5.000 e 20.000: 3 sementes por metodo (rand, ql frio, alns, val, sel, wq aquecido) e 4 por acao fixa. Folga cf: oraculo com acao escolhida nas sementes {0, 1} e avaliada nas {2, 3} (em T = 1000 vem da Entrada 209). Estatistica: a das Entradas 180-216 (d = 100 (A - B)/B; Wilcoxon pareado; IC90; margem +-0,05; Holm sobre 6 por horizonte).

PRE-REGISTRO (congelado ANTES de rodar T = 5.000 e 20.000; hashes em results/entrada217_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
(E1) a folga cf diminui com o horizonte nestas instancias: h(1000) > h(5000) > h(20000).  (E2) o Q-Learning frio ganha do aleatorio com T = 5.000 (confirmado) e seu ganho com T = 20.000 e' menor que o de T = 1000.  (E3) a validada vence o Q frio (confirmada) com T = 5.000 e com T = 20.000.
(E4) o seletor offline supera a validada em media (d < 0) nos dois horizontes longos.  (E5) o Q aquecido supera o Q frio em media (ganho > 0) nos dois horizontes longos.  (E6) com T = 20.000 todos os ganhos sobre o aleatorio (ql, alns, val, sel, wq) ficam abaixo de 0,15 pontos (a folga esta' quase esgotada).
Uso: python -m analysis.report_entrada217_longc
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada200_selector import load  # noqa: E402
from analysis.report_entrada209_cforacle import arms_table  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
METHODS = ["rand", "ql", "alns", "val", "sel", "wq"]


def load_T(T):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"longc_T{T}_s*of*_entrada217.csv")))], ignore_index=True)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method")
    a = d[d.method.str.startswith("arm")].copy()
    a["arm"] = a.method.str[3:].astype(int)
    p = a.pivot_table(index=["cls", "arm"], columns="seed_i", values="makespan").reset_index()
    p.columns = ["cls", "arm"] + [f"s{c}" for c in p.columns[2:]]
    p["sel"], p["ev"] = (p.s0 + p.s1) / 2, (p.s2 + p.s3) / 2
    g = p.groupby("cls")
    cf = p.loc[g.sel.idxmin()].set_index("cls").ev.rename("cf")
    return ms.join(cf)


def main():
    L = ["=" * 110, "HORIZONTE LONGO NA FAMILIA C (Entrada 217)", "=" * 110]
    r1000 = load("S5", "heldout")
    cfo = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "cfo_heldout_S5_s*of*_entrada209.csv")))], ignore_index=True)
    old = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "pubh_S5_s*of*_entrada190.csv")))], ignore_index=True)
    A = arms_table(old, cfo)
    q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "warmq_heldout_S5_s*of*_entrada211.csv")))], ignore_index=True)
    wq = q[q.method == "wq"].groupby("cls").makespan.mean().rename("wq")
    base = r1000.set_index("cls").join(A).join(wq)
    W = {T: load_T(T) for T in (5000, 20000)}
    base = base.loc[W[5000].index]
    W[1000] = base[METHODS + ["cf"]]
    rows, fams = [], {}
    for T in (1000, 5000, 20000):
        w = W[T]
        g = lambda col: float(-rel(w[col], w.rand).mean())
        rows.append(dict(T=T, n=len(w), cf_headroom=g("cf"), ql=g("ql"), wq=g("wq"), alns=g("alns"), val=g("val"), sel=g("sel")))
        if T > 1000:
            spec = {"ql-rand": ("ql", "rand"), "wq-rand": ("wq", "rand"), "val-ql": ("val", "ql"), "sel-val": ("sel", "val"), "wq-ql": ("wq", "ql"), "alns-rand": ("alns", "rand")}
            r = []
            for k, (a, b) in spec.items():
                t = test(w[a], w[b]); t.update(id=f"{k}-T{T}"); r.append(t)
            fams[T] = family(r)
            L += [f"\n--- T = {T}: (Holm sobre {len(r)}) ---", fams[T][COLS].to_string(index=False)]
    D = pd.DataFrame(rows)
    L += ["\n--- ganho sobre o aleatorio por horizonte (%, positivo: melhor; folga cf com validacao cruzada de sementes) ---", D.to_string(index=False)]
    h = D.set_index("T")
    e1 = bool(h.loc[1000, "cf_headroom"] > h.loc[5000, "cf_headroom"] > h.loc[20000, "cf_headroom"])
    f5 = fams[5000].set_index("id")
    e2 = bool(f5.loc["ql-rand-T5000", "confirmada"] and h.loc[20000, "ql"] < h.loc[1000, "ql"])
    e3 = all(bool((fams[T].set_index("id").loc[f"val-ql-T{T}", "confirmada"])) for T in (5000, 20000))
    e4 = all(float(fams[T].set_index("id").loc[f"sel-val-T{T}", "mean_d_pct"]) < 0 for T in (5000, 20000))
    e5 = all(float(fams[T].set_index("id").loc[f"wq-ql-T{T}", "mean_d_pct"]) < 0 for T in (5000, 20000))
    e6 = bool((h.loc[20000, ["ql", "alns", "val", "sel", "wq"]] < 0.15).all())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES PRE-FIXADAS ---", f"(E1) folga cf decrescente em 1.000 > 5.000 > 20.000: {yn(e1)} (" + ", ".join(f"{v:.3f}" for v in D.cf_headroom) + ")", f"(E2) Q frio ganha com T = 5.000 (confirmado) e menos com 20.000 que com 1.000: {yn(e2)} (ql: " + ", ".join(f"{v:+.3f}" for v in D.ql) + ")",
          f"(E3) validada vence o Q frio (confirmada) em T = 5.000 e 20.000: {yn(e3)}", f"(E4) seletor melhor que a validada em media em T = 5.000 e 20.000: {yn(e4)}", f"(E5) Q aquecido melhor que o Q frio em media em T = 5.000 e 20.000: {yn(e5)}", f"(E6) T = 20.000: todos os ganhos < 0,15: {yn(e6)} (max {float(h.loc[20000, ['ql', 'alns', 'val', 'sel', 'wq']].max()):.3f})"]
    text = "\n".join(L)
    print(text)
    (R / "entrada217_longc.txt").write_text(text, encoding="utf-8")
    D.to_csv(R / "entrada217_longc_by_T.csv", index=False)
    for T, X in fams.items():
        X.to_csv(R / f"entrada217_longc_T{T}.csv", index=False)


if __name__ == "__main__":
    main()
