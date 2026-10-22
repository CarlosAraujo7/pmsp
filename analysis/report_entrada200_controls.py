"""
analysis/report_entrada200_controls.py
========================================
Entrada 200, ESTAGIO 2: controles do seletor offline (a critica obvia: o seletor viu 5.720 instancias de treino contra 525 da validada, e roda ~15% mais avaliacoes). Runner: experiment/run_cycle18_selector_controls_entrada200.py (docstring: valbig, valclass e as versoes a avaliacoes iguais).
Estagio 1 (analysis/report_entrada200_selector.py; 7 de 7 previsoes acertaram): o seletor supera a validada estatica por 0,024 a 0,044 pontos (paridade dentro da margem em tres dos quatro pares), os aprendizes online por 0,04 a 0,14 e captura 0,79-0,84 do ganho do oraculo contra 0,58-0,66 da validada.
Estas previsoes foram registradas DEPOIS de ver o estagio 1 e ANTES de rodar os controles (hashes em results/entrada200_controls_prereg_hashes.txt).

PRE-REGISTRO DO ESTAGIO 2
-------------------------
Estatistica: a das Entradas 180-200 (media das sementes por instancia; Holm sobre 4 = S5/S15 x discovery/heldout em cada familia; equivalencia a 0 se IC90 em +-0,05). Familias: Cvb seletor contra valbig; Ccl seletor contra valclass; CvE seletor (T = 1000) contra a validada a AVALIACOES IGUAIS; CvbE e CclE idem para valbig e valclass; Svb valbig contra val; Scl valclass contra val.
(Q1) mais dados ajudam a estatica: Svb com d medio < 0 em >= 3 dos 4 pares.  (Q2) Cvb confirmada em >= 3 dos 4 pares e d medio <= -0,015 no S15 held-out.  (Q3) Ccl com d medio < 0 nos quatro pares e confirmada em >= 2 dos 4.
(Q4) a avaliacoes iguais: CvE confirmada em >= 3 dos 4 pares e d medio <= -0,015 em cada par em que e' confirmada.  (Q5) valclass melhor que valbig no held-out (Scl menos Svb: d medio de valclass contra valbig < 0 em S5 e S15), onde a cobertura de classes do treino e' de 95% (no discovery e' de 56%).
Leitura I ('condicionar na instancia agrega valor alem de mais dados e de classes mais finas') = Q2 e Q3. Se Q2 falhar, o ganho do estagio 1 e' principalmente efeito de dados; se Q3 falhar, a tabela de consulta por classe ja' fecha a folga que o seletor fecha.
Uso: python -m analysis.report_entrada200_controls
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
from analysis.report_entrada200_selector import SETS, load  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_ctl(sp, cj):
    w = load(sp, cj)
    c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"selctl_{sp}_{cj}_s*of*_entrada200.csv")))], ignore_index=True)
    m = c.groupby(["cls", "method"]).makespan.mean().unstack("method")
    e = c[c.method == "valE"].groupby("cls").evals.mean().rename("evals_valE")
    w = w.merge(m.reset_index(), on="cls").merge(e.reset_index(), on="cls")
    assert len(w) == len(load(sp, cj)), (sp, cj)
    return w


def main():
    L = ["=" * 110, "CONTROLES DO SELETOR OFFLINE (Entrada 200, estagio 2)", "=" * 110]
    W = {(sp, cj): load_ctl(sp, cj) for sp in ("S5", "S15") for cj in SETS}
    spec = {"Cvb": ("sel", "valbig"), "Ccl": ("sel", "valclass"), "CvE": ("sel", "valE"), "CvbE": ("sel", "valbigE"), "CclE": ("sel", "valclassE"), "Svb": ("valbig", "val"), "Scl": ("valclass", "val"), "Sbc": ("valclass", "valbig")}
    fams = {k: [] for k in spec}
    for (sp, cj), w in W.items():
        for fid, (a, b) in spec.items():
            t = test(w[a], w[b]); t.update(id=f"{fid}-{sp}-{cj}"); fams[fid].append(t)
    F = {k: family(v) for k, v in fams.items()}
    for k, X in F.items():
        a, b = spec[k]
        L += [f"\n--- {k}: {a} contra {b} (Holm sobre {len(X)}) ---", X[COLS].to_string(index=False)]
    rows = []
    for (sp, cj), w in W.items():
        for fam, x in list(w.groupby("fam")) + [("all", w)]:
            g = lambda col: float(-rel(x[col], x.rand).mean())
            rows.append(dict(space=sp, set=cj, fam=fam, n=len(x), sel=g("sel"), val=g("val"), valbig=g("valbig"), valclass=g("valclass"), valE=g("valE"), valbigE=g("valbigE"), valclassE=g("valclassE"), oracle=g("oracle"), cap_sel=g("sel") / g("oracle") if g("oracle") > 0 else np.nan,
                             cap_valclass=g("valclass") / g("oracle") if g("oracle") > 0 else np.nan, evals_ratio_valE=float(x.evals_valE.mean() / x.evals_val.mean())))
    D = pd.DataFrame(rows)
    L += ["\n--- ganho sobre o aleatorio (%, positivo: melhor) ---", D.to_string(index=False)]
    yn = lambda b: "SIM" if b else "NAO"
    q1 = int((F["Svb"].mean_d_pct < 0).sum()) >= 3
    q2 = int(F["Cvb"].confirmada.sum()) >= 3 and float(F["Cvb"].mean_d_pct.iloc[3]) <= -0.015
    q3 = bool((F["Ccl"].mean_d_pct < 0).all()) and int(F["Ccl"].confirmada.sum()) >= 2
    qe = F["CvE"]
    q4 = int(qe.confirmada.sum()) >= 3 and bool((qe[qe.confirmada].mean_d_pct <= -0.015).all())
    sb = F["Sbc"]
    q5 = bool((sb.mean_d_pct.iloc[[1, 3]] < 0).all())
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(Q1) valbig melhor que val em >= 3 dos 4 pares: {yn(q1)} (d medio: " + ", ".join(f"{v:+.4f}" for v in F['Svb'].mean_d_pct) + ")",
          f"(Q2) seletor contra valbig confirmada em >= 3 dos 4 e d <= -0,015 no S15 held-out: {yn(q2)} (d medio: " + ", ".join(f"{v:+.4f}" for v in F['Cvb'].mean_d_pct) + f"; confirmadas {int(F['Cvb'].confirmada.sum())})",
          f"(Q3) seletor contra valclass: d < 0 nos quatro e confirmada em >= 2: {yn(q3)} (d medio: " + ", ".join(f"{v:+.4f}" for v in F['Ccl'].mean_d_pct) + f"; confirmadas {int(F['Ccl'].confirmada.sum())})",
          f"(Q4) seletor contra a validada a avaliacoes iguais confirmada em >= 3 dos 4 e d <= -0,015 nas confirmadas: {yn(q4)} (d medio: " + ", ".join(f"{v:+.4f}" for v in qe.mean_d_pct) + f"; confirmadas {int(qe.confirmada.sum())})",
          f"(Q5) valclass melhor que valbig no held-out (S5 e S15): {yn(q5)} (d medio: " + ", ".join(f"{v:+.4f}" for v in sb.mean_d_pct) + ")", f"I (condicionar na instancia agrega valor alem de mais dados e de classes mais finas) = Q2 e Q3: {yn(q2 and q3)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada200_controls.txt").write_text(text, encoding="utf-8")
    D.to_csv(R / "entrada200_controls_by_family.csv", index=False)
    for k, X in F.items():
        X.to_csv(R / f"entrada200_controls_{k}.csv", index=False)


if __name__ == "__main__":
    main()
