"""
analysis/report_entrada203_qvariants.py
=========================================
Ciclo 18, frente B': o Q-Learning AJUSTADO e com recompensa SENSIVEL A PLATOS (pmsp/ig_numba_agents4.py; runner experiment/run_cycle18_qvariants_entrada203.py) nas instancias publicadas (2.550 discovery e 3.950 held-out; S5 e S15; T = 1000; 5 sementes). Diario: Entrada 203 (pre-registro) e 204 (resultados).
Metodos: qt (alfa, gama e decaimento do epsilon ajustados por familia em 525 instancias; grade de 27), qtp (esses tres e o bonus de plato; grade de 81), comparados com o Q-Learning padrao (ql), o aleatorio, a acao validada estatica (val), o seletor offline (sel; Entrada 200) e a roleta ALNS, a iteracoes iguais.
Estatistica: a das Entradas 180-202 (unidade = media das sementes por instancia; d = 100 (A - B)/B; Wilcoxon pareado; IC90 bootstrap; margem +-0,05; Holm sobre 4 = 2 espacos x 2 conjuntos em cada familia).

PRE-REGISTRO (congelado ANTES de rodar o ajuste e o teste; hashes em results/entrada203_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Familias: Aqt qt contra ql; Aqp qtp contra ql; Rqt qt contra o aleatorio; Rqp qtp contra o aleatorio; Vqt val contra qt; Vqp val contra qtp; Sqt sel contra qt; Sqp sel contra qtp. Gains em pontos (positivo: o primeiro e' melhor).
(R1) o ajuste de hiperparametros ajuda pouco: o ganho medio de qt sobre ql esta' em [-0,01; +0,04] pontos nos quatro pares (S5, S15 x discovery, held-out).
(R2) a recompensa sensivel a platos tambem: o ganho medio de qtp sobre ql esta' em [-0,01; +0,05] nos quatro pares, e qtp >= qt (ganho medio sobre ql maior ou igual) em >= 3 dos 4.
(R3) a validada estatica continua melhor: Vqt confirmada em >= 3 dos 4 pares e Vqp confirmada em >= 3 dos 4.
(R4) o seletor offline supera ambos: Sqt confirmada nos quatro pares e Sqp confirmada nos quatro.
(R5) nas celulas onde o Q-Learning padrao recupera mais da validada (familia C, tercis 1 e 2), a fracao do ganho da validada recuperada por qt e por qtp no held-out e' >= a do ql padrao nos dois espacos.
(R6) alguma configuracao ajustada difere do padrao: pelo menos 3 das 6 escolhas (familia x espaco) de qt diferem de (0,1; 0,9; 0,995).
Leitura T ('ajustar o Q-Learning e premiar platos muda a conclusao') = qt ou qtp batem a validada com Holm em >= 2 dos 4 pares (val - q < 0 confirmada). Previsao: NAO.
Uso: python -m analysis.report_entrada203_qvariants
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


def load_q(sp, cj):
    w = load(sp, cj)
    q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"qvar_{sp}_{cj}_s*of*_entrada203.csv")))], ignore_index=True)
    m = q.groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    w = w.merge(m, on="cls")
    assert len(w) == len(load(sp, cj))
    w["grp"] = w.cls.str.split("|").str[1].astype(int)
    return w


def main():
    L = ["=" * 110, "Q-LEARNING AJUSTADO E COM RECOMPENSA SENSIVEL A PLATOS (Entrada 203)", "=" * 110]
    ch = json.loads((R / "qvar_choice_entrada203.json").read_text())
    L.append("configuracoes escolhidas (alfa, gama, decaimento, bonus, rmode) por espaco e familia; lacuna media no ajuste da escolhida contra a do padrao:")
    n_diff = 0
    for sp in ch:
        for fam in ch[sp]:
            L.append(f"  {sp} {fam}: qt {ch[sp][fam]['qt']['cfg']} (gap {ch[sp][fam]['qt']['gap']:.4f} vs padrao {ch[sp][fam]['qt']['default_gap']:.4f}); qtp {ch[sp][fam]['qtp']['cfg']} (gap {ch[sp][fam]['qtp']['gap']:.4f})")
            n_diff += int(tuple(ch[sp][fam]["qt"]["cfg"][:3]) != (0.1, 0.9, 0.995))
    W = {(sp, cj): load_q(sp, cj) for sp in ("S5", "S15") for cj in SETS}
    spec = {"Aqt": ("qt", "ql"), "Aqp": ("qtp", "ql"), "Rqt": ("qt", "rand"), "Rqp": ("qtp", "rand"), "Vqt": ("val", "qt"), "Vqp": ("val", "qtp"), "Sqt": ("sel", "qt"), "Sqp": ("sel", "qtp")}
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
        for nm, x in (("all", w), ("C12", w[(w.fam == "C") & (w.grp >= 1)]), ("E", w[w.fam == "E"])):
            g = lambda col: float(-rel(x[col], x.rand).mean())
            rows.append(dict(space=sp, set=cj, cell=nm, n=len(x), oracle=g("oracle"), val=g("val"), sel=g("sel"), ql=g("ql"), qt=g("qt"), qtp=g("qtp"), alns=g("alns"), share_ql=g("ql") / g("val"), share_qt=g("qt") / g("val"), share_qtp=g("qtp") / g("val")))
    D = pd.DataFrame(rows)
    L += ["\n--- ganho sobre o aleatorio (%, positivo: melhor) e fracao do ganho da validada recuperada ---", D.to_string(index=False)]
    gain = lambda X: -X.mean_d_pct
    r1 = bool(((gain(F["Aqt"]) >= -0.01) & (gain(F["Aqt"]) <= 0.04)).all())
    r2 = bool(((gain(F["Aqp"]) >= -0.01) & (gain(F["Aqp"]) <= 0.05)).all()) and int((gain(F["Aqp"]) >= gain(F["Aqt"])).sum()) >= 3
    r3 = int(F["Vqt"].confirmada.sum()) >= 3 and int(F["Vqp"].confirmada.sum()) >= 3
    r4 = bool(F["Sqt"].confirmada.all() and F["Sqp"].confirmada.all())
    c = D[(D.cell == "C12") & (D.set == "heldout")]
    r5 = bool(((c.share_qt >= c.share_ql) & (c.share_qtp >= c.share_ql)).all())
    r6 = n_diff >= 3
    t_beat = int(((F["Vqt"].mean_d_pct > 0) & (F["Vqt"].p_holm < 0.05)).sum()) >= 2 or int(((F["Vqp"].mean_d_pct > 0) & (F["Vqp"].p_holm < 0.05)).sum()) >= 2
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(R1) ganho de qt sobre ql em [-0,01; +0,04] nos quatro pares: {yn(r1)} (" + ", ".join(f"{v:+.3f}" for v in gain(F['Aqt'])) + ")",
          f"(R2) ganho de qtp sobre ql em [-0,01; +0,05] nos quatro e qtp >= qt em >= 3: {yn(r2)} (" + ", ".join(f"{v:+.3f}" for v in gain(F['Aqp'])) + ")",
          f"(R3) a validada confirmada contra qt e contra qtp em >= 3 dos 4 pares: {yn(r3)} (confirmadas {int(F['Vqt'].confirmada.sum())} e {int(F['Vqp'].confirmada.sum())})", f"(R4) o seletor confirmado contra qt e qtp nos quatro pares: {yn(r4)} ({int(F['Sqt'].confirmada.sum())} e {int(F['Sqp'].confirmada.sum())} de 4)",
          f"(R5) familia C tercis 1-2, held-out: fracao recuperada de qt e qtp >= a do ql nos dois espacos: {yn(r5)} (ql " + ", ".join(f"{v:.2f}" for v in c.share_ql) + "; qt " + ", ".join(f"{v:.2f}" for v in c.share_qt) + "; qtp " + ", ".join(f"{v:.2f}" for v in c.share_qtp) + ")",
          f"(R6) >= 3 das 6 escolhas de qt diferem do padrao: {yn(r6)} ({n_diff} de 6)", f"T (ajustar e premiar platos muda a conclusao: qt ou qtp batem a validada com Holm em >= 2 dos 4 pares): {yn(t_beat)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada203_qvariants.txt").write_text(text, encoding="utf-8")
    D.to_csv(R / "entrada203_qvariants_cells.csv", index=False)
    for k, X in F.items():
        X.to_csv(R / f"entrada203_qvariants_{k}.csv", index=False)


if __name__ == "__main__":
    main()
