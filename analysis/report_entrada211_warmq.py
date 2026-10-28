"""
analysis/report_entrada211_warmq.py
=====================================
Ciclo 19, frente B: o Q-LEARNING AQUECIDO (tabela pre-treinada offline em 5.720 instancias, recompensa invariante a escala; pmsp/ig_numba_agents5.py; experiment/run_cycle19_warmq_entrada211.py). Diario: Entrada 211 (pre-registro) e 212 (resultados).
Metodos: wq (adaptacao online alfa = 0,05, epsilon 0,05 constante) e wq0 (politica pre-treinada congelada, alfa = 0, epsilon 0,05), nas 2.550 instancias da descoberta e 3.950 held-out, S5 e S15, 5 sementes, T = 1000; comparados, a iteracoes iguais, com o Q-Learning FRIO padrao (ql), o aleatorio, a acao validada (val), o seletor offline (sel) e a roleta (alns).
Estatistica: a das Entradas 180-211 (media das sementes por instancia; d = 100 (A - B)/B; Wilcoxon pareado; IC90 bootstrap; margem +-0,05; Holm sobre 4 = 2 espacos x 2 conjuntos em cada familia).

PRE-REGISTRO (congelado ANTES de rodar o pre-treino e o teste; hashes em results/entrada211_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Familias: Wql wq contra ql (frio); Wrand wq contra o aleatorio; Wfrz wq contra wq0; Vwq val contra wq; Swq sel contra wq; Wal wq contra a roleta ALNS.
(W1) tirar o handicap do inicio frio ajuda: Wql confirmada (wq melhor) nos quatro pares e com ganho medio >= 0,03 pontos em S15 held-out.  (W2) wq ganha do aleatorio: Wrand confirmada nos quatro pares.  (W3) a adaptacao online adiciona pouco: |ganho de wq sobre wq0| <= 0,01 pontos em todos os quatro pares.
(W4) wq contra a validada: wq melhor que val em media (d de Vwq > 0) em >= 2 dos 4 pares; e o seletor offline vence wq em media (d de Swq < 0) em >= 3 dos 4.  (W5) wq contra a roleta ALNS: wq melhor em S15 (discovery e held-out); em S5 a roleta fica nivelada (|diferenca| < 0,05).
Leitura K ('sem o handicap do inicio frio o aprendizado online fica competitivo com a validada') = nenhum Vwq confirmado a favor da validada em >= 3 dos 4 pares (Vwq com p_holm >= 0,05 ou d <= 0). Probabilidade a priori do autor: ~45%.
Uso: python -m analysis.report_entrada211_warmq
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


def load_w(sp, cj):
    w = load(sp, cj)
    q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"warmq_{cj}_{sp}_s*of*_entrada211.csv")))], ignore_index=True)
    m = q.groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    w = w.merge(m, on="cls")
    assert len(w) == len(load(sp, cj))
    w["grp"] = w.cls.str.split("|").str[1].astype(int)
    return w


def main():
    L = ["=" * 110, "Q-LEARNING AQUECIDO (Entrada 211)", "=" * 110]
    W = {(sp, cj): load_w(sp, cj) for sp in ("S5", "S15") for cj in SETS}
    spec = {"Wql": ("wq", "ql"), "Wrand": ("wq", "rand"), "Wfrz": ("wq", "wq0"), "Vwq": ("val", "wq"), "Swq": ("sel", "wq"), "Wal": ("wq", "alns")}
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
            rows.append(dict(space=sp, set=cj, cell=nm, n=len(x), oracle=g("oracle"), val=g("val"), sel=g("sel"), ql=g("ql"), wq=g("wq"), wq0=g("wq0"), alns=g("alns")))
    D = pd.DataFrame(rows)
    L += ["\n--- ganho sobre o aleatorio (%, positivo: melhor) ---", D.to_string(index=False)]
    gain = lambda X: -X.mean_d_pct
    w1 = bool(F["Wql"].confirmada.all()) and float(gain(F["Wql"]).iloc[3]) >= 0.03
    w2 = bool(F["Wrand"].confirmada.all())
    w3 = bool((gain(F["Wfrz"]).abs() <= 0.01).all())
    w4a = int((F["Vwq"].mean_d_pct > 0).sum()) >= 2
    w4b = int((F["Swq"].mean_d_pct < 0).sum()) >= 3
    wal = F["Wal"].set_index("id")
    w5 = bool((wal.loc[["Wal-S15-discovery", "Wal-S15-heldout"], "mean_d_pct"] < 0).all() and (wal.loc[["Wal-S5-discovery", "Wal-S5-heldout"], "mean_d_pct"].abs() < 0.05).all())
    vw = F["Vwq"]
    k_read = int(((vw.mean_d_pct < 0) & (vw.p_holm < 0.05)).sum()) <= 1
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(W1) wq confirmado melhor que o Q frio nos quatro pares e ganho >= 0,03 em S15 held-out: {yn(w1)} (ganho medio: " + ", ".join(f"{v:+.3f}" for v in gain(F['Wql'])) + ")",
          f"(W2) wq confirmado melhor que o aleatorio nos quatro pares: {yn(w2)} (" + ", ".join(f"{v:+.3f}" for v in gain(F['Wrand'])) + ")", f"(W3) |wq - wq0| <= 0,01 nos quatro: {yn(w3)} (" + ", ".join(f"{v:+.4f}" for v in gain(F['Wfrz'])) + ")",
          f"(W4a) wq melhor que a validada em media em >= 2 dos 4: {yn(w4a)} (ganho de wq sobre val: " + ", ".join(f"{v:+.3f}" for v in F['Vwq'].mean_d_pct) + ")", f"(W4b) o seletor vence wq em media em >= 3 dos 4: {yn(w4b)} (" + ", ".join(f"{-v:+.3f}" for v in F['Swq'].mean_d_pct) + ")",
          f"(W5) wq melhor que a roleta em S15 e nivelado em S5: {yn(w5)} (d medio: " + ", ".join(f"{v:+.3f}" for v in F['Wal'].mean_d_pct) + ")", f"K (sem o inicio frio o aprendizado online fica competitivo com a validada: no maximo um Vwq confirmado a favor da validada): {yn(k_read)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada211_warmq.txt").write_text(text, encoding="utf-8")
    D.to_csv(R / "entrada211_warmq_cells.csv", index=False)
    for k, X in F.items():
        X.to_csv(R / f"entrada211_warmq_{k}.csv", index=False)


if __name__ == "__main__":
    main()
