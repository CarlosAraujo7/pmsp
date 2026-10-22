"""
analysis/report_entrada205_pfsp_selector.py
=============================================
Ciclo 18, frente D': o SELETOR OFFLINE condicionado a instancia no FLOWSHOP DE PERMUTACAO (replica do ponto 2 de Carlos no segundo problema). Diario: Entrada 205 (pre-registro) e 206 (resultados).
Delineamento: experiment/run_cycle18_pfsp_selector_entrada205.py (docstring): ExtraTrees multi-saida sobre atributos escala-invariantes (pmsp/pfsp_features.py), treinado em 540 instancias sinteticas (60 por classe; sementes disjuntas das de ajuste e de teste) com cada acao fixa e 3 sementes; politica = argmin da predicao;
controle de dados 'valbig' = a acao estatica por classe (n, m) nas MESMAS 540 (no flowshop a classe e' (n, m), logo e' tambem a tabela de consulta por classe), contra a validada de 20 instancias por classe (Entrada 197). Teste: as 360 sinteticas ('syn') e as 90 de Taillard ('ta') da Entrada 197; 3 sementes; T = 1000; F5 e F15.
Validacao cruzada SO' no treino (5 dobras; results/pfspsel_cv_entrada205.json; vista ANTES das previsoes; nenhum dado de teste do seletor lido): desempenho relativo medio centrado (%, menor e' melhor) do seletor / melhor acao por classe (no proprio treino, favoravel a ela) / melhor acao global / oraculo (viesado):
F5 -0,062 / -0,071 / -0,062 / -0,108; F15 -0,092 / -0,093 / -0,081 / -0,138: o seletor fica no nivel da melhor acao GLOBAL em F5 e da melhor por classe em F15; a informacao de classe vale ~0,01 pontos e a da instancia, nada alem dela.

PRE-REGISTRO (congelado ANTES de rodar o seletor no teste; hashes em results/entrada205_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Estatistica: a das Entradas 180-203 (media das sementes por instancia; d = 100 (A - B)/B; Wilcoxon pareado bilateral; IC90 bootstrap; margem +-0,05; Holm sobre 4 = F5/F15 x syn/ta em cada familia). Familias: Cval seletor contra a validada de 20 instancias; Cvb seletor contra valbig; Svb valbig contra a validada; Cql seletor contra o Q-Learning; Crand seletor contra o aleatorio.
(F1) mais dados ajudam a estatica: Svb com d medio < 0 em >= 3 dos 4 pares.  (F2) o seletor supera a validada em media (d < 0) em >= 3 dos 4 pares, mas Cval confirmada em <= 2 dos 4 (ha' pouco a ganhar).  (F3) a instancia nao agrega alem da classe: Cvb NAO confirmada em >= 3 dos 4 pares.
(F4) o seletor vence o Q-Learning: Cql confirmada nos quatro pares.  (F5) o seletor nunca e' pior que o aleatorio: ganho medio > 0 nos quatro pares.  (F6) nas classes grandes (n >= 50, m >= 10) do conjunto 'syn' o seletor captura menos de metade da folga do oraculo (que inclui ruido de sementes), em F5 e em F15.
Leitura S ('o aprendizado offline explora a folga do flowshop acima da validada'): Cval confirmada em >= 3 dos 4 pares. Previsao: NAO. Leitura S- ('o seletor ao menos nao perde para a estatica'): nenhum Cval com d medio > 0 confirmado.
Uso: python -m analysis.report_entrada205_pfsp_selector
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
from analysis.report_entrada197_pfsp import load_test  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
LARGE = ["50x10", "50x20", "100x10", "100x20"]


def load(sp):
    w = load_test(sp)
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pfspsel_test_{sp}_s*of*_entrada205.csv")))], ignore_index=True)
    sm = s.groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    w = w.merge(sm, on="cls")
    assert len(w) == 450, (sp, len(w))
    return w


def main():
    L = ["=" * 110, "SELETOR OFFLINE NO FLOWSHOP DE PERMUTACAO (Entrada 205)", "=" * 110]
    W = {sp: load(sp) for sp in ("F5", "F15")}
    spec = {"Cval": ("sel", "val"), "Cvb": ("sel", "valbig"), "Svb": ("valbig", "val"), "Cql": ("sel", "ql"), "Crand": ("sel", "rand")}
    fams = {k: [] for k in spec}
    for sp in ("F5", "F15"):
        for setname in ("syn", "ta"):
            w = W[sp][W[sp].set == setname]
            for fid, (a, b) in spec.items():
                t = test(w[a], w[b]); t.update(id=f"{fid}-{sp}-{setname}"); fams[fid].append(t)
    F = {k: family(v) for k, v in fams.items()}
    for k, X in F.items():
        a, b = spec[k]
        L += [f"\n--- {k}: {a} contra {b} (Holm sobre {len(X)}) ---", X[COLS].to_string(index=False)]
    rows = []
    for sp in ("F5", "F15"):
        for setname in ("syn", "ta"):
            for nm, x in (("all", W[sp][W[sp].set == setname]), ("large", W[sp][(W[sp].set == setname) & W[sp].klass.isin(LARGE)])):
                g = lambda col: float(-rel(x[col], x.rand).mean())
                rows.append(dict(space=sp, set=setname, cell=nm, n=len(x), oracle=g("oracle"), val=g("val"), valbig=g("valbig"), sel=g("sel"), ql=g("ql"), alns=g("alns"), cap_sel=g("sel") / g("oracle"), cap_valbig=g("valbig") / g("oracle"), cap_val=g("val") / g("oracle")))
    D = pd.DataFrame(rows)
    L += ["\n--- ganho sobre o aleatorio (%, positivo: melhor) e captura da folga do oraculo ---", D.to_string(index=False)]
    yn = lambda b: "SIM" if b else "NAO"
    f1 = int((F["Svb"].mean_d_pct < 0).sum()) >= 3
    f2 = int((F["Cval"].mean_d_pct < 0).sum()) >= 3 and int(F["Cval"].confirmada.sum()) <= 2
    f3 = int((~F["Cvb"].confirmada).sum()) >= 3
    f4 = bool(F["Cql"].confirmada.all())
    f5 = bool((F["Crand"].mean_d_pct < 0).all())
    lg = D[(D.cell == "large") & (D.set == "syn")]
    f6 = bool((lg.cap_sel < 0.5).all())
    s_read = int(F["Cval"].confirmada.sum()) >= 3
    s_minus = not bool(((F["Cval"].mean_d_pct > 0) & (F["Cval"].p_holm < 0.05)).any())
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(F1) valbig melhor que a validada em >= 3 dos 4 pares: {yn(f1)} (d medio: " + ", ".join(f"{v:+.4f}" for v in F['Svb'].mean_d_pct) + ")",
          f"(F2) seletor melhor que a validada em media em >= 3 dos 4 e Cval confirmada em <= 2: {yn(f2)} (d medio: " + ", ".join(f"{v:+.4f}" for v in F['Cval'].mean_d_pct) + f"; confirmadas {int(F['Cval'].confirmada.sum())})",
          f"(F3) Cvb nao confirmada em >= 3 dos 4 pares: {yn(f3)} (d medio: " + ", ".join(f"{v:+.4f}" for v in F['Cvb'].mean_d_pct) + f"; confirmadas {int(F['Cvb'].confirmada.sum())})", f"(F4) Cql confirmada nos quatro pares: {yn(f4)} ({int(F['Cql'].confirmada.sum())} de 4)",
          f"(F5) o seletor melhor que o aleatorio nos quatro pares: {yn(f5)} (d medio: " + ", ".join(f"{v:+.4f}" for v in F['Crand'].mean_d_pct) + ")", f"(F6) classes grandes 'syn': captura do seletor < 0,5 em F5 e F15: {yn(f6)} (" + ", ".join(f"{v:.2f}" for v in lg.cap_sel) + ")",
          f"S (o aprendizado offline explora a folga acima da validada: Cval confirmada em >= 3 dos 4): {yn(s_read)}   |   S- (nenhum Cval com d > 0 confirmado): {yn(s_minus)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada205_pfsp_selector.txt").write_text(text, encoding="utf-8")
    D.to_csv(R / "entrada205_pfsp_selector_by_set.csv", index=False)
    for k, X in F.items():
        X.to_csv(R / f"entrada205_pfsp_selector_{k}.csv", index=False)


if __name__ == "__main__":
    main()
