"""
analysis/report_entrada212_warmq_tb.py
========================================
Ciclo 19, frente B, ESTAGIO 2: o Q aquecido com DESEMPATE ALEATORIO (experiment/run_cycle19_warmq_tb_entrada212.py; docstring). Diario: Entrada 212 (resultados do estagio 1 e pre-registro do estagio 2; os resultados do estagio 2 entram no mesmo registro). Compara wqn (adaptacao alfa 0,05) e wqn0 (congelada) com o aleatorio, o Q frio, a validada e o seletor offline, a iteracoes iguais; 2.550 + 3.950 instancias, S5 e S15; Holm sobre 4 por familia.
Estagio 1 (Entrada 211; 1 de 6 previsoes): o Q aquecido NAO supera o frio (S5: -0,006 e +0,005; S15: -0,146 e -0,143) e fica abaixo do aleatorio em S15 (-0,09); congelado e' pior (S15: -0,45). A tabela pre-treinada e' quase toda zero (recompensa esparsa) e o argmax cai na acao 0 por empate.

PRE-REGISTRO DO ESTAGIO 2 (registrado DEPOIS de ver o estagio 1 e ANTES de rodar; hashes em results/entrada212_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Familias: Tr wqn contra o aleatorio; Tq wqn contra o Q frio; Tv val contra wqn; Ts sel contra wqn; Tf wqn contra wqn0.
(T1) com o desempate aleatorio o Q aquecido nao fica pior que o aleatorio: ganho medio de wqn sobre o aleatorio >= 0 nos quatro pares.  (T2) nao supera o Q frio: Tq com wqn melhor e confirmado em <= 1 dos 4 pares.  (T3) a validada ainda vence wqn: Tv confirmada nos quatro pares.  (T4) o seletor vence wqn nos quatro pares (Ts confirmada).
(T5) a politica pre-treinada congelada com desempate fica nivelada com a adaptativa: |ganho de wqn sobre wqn0| <= 0,02 em todos os pares.
Leitura J ('a falha do Q aquecido era so' o desempate pela acao 0') = T2 NAO ou ganho de wqn sobre o Q frio >= 0,05 em S15 held-out; previsao: NAO (a recompensa esparsa e' o limite, nao o desempate).
Uso: python -m analysis.report_entrada212_warmq_tb
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada200_selector import SETS, load  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_t(sp, cj):
    w = load(sp, cj)
    q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"warmqtb_{cj}_{sp}_s*of*_entrada212.csv")))], ignore_index=True)
    w = w.merge(q.groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index(), on="cls")
    return w


def main():
    L = ["=" * 110, "Q AQUECIDO COM DESEMPATE ALEATORIO (Entrada 212)", "=" * 110]
    W = {(sp, cj): load_t(sp, cj) for sp in ("S5", "S15") for cj in SETS}
    spec = {"Tr": ("wqn", "rand"), "Tq": ("wqn", "ql"), "Tv": ("val", "wqn"), "Ts": ("sel", "wqn"), "Tf": ("wqn", "wqn0")}
    fams = {k: [] for k in spec}
    for (sp, cj), w in W.items():
        for fid, (a, b) in spec.items():
            t = test(w[a], w[b]); t.update(id=f"{fid}-{sp}-{cj}"); fams[fid].append(t)
    F = {k: family(v) for k, v in fams.items()}
    for k, X in F.items():
        a, b = spec[k]
        L += [f"\n--- {k}: {a} contra {b} (Holm sobre {len(X)}) ---", X[COLS].to_string(index=False)]
    gain = lambda X: -X.mean_d_pct
    t1 = bool((gain(F["Tr"]) >= 0).all())
    t2 = int(F["Tq"].confirmada.sum()) <= 1
    t3 = bool(F["Tv"].confirmada.all())
    t4 = bool(F["Ts"].confirmada.all())
    t5 = bool((gain(F["Tf"]).abs() <= 0.02).all())
    j = (not t2) or float(gain(F["Tq"]).iloc[3]) >= 0.05
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES E LEITURA PRE-FIXADAS ---", f"(T1) wqn >= aleatorio nos quatro: {yn(t1)} (ganho: " + ", ".join(f"{v:+.3f}" for v in gain(F['Tr'])) + ")", f"(T2) wqn melhor que o Q frio confirmado em <= 1 dos 4: {yn(t2)} (ganho sobre o frio: " + ", ".join(f"{v:+.3f}" for v in gain(F['Tq'])) + ")",
          f"(T3) a validada vence wqn nos quatro (confirmada): {yn(t3)}", f"(T4) o seletor vence wqn nos quatro: {yn(t4)}", f"(T5) |wqn - wqn0| <= 0,02 nos quatro: {yn(t5)} (" + ", ".join(f"{v:+.4f}" for v in gain(F['Tf'])) + ")", f"J (a falha era so' o desempate): {yn(j)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada212_warmq_tb.txt").write_text(text, encoding="utf-8")
    for k, X in F.items():
        X.to_csv(R / f"entrada212_warmq_tb_{k}.csv", index=False)


if __name__ == "__main__":
    main()
