"""
analysis/report_entrada249_closedloop.py
=========================================
Entrada 249 (Ciclo 27): ORACULOS DE MALHA FECHADA (dependentes do estado) no P||Cmax sintetico e no R||Cmax, espacos S5 e S15 (pmsp/closedloop_numba.py, experiment/run_cycle27_closedloop_entrada249.py).
  la8  oraculo MIOPE: a cada iteracao, 8 execucoes independentes de cada acao a partir da solucao atual estimam a melhora esperada do makespan (a recompensa que os aprendizes recebem); executa-se, com uma execucao nova, a acao de maior estimativa. 2 sementes; 100 instancias por classe.
  la0  controle do mesmo codigo sem avaliacoes (acao sorteada): deve reproduzir o aleatorio.
  ro   oraculo de ROLLOUT (melhoria de politica sobre a acao validada, nao miope): a cada 50 iteracoes, 12 execucoes independentes de cada acao por 50 iteracoes seguidas da acao validada ate' o fim das 1.000 iteracoes estimam a melhora final esperada; aplica-se por 50 iteracoes a de maior estimativa. 1 semente; 20 instancias por classe (as primeiras).
Nenhum dos dois e' justo no orcamento (centenas de milhares a milhoes de avaliacoes extras por execucao); sao referencias para o que uma politica que decide pelo estado poderia fazer. Mesmas instancias de teste das Entradas 180 e 234, pareadas com `rand` e `val` (media das 5 sementes por instancia). Ganho = -d (pontos do makespan). Famílias (Holm sobre 4 hospedeiro-espacos): A la8 contra o aleatorio; B a validada contra la8 (B': sentido contrario); C ro contra o aleatorio; D a validada contra ro (D': contrario); E controle la0 contra o aleatorio.

PRE-REGISTRO (congelado ANTES de rodar o teste; os pilotos em instancias de AJUSTE informaram so' o tempo e o sinal: la8 ~ +0,5 e ro ~ +1,2 sobre o aleatorio no R||Cmax S15, la8 ~ +0,1 no P||Cmax S15; hashes em results/entrada249_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------
(M1) A acao validada bate o oraculo miope la8 (B, confirmada) nos dois espacos do R||Cmax. (85%)
(M2) No R||Cmax S15, o ganho de la8 sobre o aleatorio e' no maximo 50% do ganho da validada sobre o aleatorio. (80%)
(M3) No P||Cmax sintetico, o ganho de la8 sobre o aleatorio e' no maximo 0,15 pontos nos dois espacos. (80%)
(M4) O oraculo de rollout ro nao bate a acao validada alem da margem (D', confirmada e nao equivalente) em nenhum dos quatro hospedeiro-espacos. (70%)
(M5) No R||Cmax S15, o ganho de ro sobre o aleatorio e' menor que o da validada. (65%)
(M6) O controle la0 e' equivalente ao aleatorio (IC90 dentro de +-0,05) nos quatro hospedeiro-espacos. (85%)
Uso: python -m analysis.report_entrada249_closedloop
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

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
HOSTS, SPACES = ("pc", "ur"), ("S5", "S15")
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load(host, sp):
    if host == "pc":
        m = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
    else:
        m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_test_{sp}_s*of*_entrada234.csv")))], ignore_index=True)
    m = m[m.method.isin(["rand", "val"])]
    c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"cl_{host}_{sp}_s*of*_entrada249.csv")))], ignore_index=True)
    a = m.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    b = c.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    return a.join(b, how="inner").reset_index()


def beats(F):
    return (F.confirmada) & (~F["equivalente_0.05"])


def main():
    W = {(h, s): load(h, s) for h in HOSTS for s in SPACES}
    T = {k: [] for k in "ABCDEFG"}
    split = []
    for (h, s), w in W.items():
        wr = w.dropna(subset=["ro"])
        a = test(w.la8, w.rand); a.update(id=f"A-{h}-{s}"); T["A"].append(a)
        b = test(w.val, w.la8); b.update(id=f"B-{h}-{s}"); T["B"].append(b)
        b2 = test(w.la8, w.val); b2.update(id=f"B'-{h}-{s}"); T["F"].append(b2)
        c = test(wr.ro, wr.rand); c.update(id=f"C-{h}-{s}"); T["C"].append(c)
        d = test(wr.val, wr.ro); d.update(id=f"D-{h}-{s}"); T["D"].append(d)
        d2 = test(wr.ro, wr.val); d2.update(id=f"D'-{h}-{s}"); T["G"].append(d2)
        e = test(w.la0, w.rand); e.update(id=f"E-{h}-{s}"); T["E"].append(e)
        split.append(dict(host=h, space=s, n=len(w), n_ro=len(wr), g_val=float(-rel(w.val, w.rand).mean()), g_la8=float(-rel(w.la8, w.rand).mean()), g_la0=float(-rel(w.la0, w.rand).mean()),
                          g_val_sub=float(-rel(wr.val, wr.rand).mean()), g_ro=float(-rel(wr.ro, wr.rand).mean())))
    F = {k: family(v) for k, v in T.items()}
    S = pd.DataFrame(split)
    L = ["=" * 110, "ORACULOS DE MALHA FECHADA (Entrada 249)", "=" * 110,
         "--- ganhos sobre o aleatorio (pontos): validada, la8 (100 instancias por classe), controle la0, e na subamostra de 20 instancias por classe: validada e ro ---", S.to_string(index=False)]
    for k, lab in (("A", "A: la8 contra o aleatorio"), ("B", "B: validada contra la8"), ("F", "B': la8 contra a validada"), ("C", "C: ro contra o aleatorio (20 instancias por classe)"),
                   ("D", "D: validada contra ro"), ("G", "D': ro contra a validada"), ("E", "E: controle la0 contra o aleatorio")):
        L += ["", f"--- Familia {lab} (Holm sobre 4) ---", F[k][COLS].to_string(index=False)]
    ur = S[S.host == "ur"].set_index("space")
    pc = S[S.host == "pc"]
    m1 = bool(F["B"][F["B"].id.str.contains("-ur-")].confirmada.all())
    m2 = bool(ur.loc["S15", "g_la8"] <= 0.5 * ur.loc["S15", "g_val"])
    m3 = bool((pc.g_la8 <= 0.15).all())
    m4 = int(beats(F["G"]).sum())
    m5 = bool(ur.loc["S15", "g_ro"] < ur.loc["S15", "g_val_sub"])
    m6 = bool(F["E"]["equivalente_0.05"].all())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["", "--- previsoes ---",
          f"(M1) a validada bate la8 nos dois espacos do R||Cmax: {yn(m1)}",
          f"(M2) R||Cmax S15: ganho de la8 / ganho da validada = {ur.loc['S15', 'g_la8'] / ur.loc['S15', 'g_val']:.3f} (previsto <= 0,5): {yn(m2)}",
          f"(M3) P||Cmax: ganho de la8 = {pc.g_la8.round(3).tolist()} (previsto <= 0,15): {yn(m3)}",
          f"(M4) ro bate a validada alem da margem (D'): {m4} de 4 (previsto 0): {yn(m4 == 0)}",
          f"(M5) R||Cmax S15: ganho de ro {ur.loc['S15', 'g_ro']:.3f} contra a validada {ur.loc['S15', 'g_val_sub']:.3f} na subamostra (previsto menor): {yn(m5)}",
          f"(M6) o controle la0 e' equivalente ao aleatorio nos quatro: {yn(m6)}"]
    pred = [m1, m2, m3, m4 == 0, m5, m6]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada249_closedloop.txt").write_text(text + "\n", encoding="utf-8")
    # so' um sentido de cada teste vai ao CSV (o controle global le os CSV; os sentidos contrarios B' e D' ficam so' no texto)
    pd.concat([F["A"], F["B"], F["C"], F["D"], F["E"]]).to_csv(R / "entrada249_closedloop_tests.csv", index=False)
    S.to_csv(R / "entrada249_closedloop_split.csv", index=False)


if __name__ == "__main__":
    main()
