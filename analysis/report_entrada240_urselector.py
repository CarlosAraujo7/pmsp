"""
analysis/report_entrada240_urselector.py
==========================================
Ciclo 22, frente C: o SELETOR OFFLINE no hospedeiro de maquinas nao relacionadas (R||Cmax), com o custo de treino cobrado, contra a acao validada, a tabela de consulta por classe e os aprendizes online. Desenho e dados: experiment/run_cycle22_urselector_entrada240.py (treino em 1.200 instancias novas x todas as acoes x 3 sementes;
teste nas 1.200 instancias da Entrada 234, 5 sementes, metodos 'sel' e 'cls'); a acao validada, o aleatorio e os aprendizes vem de results/ur_test_*_entrada234.csv. Diario: Entrada 240 (pre-registro) e 241 (resultados). Saidas: results/entrada240_urselector.txt, entrada240_urselector_*.csv.

PRE-REGISTRO (congelado ANTES de qualquer saida do teste; hashes em results/entrada240_prereg_hashes.txt, que inclui o modelo e a tabela de classe ajustados)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
Transparencia: o treino e o ajuste do modelo ja' foram feitos antes deste registro: a tabela de classe do treino coincide com a acao validada do ajuste nas 12 classes de S5 e em 9 das 12 de S15 (o que as previsoes levam em conta; os resultados no teste nao foram vistos).
Testes (Wilcoxon pareado, ties fora, IC90 por bootstrap, margem de equivalencia +-0,05; 'confirmada' = p_holm < 0,05 e o primeiro braco melhor; Holm sobre 2 espacos em cada familia): SV = seletor contra a acao validada; SC = seletor contra a tabela de classe; CV = tabela de classe contra a validada; SB = seletor contra o melhor aprendiz online do espaco (o de maior ganho agregado entre ql, ucb, pg, alns, qo); SR = seletor contra o aleatorio.
Ganho = -d (pontos percentuais; positivo: o primeiro braco e' melhor). Custo de treino cobrado: execucoes do IG de treino = 1.200 x 3 x (numero de acoes).
(V1) o seletor supera a acao validada alem da margem (confirmada e nao paridade) em pelo menos um espaco (40%).  (V2) o ganho do seletor sobre a validada (estimativa pontual) e' >= -0,05 nos dois espacos, isto e', ele nao e' pior alem da margem (75%).  (V3) o seletor vence o melhor aprendiz online (confirmada, alem da margem) nos dois espacos (92%).
(V4) a tabela de classe e' nivelada com o seletor (|diferenca| < 0,05) nos dois espacos (60%).  (V5) o seletor captura >= 0,80 da folga de uma semente nos dois espacos (45%).  (V6) o ganho do seletor sobre a validada e' maior em S15 que em S5 (60%).
Leitura R4 ('tambem num hospedeiro de folga alta o seletor offline acrescenta pouco ao conhecimento de classe'): V4 verdadeira e V1 falsa (previsao: 35%).
Nota de codigo: o relatorio falhou ao carregar (a coluna do metodo 'cls' colidia com o indice 'cls' da classe) ANTES de qualquer resultado ser visto; a coluna foi renomeada 'ctab' no relatorio (o CSV continua com 'cls') e o hash do relatorio foi gravado de novo; as previsoes nao mudaram.
Uso: python -m analysis.report_entrada240_urselector
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
LEARN = ["ql", "ucb", "pg", "alns", "qo"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load(sp):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_test_{sp}_s*of*_entrada234.csv")))], ignore_index=True)
    d = d[~d.method.str.startswith("arm")]
    ref = d[d.method.isin(["rand", "val"] + LEARN)].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    s = pd.read_csv(R / f"urselector_test_{sp}_entrada240.csv")
    sm = s.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").rename(columns={"cls": "ctab"})
    return ref.join(sm, how="inner").reset_index()


def main():
    L = ["=" * 110, "SELETOR OFFLINE NO R||Cmax (Entrada 240)", "=" * 110]
    pooled, fam = [], {k: [] for k in ("SV", "SC", "CV", "SB", "SR")}
    P0 = pd.read_csv(R / "entrada234_ur_pooled.csv").set_index("space")
    for sp, A in (("S5", 5), ("S15", 15)):
        w = load(sp)
        g = {k: float(-rel(w[k], w.rand).mean()) for k in ["sel", "ctab", "val"] + LEARN}
        best = max(LEARN, key=lambda k: g[k])
        pooled.append(dict(space=sp, n=len(w), g_sel=g["sel"], g_cls=g["ctab"], g_val=g["val"], best_learner=best, g_best=g[best], h_biased=float(P0.loc[sp, "h_biased"]), captured_sel=g["sel"] / float(P0.loc[sp, "h_biased"]), captured_val=g["val"] / float(P0.loc[sp, "h_biased"]), train_runs=1200 * 3 * A))
        for key, (a, b) in {"SV": ("sel", "val"), "SC": ("sel", "ctab"), "CV": ("ctab", "val"), "SB": ("sel", best), "SR": ("sel", "rand")}.items():
            t = test(w[a], w[b])
            t.update(id=f"{key}-{sp}")
            fam[key].append(t)
    Pd = pd.DataFrame(pooled)
    F = {k: family(v) for k, v in fam.items()}
    L += ["\n--- agregado por espaco (ganhos sobre o aleatorio, pontos percentuais; captura = ganho / folga de uma semente; treino = execucoes do IG) ---", Pd.to_string(index=False)]
    names = {"SV": "seletor contra a acao validada", "SC": "seletor contra a tabela de classe", "CV": "tabela de classe contra a validada", "SB": "seletor contra o melhor aprendiz online", "SR": "seletor contra o aleatorio"}
    for k, X in F.items():
        L += [f"\n--- {k}: {names[k]} (negativo: o primeiro e' melhor; Holm sobre 2) ---", X[COLS].to_string(index=False)]
    yn = lambda b: "SIM" if b else "NAO"
    win = lambda r: bool(r.confirmada) and not bool(r["equivalente_0.05"]) and r.mean_d_pct < 0
    v1 = any(win(F["SV"].iloc[i]) for i in range(2))
    v2 = bool(((-F["SV"].mean_d_pct) >= -0.05).all())
    v3 = all(win(F["SB"].iloc[i]) for i in range(2))
    v4 = bool((F["SC"].mean_d_pct.abs() < 0.05).all())
    v5 = bool((Pd.captured_sel >= 0.80).all())
    v6 = bool((-F["SV"].mean_d_pct.iloc[1]) > (-F["SV"].mean_d_pct.iloc[0]))
    L += ["\n--- PREVISOES E LEITURA PRE-FIXADAS ---", f"(V1) o seletor supera a validada alem da margem em algum espaco: {yn(v1)} (diferencas " + ", ".join(f"{-v:+.3f}" for v in F['SV'].mean_d_pct) + ")",
          f"(V2) o seletor nao e' pior que a validada alem da margem nos dois: {yn(v2)}", f"(V3) o seletor vence o melhor aprendiz online (confirmada, alem da margem) nos dois: {yn(v3)}",
          f"(V4) a tabela de classe e' nivelada com o seletor nos dois: {yn(v4)} (" + ", ".join(f"{-v:+.3f}" for v in F['SC'].mean_d_pct) + ")", f"(V5) o seletor captura >= 0,80 da folga de uma semente nos dois: {yn(v5)} (" + ", ".join(f"{v:.3f}" for v in Pd.captured_sel) + ")",
          f"(V6) o ganho do seletor sobre a validada e' maior em S15 que em S5: {yn(v6)}", f"R4 (o seletor acrescenta pouco ao conhecimento de classe: V4 e nao V1): {yn(v4 and not v1)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada240_urselector.txt").write_text(text, encoding="utf-8")
    Pd.to_csv(R / "entrada240_urselector_pooled.csv", index=False)
    for k, X in F.items():
        X.to_csv(R / f"entrada240_urselector_{k}.csv", index=False)


if __name__ == "__main__":
    main()
