"""
analysis/report_entrada188_new.py
===================================
Ciclo 17, frente B': Thompson com desconto, DMAB e corrida por reducao pela metade como concorrentes online do aprendizado, em S5 e S15 (P||Cmax sintetico, T = 1000). Diario: Entrada 188 (pre-registro) e 189 (resultados).
Delineamento: experiment/run_cycle17_new_entrada188.py e pmsp/ig_numba_agents3.py (59 testes).

PRE-REGISTRO (congelado ANTES de rodar o ajuste, a calibracao e o teste; hashes em results/entrada188_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------
Estatistica (a das Entradas 180 e 185): unidade = media das sementes por instancia; d = 100 (A - B) / B em makespan (negativo: A melhor); Wilcoxon pareado bilateral (empates descartados), IC90 bootstrap (2000), Holm sobre a familia, equivalencia a 0 se o IC90 cabe em +-0,05%;
'confirmada' = p_holm < 0,05 e d < 0, lida como 'paridade' se tambem equivalente. Familias com Holm sobre 6 (3 metodos x 2 espacos): Aiter (vs aleatorio a iteracoes iguais; o aleatorio e' o de results/agents_S5|S15_entrada180.csv), Acpu (vs aleatorio com T_eq = round(1000 K)),
Bval (acao validada por classe, a das Entradas 180, vs o metodo). Descritiva: o d medio dos NOVE aprendizes (ql, ucb, pg, pm, ap, alns, ts, dmab, race) contra o aleatorio a iteracoes iguais.
Leituras: R1 'um metodo novo bate o aleatorio por >= 0,05% a iteracoes iguais' = alguma Aiter confirmada e nao paridade; R2 o mesmo a CPU igual; R3 'a acao validada vence em >= 5 dos 6 pares'; R4 'a roleta tipo ALNS continua sendo o melhor aprendiz online a iteracoes iguais em S5' = nenhum dos outros oito tem d medio
menor que o da roleta por mais de 0,01; R5 'a corrida ganha >= 0,02% do aleatorio a iteracoes iguais em S5 ou S15'.
Previsoes (9; registradas antes de rodar; licao da Entrada 186: nao extrapolar de uma familia de aprendizes para outra, por isso as faixas sao largas): (1) K_CPU (media geometrica por classe) em [0,9; 2,2] para ts, [0,95; 1,25] para dmab, [0,6; 1,3] para race, em S5 e S15; (2) Aiter: ts em [-0,03; +0,06], dmab em [-0,03; +0,03],
race em [-0,06; +0,04]; (3) Acpu: todo d em [-0,10; +0,10]; (4) Bval: todo d em [-0,12; -0,04]; (5) R1 NAO; (6) R2 NAO; (7) R3 SIM; (8) R4 SIM; (9) R5 NAO.
Transparencia: um teste de fumaca em pasta temporaria (2 classes de n <= 30, 2-3 instancias; fora dos dados) validou o runner e o relatorio de ponta a ponta e mostrou K_CPU do Thompson de 1,6 a 2,2 nessas classes pequenas (duas gamas por acao e iteracao pesam onde a iteracao e' barata); por isso a faixa de ts na previsao (1) foi
alargada de [0,9; 1,8] para [0,9; 2,2]. Nenhum resultado de qualidade entrou nas previsoes.
Uso: python -m analysis.report_entrada188_new
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, geo, rel, test  # noqa: E402

R = ROOT / "results"
EXPECT = int(os.environ.get("C17_EXPECT", "1200"))
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def per_instance(d):
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
    orc = d[d.method.str.startswith("arm")].groupby(["cls", "rep"]).makespan.min().rename("oracle").reset_index()
    return ms.merge(orc, on=["cls", "rep"]) if len(orc) else ms


def main():
    L = ["=" * 110, "THOMPSON, DMAB E CORRIDA -- Ciclo 17, frente B'", "=" * 110]
    K = json.loads((R / "new_cpu_ratio_entrada188.json").read_text())
    ri, ra, rb, kk, desc = [], [], [], [], []
    for sp in ("S5", "S15"):
        d0 = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        d5 = pd.read_csv(R / f"aos_{sp}_entrada185.csv")
        d8 = pd.read_csv(R / f"new_{sp}_entrada188.csv")
        w = per_instance(d0).merge(per_instance(d5), on=["cls", "rep"]).merge(per_instance(d8), on=["cls", "rep"])
        assert len(w) == EXPECT, (sp, len(w))
        ch = json.loads((R / f"new_choice_{sp}_entrada188.json").read_text())
        L.append(f"\n{sp}: hiperparametros escolhidos no ajuste: " + ", ".join(f"{k} {ch[k]}" for k in ("ts", "dmab", "race")))
        for k in ("ts", "dmab", "race"):
            t = test(w[k], w["rand"]); t.update(space=sp, learner=k, id=f"Aiter-{sp}-{k}"); ri.append(t)
            t = test(w[k], w[f"rand_eq_{k}"]); t.update(space=sp, learner=k, id=f"Acpu-{sp}-{k}"); ra.append(t)
            t = test(w["val"], w[k]); t.update(space=sp, learner=k, id=f"Bval-{sp}-{k}"); rb.append(t)
            ev = pd.concat([d0[(d0.method == "rand") & d0.cls.isin(d8.cls.unique())], d8[d8.method == k]]).groupby(["cls", "method"]).evals.mean().unstack()
            ke = ev[k] / ev["rand"]
            kc = pd.Series({c: K[sp][c.replace("rand_", "").replace("j_", "x").replace("m", "")][k] for c in ke.index})
            kk.append(dict(space=sp, learner=k, K_cpu=geo(kc), K_avaliacoes=geo(ke), sobrecusto_por_avaliacao=geo(kc / ke)))
        for k in ("ql", "ucb", "pg", "pm", "ap", "alns", "ts", "dmab", "race", "val"):
            desc.append(dict(space=sp, method=k, d_vs_aleatorio_pct=float(rel(w[k], w["rand"]).mean())))
    Ai, Ac, Bv = family(ri), family(ra), family(rb)
    KK, DD = pd.DataFrame(kk), pd.DataFrame(desc)
    L += ["\n--- Aiter: metodo novo vs aleatorio a iteracoes iguais (Holm sobre 6) ---", Ai[COLS].to_string(index=False), "\n--- Acpu: vs aleatorio a CPU igual (Holm sobre 6) ---", Ac[COLS].to_string(index=False),
          "\n--- Bval: acao validada vs metodo novo (Holm sobre 6; negativo: a validada vence) ---", Bv[COLS].to_string(index=False),
          "\n--- custo: K_CPU, razao de avaliacoes e sobrecusto por avaliacao ---", KK.to_string(index=False),
          "\n--- descritiva: d medio de cada metodo contra o aleatorio a iteracoes iguais (%; os nove aprendizes e a validada) ---", DD.pivot(index="method", columns="space", values="d_vs_aleatorio_pct").to_string()]
    r1 = bool((Ai.confirmada & ~Ai["equivalente_0.05"]).any())
    r2 = bool((Ac.confirmada & ~Ac["equivalente_0.05"]).any())
    r3 = int(Bv.confirmada.sum()) >= 5
    s5 = DD[DD.space == "S5"].set_index("method").d_vs_aleatorio_pct
    others = [k for k in ("ql", "ucb", "pg", "pm", "ap", "ts", "dmab", "race")]
    r4 = bool(all(s5[k] > s5["alns"] - 0.01 for k in others))
    r5 = bool(any(-DD[(DD.space == sp) & (DD.method == "race")].d_vs_aleatorio_pct.iloc[0] >= 0.02 for sp in ("S5", "S15")))
    L += ["\n--- LEITURAS PRE-FIXADAS ---", f"R1 (um metodo novo bate o aleatorio por >= 0,05% a iteracoes iguais): {'SIM' if r1 else 'NAO'}", f"R2 (o mesmo a CPU igual): {'SIM' if r2 else 'NAO'}", f"R3 (a validada vence em >= 5 dos 6 pares; {int(Bv.confirmada.sum())}/6): {'SIM' if r3 else 'NAO'}",
          f"R4 (a roleta tipo ALNS continua o melhor aprendiz online em S5): {'SIM' if r4 else 'NAO'}", f"R5 (a corrida ganha >= 0,02% a iteracoes iguais em S5 ou S15): {'SIM' if r5 else 'NAO'}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada188_new.txt").write_text(text, encoding="utf-8")
    for name, tb in dict(aiter=Ai, acpu=Ac, bval=Bv, k=KK, desc=DD).items():
        tb.to_csv(R / f"entrada188_new_{name}.csv", index=False)


if __name__ == "__main__":
    main()
