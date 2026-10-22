"""
analysis/report_entrada185_cycle16.py
=======================================
Ciclo 16 (frentes A2, B, C e D do plano da Entrada 183). Diario: Entrada 184 (pre-registro e poder, A1) e 185 (pre-registro das frentes B, A2, C, D) e 186 (resultados). Delineamento: experiment/run_cycle16_entrada185.py, pmsp/ig_numba_agents2.py (54 testes).

PRE-REGISTRO (congelado ANTES de rodar o ajuste, a calibracao e o teste; hashes em results/entrada185_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------
Estatistica (a mesma da Entrada 180): unidade = media das sementes por instancia; d = 100 (A - B) / B em makespan (negativo: A melhor); Wilcoxon pareado bilateral (empates descartados), IC90 bootstrap (2000), Holm sobre a familia, equivalencia a 0 se o IC90 cabe em +-0,05%;
'confirmada' = p_holm < 0,05 e d < 0, lida como 'paridade' se tambem equivalente (licao II.24).

B -- AOS classicos (PM, AP, roleta tipo ALNS) em S5 e S15, T = 1000. Familias com Holm sobre 6 (3 aprendizes x 2 espacos): Aiter (aprendiz vs aleatorio a iteracoes iguais; o aleatorio e' o de results/agents_S5|S15_entrada180.csv), Acpu (vs aleatorio com T_eq = round(1000 K)),
  Bval (acao validada por classe, a de results/agents_choice_*_entrada180.json, vs aprendiz). Leituras: N1 'um AOS classico bate o aleatorio por >= 0,05% a iteracoes iguais' = alguma Aiter confirmada e nao paridade; N2 o mesmo a CPU igual; N3 'a acao validada vence em >= 5 dos 6 pares'.
  Previsoes: K_CPU (media geometrica por classe) de PM, AP e ALNS em [0,8; 1,3] em S5 e S15; Aiter: todo d em [-0,03; +0,10]; Acpu: todo d em [-0,05; +0,10]; Bval: todo d em [-0,12; -0,04]; N1 NAO; N2 NAO; N3 SIM; a media dos seis d de Aiter > -0,01.
A2 -- CONTROLE POSITIVO (bonus plantado; fracao de escolhas da acao plantada na segunda metade de T = 1000, em multiplos do uniforme 1/A). Previsoes: modo 1 (acao plantada igual em todo estado): bonus 5: todo aprendiz >= 2,0 vezes o uniforme em S5 e em S15; bonus 2: >= 1,5 vezes;
  bonus 0,5: >= 1,0 vez em pelo menos 4 dos 6 aprendizes (dose-resposta); modo 2 (depende do estado), bonus 5: Q-Learning e gradiente de politica >= 1,2 vez o uniforme em S5; para os aprendizes sem estado (UCB1, PM, AP, ALNS) NAO ha' previsao (descritivo): um teste de fumaca em pasta temporaria (2 classes de n <= 30, fora dos dados) mostrou que eles tambem capturam o modo 2, porque poucos estados concentram as visitas e a acao plantada do estado dominante
  e' encontrada; a previsao 'entre 0,5 e 2,0 vezes' que seria natural foi retirada por causa dessa observacao. Leitura P1: 'os seis aprendizes
  capturam a estrutura plantada (modo 1, bonus 5, ambos os espacos)' = SIM; P2: 'os aprendizes com estado capturam a estrutura que depende do estado' = Q e PG >= 1,2 vez em S5.
C -- HORIZONTE LONGO (S5; T em {5000, 20000, 100000}; 12 x 100 instancias; 5 sementes, 3 em T = 100000; aleatorio, ql, ucb, pg com lr ajustado em T, validada ajustada em T, oraculo de todas as acoes). Familias com Holm sobre 9 (3 aprendizes x 3 T): Aiter e Bval.
  Previsoes: Aiter: todo |d| <= 0,05 e nenhum confirmado e nao paridade (L1 NAO: 'a politica ganha >= 0,05% em algum T'); folga do oraculo em valor absoluto em T = 20000 menor que 0,6 vez a de T = 1000 (-0,125%) e em T = 100000 menor que 0,4 vez;
  fracao de instancias em que a validada atinge o limite inferior classico (otimalidade provada) cresce com T: f(1000) < f(5000) < f(20000) < f(100000) (L2 SIM); Bval em T = 5000 em [-0,10; 0,0]; em T = 100000 |d| < 0,03 para os tres aprendizes (L3 'a vantagem da validada some no horizonte longo' = SIM).
D -- TEMPOS FINOS U[1, 10^4] (S5 e S15, T = 1000, 12 x 100 instancias, 5 sementes; ql, ucb, pg; validada; oraculo; K_CPU calibrado nos tempos finos). Familias com Holm sobre 6: Aiter, Acpu, Bval. Previsoes: a fracao de instancias em que a validada atinge o limite inferior
  e' menor que nos tempos grossos (S5: E180) por >= 20 pontos; folga do oraculo (S5) em [-0,60; -0,05]%; Aiter: todo d em [-0,05; +0,10] e nenhum confirmado e nao paridade (D1 NAO: 'o regime com mais folga resgata o aprendiz a iteracoes iguais'); K_CPU do Q-Learning em [0,3; 0,95]
  em S5 e S15 e Acpu do Q-Learning confirmado em S5 e S15 com d em [-0,30; -0,02]; Bval confirmada em >= 5 dos 6 pares, d em [-0,40; -0,04] (D2 SIM).
Uso: python -m analysis.report_entrada185_cycle16 {aos|plant|long|fine}
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
EXPECT = int(os.environ.get("C16_EXPECT", "1200"))   # pares por comparacao (so muda em testes de fumaca)
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def per_instance(d, drop_arms=True):
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
    orc = d[d.method.str.startswith("arm")].groupby(["cls", "rep"]).makespan.min().rename("oracle").reset_index()
    return ms.merge(orc, on=["cls", "rep"]) if len(orc) else ms


def lb_of(d):
    return d.drop_duplicates(["cls", "rep"]).set_index(["cls", "rep"]).lb


def frac_at_lb(d, method):
    m = d[d.method == method].groupby(["cls", "rep"]).makespan.max()
    return float((m == lb_of(d).reindex(m.index)).mean())


def finish(name, L, tables):
    text = "\n".join(L)
    print(text)
    (R / f"entrada185_{name}.txt").write_text(text, encoding="utf-8")
    for k, v in tables.items():
        v.to_csv(R / f"entrada185_{name}_{k}.csv", index=False)


# ------------------------------------------------------------------------------------------------- B
def report_aos():
    L = ["=" * 110, "AOS CLASSICOS (PM, AP, ROLETA TIPO ALNS) -- Ciclo 16, frente B", "=" * 110]
    K = json.loads((R / "aos_cpu_ratio_entrada185.json").read_text())
    ra, ri, rb, kk, desc = [], [], [], [], []
    for sp in ("S5", "S15"):
        d0 = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        d1 = pd.read_csv(R / f"aos_{sp}_entrada185.csv")
        w = per_instance(d0).merge(per_instance(d1, False), on=["cls", "rep"])
        assert len(w) == EXPECT, (sp, len(w))
        ch = json.loads((R / f"aos_choice_{sp}_entrada185.json").read_text())
        L.append(f"\n{sp}: hiperparametros escolhidos no ajuste: " + ", ".join(f"{k} {ch[k]}" for k in ("pm", "ap", "alns")))
        for k in ("pm", "ap", "alns"):
            t = test(w[k], w["rand"]); t.update(space=sp, learner=k, id=f"Aiter-{sp}-{k}"); ri.append(t)
            t = test(w[k], w[f"rand_eq_{k}"]); t.update(space=sp, learner=k, id=f"Acpu-{sp}-{k}"); ra.append(t)
            t = test(w["val"], w[k]); t.update(space=sp, learner=k, id=f"Bval-{sp}-{k}"); rb.append(t)
            ev = pd.concat([d0[(d0.method == "rand") & d0.cls.isin(d1.cls.unique())], d1[d1.method == k]]).groupby(["cls", "method"]).evals.mean().unstack()
            ke = ev[k] / ev["rand"]
            kc = pd.Series({c: K[sp][c.replace("rand_", "").replace("j_", "x").replace("m", "")][k] for c in ke.index})
            kk.append(dict(space=sp, learner=k, K_cpu=geo(kc), K_avaliacoes=geo(ke), sobrecusto_por_avaliacao=geo(kc / ke)))
        for k in ("ql", "ucb", "pg", "pm", "ap", "alns", "val"):
            desc.append(dict(space=sp, method=k, d_vs_aleatorio_pct=float(rel(w[k], w["rand"]).mean())))
    Ai, Ac, Bv = family(ri), family(ra), family(rb)
    KK = pd.DataFrame(kk)
    DD = pd.DataFrame(desc)
    L += ["\n--- Aiter: AOS classico vs aleatorio a iteracoes iguais (Holm sobre 6) ---", Ai[COLS].to_string(index=False), "\n--- Acpu: vs aleatorio a CPU igual (Holm sobre 6) ---", Ac[COLS].to_string(index=False),
          "\n--- Bval: acao validada vs AOS classico (Holm sobre 6; negativo: a validada vence) ---", Bv[COLS].to_string(index=False),
          "\n--- custo: K_CPU, razao de avaliacoes e sobrecusto por avaliacao (media geometrica por classe) ---", KK.to_string(index=False),
          "\n--- descritiva: d medio de cada metodo contra o aleatorio a iteracoes iguais (%; os seis aprendizes e a validada) ---", DD.pivot(index="method", columns="space", values="d_vs_aleatorio_pct").to_string()]
    n1 = bool((Ai.confirmada & ~Ai["equivalente_0.05"]).any())
    n2 = bool((Ac.confirmada & ~Ac["equivalente_0.05"]).any())
    n3 = int(Bv.confirmada.sum()) >= 5
    L += ["\n--- LEITURAS PRE-FIXADAS ---", f"N1 (um AOS classico bate o aleatorio por >= 0,05% a iteracoes iguais): {'SIM' if n1 else 'NAO'}", f"N2 (o mesmo a CPU igual): {'SIM' if n2 else 'NAO'}",
          f"N3 (a validada vence em >= 5 dos 6 pares; {int(Bv.confirmada.sum())}/6): {'SIM' if n3 else 'NAO'}", f"media dos seis d de Aiter: {Ai.mean_d_pct.mean():.4f}"]
    finish("aos", L, dict(aiter=Ai, acpu=Ac, bval=Bv, k=KK))


# ------------------------------------------------------------------------------------------------- A2
def report_plant():
    d = pd.read_csv(R / "plant_entrada185.csv")
    g = d.groupby(["space", "mode", "learner", "bonus"]).apply(lambda x: pd.Series(dict(share=x.hit2.sum() / (len(x) * (x["T"].iloc[0] - x["T"].iloc[0] // 2)), A=x.A.iloc[0], n=len(x))), include_groups=False).reset_index()
    g["x_uniforme"] = g.share * g.A
    L = ["=" * 110, "CONTROLE POSITIVO: recompensa plantada -- Ciclo 16, frente A2", "=" * 110,
         "fracao de escolhas da acao plantada na segunda metade de T = 1000, em multiplos do uniforme (1/A); 12 classes x 10 instancias x 3 sementes por celula", ""]
    for (sp, mode), x in g.groupby(["space", "mode"]):
        L += [f"--- {sp}, modo {mode} ({'mesma acao em todo estado' if mode == 1 else 'acao plantada = estado mod A'}) ---", x.pivot(index="learner", columns="bonus", values="x_uniforme").to_string(), ""]
    m1 = g[(g["mode"] == 1)]
    p1_5 = bool((m1[m1.bonus == 5.0].x_uniforme >= 2.0).all())
    p1_2 = bool((m1[m1.bonus == 2.0].x_uniforme >= 1.5).all())
    p1_h = {sp: int((m1[(m1.bonus == 0.5) & (m1.space == sp)].x_uniforme >= 1.0).sum()) for sp in ("S5", "S15")}
    m2 = g[(g["mode"] == 2) & (g.bonus == 5.0) & (g.space == "S5")].set_index("learner").x_uniforme
    p2 = bool(m2["ql"] >= 1.2 and m2["pg"] >= 1.2)
    blind = {k: float(m2[k]) for k in ("ucb", "pm", "ap", "alns")}
    L += ["--- LEITURAS PRE-FIXADAS ---", f"modo 1, bonus 5: todo aprendiz >= 2,0 vezes o uniforme nos dois espacos: {'SIM' if p1_5 else 'NAO'} (minimo {m1[m1.bonus == 5.0].x_uniforme.min():.2f})",
          f"modo 1, bonus 2: todo aprendiz >= 1,5 vez: {'SIM' if p1_2 else 'NAO'} (minimo {m1[m1.bonus == 2.0].x_uniforme.min():.2f})", f"modo 1, bonus 0,5: aprendizes com >= 1,0 vez: S5 {p1_h['S5']}/6, S15 {p1_h['S15']}/6 (previsto >= 4/6)",
          f"P1 (os seis capturam a estrutura plantada, modo 1, bonus 5): {'SIM' if p1_5 else 'NAO'}", f"P2 (Q-Learning e PG capturam a estrutura que depende do estado, S5, bonus 5; ql {m2['ql']:.2f}, pg {m2['pg']:.2f}): {'SIM' if p2 else 'NAO'}",
          "aprendizes sem estado no modo 2 (descritivo, sem previsao): " + ", ".join(f"{k} {v:.2f}" for k, v in blind.items())]
    finish("plant", L, dict(shares=g))


# ------------------------------------------------------------------------------------------------- C
def report_long():
    L = ["=" * 110, "HORIZONTE LONGO (S5) -- Ciclo 16, frente C", "=" * 110]
    ri, rb, hd, fr = [], [], [], []
    d0 = pd.read_csv(R / "agents_S5_entrada180.csv")
    w0 = per_instance(d0)
    hd.append(dict(T=1000, folga_oraculo_pct=float(rel(w0.oracle, w0.rand).mean()), frac_validada_no_lb=frac_at_lb(d0, "val"), frac_aleatorio_no_lb=frac_at_lb(d0, "rand")))
    for T in (5000, 20000, 100000):
        d = pd.read_csv(R / f"long_S5_T{T}_entrada185.csv")
        w = per_instance(d)
        assert len(w) == EXPECT, (T, len(w))
        for k in ("ql", "ucb", "pg"):
            t = test(w[k], w["rand"]); t.update(T=T, learner=k, id=f"Aiter-T{T}-{k}"); ri.append(t)
            t = test(w["val"], w[k]); t.update(T=T, learner=k, id=f"Bval-T{T}-{k}"); rb.append(t)
        hd.append(dict(T=T, folga_oraculo_pct=float(rel(w.oracle, w.rand).mean()), frac_validada_no_lb=frac_at_lb(d, "val"), frac_aleatorio_no_lb=frac_at_lb(d, "rand")))
        ch = json.loads((R / f"long_S5_T{T}_choice_entrada185.json").read_text())
        L.append(f"T = {T}: lr do PG ajustado {ch['lr']}")
    Ai, Bv = family(ri), family(rb)
    H = pd.DataFrame(hd)
    L += ["\n--- Aiter: aprendiz vs aleatorio a iteracoes iguais, por T (Holm sobre 9) ---", Ai[COLS].to_string(index=False), "\n--- Bval: validada (ajustada em T) vs aprendiz (Holm sobre 9; negativo: a validada vence) ---", Bv[COLS].to_string(index=False),
          "\n--- folga do oraculo (todas as acoes fixas, 1 semente) contra o aleatorio e fracao de instancias que atingem o limite inferior classico ---", H.to_string(index=False)]
    f1 = float(H.loc[H["T"] == 1000, "folga_oraculo_pct"].iloc[0])
    r20 = float(H.loc[H["T"] == 20000, "folga_oraculo_pct"].iloc[0]) / f1
    r100 = float(H.loc[H["T"] == 100000, "folga_oraculo_pct"].iloc[0]) / f1
    fl = H.frac_validada_no_lb.tolist()
    l1 = bool((Ai.confirmada & ~Ai["equivalente_0.05"]).any())
    l2 = all(fl[i] < fl[i + 1] for i in range(3))
    b100 = Bv[Bv.id.str.contains("T100000")].mean_d_pct.abs()
    l3 = bool((b100 < 0.03).all())
    L += ["\n--- LEITURAS PRE-FIXADAS ---", f"L1 (a politica ganha >= 0,05% em algum T): {'SIM' if l1 else 'NAO'}; max |d| de Aiter = {Ai.mean_d_pct.abs().max():.4f}",
          f"razao da folga T = 20000 / T = 1000: {r20:.2f} (previsto < 0,6); T = 100000 / T = 1000: {r100:.2f} (previsto < 0,4)", f"L2 (fracao da validada no limite inferior cresce com T; {[round(x, 3) for x in fl]}): {'SIM' if l2 else 'NAO'}",
          f"L3 (a vantagem da validada some em T = 100000, |d| < 0,03 nos tres; {[round(x, 4) for x in b100]}): {'SIM' if l3 else 'NAO'}"]
    finish("long", L, dict(aiter=Ai, bval=Bv, headroom=H))


# ------------------------------------------------------------------------------------------------- D
def report_fine():
    L = ["=" * 110, "TEMPOS FINOS U[1, 10^4] (S5 e S15) -- Ciclo 16, frente D", "=" * 110]
    K = json.loads((R / "fine_cpu_ratio_entrada185.json").read_text())
    ra, ri, rb, kk, hd = [], [], [], [], []
    for sp in ("S5", "S15"):
        d = pd.read_csv(R / f"fine_{sp}_entrada185.csv")
        w = per_instance(d)
        assert len(w) == EXPECT, (sp, len(w))
        c0 = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        w0 = per_instance(c0)
        hd.append(dict(space=sp, folga_fina_pct=float(rel(w.oracle, w.rand).mean()), folga_grossa_pct=float(rel(w0.oracle, w0.rand).mean()), frac_validada_no_lb_fina=frac_at_lb(d, "val"), frac_validada_no_lb_grossa=frac_at_lb(c0, "val")))
        for k in ("ql", "ucb", "pg"):
            t = test(w[k], w["rand"]); t.update(space=sp, learner=k, id=f"Aiter-{sp}-{k}"); ri.append(t)
            t = test(w[k], w[f"rand_eq_{k}"]); t.update(space=sp, learner=k, id=f"Acpu-{sp}-{k}"); ra.append(t)
            t = test(w["val"], w[k]); t.update(space=sp, learner=k, id=f"Bval-{sp}-{k}"); rb.append(t)
            ev = d[d.method.isin([k, "rand"])].groupby(["cls", "method"]).evals.mean().unstack()
            ke = ev[k] / ev["rand"]
            kc = pd.Series({c: K[sp][c.replace("rand_", "").replace("j_", "x").replace("m", "")][k] for c in ke.index})
            kk.append(dict(space=sp, learner=k, K_cpu=geo(kc), K_avaliacoes=geo(ke), sobrecusto_por_avaliacao=geo(kc / ke)))
    Ai, Ac, Bv = family(ri), family(ra), family(rb)
    KK, HD = pd.DataFrame(kk), pd.DataFrame(hd)
    L += ["\n--- Aiter (Holm sobre 6) ---", Ai[COLS].to_string(index=False), "\n--- Acpu (Holm sobre 6) ---", Ac[COLS].to_string(index=False), "\n--- Bval (validada vs aprendiz; Holm sobre 6) ---", Bv[COLS].to_string(index=False),
          "\n--- custo (K_CPU nos tempos finos) ---", KK.to_string(index=False), "\n--- folga e otimalidade provada: tempos finos vs grossos ---", HD.to_string(index=False)]
    d1 = bool((Ai.confirmada & ~Ai["equivalente_0.05"]).any())
    d2 = int(Bv.confirmada.sum()) >= 5
    s5 = HD[HD.space == "S5"].iloc[0]
    L += ["\n--- LEITURAS PRE-FIXADAS ---", f"D1 (o regime com mais folga resgata o aprendiz a iteracoes iguais): {'SIM' if d1 else 'NAO'}", f"D2 (a validada vence em >= 5 dos 6 pares; {int(Bv.confirmada.sum())}/6): {'SIM' if d2 else 'NAO'}",
          f"queda da fracao da validada no limite inferior em S5: {100 * (s5.frac_validada_no_lb_grossa - s5.frac_validada_no_lb_fina):.1f} pontos (previsto >= 20)", f"folga do oraculo em S5: {s5.folga_fina_pct:.3f}% (previsto em [-0,60; -0,05])"]
    finish("fine", L, dict(aiter=Ai, acpu=Ac, bval=Bv, k=KK, headroom=HD))


if __name__ == "__main__":
    {"aos": report_aos, "plant": report_plant, "long": report_long, "fine": report_fine}[sys.argv[1]]()
