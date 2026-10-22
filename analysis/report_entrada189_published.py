"""
analysis/report_entrada189_published.py
=========================================
Ciclo 17, frente A': os nove aprendizes compilados e a acao validada nas instancias PUBLICADAS (benchmark de Ostojic et al. 2025, familias B, C e E), S5 e S15, T = 1000. Diario: Entrada 189 (resultados do estagio 1 sintetico e pre-registro deste estagio) e 190 (resultados).
Delineamento: experiment/run_cycle17_published_entrada189.py (instancias de teste = manifest_confirm2_entrada115.csv, ja' usadas na Entrada 115 para outra pergunta; ajuste = manifest_calib_entrada115.csv, 553 instancias, disjuntas).
Correcao de VIABILIDADE feita antes de qualquer execucao (a primeira tentativa do pipeline parou no ajuste com OverflowError, sem gravar nenhum resultado): instancias da familia B com tempos de ate ~10^18 (soma ate ~10^20) nao cabem no int64 do nucleo compilado; as de soma >= 2^62 sao excluidas do ajuste e do teste e contadas em
results/pub_groups_entrada189.json (chave 'excluded'); as previsoes abaixo nao mudam.

PRE-REGISTRO (congelado ANTES de rodar o ajuste, a calibracao e o teste; hashes em results/entrada189_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------
Estatistica (a das Entradas 180, 185 e 188): unidade = media das 5 sementes por instancia (um par por instancia de teste); d = 100 (A - B) / B em makespan (negativo: A melhor); Wilcoxon pareado bilateral (empates descartados), IC90 bootstrap (2000), Holm, equivalencia a 0 se o IC90 cabe em +-0,05%;
'confirmada' = p_holm < 0,05 e d < 0, lida como 'paridade' se tambem equivalente. Familias com Holm sobre 18 (9 aprendizes x 2 espacos): Aiter (vs aleatorio a iteracoes iguais), Acpu (vs aleatorio com T_eq = round(1000 K), K por grupo (familia, tercil de n) calibrado em sequencia), Bval (acao validada por grupo,
escolhida nas instancias de ajuste, vs aprendiz). Descritivas: folga do oraculo (todas as acoes fixas, 1 semente) e fracao de instancias em que a validada atinge o limite inferior classico; d medio por familia (B, C, E) de ql, alns, ts, dmab, race e da validada contra o aleatorio.
Leituras: R1' 'um aprendiz bate o aleatorio por >= 0,05% a iteracoes iguais' = alguma Aiter confirmada e nao paridade; R2' o mesmo a CPU igual; R3' 'a acao validada vence em >= 12 dos 18 pares' (confirmada); R4' 'a roleta tipo ALNS ganha em S5 menos que na Entrada 185' = d de Aiter-S5-alns >= -0,039; R5' 'a corrida nao bate o aleatorio' = d >= 0 em S5 e em S15.
Previsoes (9; registradas depois dos resultados do estagio 1 sintetico (Thompson, DMAB e corrida; results/entrada188_new.txt) e antes de rodar o estagio 2; faixas largas porque as instancias publicadas sao heterogeneas (n de 6 a 261; muitas triviais, com empates)): (1) K_CPU do Q-Learning em [0,3; 1,15] nos 18 grupos (familia, tercil, espaco);
(2) Aiter dos oito aprendizes que nao sao a corrida em [-0,06; +0,06] nos dois espacos; (3) Aiter da corrida >= -0,01 nos dois espacos; (4) R1' NAO; (5) R2' NAO; (6) Acpu de todos em [-0,12; +0,12]; (7) R3' SIM; (8) folga do oraculo em S5 em [-0,30; -0,02]%; (9) R4' SIM.
Uso: python -m analysis.report_entrada189_published
"""
from __future__ import annotations

import glob
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
EXPECT = int(os.environ.get("C17_EXPECT", "0"))   # 0: o numero de instancias de teste vem de pub_groups_entrada189.json (so muda em testes de fumaca)
LEARNERS = ["ql", "ucb", "pg", "pm", "ap", "alns", "ts", "dmab", "race"]
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def per_instance(d):
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    orc = d[d.method.str.startswith("arm")].groupby("cls").makespan.min().rename("oracle").reset_index()
    lb = d.drop_duplicates("cls")[["cls", "lb"]]
    return ms.merge(orc, on="cls").merge(lb, on="cls")


def main():
    L = ["=" * 110, "INSTANCIAS PUBLICADAS (B, C, E): nove aprendizes e a acao validada -- Ciclo 17, frente A'", "=" * 110]
    K = json.loads((R / "pub_cpu_ratio_entrada189.json").read_text())
    ri, ra, rb, kk, hd, fam_rows = [], [], [], [], [], []
    W = {}
    for sp in ("S5", "S15"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pub_{sp}_s*of*_entrada189.csv")))], ignore_index=True)
        w = per_instance(d)
        assert len(w) == (EXPECT or len(json.loads((R / 'pub_groups_entrada189.json').read_text())['test'])), (sp, len(w))
        w["fam"] = w.cls.str.split("|").str[0]
        W[sp] = w
        for k in LEARNERS:
            t = test(w[k], w["rand"]); t.update(space=sp, learner=k, id=f"Aiter-{sp}-{k}"); ri.append(t)
            t = test(w[k], w[f"rand_eq_{k}"]); t.update(space=sp, learner=k, id=f"Acpu-{sp}-{k}"); ra.append(t)
            t = test(w["val"], w[k]); t.update(space=sp, learner=k, id=f"Bval-{sp}-{k}"); rb.append(t)
            ks = [v[k] for v in K[sp].values()]
            kk.append(dict(space=sp, learner=k, K_min=min(ks), K_geomean=geo(ks), K_max=max(ks)))
        hd.append(dict(space=sp, folga_oraculo_pct=float(rel(w.oracle, w.rand).mean()), folga_validada_pct=float(rel(w.val, w.rand).mean()), frac_validada_no_lb=float((w.val == w.lb).mean()), frac_aleatorio_no_lb=float((w.rand == w.lb).mean())))
        for f in ("B", "C", "E"):
            x = w[w.fam == f]
            fam_rows.append(dict(space=sp, familia=f, n=len(x), **{k: float(rel(x[k], x["rand"]).mean()) for k in ("ql", "alns", "ts", "dmab", "race", "val")}))
    Ai, Ac, Bv = family(ri), family(ra), family(rb)
    KK, HD, FR = pd.DataFrame(kk), pd.DataFrame(hd), pd.DataFrame(fam_rows)
    L += ["\n--- Aiter: aprendiz vs aleatorio a iteracoes iguais (Holm sobre 18) ---", Ai[COLS].to_string(index=False), "\n--- Acpu: vs aleatorio a CPU igual (Holm sobre 18) ---", Ac[COLS].to_string(index=False),
          "\n--- Bval: acao validada vs aprendiz (Holm sobre 18; negativo: a validada vence) ---", Bv[COLS].to_string(index=False), "\n--- K_CPU por grupo (familia, tercil de n): minimo, media geometrica e maximo ---", KK.to_string(index=False),
          "\n--- folga do oraculo, ganho da validada e fracao no limite inferior ---", HD.to_string(index=False), "\n--- d medio contra o aleatorio a iteracoes iguais, por familia (descritivo) ---", FR.to_string(index=False)]
    qk = KK[KK.learner == "ql"]
    p1 = bool((qk.K_min >= 0.3).all() and (qk.K_max <= 1.15).all())
    nr = Ai[~Ai.learner.eq("race")]
    p2 = bool(((nr.mean_d_pct >= -0.06) & (nr.mean_d_pct <= 0.06)).all())
    rc = Ai[Ai.learner.eq("race")]
    p3 = bool((rc.mean_d_pct >= -0.01).all())
    r1 = bool((Ai.confirmada & ~Ai["equivalente_0.05"]).any())
    r2 = bool((Ac.confirmada & ~Ac["equivalente_0.05"]).any())
    p6 = bool(((Ac.mean_d_pct >= -0.12) & (Ac.mean_d_pct <= 0.12)).all())
    r3 = int(Bv.confirmada.sum()) >= 12
    p8 = bool(-0.30 <= float(HD.loc[HD.space == "S5", "folga_oraculo_pct"].iloc[0]) <= -0.02)
    r4 = bool(Ai.loc[Ai.id == "Aiter-S5-alns", "mean_d_pct"].iloc[0] >= -0.039)
    r5 = bool((rc.mean_d_pct >= 0).all())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(1) K_CPU do Q-Learning em [0,3; 1,15] nos grupos: {yn(p1)} (min {qk.K_min.min():.2f}, max {qk.K_max.max():.2f})", f"(2) Aiter dos oito aprendizes (sem a corrida) em [-0,06; +0,06]: {yn(p2)} (min {nr.mean_d_pct.min():.4f}, max {nr.mean_d_pct.max():.4f})",
          f"(3) Aiter da corrida >= -0,01: {yn(p3)} ({[round(v, 4) for v in rc.mean_d_pct]})", f"(4) R1' (um aprendiz bate o aleatorio por >= 0,05% a iteracoes iguais; previsto NAO): {yn(r1)}", f"(5) R2' (o mesmo a CPU igual; previsto NAO): {yn(r2)}", f"(6) Acpu de todos em [-0,12; +0,12]: {yn(p6)} (min {Ac.mean_d_pct.min():.4f}, max {Ac.mean_d_pct.max():.4f})",
          f"(7) R3' (a validada vence em >= 12 dos 18 pares; {int(Bv.confirmada.sum())}/18; previsto SIM): {yn(r3)}", f"(8) folga do oraculo em S5 em [-0,30; -0,02]: {yn(p8)} ({float(HD.loc[HD.space == 'S5', 'folga_oraculo_pct'].iloc[0]):.4f})",
          f"(9) R4' (a roleta ganha em S5 menos que na Entrada 185: d >= -0,039; previsto SIM): {yn(r4)} ({float(Ai.loc[Ai.id == 'Aiter-S5-alns', 'mean_d_pct'].iloc[0]):.4f})", f"R5' (a corrida nao bate o aleatorio, d >= 0 nos dois espacos): {yn(r5)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada189_published.txt").write_text(text, encoding="utf-8")
    for name, tb in dict(aiter=Ai, acpu=Ac, bval=Bv, k=KK, headroom=HD, family=FR).items():
        tb.to_csv(R / f"entrada189_published_{name}.csv", index=False)


if __name__ == "__main__":
    main()
