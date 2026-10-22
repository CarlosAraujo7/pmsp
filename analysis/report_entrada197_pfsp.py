"""
analysis/report_entrada197_pfsp.py
====================================
Ciclo 18, frente D (e validacao FORA DA AMOSTRA do pre-teste, frente C): o flowshop de permutacao com o IG de Ruiz e Stutzle, compilado (pmsp/pfsp_numba.py; 39 testes), com os mesmos seletores e o mesmo protocolo. Diario: Entrada 197 (pre-registro) e 198 (resultados).
Delineamento: experiment/run_cycle18_pfsp_entrada197.py. Acoes F5 (tamanho de destruicao d em {2, 3, 4, 6, 8}) e F15 (d x fator de temperatura {0,1; 0,4; 0,8}); 9 classes (n em {20, 50, 100} x m em {5, 10, 20}); ajuste: 20 instancias sinteticas por classe (3 sementes do aleatorio e de cada acao, 1 semente de cada
configuracao de hiperparametros); teste: 40 instancias sinteticas por classe (conjunto 'syn') e as 90 de Taillard ta001-ta090 (conjunto 'ta'); 3 sementes; T = 1000; aprendizes ql, ucb, pg, alns, ts; acao validada por classe tambem a CPU igual (K_val calibrado).

PRE-REGISTRO (congelado ANTES de rodar o ajuste, a calibracao e o teste; hashes em results/entrada197_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------
Estatistica (a das Entradas 180-195): unidade = media das sementes por instancia; d = 100 (A - B) / B em makespan (negativo: A melhor); Wilcoxon pareado bilateral (empates descartados), IC90 bootstrap (2000), equivalencia a 0 se o IC90 cabe em +-0,05%; 'confirmada' = p_holm < 0,05 e d < 0.
Familias (Holm sobre 10 = 5 aprendizes x 2 espacos, em cada conjunto): Aiter (aprendiz vs aleatorio a iteracoes iguais), Bit (validada vs aprendiz a iteracoes iguais), Beq (validada a CPU igual do aprendiz, T_v = round(1000 K_aprendiz / K_val), vs aprendiz); Acpu (Holm sobre 6: ql, alns e ts vs aleatorio com T_eq = round(1000 K)).
PRE-TESTE (frente C): VNR de cada celula (classe x espaco) = pmsp.budgetfair.validated_to_noise sobre as sondas do AJUSTE (analysis/pretest_vnr_entrada198.cell_vnr); o limiar tau = 0,877 foi CONGELADO em results/pretest_tau_entrada198.json (calibrado nas 42 celulas sinteticas e de descoberta do P||Cmax) e a regra e'
'o Q-Learning online paga (ganha do aleatorio a iteracoes iguais) na celula se VNR >= tau'. Desfecho da celula: g_Q = -100 media[(ql - rand) / rand] nas instancias de teste do conjunto 'syn' (40 por classe).
Transparencia (medida de folga em instancias de AJUSTE, antes de congelar; nenhum dado de teste visto; T = 1000, F15, 5 a 6 instancias por classe, 2 sementes): a folga do oraculo (melhor acao fixa contra o aleatorio) e' 0,000% nas tres classes com m = 5 (o aleatorio e as acoes alcancam o limite inferior em 33-50% das instancias),
0,03-0,04% com n = 20, e 0,2-0,3% com n >= 50 e m >= 10; portanto as classes com m = 5 sao TETO (VNR indefinido, ruido 0), como as sinteticas do P||Cmax; elas ficam na bateria e sao contadas. A POPULACAO de avaliacao do pre-teste e' a das celulas com VNR definido.
Previsoes (10; registradas antes de rodar; o IG de flowshop tem custo por iteracao dominado pela busca local, igual entre acoes, por isso K ~ 1): (1) K_CPU do Q-Learning em [0,6; 1,1] nas 18 celulas (as medidas de tempo de uma execucao de T = 1000 em instancias de ajuste, vistas antes, davam 0,67-0,95: o Q-Learning escolhe tamanhos de destruicao menores e custa menos); (2) K_val em [0,5; 1,45] nas 18 celulas (a acao de d = 2 custa ~0,6 e a de d = 8 ~1,3 do aleatorio); (3) a validada vence o aleatorio a iteracoes iguais (d medio < 0) em >= 75% das celulas avaliaveis do conjunto 'syn';
(4) Beq (a validada a CPU igual vence o aprendiz, confirmada) em >= 8 dos 10 pares do conjunto 'syn'; (5) algum aprendiz e' confirmado melhor que o aleatorio a iteracoes iguais em pelo menos um espaco do conjunto 'syn' (um segundo problema em que o aprendizado online paga em algum lugar);
PRE-TESTE FORA DA AMOSTRA no conjunto 'syn' (celulas avaliaveis, VNR definido; 18 menos as de teto): (6) acuracia da regra 'VNR >= tau' para 'g_Q > 0' >= 0,70; (7) AUC do VNR para 'g_Q > 0' >= 0,70; (8) Spearman(VNR, g_Q) >= +0,35; (9) nas celulas com VNR >= 2 tau o Q-Learning ganha em >= 75% (se houver ao menos 3 celulas assim; senao, previsao nao avaliavel);
(10) o conjunto 'ta' (Taillard) concorda em sinal de g_Q com o 'syn' em >= 75% das celulas avaliaveis. Leitura L: 'o pre-teste VNR generaliza para um segundo problema' = (6), (7) e (8) cumpridas.
Uso: python -m analysis.report_entrada197_pfsp
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
from analysis.pretest_vnr_entrada198 import cell_vnr, evaluate  # noqa: E402
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
LEARNERS = ["ql", "ucb", "pg", "alns", "ts"]
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
EXPECT = {"syn": int(os.environ.get("C18_EXPECT_SYN", "360")), "ta": int(os.environ.get("C18_EXPECT_TA", "90"))}
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_test(sp):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pfsp_test_{sp}_s*of*_entrada197.csv")))], ignore_index=True)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    orc = d[d.method.str.startswith("arm")].groupby("cls").makespan.min().rename("oracle").reset_index()
    w = ms.merge(orc, on="cls")
    w["set"] = w.cls.str.split("|").str[0]
    w["klass"] = w.cls.str.split("|").str[1]
    return w


def main():
    tau = json.loads((R / "pretest_tau_entrada198.json").read_text())["tau"]
    K = json.loads((R / "pfsp_cpu_ratio_entrada197.json").read_text())
    L = ["=" * 110, "FLOWSHOP DE PERMUTACAO (IG de Ruiz e Stutzle) -- Ciclo 18, frente D", "=" * 110, f"tau congelado = {tau:.4f}"]
    W = {sp: load_test(sp) for sp in ("F5", "F15")}
    fam, cell_rows = {}, []
    for setname in ("syn", "ta"):
        ri, rb, re_, ra = [], [], [], []
        for sp in ("F5", "F15"):
            w = W[sp][W[sp].set == setname]
            assert len(w) == EXPECT[setname], (setname, sp, len(w))
            for k in LEARNERS:
                t = test(w[k], w["rand"]); t.update(space=sp, learner=k, id=f"Aiter-{setname}-{sp}-{k}"); ri.append(t)
                t = test(w["val"], w[k]); t.update(space=sp, learner=k, id=f"Bit-{setname}-{sp}-{k}"); rb.append(t)
                t = test(w[f"val_eq_{k}"], w[k]); t.update(space=sp, learner=k, id=f"Beq-{setname}-{sp}-{k}"); re_.append(t)
                if k in ("ql", "alns", "ts"):
                    t = test(w[k], w[f"rand_eq_{k}"]); t.update(space=sp, learner=k, id=f"Acpu-{setname}-{sp}-{k}"); ra.append(t)
            for klass, x in w.groupby("klass"):
                cell_rows.append(dict(set=setname, space=sp, cell=klass, n=len(x), folga=float(-rel(x.oracle, x.rand).mean()), val=float(-rel(x.val, x.rand).mean()), ql=float(-rel(x.ql, x.rand).mean()), alns=float(-rel(x.alns, x.rand).mean()), ts=float(-rel(x.ts, x.rand).mean())))
        fam[setname] = dict(Aiter=family(ri), Bit=family(rb), Beq=family(re_), Acpu=family(ra))
        for nm, F in fam[setname].items():
            L += [f"\n--- {setname}: {nm} (Holm sobre {len(F)}) ---", F[COLS].to_string(index=False)]
    # ---- celulas: VNR do ajuste e desfecho do teste
    vrow = []
    for sp in ("F5", "F15"):
        t = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pfsp_tune_{sp}_s*of*_entrada197.csv")))], ignore_index=True)
        for klass, x in t.groupby("cls"):
            vrow.append(dict(space=sp, cell=klass, **cell_vnr(x[x.method.isin(["rand"]) | x.method.str.startswith("arm")])))
    V = pd.DataFrame(vrow)
    C = pd.DataFrame(cell_rows).merge(V, on=["space", "cell"])
    C["gq"] = C.ql
    syn, ta = C[C.set == "syn"], C[C.set == "ta"]
    ev = evaluate(syn.vnr, syn.gq, tau)
    L += ["\n--- celulas (classe x espaco): VNR do ajuste (ruido e ganho validado em %), folga do oraculo e ganho sobre o aleatorio no teste (%; positivo: melhor) ---", C.sort_values(["set", "space", "cell"]).to_string(index=False)]
    kq = [K[sp][c]["ql"] for sp in K for c in K[sp]]
    kv = [K[sp][c]["val"] for sp in K for c in K[sp]]
    merged = syn.merge(ta, on=["space", "cell"], suffixes=("_syn", "_ta"))
    merged = merged[merged.vnr_syn.notna()]
    agree = int(((merged.gq_syn > 0) == (merged.gq_ta > 0)).sum())
    n_ev = int(syn.vnr.notna().sum())
    hi = syn[syn.vnr >= 2 * tau]
    yn = lambda b: "SIM" if b else "NAO"
    Bs = fam["syn"]["Beq"]
    nb = int(Bs.confirmada.sum())
    some = bool((fam["syn"]["Aiter"].confirmada).any())
    ov = int((syn[syn.vnr.notna()].val > 0).sum())
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(1) K_CPU do Q-Learning em [0,6; 1,1] nas 18 celulas: {yn(min(kq) >= 0.6 and max(kq) <= 1.1)} (min {min(kq):.2f}, max {max(kq):.2f})", f"(2) K_val em [0,5; 1,45]: {yn(min(kv) >= 0.5 and max(kv) <= 1.45)} (min {min(kv):.2f}, max {max(kv):.2f})",
          f"(3) 'syn': a validada vence o aleatorio a iteracoes iguais em >= 75% das celulas avaliaveis ({ov}/{n_ev}): {yn(ov >= 0.75 * n_ev)}", f"(4) 'syn': a validada a CPU igual vence o aprendiz (confirmada) em >= 8 dos 10 pares ({nb}/10): {yn(nb >= 8)}",
          f"(5) 'syn': algum aprendiz confirmado melhor que o aleatorio a iteracoes iguais em algum espaco: {yn(some)}", f"(6) pre-teste, acuracia 'VNR >= tau' para 'g_Q > 0' >= 0,70: {yn(ev['acc'] >= 0.70)} ({ev['acc']:.3f})", f"(7) AUC >= 0,70: {yn(ev['auc'] >= 0.70)} ({ev['auc']:.3f})",
          f"(8) Spearman(VNR, g_Q) >= +0,35: {yn(ev['spearman'] >= 0.35)} ({ev['spearman']:+.3f})", f"(9) celulas com VNR >= 2 tau: {len(hi)}; o Q-Learning ganha em {int((hi.gq > 0).sum())}: " + (yn((hi.gq > 0).mean() >= 0.75) if len(hi) >= 3 else "nao avaliavel"),
          f"(10) 'ta' concorda em sinal de g_Q com 'syn' em >= 75% das celulas avaliaveis ({agree}/{len(merged)}): {yn(agree >= 0.75 * len(merged))}", f"L (o pre-teste VNR generaliza para o flowshop): {yn(ev['acc'] >= 0.70 and ev['auc'] >= 0.70 and ev['spearman'] >= 0.35)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada197_pfsp.txt").write_text(text, encoding="utf-8")
    C.to_csv(R / "entrada197_pfsp_cells.csv", index=False)
    for sname, d_ in fam.items():
        for nm, F in d_.items():
            F.to_csv(R / f"entrada197_pfsp_{sname}_{nm}.csv", index=False)


if __name__ == "__main__":
    main()
