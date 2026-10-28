"""
analysis/report_entrada234_ur.py
==================================
Ciclo 22, frente A: o HOSPEDEIRO NOVO, o IG de maquinas nao relacionadas (R||Cmax), com a regra da folga CONGELADA (h* das Entradas 209/210) e o resto das conclusoes testadas fora da amostra. Desenho e dados: experiment/run_cycle22_ur_entrada234.py (12 classes x 100 instancias de teste, S5 e S15, T = 1000,
5 sementes por metodo e 4 por acao fixa). Diario: Entrada 234 (pre-registro) e 235 (resultados). Saidas: results/entrada234_ur.txt, entrada234_ur_cells.csv, entrada234_ur_pooled.csv. Com --pilot le' results/ur_pilot_* (instancias de AJUSTE; so' para depurar este relatorio, NAO e' resultado).

PRE-REGISTRO (congelado ANTES de qualquer saida do teste; hashes em results/entrada234_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Transparencia: antes deste registro rodei um PILOTO para depurar o codigo (36 instancias de AJUSTE por espaco, 3 por classe; NAO e' resultado nem toca nas instancias de teste): a folga de uma semente do oraculo e' 1,9 (S5) e 3,1 (S15) pontos e a acao validada ganha 1,0 e 2,5 pontos sobre o aleatorio, isto e', o hospedeiro tem uma folga
de 20 a 50 vezes a do P||Cmax sintetico; o Q-Learning padrao PERDE (-1,4 e -0,3), UCB1, o gradiente de politica e a roleta ALNS ficam perto do aleatorio e o Q otimista ganha (+0,5 e +0,3); a folga cf agregada em S5 ficou perto de zero com 36 instancias (ruido). As previsoes abaixo foram fixadas DEPOIS do piloto e o levam em conta.
Celula = classe (n, m) x espaco (12 por espaco, 24 no total). Folga cf = ganho sobre o aleatorio (media das 5 sementes por instancia) da acao fixa escolhida por instancia nas sementes {0,1} e avaliada nas {2,3}; ganho do Q = ganho do Q-Learning padrao sobre o aleatorio a iteracoes iguais. Regra: 'o Q ganha' (ganho > 0) se folga cf >= h* = 0,0475 (congelado; NAO reajustado aqui).
Leitura unilateral: abaixo de h*, o ganho do Q-Learning passa de 0,05 (a margem) em alguma celula? Testes: aprendiz contra o aleatorio (Wilcoxon pareado, ties fora, Holm sobre 10 = 5 aprendizes x 2 espacos), validada contra aprendiz (Holm sobre 10); 'confirmada' = p_holm < 0,05 e o aprendiz melhor; 'paridade' = IC90 dentro de +-0,05.
(U1) a folga cf agregada fica >= h* nos dois espacos (65%).  (U2) a regra congelada classifica corretamente >= 75% das 24 celulas (20%).  (U3) a acuracia balanceada da regra e' >= 0,65 (30%).  (U4) leitura unilateral: nenhuma celula com folga cf < h* tem ganho do Q > 0,05 (75%).
(U5) o Q-Learning padrao bate o aleatorio por mais que a margem (confirmado e nao paridade) em pelo menos um espaco, agregado (10%).  (U6) a acao validada vence o Q-Learning padrao, confirmada, nos dois espacos (95%).  (U7) as execucoes do aleatorio chegam ao limite inferior em menos de 40% dos casos nos dois espacos (97%).
(U8) a folga cf agregada supera a do P||Cmax sintetico (0,063 em S5; 0,043 em S15) nos dois espacos (70%).  (U9) o Spearman entre a folga cf e o ganho do Q nas 12 classes e' >= 0,5 nos dois espacos (35%).
(U10) o Q otimista (qo) bate o aleatorio por mais que a margem (confirmado e nao paridade) em pelo menos um espaco, agregado (40%).  (U11) o ganho agregado de qo supera o do Q padrao nos dois espacos (85%).  (U12) a acao validada vence qo, confirmada, nos dois espacos (90%).
(U13) o ganho agregado da acao validada sobre o aleatorio passa de 0,5 ponto nos dois espacos (85%).  (U14) pelo menos um dos cinco aprendizes (ql, ucb, pg, alns, qo) bate o aleatorio por mais que a margem (confirmado e nao paridade) em pelo menos um espaco, agregado (50%).
Leituras: R1 ('a regra da folga, congelada, se transfere a um problema novo'): U2 e U4 verdadeiras (previsao NAO).  R2 ('o hospedeiro tem folga real'): U1 e U8 e U13 verdadeiras (previsao SIM).  R3 ('aqui o aprendizado online paga'): U14 verdadeira.
Uso: python -m analysis.report_entrada234_ur [--pilot]
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
PILOT = "--pilot" in sys.argv
LEARN = ["ql", "ucb", "pg", "alns", "qo"]
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load(sp):
    pat = f"ur_pilot_{sp}_entrada234.csv" if PILOT else f"ur_test_{sp}_s*of*_entrada234.csv"
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat)))], ignore_index=True)


def per_instance(d):
    d = d.copy()
    d["inst"] = d.cls + "#" + d.rep.astype(str)
    ms = d[~d.method.str.startswith("arm")].groupby(["inst", "method"]).makespan.mean().unstack("method")
    a = d[d.method.str.startswith("arm")].copy()
    a["arm"] = a.method.str[3:].astype(int)
    X = a.pivot_table(index=["inst", "arm"], columns="seed_i", values="makespan")
    insts = sorted(ms.index)
    A = int(a.arm.max()) + 1
    M = np.stack([X.loc[(i, slice(None))].sort_index().to_numpy() for i in insts])  # (instancia, acao, semente)
    ix = np.arange(len(insts))
    sel = M[:, :, [0, 1]].mean(axis=2)
    ev = M[:, :, [2, 3]].mean(axis=2)
    ms = ms.loc[insts].copy()
    ms["cf"] = ev[ix, sel.argmin(axis=1)]
    ms["biased"] = M[:, :, 0].min(axis=1)
    ms["cell"] = [i.split("#")[0] for i in insts]
    ms["lb"] = d.groupby("inst").lb.first().loc[insts].to_numpy()
    return ms.reset_index()


def main():
    hstar = float(re.search(r"limiar h\* escolhido na descoberta \(acuracia balanceada [\d.]+\): ([\d.]+)", (R / "entrada209_cforacle.txt").read_text(encoding="utf-8")).group(1))
    L = ["=" * 110, "HOSPEDEIRO NOVO: IG DE MAQUINAS NAO RELACIONADAS (Entrada 234)" + ("  [PILOTO: instancias de AJUSTE, nao e' resultado]" if PILOT else ""), "=" * 110, f"limiar h* (congelado) = {hstar:.4f}"]
    W, raw, cells, pooled = {}, {}, [], []
    for sp in ("S5", "S15"):
        d = load(sp)
        raw[sp] = d
        w = per_instance(d)
        W[sp] = w
        r = d[d.method == "rand"]
        pooled.append(dict(space=sp, n=len(w), gap_rand_pct=float((100 * (r.makespan - r.lb) / r.lb).mean()), frac_rand_at_lb=float((r.makespan <= r.lb).mean()),
                           h_biased=float(-rel(w.biased, w.rand).mean()), h_cf=float(-rel(w.cf, w.rand).mean()), **{f"g_{k}": float(-rel(w[k], w.rand).mean()) for k in LEARN + ["val"]}))
        for c, x in w.groupby("cell"):
            cells.append(dict(space=sp, cell=c, n=len(x), h_biased=float(-rel(x.biased, x.rand).mean()), h_cf=float(-rel(x.cf, x.rand).mean()), gq=float(-rel(x.ql, x.rand).mean()), gval=float(-rel(x.val, x.rand).mean())))
    Pd, C = pd.DataFrame(pooled), pd.DataFrame(cells)
    L += ["\n--- agregado por espaco (pontos percentuais; gap do aleatorio ao limite inferior em %; fracao das execucoes do aleatorio no limite) ---", Pd.to_string(index=False)]
    ta, tb = [], []
    for sp, w in W.items():
        for k in LEARN:
            t = test(w[k], w.rand); t.update(id=f"A-{sp}-{k}"); ta.append(t)
            t = test(w.val, w[k]); t.update(id=f"B-{sp}-{k}"); tb.append(t)
    FA, FB = family(ta), family(tb)
    L += ["\n--- A: aprendiz contra o aleatorio, iteracoes iguais (negativo: o aprendiz e' melhor; Holm sobre 10) ---", FA[COLS].to_string(index=False), "\n--- B: acao validada contra o aprendiz (negativo: a validada e' melhor; Holm sobre 10) ---", FB[COLS].to_string(index=False)]
    C["pred_gain"] = C.h_cf >= hstar
    C["true_gain"] = C.gq > 0
    acc = float((C.pred_gain == C.true_gain).mean())
    tpr = float((C.pred_gain & C.true_gain).sum() / max(C.true_gain.sum(), 1))
    tnr = float((~C.pred_gain & ~C.true_gain).sum() / max((~C.true_gain).sum(), 1))
    bal = 0.5 * (tpr + tnr)
    rho = {sp: sstats.spearmanr(x.h_cf, x.gq).statistic for sp, x in C.groupby("space")}
    lo, hi = C[C.h_cf < hstar], C[C.h_cf >= hstar]
    L += ["\n--- celulas (classe x espaco) ---", C.to_string(index=False),
          f"\nregra congelada: acuracia {acc:.3f}, balanceada {bal:.3f} (fracao de celulas em que o Q ganha {C.true_gain.mean():.3f}); Spearman folga cf x ganho do Q: " + ", ".join(f"{k} {v:+.3f}" for k, v in rho.items()),
          f"leitura unilateral: abaixo de h* {len(lo)} celulas (ganho do Q > 0,05 em {int((lo.gq > 0.05).sum())}; maior ganho {lo.gq.max() if len(lo) else float('nan'):+.3f}); acima {len(hi)} celulas (ganho > 0,05 em {int((hi.gq > 0.05).sum())}, > 0 em {int((hi.gq > 0).sum())}, negativo em {int((hi.gq < 0).sum())})"]
    yn = lambda b: "SIM" if b else "NAO"
    u1 = bool((Pd.h_cf >= hstar).all())
    u2 = acc >= 0.75
    u3 = bal >= 0.65
    u4 = bool((lo.gq <= 0.05).all()) if len(lo) else True
    qa = FA[FA.id.str.endswith("-ql")]
    u5 = bool(((qa.confirmada) & (~qa["equivalente_0.05"])).any())
    u6 = bool(FB[FB.id.str.endswith("-ql")].confirmada.all())
    u7 = bool((Pd.frac_rand_at_lb < 0.40).all())
    u8 = bool(Pd.set_index("space").loc["S5", "h_cf"] > 0.063 and Pd.set_index("space").loc["S15", "h_cf"] > 0.043)
    u9 = bool(min(rho.values()) >= 0.5)
    qoa = FA[FA.id.str.endswith("-qo")]
    u10 = bool(((qoa.confirmada) & (~qoa["equivalente_0.05"])).any())
    u11 = bool((Pd.g_qo > Pd.g_ql).all())
    u12 = bool(FB[FB.id.str.endswith("-qo")].confirmada.all())
    u13 = bool((Pd.g_val > 0.5).all())
    u14 = bool(((FA.confirmada) & (~FA["equivalente_0.05"])).any())
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---",
          f"(U1) folga cf agregada >= h* nos dois espacos: {yn(u1)} (" + ", ".join(f"{v:.3f}" for v in Pd.h_cf) + ")", f"(U2) acuracia da regra congelada >= 0,75 nas 24 celulas: {yn(u2)} ({acc:.3f})",
          f"(U3) acuracia balanceada >= 0,65: {yn(u3)} ({bal:.3f})", f"(U4) nenhuma celula abaixo de h* com ganho do Q > 0,05: {yn(u4)}",
          f"(U5) o Q bate o aleatorio por mais que a margem em algum espaco: {yn(u5)} (ganho agregado " + ", ".join(f"{v:+.3f}" for v in Pd.g_ql) + ")",
          f"(U6) a validada vence o Q (confirmada) nos dois espacos: {yn(u6)}", f"(U7) o aleatorio chega ao limite inferior em < 40% das execucoes nos dois espacos: {yn(u7)} (" + ", ".join(f"{v:.3f}" for v in Pd.frac_rand_at_lb) + ")",
          f"(U8) folga cf agregada maior que a do P||Cmax sintetico (0,063; 0,043): {yn(u8)}", f"(U9) Spearman >= 0,5 nos dois espacos: {yn(u9)} (" + ", ".join(f"{v:+.3f}" for v in rho.values()) + ")",
          f"(U10) o Q otimista bate o aleatorio por mais que a margem em algum espaco: {yn(u10)} (ganho agregado " + ", ".join(f"{v:+.3f}" for v in Pd.g_qo) + ")", f"(U11) ganho agregado de qo > o do Q padrao nos dois espacos: {yn(u11)}",
          f"(U12) a validada vence qo (confirmada) nos dois espacos: {yn(u12)}", f"(U13) ganho agregado da validada > 0,5 nos dois espacos: {yn(u13)} (" + ", ".join(f"{v:+.3f}" for v in Pd.g_val) + ")",
          f"(U14) algum aprendiz bate o aleatorio por mais que a margem em algum espaco: {yn(u14)}",
          f"R1 (a regra congelada se transfere: U2 e U4): {yn(u2 and u4)}", f"R2 (folga real: U1, U8 e U13): {yn(u1 and u8 and u13)}", f"R3 (aqui o aprendizado online paga: U14): {yn(u14)}"]
    text = "\n".join(L)
    print(text)
    if not PILOT:
        (R / "entrada234_ur.txt").write_text(text, encoding="utf-8")
        C.to_csv(R / "entrada234_ur_cells.csv", index=False)
        Pd.to_csv(R / "entrada234_ur_pooled.csv", index=False)
        FA.to_csv(R / "entrada234_ur_A.csv", index=False)
        FB.to_csv(R / "entrada234_ur_B.csv", index=False)


if __name__ == "__main__":
    main()
