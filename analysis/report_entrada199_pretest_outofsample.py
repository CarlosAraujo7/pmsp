"""
analysis/report_entrada199_pretest_outofsample.py
====================================================
Ciclo 18, frente C: VALIDACAO FORA DA AMOSTRA do pre-teste de folga/ruido (VNR) na familia F publicada de P||Cmax (FFU, FFNU; 26 classes (subpasta, m, n) de 30 instancias). Diario: Entrada 199 (pre-registro) e 201 (resultados).
Delineamento: experiment/run_cycle18_famF_entrada199.py. Celula = classe (26 por espaco). Sonda = 12 primeiras instancias da classe (por inst_id): 3 sementes do aleatorio e 3 de cada acao fixa; o VNR (pmsp.budgetfair.validated_to_noise: ganho validado em validacao cruzada / desvio padrao relativo do aleatorio) e a acao validada
da celula saem SO' delas. Teste = as 18 instancias restantes: 5 sementes do aleatorio, Q-Learning (alfa = gama = 0,1), UCB1, roleta ALNS (rho 0,3) e a acao validada, e 1 semente de cada acao fixa (oraculo); T = 1000; espacos S5 e S15. Limiar tau = 0,877 CONGELADO em results/pretest_tau_entrada198.json (calibrado nas celulas
sinteticas e de descoberta do B/C/E; a familia F nao entrou em nenhum ajuste de aprendiz, acao ou limiar). Desfecho da celula: g_Q = -100 media[(ql - rand)/rand] nas 18 instancias de teste (positivo: o Q-Learning ganha do aleatorio a iteracoes iguais). Estatistica das Entradas 180-198 (unidade = media das sementes por instancia).

PRE-REGISTRO (congelado ANTES de rodar as sondas e o teste; hashes em results/entrada199_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Transparencia: as SONDAS (so' o lado de ajuste; nenhum desfecho de teste) foram vistas antes de congelar estas previsoes: em S5, 21 das 26 celulas tem VNR definido (5 com ruido 0: o aleatorio sempre acha o mesmo makespan, o teto) e 13 delas tem VNR >= tau; em S15, 21 definidas e 8 >= tau; mas o ganho validado absoluto e' minusculo na maioria (v_hat >= 0,01% em 10 celulas de S5 e 8 de S15), de modo que o sinal de g_Q em muitas celulas e' praticamente ruido. A POPULACAO de avaliacao e' a das celulas com VNR definido (as demais sao excluidas e contadas). Previsao a priori honesta do autor: probabilidade ~40% de (2) valer nos dois espacos, porque o teto da familia F deixa pouca folga; a analise SECUNDARIA (descritiva) restringe-se as celulas materiais (v_hat >= 0,01%).
Previsoes (cada uma avaliada por espaco; 26 celulas independentes por espaco, S5 e S15 compartilham as instancias, por isso nao se agrupam):
(1) Spearman(VNR, g_Q) >= +0,35 em S5 e em S15;  (2) AUC do VNR para 'g_Q > 0' >= 0,70 em S5 e em S15;  (3) acuracia da regra 'VNR >= tau' para 'g_Q > 0' >= 0,65 em S5 e em S15 (limiar mais baixo que o das sinteticas porque na familia F muitas celulas tem g_Q ~ 0 e o sinal e' ruidoso);
(4) Spearman(VNR, folga do oraculo na celula) >= +0,50 em S5 e em S15 (o VNR mede folga/ruido);  (5) a acao validada tem ganho sobre o aleatorio >= o do Q-Learning em >= 75% das celulas em S5 e em S15;  (6) restrito as celulas DECISIVAS (Wilcoxon bilateral do Q-Learning contra o aleatorio com p < 0,10 nas 18 instancias), a acuracia da regra e' >= 0,75 (se houver >= 6 celulas decisivas em um espaco;
senao, nao avaliavel nesse espaco);  (7) o Q-Learning ganha do aleatorio (g_Q > 0) em >= 50% das celulas com VNR >= tau e em <= 50% das com VNR < tau, em cada espaco.
Leitura L ('o pre-teste VNR generaliza para uma familia nunca vista'): (1), (2) e (3) cumpridas nos dois espacos. Leitura L- ('generalizacao parcial'): (2) cumprida nos dois espacos mas nao (1) ou (3). Se (2) falhar em algum espaco, o pre-teste NAO generaliza como regra de decisao e o artigo deve apresenta-lo so' como descritivo (calibracao).
Familias de testes de hipotese (descritivas sobre as instancias de teste agrupadas, Holm sobre 6 = 3 aprendizes x 2 espacos): Aiter aprendiz contra aleatorio; Bit validada contra aprendiz.
Uso: python -m analysis.report_entrada199_pretest_outofsample
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.pretest_vnr_entrada198 import cell_vnr, evaluate  # noqa: E402
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
LEARNERS = ["ql", "ucb", "alns"]
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_test(sp):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"famF_test_{sp}_s*of*_entrada199.csv")))], ignore_index=True)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep_key", "method"]).makespan.mean().unstack("method").reset_index()
    orc = d[d.method.str.startswith("arm")].groupby(["cls", "rep_key"]).makespan.min().rename("oracle").reset_index()
    return ms.merge(orc, on=["cls", "rep_key"])


def main():
    tau = json.loads((R / "pretest_tau_entrada198.json").read_text())["tau"]
    L = ["=" * 110, "PRE-TESTE VNR FORA DA AMOSTRA: familia F publicada (Entrada 199)", "=" * 110, f"tau congelado = {tau:.4f}"]
    W, rows, ev = {}, [], {}
    for sp in ("S5", "S15"):
        W[sp] = load_test(sp)
        pr = pd.read_csv(R / f"famF_probe_{sp}_entrada199.csv")
        assert W[sp].cls.nunique() == 26 and len(W[sp]) == 26 * 18, (sp, W[sp].cls.nunique(), len(W[sp]))
        for cell, x in W[sp].groupby("cls"):
            v = cell_vnr(pr[pr.cls == cell])
            dq = rel(x.ql, x.rand)
            nz = dq[dq != 0]
            rows.append(dict(space=sp, cell=cell, n=len(x), noise=v["noise"], v_hat=v["v_hat"], vnr=v["vnr"], folga=float(-rel(x.oracle, x.rand).mean()), gq=float(-dq.mean()), gval=float(-rel(x.val, x.rand).mean()), galns=float(-rel(x.alns, x.rand).mean()),
                             p_q=float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")))
    C = pd.DataFrame(rows)
    L += ["\n--- celulas (classe x espaco) ---", C.to_string(index=False)]
    fam = {"Aiter": [], "Bit": []}
    for sp in ("S5", "S15"):
        w = W[sp]
        for k in LEARNERS:
            t = test(w[k], w["rand"]); t.update(id=f"Aiter-{sp}-{k}"); fam["Aiter"].append(t)
            t = test(w["val"], w[k]); t.update(id=f"Bit-{sp}-{k}"); fam["Bit"].append(t)
    F = {nm: family(r) for nm, r in fam.items()}
    for nm, X in F.items():
        L += [f"\n--- {nm} (Holm sobre {len(X)}; instancias de teste agrupadas) ---", X[COLS].to_string(index=False)]
    P = {}
    for sp in ("S5", "S15"):
        c = C[(C.space == sp) & C.vnr.notna()]
        e = evaluate(c.vnr, c.gq, tau)
        mat = c[c.v_hat >= 0.01]
        e_mat = evaluate(mat.vnr, mat.gq, tau) if len(mat) >= 6 else None
        dec = c[c.p_q < 0.10]
        e6 = evaluate(dec.vnr, dec.gq, tau) if len(dec) >= 6 else None
        hi, lo = c[c.vnr >= tau], c[c.vnr < tau]
        ev[sp] = dict(e=e, e_mat=e_mat, n_mat=len(mat), n_undef=int(((C.space == sp) & C.vnr.isna()).sum()), n_dec=len(dec), e6=e6, sp_folga=float(sstats.spearmanr(c.vnr, c.folga).statistic), frac_val_ge=float((c.gval >= c.gq).mean()), hi_pay=float((hi.gq > 0).mean()) if len(hi) else float("nan"), lo_pay=float((lo.gq > 0).mean()) if len(lo) else float("nan"), n_hi=len(hi), n_lo=len(lo))
    yn = lambda b: "SIM" if b else "NAO"
    L.append("\n--- PREVISOES E LEITURAS PRE-FIXADAS ---")
    for sp in ("S5", "S15"):
        d = ev[sp]
        e = d["e"]
        L += [f"[{sp}] celulas avaliaveis {e['n']} (VNR indefinido, excluidas: {d['n_undef']}), fracao em que o Q-Learning ganha {e['frac_pays']:.2f}; celulas com VNR >= tau: {d['n_hi']}",
              f"  (1) Spearman(VNR, g_Q) >= +0,35: {yn(e['spearman'] >= 0.35)} ({e['spearman']:+.3f})", f"  (2) AUC >= 0,70: {yn(e['auc'] >= 0.70)} ({e['auc']:.3f})", f"  (3) acuracia da regra >= 0,65: {yn(e['acc'] >= 0.65)} ({e['acc']:.3f}; acuracia balanceada {e['bacc']:.3f})",
              f"  (4) Spearman(VNR, folga do oraculo) >= +0,50: {yn(d['sp_folga'] >= 0.50)} ({d['sp_folga']:+.3f})", f"  (5) validada >= Q-Learning em >= 75% das celulas: {yn(d['frac_val_ge'] >= 0.75)} ({d['frac_val_ge']:.2f})",
              "  (6) celulas decisivas: " + (f"{d['n_dec']}; acuracia {d['e6']['acc']:.3f} -> {yn(d['e6']['acc'] >= 0.75)}" if d["e6"] is not None else f"{d['n_dec']}; nao avaliavel"),
              f"  (7) Q-Learning ganha em >= 50% das celulas com VNR >= tau ({d['hi_pay']:.2f}) e em <= 50% das com VNR < tau ({d['lo_pay']:.2f}): {yn(d['hi_pay'] >= 0.5 and d['lo_pay'] <= 0.5)}"]
    for sp in ("S5", "S15"):
        d = ev[sp]
        L.append(f"[secundaria, descritiva, {sp}] celulas materiais (v_hat >= 0,01%): {d['n_mat']}" + (f"; AUC {d['e_mat']['auc']:.3f}, acuracia {d['e_mat']['acc']:.3f}, Spearman {d['e_mat']['spearman']:+.3f}" if d["e_mat"] is not None else "; poucas para avaliar"))
    ok = lambda sp: ev[sp]["e"]["spearman"] >= 0.35 and ev[sp]["e"]["auc"] >= 0.70 and ev[sp]["e"]["acc"] >= 0.65
    auc_ok = all(ev[sp]["e"]["auc"] >= 0.70 for sp in ev)
    L.append(f"L (generaliza para uma familia nunca vista; (1), (2) e (3) nos dois espacos): {yn(ok('S5') and ok('S15'))}   |   L- (so' (2) nos dois espacos): {yn(auc_ok and not (ok('S5') and ok('S15')))}   |   (2) falha em algum espaco: {yn(not auc_ok)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada199_pretest_outofsample.txt").write_text(text, encoding="utf-8")
    C.to_csv(R / "entrada199_pretest_cells.csv", index=False)
    for nm, X in F.items():
        X.to_csv(R / f"entrada199_famF_{nm}.csv", index=False)


if __name__ == "__main__":
    main()
