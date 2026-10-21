"""
analysis/report_entrada180_agents.py
======================================
Agentes compilados e repeticoes em massa em espacos de acao de 3, 5, 15 e 30 acoes (Ciclo 15, frentes L e M). Diario: Entrada 180 (pre-registro) e 181 (resultados).

PRE-REGISTRO (congelado ANTES de rodar o ajuste, a calibracao e o teste; hashes em results/entrada180_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------
Contexto e delineamento: ver experiment/run_agents_numba_entrada180.py e pmsp/ig_numba_agents.py (35 testes: estado identico ao de pmsp/bounds.compute_state; agente fixo e aleatorio exatamente iguais ao ig_core). Espacos S3, S5, S15, S30; aprendizes ql, ucb, pg; aleatorio uniforme; braco validado por classe
(30 instancias de ajuste por classe); 12 classes x 100 instancias de teste x 5 sementes; unidade de analise = a MEDIA das 5 sementes por instancia (1.200 pares por espaco). d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal; nulos descartados;
bootstrap IC90 (2000); equivalencia a 0: IC90 dentro de +-0,05%; 'confirmada' = p_holm < 0,05 e d < 0, com leitura 'paridade' se tambem equivalente (licao II.24).
Transparencia: antes do pre-registro foi feito um teste de fumaca (uma instancia n = 100, m = 20, uma semente, T curto) que mostrou que o Q-Learning compilado usa menos avaliacoes que o aleatorio (acoes baratas: tamanhos de destruicao pequenos) e portanto custa MENOS CPU; as previsoes de K usam isso; as de qualidade nao.
Verificacao (V; regra de parada): o compilado reproduz a referencia Python a T = 1000: V1 S5 ql vs igd_ql_def; V2 S5 ucb vs igd_ucb; V3 S5 rand vs igd_random (results/dsel_test_entrada114.csv); V4 S3 ql vs ig_ql; V5 S3 rand vs ig_random (results/results.csv); semente unica por instancia; 'equivalente' = IC90 da diferenca media
pareada dentro de +-0,05. Se falhar, o compilado e' tratado como outra implementacao.
Familias (cada uma com Holm sobre 12 = 4 espacos x 3 aprendizes, alpha = 0,05):
  Acpu  aprendiz vs aleatorio a CPU igual (o aleatorio roda T_eq = round(1000 K) iteracoes, K = CPU do aprendiz / CPU do aleatorio, calibrado em sequencia por classe)
  Aiter aprendiz vs aleatorio a iteracoes iguais (T = 1000)
  B     braco validado vs aprendiz, a iteracoes iguais (negativo: a validacao vence)
Descritivas: K (CPU), razao de avaliacoes e sobrecusto do agente por avaliacao (K_CPU / K_avaliacoes); folga (oraculo de todas as acoes fixas, 1 semente, contra o aleatorio); fracao de acoes com busca local escolhida em S30; Spearman (folga da celula, ganho do Q-Learning a iteracoes iguais) em 48 celulas.
Leituras pre-fixadas:
  M1 'o agente compilado e' barato por unidade de trabalho' = media geometrica de K_CPU/K_avaliacoes <= 1,5 para ql e para ucb em cada espaco.
  M2 'um aprendiz online bate a mistura aleatoria por >= 0,05% a iteracoes iguais em algum espaco' = alguma Aiter confirmada e nao paridade.
  M3 'a CPU igual, algum aprendiz bate o aleatorio por >= 0,05% em algum espaco' = alguma Acpu confirmada e nao paridade.
  M4 'uma configuracao estatica vence o aprendizado online' = B confirmada em >= 8 dos 12 pares.
Previsoes registradas antes de rodar:
  V1-V5 todas equivalentes.
  K_CPU (media geometrica por classe): S3 ql, ucb, pg em [0,85; 1,3]; S5 ql [0,3; 0,9], ucb [0,8; 1,3], pg [0,7; 1,3]; S15 ql [0,3; 0,95], ucb [0,8; 1,3], pg [0,7; 1,3]; S30 ql [0,4; 1,1], ucb [0,8; 1,3], pg [0,7; 1,3]. Sobrecusto por avaliacao em [1,0; 2,0] para ql e ucb.
  Aiter: d em [-0,03; +0,08]% em todos os 12 pares, nenhum confirmado e nao paridade (M2 NAO). Acpu: ql confirmado em S5, S15 e S30 por custo (d em [-0,25; -0,02]%), nao confirmado em S3; M3 SIM. B: confirmada em S5 e S15 para os tres aprendizes, nao confirmada em S3 para ao menos dois, confirmada em S30 para ao menos dois (M4 SIM).
  Folga de S30 (oraculo contra o aleatorio) em [-0,6; -0,1]%; Spearman(folga, ganho do Q-Learning) <= 0,3.
Uso: python -m analysis.report_entrada180_agents
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni, rank_biserial_paired  # noqa: E402

R = ROOT / "results"
SPACES = ["S3", "S5", "S15", "S30"]
LEARNERS = ["ql", "ucb", "pg"]


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def geo(x):
    return float(np.exp(np.log(np.asarray(x, float)).mean()))


def family(rows):
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    P["equivalente_0.05"] = (P.ci90_lo > -0.05) & (P.ci90_hi < 0.05)
    P["leitura"] = np.where(P.confirmada & P["equivalente_0.05"], "paridade", np.where(P.confirmada, "confirmada", "nao confirmada"))
    return P


def main() -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "AGENTES COMPILADOS E ESPACOS DE ACAO MAIORES -- Entrada 180/181", "=" * 110]
    # ---- verificacao
    vf = pd.read_csv(R / "agents_verify_entrada180.csv")
    dsel = pd.read_csv(R / "dsel_test_entrada114.csv")
    res = pd.read_csv(R / "results.csv")
    refs = {"V1": ("S5", "ql", dsel[dsel.method == "igd_ql_def"]), "V2": ("S5", "ucb", dsel[dsel.method == "igd_ucb"]), "V3": ("S5", "rand", dsel[dsel.method == "igd_random"]),
            "V4": ("S3", "ql", res[res.method == "ig_ql"]), "V5": ("S3", "rand", res[res.method == "ig_random"])}
    vr = []
    for vid, (sp, me, ref) in refs.items():
        a = vf[(vf.space == sp) & (vf.method == me)][["cls", "rep", "makespan"]].rename(columns={"cls": "instance"})
        m = a.merge(ref[["instance", "rep", "makespan"]].astype({"rep": int}), on=["instance", "rep"], suffixes=("_c", "_p"))
        t = test(m.makespan_c, m.makespan_p)
        t.update(id=vid, comparison=f"{sp} {me}: compilado vs Python")
        vr.append(t)
    V = pd.DataFrame(vr)
    V["equivalente_0.05"] = (V.ci90_lo > -0.05) & (V.ci90_hi < 0.05)
    L += ["\n--- VERIFICACAO: agentes compilados vs referencia Python a T = 1000 (diferenca media pareada em %) ---", V[["id", "comparison", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "equivalente_0.05"]].to_string(index=False),
          "Todas equivalentes (+-0,05): " + ("SIM" if V["equivalente_0.05"].all() else "NAO -- o compilado e tratado como outra implementacao")]
    # ---- dados de teste
    K = json.loads((R / "agents_cpu_ratio_entrada180.json").read_text())
    data, rows_ac, rows_ai, rows_b, kk, hd, lsfrac = {}, [], [], [], [], [], []
    for sp in SPACES:
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        mean_ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
        orc = d[d.method.str.startswith("arm")].groupby(["cls", "rep"]).makespan.min().rename("oracle").reset_index()
        w = mean_ms.merge(orc, on=["cls", "rep"])
        assert len(w) == 1200, (sp, len(w))
        data[sp] = (w, d)
        for k in LEARNERS:
            t = test(w[k], w[f"rand_eq_{k}"]); t.update(space=sp, learner=k, id=f"Acpu-{sp}-{k}"); rows_ac.append(t)
            t = test(w[k], w["rand"]); t.update(space=sp, learner=k, id=f"Aiter-{sp}-{k}"); rows_ai.append(t)
            t = test(w["val"], w[k]); t.update(space=sp, learner=k, id=f"B-{sp}-{k}"); rows_b.append(t)
            ev = d[d.method.isin([k, "rand"])].groupby(["cls", "method"]).evals.mean().unstack()
            ke = ev[k] / ev["rand"]
            kc = pd.Series({c: K[sp][c.replace("rand_", "").replace("j_", "x").replace("m", "")][k] for c in ke.index})
            kk.append(dict(space=sp, learner=k, K_cpu=geo(kc), K_avaliacoes=geo(ke), sobrecusto_por_avaliacao=geo(kc / ke)))
        hd.append(dict(space=sp, folga_oraculo_vs_aleatorio=float(rel(w.oracle, w.rand).mean())))
        if sp == "S30":
            for k in LEARNERS:
                lsfrac.append(dict(learner=k, frac_acoes_com_busca_local=float(d[d.method == k].frac_ls.mean())))
            lsfrac.append(dict(learner="rand", frac_acoes_com_busca_local=float(d[d.method == "rand"].frac_ls.mean())))
    Ac, Ai, B = family(rows_ac), family(rows_ai), family(rows_b)
    cols = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
    L += ["\n--- Acpu: aprendiz vs aleatorio a CPU igual (Holm sobre 12) ---", Ac[cols].to_string(index=False), "\n--- Aiter: aprendiz vs aleatorio a iteracoes iguais (Holm sobre 12) ---", Ai[cols].to_string(index=False),
          "\n--- B: braco validado vs aprendiz a iteracoes iguais (Holm sobre 12; negativo: a validacao vence) ---", B[cols].to_string(index=False)]
    KK = pd.DataFrame(kk)
    L += ["\n--- custo do agente compilado: K_CPU (CPU do aprendiz / CPU do aleatorio), razao de avaliacoes e sobrecusto por avaliacao (medias geometricas por classe) ---", KK.to_string(index=False),
          "\n--- folga: oraculo (todas as acoes fixas, 1 semente) contra o aleatorio (%) ---", pd.DataFrame(hd).to_string(index=False), "\n--- S30: fracao de iteracoes com acao que aplica busca local ---", pd.DataFrame(lsfrac).to_string(index=False)]
    cells = []
    for sp in SPACES:
        w, _ = data[sp]
        g = pd.DataFrame({"hd": rel(w.oracle, w.rand), "gain": rel(w["ql"], w["rand"]), "cls": w.cls}).groupby("cls").mean()
        g["space"] = sp
        cells.append(g)
    cells = pd.concat(cells)
    rho = sstats.spearmanr(cells.hd, cells.gain)
    L.append(f"\nSpearman(folga da celula, ganho do Q-Learning a iteracoes iguais) em {len(cells)} celulas: rho = {rho.statistic:.3f} (p = {rho.pvalue:.3g})")
    m1 = bool(all(KK[KK.learner.isin(["ql", "ucb"])].sobrecusto_por_avaliacao <= 1.5))
    m2 = bool((Ai.confirmada & ~Ai["equivalente_0.05"]).any())
    m3 = bool((Ac.confirmada & ~Ac["equivalente_0.05"]).any())
    m4 = bool(int((B.confirmada).sum()) >= 8)
    L += ["\n--- LEITURAS PRE-FIXADAS ---", f"M1 (agente compilado barato por unidade de trabalho): {'SIM' if m1 else 'NAO'}", f"M2 (um aprendiz bate a mistura aleatoria por >= 0,05% a iteracoes iguais em algum espaco): {'SIM' if m2 else 'NAO'}",
          f"M3 (a CPU igual, algum aprendiz bate o aleatorio por >= 0,05% em algum espaco): {'SIM' if m3 else 'NAO'}", f"M4 (a validacao vence o aprendizado online em >= 8 dos 12 pares; {int(B.confirmada.sum())}/12): {'SIM' if m4 else 'NAO'}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada180_agents.txt").write_text(text, encoding="utf-8")
    Ac.to_csv(R / "entrada180_agents_acpu.csv", index=False)
    Ai.to_csv(R / "entrada180_agents_aiter.csv", index=False)
    B.to_csv(R / "entrada180_agents_b.csv", index=False)
    KK.to_csv(R / "entrada180_agents_k.csv", index=False)


if __name__ == "__main__":
    main()
