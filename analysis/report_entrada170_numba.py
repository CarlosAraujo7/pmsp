"""
analysis/report_entrada170_numba.py
=====================================
O orcamento 'a CPU igual' com uma implementacao COMPILADA e com uma unidade de orcamento independente de implementacao (Ciclo 14, frente H). Diario: Entrada 170 (pre-registro) e 171 (resultados).

PRE-REGISTRO (congelado ANTES de calibrar e de executar; hashes em results/entrada170_prereg_hashes.txt)
--------------------------------------------------------------------------------------------------------
Contexto: a conclusao 'a CPU igual' sobre a busca local do IG de P||Cmax mudou com a implementacao (lacos Python: K = 9,6, ganho a CPU igual -0,010%, 93% do ganho era orcamento; vetorizada: K = 1,8, -0,105%, 30%).
pmsp/ig_numba.py compila o IG e a busca local (busca local e reconstrucao exatamente equivalentes: tests/test_ig_numba.py, 403 testes; os sorteios usam o gerador do numba, logo a equivalencia das trajetorias e' estatistica) e conta
avaliacoes (candidatos de movimento da busca local + comparacoes de maquina na reconstrucao; peso w da reconstrucao: 1 principal, 0,25 e 4 sensibilidade). Dados: experiment/run_numba_entrada170.py -> results/numba_entrada170.csv
(cenarios H e R de Entrada 129/127/162; 12 classes x 100 reps, as mesmas instancias). d = 100*(A-B)/B em makespan; negativo: A melhor. Wilcoxon pareado bicaudal pooled (nulos descartados), bootstrap IC90 (2000), equivalencia a 0 se o IC90 cabe em +-0,05%.
Aviso de transparencia: antes deste pre-registro foi feito um piloto de tempo (5 classes, T = 1000, ig vs igls compilados): K_cpu 1,9 a 2,4 e K_ev 2,6 a 6,2; as previsoes de K abaixo usam esse piloto; as de qualidade (d) nao.
Verificacao (V; regra de parada): o compilado reproduz a referencia Python a iteracoes iguais (T = 1000): V1 H igls vs igls_random (Entrada 124); V2 H ig_T1000 vs ig_random (results.csv); V3 R igls vs igcvls_seed (Entrada 127);
  V4 R ig_T1000 vs igcv_seed (Entrada 121). 'Equivalente' = IC90 da diferenca media pareada (compilado - Python, em %) dentro de +-0,05. Se alguma V falhar, o compilado e' tratado como OUTRA implementacao (nao como a mesma) e as conclusoes
  C1-C4 sao relatadas com essa ressalva; nenhuma afirmacao de 'equivalencia' e' feita.
Familia primaria (Holm sobre 4, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0, isto e', a busca local ganha com orcamento igual):
  C1  H: igls vs ig_eqcpu (T_eq = max(1000, round(1000 K_cpu)), K_cpu calibrado sequencialmente por classe)       C2  H: igls vs ig_eqev_w1 (mesmas avaliacoes por instancia)
  C3  R: igls vs ig_eqcpu                                                                                        C4  R: igls vs ig_eqev_w1
Leitura quando o teste de postos e a media discordam (licao II.24, pre-fixada): se p_holm < 0,05 e o IC90 da media cabe em +-0,05, o veredito e' 'detectavel mas equivalente a zero (paridade)', nao 'a busca local ganha'.
Descritivas (fora da familia): d a iteracoes iguais (igls vs ig_T1000) em H e R; sensibilidade ao peso w (0,25; 4); K_cpu e K_ev por classe; Spearman(K_cpu, K_ev) nas 24 classes-cenarios; razao realizada de tempo de parede (ig_eqcpu/igls) por classe;
fracao de orcamento B = 1 - d_orcamento_igual / d_iteracoes_iguais para CPU e para avaliacoes; tabela das tres implementacoes (lacos, vetorizada, compilada) no cenario H.
Leituras pre-fixadas:
  L1 'a conclusao da implementacao vetorizada (a busca local paga a CPU igual no IG, neutra na receita) e' robusta a uma implementacao compilada' = C1 confirmada com d <= -0,05 E (C3 nao confirmada OU |d(C3)| < 0,05).
  L2 'a contagem de avaliacoes e' uma unidade de orcamento utilizavel' = C1 e C2 com o mesmo veredito E |d(C1) - d(C2)| < 0,05.
Previsoes registradas antes de calibrar/rodar:
  V1-V4: todas equivalentes.
  K_cpu: media geometrica em H em [1,6; 2,8] e em R em [1,6; 3,0] (contra 9,6 com lacos e 1,8 vetorizado); K_ev (razao das avaliacoes igls/ig a T = 1000): media geometrica em H em [2,5; 6,0]; K_ev/K_cpu > 1,3 (a contagem superestima o custo da
  busca local em relacao a' CPU); Spearman(K_cpu, K_ev) < 0,8.
  Iteracoes iguais: H d em [-0,19; -0,11]; R d em [-0,07; 0,00].
  C1 confirmada com d em [-0,14; -0,06], B_cpu em [0,15; 0,55]; C2 confirmada com d em [-0,12; -0,03] (mais proxima de zero que C1) e B_ev em [0,2; 0,8]; C3 e C4 nao confirmadas com |d| < 0,05.
  L1: SIM. L2: SIM com ressalva (|d(C1) - d(C2)| entre 0,02 e 0,05).
Uso: python -m analysis.report_entrada170_numba
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


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def geo(x):
    return float(np.exp(np.log(np.asarray(x, float)).mean()))


def main() -> None:
    nb = pd.read_csv(R / "numba_entrada170.csv")
    ratio = json.loads((R / "numba_cpu_ratio_entrada170.json").read_text())
    W = nb.pivot(index=["scenario", "instance", "rep"], columns="method", values="makespan").reset_index()
    E = nb.pivot(index=["scenario", "instance", "rep"], columns="method", values="evals").reset_index()
    T = nb.pivot(index=["scenario", "instance", "rep"], columns="method", values="wall_s").reset_index()
    H, Rr = W[W.scenario == "H"].reset_index(drop=True), W[W.scenario == "R"].reset_index(drop=True)
    assert len(H) == 1200 and len(Rr) == 1200, (len(H), len(Rr))
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "ORCAMENTO 'A CPU IGUAL' COM IMPLEMENTACAO COMPILADA E UNIDADE INDEPENDENTE DE IMPLEMENTACAO -- Entrada 170/171", "=" * 110]

    # ---- verificacao: compilado vs referencia Python a iteracoes iguais
    res = pd.read_csv(R / "results.csv")
    hy = pd.read_csv(R / "igls_p_entrada124.csv")
    rc = pd.read_csv(R / "igcvls_seed_entrada127.csv")
    rb = pd.read_csv(R / "igcv_seed_entrada121.csv")
    refs = {"V1": ("H", "igls", hy[hy.method == "igls_random"]), "V2": ("H", "ig_T1000", res[res.method == "ig_random"]),
            "V3": ("R", "igls", rc[rc.method == "igcvls_seed"]), "V4": ("R", "ig_T1000", rb[rb.method == "igcv_seed"])}
    vrows = []
    for vid, (sc, col, ref) in refs.items():
        base = (H if sc == "H" else Rr)[["instance", "rep", col]].merge(ref[["instance", "rep", "makespan"]].astype({"rep": int}), on=["instance", "rep"])
        assert len(base) == 1200, (vid, len(base))
        t = test(base[col], base["makespan"])
        t.update(id=vid, comparison=f"{sc}: {col} compilado vs Python")
        vrows.append(t)
    V = pd.DataFrame(vrows)
    V["equivalente_0.05"] = (V.ci90_lo > -0.05) & (V.ci90_hi < 0.05)
    L += ["\n--- VERIFICACAO: compilado vs referencia Python a iteracoes iguais (T = 1000; diferenca media pareada em %) ---",
          V[["id", "comparison", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "equivalente_0.05"]].to_string(index=False),
          "Todas equivalentes (+-0,05): " + ("SIM" if V["equivalente_0.05"].all() else "NAO -- o compilado e tratado como outra implementacao")]

    # ---- familia primaria
    rows = []
    for pid, sc, df, b, lab in [("C1", "H", H, "ig_eqcpu", "H: igls vs ig_eqcpu (CPU igual)"), ("C2", "H", H, "ig_eqev_w1", "H: igls vs ig_eqev_w1 (avaliacoes iguais)"),
                                ("C3", "R", Rr, "ig_eqcpu", "R: igls vs ig_eqcpu (CPU igual)"), ("C4", "R", Rr, "ig_eqev_w1", "R: igls vs ig_eqev_w1 (avaliacoes iguais)")]:
        t = test(df["igls"], df[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    P["equivalente_0.05"] = (P.ci90_lo > -0.05) & (P.ci90_hi < 0.05)
    P["leitura"] = np.where(P.confirmada & P["equivalente_0.05"], "paridade (detectavel mas equivalente a zero)", np.where(P.confirmada, "a busca local ganha", "sem ganho confirmado"))
    L += ["\n--- PRIMARIA (Holm sobre 4; confirmada = a busca local ganha com orcamento igual) ---",
          P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]].to_string(index=False)]
    Pi = P.set_index("id")

    # ---- descritivas
    it_h, it_r = test(H.igls, H.ig_T1000), test(Rr.igls, Rr.ig_T1000)
    L.append(f"\nIteracoes iguais (T = 1000): H d = {it_h['mean_d_pct']:+.4f}% (IC90 {it_h['ci90_lo']:+.3f}; {it_h['ci90_hi']:+.3f}; {it_h['better']}/{it_h['worse']}/{it_h['tie']}); "
             f"R d = {it_r['mean_d_pct']:+.4f}% (IC90 {it_r['ci90_lo']:+.3f}; {it_r['ci90_hi']:+.3f}; {it_r['better']}/{it_r['worse']}/{it_r['tie']})")
    b_cpu = 1 - abs(Pi.loc["C1", "mean_d_pct"]) / abs(it_h["mean_d_pct"])
    b_ev = 1 - abs(Pi.loc["C2", "mean_d_pct"]) / abs(it_h["mean_d_pct"])
    L.append(f"Fracao de orcamento do ganho da busca local no IG (H): B_cpu = {b_cpu:.2f}; B_ev = {b_ev:.2f}")
    sens = []
    for meth in ("ig_eqev_w0.25", "ig_eqev_w1", "ig_eqev_w4"):
        for sc, df in (("H", H), ("R", Rr)):
            t = test(df.igls, df[meth])
            sens.append(dict(cenario=sc, metodo=meth, mean_d_pct=t["mean_d_pct"], ci90_lo=t["ci90_lo"], ci90_hi=t["ci90_hi"], melhor_igls=t["better"], pior_igls=t["worse"], p_raw=t["p_raw"]))
    L += ["\n--- sensibilidade ao peso w da avaliacao da reconstrucao (igls vs ig a avaliacoes iguais; d %) ---", pd.DataFrame(sens).to_string(index=False)]

    kk = []
    for sc, Ed, Td in (("H", E[E.scenario == "H"], T[T.scenario == "H"]), ("R", E[E.scenario == "R"], T[T.scenario == "R"])):
        for c, g in Ed.groupby("instance"):
            n, m = c.replace("rand_", "").replace("m", "").split("j_")
            key = f"{n}x{m}"
            tg = Td[Td.instance == c]
            kk.append(dict(cenario=sc, classe=key, K_cpu=ratio[sc][key]["K"], T_eq=ratio[sc][key]["T_eq"], K_ev=float((g.igls / g.ig_T1000).mean()),
                           razao_parede_eqcpu_igls=float((tg.ig_eqcpu / tg.igls).median())))
    K = pd.DataFrame(kk)
    K["K_ev/K_cpu"] = K.K_ev / K.K_cpu
    L += ["\n--- razoes de custo da busca local por classe: K_cpu (calibrado), K_ev (contagem de avaliacoes, media por instancia) e razao realizada de tempo de parede (ig_eqcpu/igls, deve ficar perto de 1) ---", K.to_string(index=False)]
    rho = sstats.spearmanr(K.K_cpu, K.K_ev)
    L.append(f"Medias geometricas: K_cpu H {geo(K[K.cenario == 'H'].K_cpu):.2f}, R {geo(K[K.cenario == 'R'].K_cpu):.2f}; K_ev H {geo(K[K.cenario == 'H'].K_ev):.2f}, R {geo(K[K.cenario == 'R'].K_ev):.2f}; "
             f"K_ev/K_cpu H {geo(K[K.cenario == 'H']['K_ev/K_cpu']):.2f}. Spearman(K_cpu, K_ev) nas 24 classes-cenarios = {rho.statistic:.2f} (p = {rho.pvalue:.3g})")

    # ---- tres implementacoes (cenario H)
    k129 = json.loads((R / "igls_eq_cpu_ratio_entrada129.json").read_text())
    k162 = json.loads((R / "igls_fast_cpu_ratio_entrada162.json").read_text())
    eq129 = pd.read_csv(R / "igls_eq_entrada129.csv")
    eq162 = pd.read_csv(R / "igls_eq_fast_entrada162.csv")
    ref_ls = hy[hy.method == "igls_random"][["instance", "rep", "makespan"]].rename(columns={"makespan": "igls"})
    ref_ig = res[res.method == "ig_random"][["instance", "rep", "makespan"]].rename(columns={"makespan": "ig"})
    impl = []
    for lab, df, ks in [("lacos Python (Ent. 129)", eq129[eq129.method == "ig_random_eq"], [k129[c]["K"] for c in k129]),
                        ("vetorizada (Ent. 162)", eq162[eq162.method == "ig_random_eq_fast"], [k162["H"][c]["K"] for c in k162["H"]]),
                        ("compilada (esta)", None, [ratio["H"][c]["K"] for c in ratio["H"]])]:
        if df is None:
            t = test(H.igls, H.ig_eqcpu)
        else:
            mm = ref_ls.merge(df[["instance", "rep", "makespan"]].astype({"rep": int}), on=["instance", "rep"])
            t = test(mm.igls, mm.makespan)
        impl.append(dict(implementacao=lab, K_geo=geo(ks), d_CPU_igual=t["mean_d_pct"], ci90_lo=t["ci90_lo"], ci90_hi=t["ci90_hi"], B=1 - abs(t["mean_d_pct"]) / abs(it_h["mean_d_pct"])))
    L += ["\n--- a mesma pergunta (IG com busca local vs IG sem, a CPU igual, cenario H) em tres implementacoes ---", pd.DataFrame(impl).to_string(index=False)]

    sec = []
    for c, g in H.groupby("instance"):
        r = Rr[Rr.instance == c]
        sec.append(dict(classe=c, H_iter=rel(g.igls, g.ig_T1000).mean(), H_CPU=rel(g.igls, g.ig_eqcpu).mean(), H_aval=rel(g.igls, g.ig_eqev_w1).mean(),
                        R_iter=rel(r.igls, r.ig_T1000).mean(), R_CPU=rel(r.igls, r.ig_eqcpu).mean(), R_aval=rel(r.igls, r.ig_eqev_w1).mean()))
    L += ["\n--- por classe (d %; negativo: a busca local ganha) ---", pd.DataFrame(sec).to_string(index=False)]

    # ---- leituras pre-fixadas
    c1, c3, c2 = Pi.loc["C1"], Pi.loc["C3"], Pi.loc["C2"]
    l1 = bool(c1.confirmada and c1.mean_d_pct <= -0.05 and ((not c3.confirmada) or abs(c3.mean_d_pct) < 0.05))
    l2 = bool((c1.confirmada == c2.confirmada) and abs(c1.mean_d_pct - c2.mean_d_pct) < 0.05)
    L += ["\n--- LEITURAS PRE-FIXADAS ---",
          f"L1 (a conclusao da implementacao vetorizada e' robusta a uma implementacao compilada): {'SIM' if l1 else 'NAO'}",
          f"L2 (a contagem de avaliacoes e' uma unidade de orcamento utilizavel: C1 e C2 com o mesmo veredito e |d1 - d2| < 0,05; |d1 - d2| = {abs(c1.mean_d_pct - c2.mean_d_pct):.3f}): {'SIM' if l2 else 'NAO'}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada170_numba.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada170_numba_primary.csv", index=False)
    K.to_csv(R / "entrada170_numba_k.csv", index=False)


if __name__ == "__main__":
    main()
