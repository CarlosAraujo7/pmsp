"""
analysis/report_entrada264_qperm_pub.py
========================================
Entrada 264 (Ciclo 38): o Q-LEARNING INVARIANTE A ORDEM (ql_perm) nas instancias publicadas HELD-OUT de P||Cmax (Entrada 190: familias B, C, E do benchmark de Ostojic et al.; espacos S5 e S15; T = 1.000; 5 sementes; 3.950 instancias
menos as excluidas). As hipoteses confirmatorias H1-H3 da Entrada 190 ('o Q-Learning ganha na familia C, tercis 1-2, e perde na familia E') e a correlacao de Spearman entre a folga e o ganho do Q-Learning foram medidas com o Q-Learning padrao,
que repete a acao listada primeiro (Entrada 258); no P||Cmax a acao 0 e' a de menor destruicao (a mais barata; Entrada 261: o padrao avalia 0,44-0,46 das solucoes do aleatorio, o ql_perm 0,99-1,01). Pergunta: o que sobra com a ordem
invariante? Mesmas instancias, mesmos hiperparametros do Q-Learning por familia (Entrada 189), o aleatorio, o ql padrao, a validada e as acoes fixas sao os da Entrada 190 (results/pubh_*_entrada190.csv); o ql_perm tem sementes proprias.
Unidade = media das 5 sementes por instancia; d = 100 (A - B) / B em makespan (negativo: A melhor); testes como nas Entradas 180-190 (Wilcoxon pareado bilateral, IC90 bootstrap, equivalencia a 0 em +-0,05%); 'alem da margem' = confirmada e nao
equivalente a zero. Familias: H' (as quatro hipoteses H1-H3b da Entrada 190 com o ql_perm no lugar do ql; Holm sobre 4), A (ql_perm contra o aleatorio nas 18 celulas familia x grupo x espaco; Holm sobre 18), B (a validada contra o ql_perm, 18 celulas).

PRE-REGISTRO (congelado ANTES de rodar; hashes em results/entrada264_prereg_hashes.txt; nenhum resultado do ql_perm nestas instancias visto; o ql padrao vem da Entrada 190)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(U1) H1' (S15, familia C, grupos 1-2: ql_perm melhora o aleatorio em pelo menos 0,10%, confirmada) NAO se confirma. (85%)
(U2) H2' (S5, o mesmo, melhora de pelo menos 0,05%) NAO se confirma. (75%)
(U3) O ql_perm ganha do aleatorio alem da margem (A) em no maximo 1 das 18 celulas (o ql padrao: H1 e H2 confirmadas, +0,245 e +0,149 pontos). (75%)
(U4) O Spearman entre a folga do oraculo (em modulo) e o ganho do ql_perm nas 18 celulas e' menor que +0,30 (o do ql padrao: ver a Entrada 190). (55%)
(U5) A validada bate o ql_perm (B, confirmada) em pelo menos 15 das 18 celulas. (70%)
(U6) H3a' e H3b' (familia E: ql_perm PIOR que o aleatorio, confirmada, em S5 e em S15) se confirmam as duas. (50%)
(U7) A razao entre as avaliacoes do ql_perm e as do aleatorio, media das instancias, e' de pelo menos 0,9 em S5 e em S15 (o ql padrao: ver abaixo). (80%)
Uso: python -m analysis.report_entrada264_qperm_pub
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
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada189_published import per_instance  # noqa: E402
from analysis.stats import holm_bonferroni  # noqa: E402

R = ROOT / "results"
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def main():
    info = json.loads((R / "pubh_groups_entrada190.json").read_text())
    L = ["=" * 110, "Q-LEARNING INVARIANTE A ORDEM NAS INSTANCIAS PUBLICADAS HELD-OUT (Entrada 264)", "=" * 110, f"instancias held-out: {len(info['test'])}; excluidas (soma >= 2^62): {info['excluded']}"]
    W, cells, TA, TB, evr = {}, [], [], [], {}
    for sp in ("S5", "S15"):
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pubh_{sp}_s*of*_entrada190.csv")))], ignore_index=True)
        w = per_instance(d)
        q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"qpermpub_{sp}_s*of*_entrada264.csv")))], ignore_index=True)
        qm = q.groupby("cls").agg(qp=("makespan", "mean"), qp_ev=("evals", "mean")).reset_index()
        ev = d[d.method.isin(["rand", "ql"])].groupby(["cls", "method"]).evals.mean().unstack().reset_index()
        w = w.merge(qm, on="cls").merge(ev.rename(columns={"rand": "rand_ev", "ql": "ql_ev"}), on="cls")
        w["fam"] = w.cls.str.split("|").str[0]
        w["grp"] = w.cls.str.split("|").str[1].astype(int)
        assert len(w) == len(info["test"]), (sp, len(w), len(info["test"]))
        W[sp] = w
        evr[sp] = (float((w.qp_ev / w.rand_ev).mean()), float((w.ql_ev / w.rand_ev).mean()))
        for (f, g), x in w.groupby(["fam", "grp"]):
            cells.append(dict(space=sp, fam=f, grp=g, n=len(x), folga=float(-rel(x.oracle, x.rand).mean()), g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean())))
            a = test(x.qp, x.rand); a.update(id=f"A-{f}-{g}-{sp}"); TA.append(a)
            b = test(x.val, x.qp); b.update(id=f"B-{f}-{g}-{sp}"); TB.append(b)
    C = pd.DataFrame(cells)
    FA, FB = family(TA), family(TB)
    # ---- H' (as hipoteses H1-H3b da Entrada 190 com o ql_perm)
    specs = [("H1p", "S15", lambda w: (w.fam == "C") & w.grp.isin([1, 2]), -0.10, "ate"), ("H2p", "S5", lambda w: (w.fam == "C") & w.grp.isin([1, 2]), -0.05, "ate"),
             ("H3ap", "S5", lambda w: w.fam == "E", 0.0, "pior"), ("H3bp", "S15", lambda w: w.fam == "E", 0.0, "pior")]
    hrows = []
    for hid, sp, m, thr, mode in specs:
        x = W[sp][m(W[sp])]
        t = test(x.qp, x.rand); t.update(id=hid, space=sp, thr=thr, mode=mode)
        hrows.append(t)
    H = pd.DataFrame(hrows)
    h = holm_bonferroni(H.p_raw.tolist(), alpha=0.05)
    H["p_holm"], H["reject"] = h["p_adjusted"], h["reject"]
    H["confirmada_h"] = [bool(r.reject and ((r.mean_d_pct <= r.thr) if r["mode"] == "ate" else (r.mean_d_pct > 0))) for _, r in H.iterrows()]
    rho = sstats.spearmanr(C.folga, C.g_qp)
    rho_ql = sstats.spearmanr(C.folga, C.g_ql)
    yn = lambda b: "SIM" if b else "NAO"
    Hc = H.set_index("id").confirmada_h
    n_a = int(beyond(FA).sum())
    n_b = int(FB.confirmada.sum())
    L += ["\n--- celulas (familia x grupo x espaco): folga do oraculo e ganhos contra o aleatorio (pontos; positivo = melhor) ---", C.round(3).to_string(index=False),
          "\n--- A (ql_perm contra o aleatorio; Holm sobre 18) ---", FA.round(4).to_string(index=False),
          "\n--- H' (hipoteses H1-H3b com o ql_perm) ---", H[["id", "space", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "thr", "mode", "confirmada_h"]].round(4).to_string(index=False),
          f"\nSpearman(folga do oraculo, ganho do ql_perm) em {len(C)} celulas: rho = {rho.statistic:+.3f} (p = {rho.pvalue:.3g}); com o ql padrao: rho = {rho_ql.statistic:+.3f} (p = {rho_ql.pvalue:.3g})",
          f"razao de avaliacoes ao aleatorio (media das instancias): ql_perm S5 {evr['S5'][0]:.3f}, S15 {evr['S15'][0]:.3f}; ql padrao S5 {evr['S5'][1]:.3f}, S15 {evr['S15'][1]:.3f}",
          f"ganho medio nas 18 celulas: validada {C.g_val.mean():.3f}; ql padrao {C.g_ql.mean():.3f}; ql_perm {C.g_qp.mean():.3f}",
          "\n--- PREVISOES ---",
          f"(U1) H1' confirmada: {yn(Hc['H1p'])} (previsto NAO): {yn(not Hc['H1p'])}",
          f"(U2) H2' confirmada: {yn(Hc['H2p'])} (previsto NAO): {yn(not Hc['H2p'])}",
          f"(U3) celulas em que o ql_perm ganha alem da margem: {n_a} de 18 (previsto <= 1): {yn(n_a <= 1)}",
          f"(U4) Spearman do ql_perm: {rho.statistic:+.3f} (previsto < +0,30): {yn(rho.statistic < 0.30)}",
          f"(U5) a validada bate o ql_perm: {n_b} de 18 (previsto >= 15): {yn(n_b >= 15)}",
          f"(U6) H3a' e H3b' confirmadas (ql_perm pior que o aleatorio na familia E): {yn(Hc['H3ap'] and Hc['H3bp'])} (S5 {Hc['H3ap']}, S15 {Hc['H3bp']}) (previsto SIM): {yn(bool(Hc['H3ap'] and Hc['H3bp']))}",
          f"(U7) razao de avaliacoes do ql_perm >= 0,9 nos dois espacos: {yn(min(evr['S5'][0], evr['S15'][0]) >= 0.9)} ({evr['S5'][0]:.3f}, {evr['S15'][0]:.3f})"]
    pred = [not Hc["H1p"], not Hc["H2p"], n_a <= 1, rho.statistic < 0.30, n_b >= 15, bool(Hc["H3ap"] and Hc["H3bp"]), min(evr["S5"][0], evr["S15"][0]) >= 0.9]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada264_qperm_pub.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([FA, FB]).to_csv(R / "entrada264_qperm_pub_tests.csv", index=False)
    H.to_csv(R / "entrada264_qperm_pub_h.csv", index=False)
    C.to_csv(R / "entrada264_qperm_pub_cells.csv", index=False)


if __name__ == "__main__":
    main()
