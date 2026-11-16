"""
analysis/report_entrada280_tiesall.py
=====================================
Entrada 280 (Ciclo 48): O Q-LEARNING COM DESEMPATE ALEATORIO NAS 85 CELULAS DO 'INVARIANTE A ORDEM'. Entrada 279 (B15 e B30): o Q-Learning padrao com desempate aleatorio a cada decisao (ql_rt) ganha do aleatorio alem da margem em 12 de 14 celulas, enquanto o ql_perm (permutacao por execucao;
Entradas 260-267) perde em todas: o ql_perm nao era 'invariante a ordem' no sentido relevante, e a afirmacao 'o Q-Learning invariante a ordem nao ganha em nenhuma das 85 celulas' precisa ser refeita com o desempate aleatorio. Esta entrada roda o ql_rt em TODAS as colecoes das 85 celulas:
o benchmark padrao (B5 aqui, B15 e B30 da Entrada 279; 21 celulas = 7 tipos x 3 espacos), as familias publicadas de R||Cmax (6 celulas: 3 familias x S5/S15), o P||Cmax sintetico (24: 12 classes x S5/S15), o P||Cmax publicado held-out (18: familia x tercil x espaco) e o ILS do TSP (16: 8 classes x T5/T15),
com as mesmas instancias e os mesmos hiperparametros do Q-Learning padrao de cada bateria (experiment/run_cycle48_tiesall_entrada280.py; sementes proprias). Mesmos testes das Entradas 260-267 (Wilcoxon pareado bilateral, IC90 bootstrap, equivalencia em +-0,05, Holm sobre as celulas de cada colecao; 'alem da margem' = confirmada e nao equivalente);
ganho = -d sobre o aleatorio a iteracoes iguais. Spearman entre a folga do oraculo de uma semente e o ganho: nas 18 celulas publicadas (as mesmas da Entrada 264).

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada280_prereg_hashes.txt; nenhum resultado do ql_rt fora de B15 e B30 (Entrada 279) foi visto; o ql padrao, o ql_perm, o aleatorio e a validada vem das Entradas 180, 190, 254, 256, 260, 261, 264 e 267):
(S1) Em B5 o ql_rt ganha do aleatorio alem da margem em pelo menos 4 dos 7 tipos. (60%)
(S2) Nas familias publicadas de R||Cmax o ql_rt ganha alem da margem em pelo menos 4 das 6 celulas. (65%)
(S3) Nas familias o maior valor de ql_rt / validada e' de no maximo 0,45. (65%)
(S4) Nas 18 celulas publicadas held-out de P||Cmax a correlacao de Spearman entre a folga de uma semente e o ganho do ql_rt e' de pelo menos 0,50. (60%)
(S5) No P||Cmax sintetico o ql_rt ganha alem da margem em no maximo 6 das 24 celulas. (60%)
(S6) No TSP o ql_rt ganha alem da margem em pelo menos 6 das 16 celulas. (50%)
(S7) No total das 85 celulas o ql_rt ganha alem da margem em pelo menos 30. (60%)
(S8) A validada bate o ql_rt (confirmada) em pelo menos 75 das 85 celulas. (65%)
Uso: python -m analysis.report_entrada280_tiesall
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

R = ROOT / "results"
TYPES = ["U1a100", "U10a100", "U100a120", "U100a200", "U1000a1100", "JobsCorre", "MaqCorre"]
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def cell_tests(x, tag, TA, TB):
    a = test(x.rt, x.rand); a.update(id=f"A-{tag}"); TA.append(a)
    b = test(x.val, x.rt); b.update(id=f"B-{tag}"); TB.append(b)


def main():
    L = ["=" * 110, "O Q-LEARNING COM DESEMPATE ALEATORIO NAS 85 CELULAS (Entrada 280)", "=" * 110]
    out = {}
    # ---------------------------------------------------------------- benchmark padrao (21 celulas)
    cells, TA, TB = [], [], []
    for sp in ("B5", "B15", "B30"):
        d = read(f"fr_test_{sp}_s*of4_entrada256.csv")
        w = d.groupby(["type", "instance", "method"]).makespan.mean().unstack("method").reset_index()
        q = read(f"qperm_fanjul_{sp}_s*of*_entrada260.csv").groupby(["type", "instance"]).makespan.mean().rename("qp").reset_index()
        rt = read("ties280_b5_B5_s*of12_entrada280.csv" if sp == "B5" else f"ties_fanjul_{sp}_s*of12_entrada279.csv")
        rt = rt[rt.method == "ql_rt"].groupby(["type", "instance"]).makespan.mean().rename("rt").reset_index()
        w = w.merge(q, on=["type", "instance"]).merge(rt, on=["type", "instance"])
        assert len(w) == 700, (sp, len(w))
        for ty in TYPES:
            x = w[w.type == ty]
            cells.append(dict(coll="padrao", cell=f"{sp}|{ty}", g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean()), g_rt=float(-rel(x.rt, x.rand).mean())))
            cell_tests(x, f"{sp}|{ty}", TA, TB)
    out["padrao"] = (pd.DataFrame(cells), family(TA), family(TB))
    # ---------------------------------------------------------------- familias publicadas de R||Cmax (6)
    cells, TA, TB = [], [], []
    for sp in ("S5", "S15"):
        d = read(f"ur_ext_test_{sp}_s*of*_entrada254.csv")
        d = d[d.method.isin(["rand", "ql", "val"])]
        w = d.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
        q = read(f"qperm_families_{sp}_s*of*_entrada260.csv").groupby(["cls", "rep"]).makespan.mean().rename("qp").reset_index()
        rt = read(f"ties280_families_{sp}_s*of12_entrada280.csv").groupby(["cls", "rep"]).makespan.mean().rename("rt").reset_index()
        w = w.merge(q, on=["cls", "rep"]).merge(rt, on=["cls", "rep"])
        w["fam"] = w.cls.str.split("|").str[0]
        for fam in ("unc", "cj", "cm"):
            x = w[w.fam == fam]
            cells.append(dict(coll="familias", cell=f"{sp}|{fam}", g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean()), g_rt=float(-rel(x.rt, x.rand).mean())))
            cell_tests(x, f"{sp}|{fam}", TA, TB)
    out["familias"] = (pd.DataFrame(cells), family(TA), family(TB))
    # ---------------------------------------------------------------- P||Cmax sintetico (24)
    cells, TA, TB = [], [], []
    for sp in ("S5", "S15"):
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        w = d.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
        q = read(f"qperm_pcmax_{sp}_s*of*_entrada261.csv").groupby(["cls", "rep"]).makespan.mean().rename("qp").reset_index()
        rt = read(f"ties280_pcmax_{sp}_s*of12_entrada280.csv").groupby(["cls", "rep"]).makespan.mean().rename("rt").reset_index()
        w = w.merge(q, on=["cls", "rep"]).merge(rt, on=["cls", "rep"])
        for cl in sorted(w.cls.unique()):
            x = w[w.cls == cl]
            cells.append(dict(coll="sintetico", cell=f"{sp}|{cl}", g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean()), g_rt=float(-rel(x.rt, x.rand).mean())))
            cell_tests(x, f"{sp}|{cl}", TA, TB)
    out["sintetico"] = (pd.DataFrame(cells), family(TA), family(TB))
    # ---------------------------------------------------------------- P||Cmax publicado held-out (18)
    info = json.loads((R / "pubh_groups_entrada190.json").read_text())
    cells, TA, TB = [], [], []
    for sp in ("S5", "S15"):
        d = read(f"pubh_{sp}_s*of*_entrada190.csv")
        w = per_instance(d)
        q = read(f"qpermpub_{sp}_s*of*_entrada264.csv").groupby("cls").makespan.mean().rename("qp").reset_index()
        rt = read(f"ties280_pub_{sp}_s*of12_entrada280.csv").groupby("cls").makespan.mean().rename("rt").reset_index()
        w = w.merge(q, on="cls").merge(rt, on="cls")
        assert len(w) == len(info["test"]), (sp, len(w), len(info["test"]))
        w["fam"] = w.cls.str.split("|").str[0]
        w["grp"] = w.cls.str.split("|").str[1].astype(int)
        for (f, g), x in w.groupby(["fam", "grp"]):
            cells.append(dict(coll="publicado", cell=f"{sp}|{f}|{g}", folga=float(-rel(x.oracle, x.rand).mean()), g_val=float(-rel(x.val, x.rand).mean()), g_ql=float(-rel(x.ql, x.rand).mean()), g_qp=float(-rel(x.qp, x.rand).mean()), g_rt=float(-rel(x.rt, x.rand).mean())))
            cell_tests(x, f"{sp}|{f}|{g}", TA, TB)
    out["publicado"] = (pd.DataFrame(cells), family(TA), family(TB))
    # ---------------------------------------------------------------- TSP (16)
    d = read("tsp_test_s*of*_entrada267.csv")
    ch = json.loads((R / "tsp_choice_entrada267.json").read_text())
    rtd = read("ties280_tsp_s*of12_entrada280.csv")
    key = ["inst", "kind", "n"]
    arms = d[d.method.str.startswith("arm")].copy()
    arms["a"] = arms.method.str[3:].astype(int)
    WA = arms.groupby(key + ["a"]).length.mean().unstack("a")
    WA.columns = [f"arm{c}" for c in WA.columns]
    WA = WA.reset_index()
    cells, TA, TB = [], [], []
    for sp in ("T5", "T15"):
        x = d[(d.space == sp) & ~d.method.str.startswith("arm")]
        w = x.groupby(key + ["method"]).length.mean().unstack("method").reset_index().merge(WA, on=key)
        w["cls"] = w.kind + "|" + w.n.astype(str)
        w["val"] = [float(r[f"arm{ch[sp][r.cls]}"]) for _, r in w.iterrows()]
        w = w.merge(rtd[rtd.space == sp].groupby(key).length.mean().rename("rt").reset_index(), on=key)
        for (k, n), g in w.groupby(["kind", "n"]):
            cells.append(dict(coll="tsp", cell=f"{sp}|{k}|{n}", g_val=float(-rel(g.val, g.rand).mean()), g_ql=float(-rel(g.ql, g.rand).mean()), g_qp=float(-rel(g.ql_perm, g.rand).mean()), g_rt=float(-rel(g.rt, g.rand).mean())))
            cell_tests(g, f"{sp}|{k}|{n}", TA, TB)
    out["tsp"] = (pd.DataFrame(cells), family(TA), family(TB))
    # ---------------------------------------------------------------- resumo
    summ, tot_a, tot_b, tot_n = [], 0, 0, 0
    for k, (C, FA, FB) in out.items():
        na, nb = int(beyond(FA).sum()), int(FB.confirmada.sum())
        tot_a, tot_b, tot_n = tot_a + na, tot_b + nb, tot_n + len(C)
        summ.append(dict(colecao=k, celulas=len(C), ql_padrao=C.g_ql.mean(), ql_perm=C.g_qp.mean(), ql_rt=C.g_rt.mean(), validada=C.g_val.mean(), rt_alem_da_margem=na, validada_bate_rt=nb,
                         rt_sobre_validada_max=float((C.g_rt / C.g_val).max())))
    S = pd.DataFrame(summ)
    L += ["--- resumo por colecao: ganho medio sobre o aleatorio (pontos) e celulas ---", S.round(3).to_string(index=False), "",
          f"total: {tot_n} celulas; o ql_rt ganha do aleatorio alem da margem em {tot_a}; a validada bate o ql_rt em {tot_b}", ""]
    for k, (C, FA, FB) in out.items():
        C2 = C.copy()
        C2["rt_beyond"] = beyond(FA).to_numpy()
        C2["val_beats_rt"] = FB.confirmada.to_numpy()
        L += [f"--- {k}: celulas ---", C2.round(3).to_string(index=False), ""]
    Cp = out["publicado"][0]
    rho = {m: sstats.spearmanr(Cp.folga, Cp[f"g_{m}"]) for m in ("ql", "qp", "rt")}
    L += ["--- publicado held-out (18 celulas): Spearman entre a folga do oraculo de uma semente e o ganho ---", "; ".join(f"{m}: {rho[m].statistic:+.3f} (p = {rho[m].pvalue:.3g})" for m in rho), ""]
    Ca = pd.concat([v[0] for v in out.values()])
    # ---------------------------------------------------------------- previsoes
    nb5 = int(beyond(out["padrao"][1]).sum())
    s1 = int(beyond(out["padrao"][1]).to_numpy()[:7].sum())
    s2 = int(beyond(out["familias"][1]).sum())
    s3 = float(out["familias"][0].pipe(lambda c: (c.g_rt / c.g_val).max()))
    s4 = float(rho["rt"].statistic)
    s5 = int(beyond(out["sintetico"][1]).sum())
    s6 = int(beyond(out["tsp"][1]).sum())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- previsoes ---",
          f"(S1) tipos de B5 em que o ql_rt ganha do aleatorio alem da margem: {s1} de 7 (previsto >= 4): {yn(s1 >= 4)}",
          f"(S2) celulas das familias em que o ql_rt ganha alem da margem: {s2} de 6 (previsto >= 4): {yn(s2 >= 4)}",
          f"(S3) maior ql_rt / validada nas familias: {s3:.3f} (previsto <= 0,45): {yn(s3 <= 0.45)}",
          f"(S4) Spearman folga-ganho do ql_rt nas 18 celulas publicadas: {s4:+.3f} (previsto >= 0,50): {yn(s4 >= 0.5)}",
          f"(S5) celulas do P||Cmax sintetico em que o ql_rt ganha alem da margem: {s5} de 24 (previsto <= 6): {yn(s5 <= 6)}",
          f"(S6) celulas do TSP em que o ql_rt ganha alem da margem: {s6} de 16 (previsto >= 6): {yn(s6 >= 6)}",
          f"(S7) celulas (de 85) em que o ql_rt ganha alem da margem: {tot_a} (previsto >= 30): {yn(tot_a >= 30)}",
          f"(S8) celulas (de 85) em que a validada bate o ql_rt: {tot_b} (previsto >= 75): {yn(tot_b >= 75)}"]
    pred = [s1 >= 4, s2 >= 4, s3 <= 0.45, s4 >= 0.5, s5 <= 6, s6 >= 6, tot_a >= 30, tot_b >= 75]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada280_tiesall.txt").write_text(text + "\n", encoding="utf-8")
    Ca.to_csv(R / "entrada280_tiesall_cells.csv", index=False)
    pd.concat([pd.concat([v[1], v[2]]).assign(coll=k) for k, v in out.items()]).to_csv(R / "entrada280_tiesall_tests.csv", index=False)
    S.to_csv(R / "entrada280_tiesall_summary.csv", index=False)


if __name__ == "__main__":
    main()
