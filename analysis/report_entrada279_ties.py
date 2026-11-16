"""
analysis/report_entrada279_ties.py
==================================
Entrada 279 (Ciclo 47): APRENDIZES SEM TRAVAMENTO NA PRIMEIRA ACAO E A ORDEM DAS ACOES. Resposta as revisoes cegas de 15/11/2026 (os dois revisores): (a) o Q-Learning 'invariante a ordem' (ql_perm) fixa UMA acao aleatoria por execucao, nao desempata ao acaso a cada decisao, e perde para o
aleatorio; logo '0 de 85 celulas' mostra que um aprendiz travado e' ruim, nao que aprender nao paga; (b) UCB1, gradiente de politica, roleta, Q-Learning episodico e FRRMAB nunca foram repetidos com a ordem permutada; (c) a relacao folga-ganho nunca foi
vista para aprendizes sem travamento. Execucao: experiment/run_cycle47_ties_entrada279.py (700 instancias de teste do benchmark de Fanjul-Peyro e Ruiz, B15 e B30, 3 sementes, T = 1.000; os mesmos hiperparametros congelados da Entrada 256; sementes proprias):
ql_rt (Q-Learning padrao com desempate aleatorio a cada decisao), qo_rt (Q-Learning otimista com desempate aleatorio), qo_perm (otimista com as acoes permutadas por execucao), ucb_perm, pg_perm, alns_perm, iq_perm, fr_perm.
Unidade = media das 3 sementes por instancia; celulas = os 7 tipos (100 instancias cada) em cada espaco; ganho = -d sobre o aleatorio (d = 100 (A - B) / B); testes como nas Entradas 180-278 (Wilcoxon pareado bilateral, IC90 bootstrap, Holm sobre os 7 tipos, equivalencia a 0 em +-0,05;
'alem da margem' = confirmada e nao equivalente). 'Difere' = rejeita H0 (Holm) e nao e' equivalente. Folga por classe: HCF (acao estatica escolhida nas outras duas sementes e avaliada na terceira, media dos tres rodizios), como na Entrada 266; a correlacao de postos de Spearman
entre a folga e o ganho de cada aprendiz e' calculada com a CLASSE (tipo, n; 28 em B30) como unidade, com IC90 por bootstrap das classes.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada279_prereg_hashes.txt; a fumaca usou instancias de AJUSTE; nenhum resultado das variantes no teste foi visto; os resultados de rand, ql, ucb, pg, alns, qo, iq, fr, val e ql_perm sao os das Entradas 256 e 260):
(R1) ql_rt ganha do aleatorio alem da margem em no maximo 1 das 14 celulas (B15 e B30; 7 tipos cada). (70%)
(R2) O ganho medio do ql_rt sobre o aleatorio em B30 (media dos 7 tipos) e' negativo. (65%)
(R3) qo_rt e' equivalente ao qo (IC90 dentro de +-0,05) em pelo menos 10 das 14 celulas. (50%)
(R4) O ganho medio do qo em B30 excede o do qo_perm em pelo menos 0,05 ponto (o otimista depende em parte da ordem). (50%)
(R5) ucb_perm, pg_perm e alns_perm sao, cada um, equivalentes ao seu aprendiz sem permutacao em pelo menos 5 dos 7 tipos de B30 (os tres). (55%)
(R6) iq_perm DIFERE do iq (alem da margem) em pelo menos 3 dos 7 tipos de B30. (50%)
(R7) Nenhuma das 8 variantes bate a acao validada alem da margem em nenhum dos 112 pares (8 variantes x 14 celulas). (85%)
(R8) Em B30 a correlacao de Spearman entre a folga por classe (28 classes) e o ganho do qo_rt e' pelo menos 0,50. (50%)
(R9) Em B30 a correlacao de Spearman entre a folga por classe e o ganho do ql_rt e' no maximo 0,30. (55%)
Uso: python -m analysis.report_entrada279_ties
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
TYPES = ["U1a100", "U10a100", "U100a120", "U100a200", "U1000a1100", "JobsCorre", "MaqCorre"]
NARMS = {"B15": 15, "B30": 30}
NEW = ["ql_rt", "qo_rt", "qo_perm", "ucb_perm", "pg_perm", "alns_perm", "iq_perm", "fr_perm"]
BASE = {"ql_rt": "ql", "qo_rt": "qo", "qo_perm": "qo", "ucb_perm": "ucb", "pg_perm": "pg", "alns_perm": "alns", "iq_perm": "iq", "fr_perm": "fr"}
OLD = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr", "ql_perm"]
pd.set_option("display.width", 230)
KEY = ["type", "instance"]


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def hcf_by_instance(arms, na):
    a2 = arms[arms.a < na].copy()
    a2["cls"] = a2["type"] + "|" + a2.n.astype(str)
    a2["ratio"] = a2.makespan / a2.bks
    out = []
    for cl, g in a2.groupby("cls"):
        vals = []
        for s in range(3):
            tr, ev = g[g.seed_i != s], g[g.seed_i == s]
            best = int(tr.groupby("a").ratio.mean().idxmin())
            vals.append(ev[ev.a == best].set_index(KEY).makespan)
        out.append(pd.concat(vals, axis=1).mean(axis=1))
    return pd.concat(out).rename("HCF")


def main():
    arms = read("arms_test_B30_s*of*_entrada266.csv")
    arms["a"] = arms.arm.astype(int)
    L = ["=" * 110, "APRENDIZES SEM TRAVAMENTO NA PRIMEIRA ACAO E A ORDEM DAS ACOES (Entrada 279)", "=" * 110]
    cells_all, tests_all, S = [], [], {}
    for sp in ("B30", "B15"):
        fr = read(f"fr_test_{sp}_s*of4_entrada256.csv")
        W = fr.groupby(KEY + ["method"]).makespan.mean().unstack("method").reset_index()
        meta = fr.drop_duplicates(KEY)[KEY + ["n"]]
        W = W.merge(meta, on=KEY)
        qp = read(f"qperm_fanjul_{sp}_s*of*_entrada260.csv").groupby(KEY).makespan.mean().rename("ql_perm").reset_index()
        nw = read(f"ties_fanjul_{sp}_s*of12_entrada279.csv").groupby(KEY + ["method"]).makespan.mean().unstack("method").reset_index()
        W = W.merge(qp, on=KEY).merge(nw, on=KEY)
        assert len(W) == 700, len(W)
        rows, TA, TB, TC = [], [], [], []
        for t in TYPES:
            x = W[W["type"] == t]
            rows.append(dict(space=sp, type=t, **{f"g_{k}": float(-rel(x[k], x.rand).mean()) for k in OLD + NEW}, g_val=float(-rel(x.val, x.rand).mean())))
            for v in NEW:
                a = test(x[v], x.rand); a.update(id=f"A-{v}-{t}-{sp}"); TA.append(a)
                b = test(x[v], x[BASE[v]]); b.update(id=f"B-{v}-{t}-{sp}"); TB.append(b)
                c = test(x[v], x.val); c.update(id=f"C-{v}-{t}-{sp}"); TC.append(c)
        C = pd.DataFrame(rows)
        FA = pd.concat([family([r for r in TA if r["id"].split("-")[1] == v]) for v in NEW], ignore_index=True)
        FB = pd.concat([family([r for r in TB if r["id"].split("-")[1] == v]) for v in NEW], ignore_index=True)
        FC = pd.concat([family([r for r in TC if r["id"].split("-")[1] == v]) for v in NEW], ignore_index=True)
        cells_all.append(C)
        tests_all += [FA, FB, FC]
        mean_g = {k: float(C[f"g_{k}"].mean()) for k in OLD + NEW + ["val"]}
        vid = lambda F_, v: F_[F_.id.str.split("-").str[1] == v]
        L += [f"--- {sp}: ganho medio sobre o aleatorio (pontos; media dos 7 tipos) ---",
              "aprendizes originais: " + "; ".join(f"{k} {mean_g[k]:.3f}" for k in OLD) + f"; validada {mean_g['val']:.3f}",
              "variantes novas:      " + "; ".join(f"{k} {mean_g[k]:.3f}" for k in NEW), "",
              C.round(3).to_string(index=False), "",
              "variantes que ganham do aleatorio alem da margem (de 7 tipos): " + "; ".join(f"{v} {int(beyond(vid(FA, v)).sum())}" for v in NEW),
              "variantes equivalentes ao aprendiz de base (de 7 tipos): " + "; ".join(f"{v}~{BASE[v]} {int(vid(FB, v)['equivalente_0.05'].sum())}" for v in NEW),
              "variantes que DIFEREM do aprendiz de base alem da margem (de 7 tipos): " + "; ".join(f"{v} {int((vid(FB, v).reject & ~vid(FB, v)['equivalente_0.05']).sum())}" for v in NEW),
              "variantes que batem a validada alem da margem (de 7 tipos): " + "; ".join(f"{v} {int(beyond(vid(FC, v)).sum())}" for v in NEW), ""]
        S[sp] = dict(C=C, W=W, FA=FA, FB=FB, FC=FC, mean_g=mean_g)
    # ---- folga por classe e correlacao de postos (B30; classe = unidade)
    W = S["B30"]["W"]
    H = hcf_by_instance(arms, 30)
    W = W.merge(H.reset_index(), on=KEY)
    W["cls"] = W["type"] + "|" + W.n.astype(str)
    cls = sorted(W.cls.unique())
    gain_cls = {k: np.array([float(-rel(W[W.cls == c][k], W[W.cls == c].rand).mean()) for c in cls]) for k in OLD + NEW}
    head = np.array([float(-rel(W[W.cls == c].HCF, W[W.cls == c].rand).mean()) for c in cls])
    rng = np.random.default_rng(0)
    corr = {}
    for k in OLD + NEW:
        rho = float(sstats.spearmanr(head, gain_cls[k]).statistic)
        bs = []
        for _ in range(2000):
            ix = rng.integers(0, len(cls), len(cls))
            bs.append(sstats.spearmanr(head[ix], gain_cls[k][ix]).statistic)
        lo, hi = np.nanpercentile(bs, [5, 95])
        corr[k] = (rho, float(lo), float(hi), int((gain_cls[k] > 0.05).sum()))
    CR = pd.DataFrame([dict(learner=k, spearman=v[0], ci90_lo=v[1], ci90_hi=v[2], classes_gain_gt_0_05=v[3]) for k, v in corr.items()])
    L += [f"--- B30: correlacao de postos entre a folga por classe (HCF) e o ganho do aprendiz, {len(cls)} classes (IC90 por bootstrap das classes; 'classes_gain_gt_0_05' = classes com ganho > 0,05) ---", CR.round(3).to_string(index=False), ""]
    # ---- previsoes
    cnt = lambda F_, v, f: int(f(vid2(F_, v)).sum())
    vid2 = lambda F_, v: F_[F_.id.str.split("-").str[1] == v]
    both = ("B30", "B15")
    r1 = sum(int(beyond(vid2(S[sp]["FA"], "ql_rt")).sum()) for sp in both)
    r2 = S["B30"]["mean_g"]["ql_rt"]
    r3 = sum(int(vid2(S[sp]["FB"], "qo_rt")["equivalente_0.05"].sum()) for sp in both)
    r4 = S["B30"]["mean_g"]["qo"] - S["B30"]["mean_g"]["qo_perm"]
    eq = lambda v: int(vid2(S["B30"]["FB"], v)["equivalente_0.05"].sum())
    r5 = [eq(v) for v in ("ucb_perm", "pg_perm", "alns_perm")]
    dif = lambda F_, v: int((vid2(F_, v).reject & ~vid2(F_, v)["equivalente_0.05"]).sum())
    r6 = dif(S["B30"]["FB"], "iq_perm")
    r7 = sum(int(beyond(vid2(S[sp]["FC"], v)).sum()) for sp in both for v in NEW)
    r8, r9 = corr["qo_rt"][0], corr["ql_rt"][0]
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- previsoes ---",
          f"(R1) celulas em que o ql_rt ganha do aleatorio alem da margem: {r1} de 14 (previsto <= 1): {yn(r1 <= 1)}",
          f"(R2) ganho medio do ql_rt em B30: {r2:.3f} (previsto < 0): {yn(r2 < 0)}",
          f"(R3) celulas em que o qo_rt e' equivalente ao qo: {r3} de 14 (previsto >= 10): {yn(r3 >= 10)}",
          f"(R4) ganho medio do qo menos o do qo_perm em B30: {r4:.3f} (previsto >= 0,05): {yn(r4 >= 0.05)}",
          f"(R5) tipos de B30 em que ucb_perm, pg_perm, alns_perm sao equivalentes ao original: {r5} (previsto >= 5 em cada um): {yn(all(x >= 5 for x in r5))}",
          f"(R6) tipos de B30 em que o iq_perm difere do iq alem da margem: {r6} de 7 (previsto >= 3): {yn(r6 >= 3)}",
          f"(R7) pares (variante, celula) em que a variante bate a validada alem da margem: {r7} de 112 (previsto 0): {yn(r7 == 0)}",
          f"(R8) Spearman folga-ganho do qo_rt em B30: {r8:.3f} (previsto >= 0,50): {yn(r8 >= 0.5)}",
          f"(R9) Spearman folga-ganho do ql_rt em B30: {r9:.3f} (previsto <= 0,30): {yn(r9 <= 0.30)}"]
    pred = [r1 <= 1, r2 < 0, r3 >= 10, r4 >= 0.05, all(x >= 5 for x in r5), r6 >= 3, r7 == 0, r8 >= 0.5, r9 <= 0.30]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada279_ties.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat(cells_all).to_csv(R / "entrada279_ties_cells.csv", index=False)
    pd.concat(tests_all).to_csv(R / "entrada279_ties_tests.csv", index=False)
    CR.to_csv(R / "entrada279_ties_corr.csv", index=False)


if __name__ == "__main__":
    main()
