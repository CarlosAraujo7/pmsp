"""
analysis/report_entrada266_granularity.py
==========================================
Entrada 266 (Ciclo 39): O VALOR DO CONHECIMENTO OFFLINE POR GRANULARIDADE contra o aprendizado online, no benchmark de Fanjul-Peyro e Ruiz (700 instancias de teste, T = 1.000, 3 sementes; espacos B5, B15, B30; o B30 e' o principal).
Nas Entradas 180-264 o competidor estatico era a acao validada POR CLASSE (tipo, n), isto e', conhecimento offline com a granularidade maxima. Aqui as 30 acoes fixas foram rodadas no teste (experiment/run_cycle39_arms_entrada266.py) e
politicas estaticas sao escolhidas SO' nas instancias de ajuste (menor razao media makespan / melhor conhecido) com granularidades decrescentes de conhecimento: G0 (uma acao global), Gn (por tamanho n), Gt (por tipo), Gtn (por classe =
a validada da Entrada 256). Regimes de TRANSFERENCIA: LOTO (a acao e' escolhida SEM os dados do tipo da instancia testada: G0 e Gn sobre os outros seis tipos) e LOSO (SEM os dados do tamanho n testado: G0 sobre os outros tres tamanhos e Gt
por tipo sobre os outros tamanhos). Pergunta: a partir de que quantidade de conhecimento offline uma acao estatica vence os aprendizes online (os sete da Entrada 256, o Q-Learning invariante a ordem, a politica neural treinada offline)?
Unidade = media das 3 sementes por instancia; celulas = os 7 tipos (100 instancias de teste cada); ganho = -d sobre o aleatorio (pontos); d = 100 (A - B) / B; testes como nas Entradas 180-265 (Wilcoxon pareado bilateral, IC90 bootstrap,
equivalencia a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente a zero). Familias (Holm sobre os 7 tipos): A (politica contra o aleatorio), B' (aprendiz contra a politica estatica; positivo/confirmada = o aprendiz bate a estatica).
HCF = folga estatica por classe com validacao cruzada de sementes NO TESTE (escolhe a acao nas outras duas sementes da classe e avalia na terceira; media dos tres rodizios): o teto do que uma acao estatica por classe pode dar.

PRE-REGISTRO (congelado ANTES de rodar as acoes no teste; hashes em results/entrada266_prereg_hashes.txt; as acoes fixas nao foram vistas no teste; as escolhas vem so' do ajuste; os resultados dos aprendizes sao os das Entradas 256, 260 e 262)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(K1) Em B30, G0 ganha do aleatorio alem da margem (A) em pelo menos 5 dos 7 tipos. (65%)
(K2) Em B30, o ganho medio de G0 nos 7 tipos e' de pelo menos 50% do de Gtn (a validada por classe). (60%)
(K3) Em B30, o numero de pares (aprendiz, tipo) em que um dos 7 aprendizes online bate G0 alem da margem (B') e' de no maximo 3 de 49. (60%)
(K4) Em B30, ordem de granularidade: o ganho medio de Gtn >= o de Gt >= o de G0 - 0,05 e o de Gtn >= o de Gn >= o de G0 - 0,05. (70%)
(K5) Em B30, LOTO-G0 ganha do aleatorio alem da margem em pelo menos 5 dos 7 tipos. (60%)
(K6) Em B30, o melhor aprendiz online de cada tipo bate LOTO-G0 alem da margem em no maximo 2 dos 7 tipos. (60%)
(K7) Em B30, LOSO-G0 ganha do aleatorio alem da margem em pelo menos 3 dos 4 tamanhos. (65%)
(K8) Em B30, a folga estatica por classe com validacao cruzada no teste (HCF) excede o ganho de Gtn em no maximo 0,30 ponto (media dos 7 tipos). (55%)
(K9) Em B30, a politica neural (nn) bate G0 alem da margem em no maximo 2 dos 7 tipos. (60%)
Uso: python -m analysis.report_entrada266_granularity
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
LEARN = ["ql", "ucb", "pg", "alns", "qo", "iq", "fr"]
TYPES = ["U1a100", "U10a100", "U100a120", "U100a200", "U1000a1100", "JobsCorre", "MaqCorre"]
NSIZES = [100, 200, 500, 1000]
NARMS = {"B5": 5, "B15": 15, "B30": 30}
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def tuning(space):
    d = read(f"fr_tune_{space}_s*of*_entrada256.csv")
    d["type"] = d.cls.str.split("|").str[0]
    d["n"] = d.cls.str.split("|").str[1].astype(int)
    return d


def pick(d, by=None):
    """Acao de menor razao media (uma por grupo; uma so' se by is None)."""
    if by is None:
        return int(d.groupby("arm").ratio.mean().idxmin())
    return {k: int(g.groupby("arm").ratio.mean().idxmin()) for k, g in d.groupby(by)}


def main():
    arms = read("arms_test_B30_s*of*_entrada266.csv")
    arms["a"] = arms.arm.astype(int)
    key = ["type", "instance"]
    meta = arms.drop_duplicates(key + ["seed_i", "arm"])[key + ["n", "bks"]].drop_duplicates(key).set_index(key)
    WA = arms.groupby(key + ["a"]).makespan.mean().unstack("a")           # media das 3 sementes por instancia e acao
    WA.columns = [f"arm{c}" for c in WA.columns]
    assert len(WA) == 700 and WA.shape[1] == 30, WA.shape
    L = ["=" * 110, "O VALOR DO CONHECIMENTO OFFLINE POR GRANULARIDADE (Entrada 266)", "=" * 110]
    cells_all, tests_all, summ = [], [], {}
    for sp in ("B30", "B15", "B5"):
        na = NARMS[sp]
        T_ = tuning(sp)
        fr = read(f"fr_test_{sp}_s*of4_entrada256.csv")
        W = fr.groupby(key + ["method"]).makespan.mean().unstack("method").reset_index().merge(meta.reset_index(), on=key, suffixes=("", "_m"))
        qp = read(f"qperm_fanjul_{sp}_s*of*_entrada260.csv").groupby(key).makespan.mean().rename("ql_perm").reset_index()
        W = W.merge(qp, on=key)
        if sp in ("B15", "B30"):
            nn = read(f"nn_test_{sp}_s*of4_entrada262.csv").groupby(key + ["method"]).makespan.mean().unstack("method").reset_index()
            W = W.merge(nn, on=key)
        W = W.merge(WA.reset_index()[key + [f"arm{a}" for a in range(na)]], on=key)
        # ---- escolhas offline (so' do ajuste)
        g0 = pick(T_)
        gn = pick(T_, "n")
        gt = pick(T_, "type")
        gtn = pick(T_, "cls")
        val = json.loads((R / f"fr_choice_{sp}_entrada256.json").read_text())["val"]
        assert gtn == val, "a escolha por classe nao reproduz a validada da Entrada 256"
        loto0 = {t: pick(T_[T_.type != t]) for t in TYPES}
        lotoN = {t: pick(T_[T_.type != t], "n") for t in TYPES}
        loso0 = {n: pick(T_[T_.n != n]) for n in NSIZES}
        losoT = {n: pick(T_[T_.n != n], "type") for n in NSIZES}
        lk = lambda r, tab: f"arm{tab[r.n]}"
        pol = {"G0": lambda r: f"arm{g0}", "Gn": lambda r: f"arm{gn[r.n]}", "Gt": lambda r: f"arm{gt[r['type']]}", "Gtn": lambda r: f"arm{gtn[r['type'] + '|' + str(int(r.n))]}",
               "LOTO_G0": lambda r: f"arm{loto0[r['type']]}", "LOTO_Gn": lambda r: f"arm{lotoN[r['type']][r.n]}", "LOSO_G0": lambda r: f"arm{loso0[r.n]}", "LOSO_Gt": lambda r: f"arm{losoT[r.n][r['type']]}"}
        for name, f in pol.items():
            W[name] = [float(r[f(r)]) for _, r in W.iterrows()]
        # ---- folga estatica por classe com validacao cruzada de sementes no teste
        a2 = arms[arms.a < na].merge(meta.reset_index()[key + ["bks"]], on=key, suffixes=("", "_b"))
        a2["cls"] = a2["type"] + "|" + a2.n.astype(str)
        a2["ratio"] = a2.makespan / a2.bks
        hcf = {}
        for cl, g in a2.groupby("cls"):
            vals = []
            for s in range(3):
                tr, ev = g[g.seed_i != s], g[g.seed_i == s]
                best = int(tr.groupby("a").ratio.mean().idxmin())
                vals.append(ev[ev.a == best].set_index(key).makespan)
            hcf[cl] = pd.concat(vals, axis=1).mean(axis=1)
        H = pd.concat(hcf.values()).rename("HCF")
        W = W.merge(H.reset_index(), on=key)
        pols = list(pol)
        learners = LEARN + ["ql_perm"] + (["nn", "nn_const"] if sp in ("B15", "B30") else [])
        # ---- celulas por tipo
        TA, TB, rows = [], [], []
        for t in TYPES:
            x = W[W["type"] == t]
            rows.append(dict(space=sp, type=t, **{f"g_{p}": float(-rel(x[p], x.rand).mean()) for p in pols + ["HCF"]}, **{f"g_{k}": float(-rel(x[k], x.rand).mean()) for k in learners}, g_valrun=float(-rel(x.val, x.rand).mean())))
            for p in pols:
                a = test(x[p], x.rand); a.update(id=f"A-{p}-{t}-{sp}"); TA.append(a)
            for k in learners:
                for p in ("G0", "LOTO_G0", "Gtn"):
                    b = test(x[k], x[p]); b.update(id=f"Bp-{k}-{p}-{t}-{sp}"); TB.append(b)
        C = pd.DataFrame(rows)
        # famílias de Holm: A por politica (sobre os 7 tipos); B' por (aprendiz, politica) sobre os 7 tipos
        FA = pd.concat([family([r for r in TA if r["id"].split("-")[1] == p]) for p in pols], ignore_index=True)
        FB = pd.concat([family([r for r in TB if r["id"].split("-")[1] == k and r["id"].split("-")[2] == p]) for k in learners for p in ("G0", "LOTO_G0", "Gtn")], ignore_index=True)
        cells_all.append(C)
        tests_all += [FA, FB]
        # ---- tamanhos (LOSO por n): ganhos de G0 e LOSO_G0 por n
        bysize = []
        for n in NSIZES:
            x = W[W.n == n]
            a = test(x["LOSO_G0"], x.rand); a.update(id=f"A-LOSO_G0-n{n}-{sp}")
            b = test(x["G0"], x.rand); b.update(id=f"A-G0-n{n}-{sp}")
            bysize += [a, b]
        FS = family(bysize)
        tests_all.append(FS)
        mean_g = {p: float(C[f"g_{p}"].mean()) for p in pols + ["HCF"]}
        mean_l = {k: float(C[f"g_{k}"].mean()) for k in learners}
        L += [f"--- {sp}: ganhos medios por politica estatica e por aprendiz (pontos; media dos 7 tipos) ---",
              "estaticas: " + "; ".join(f"{p} {v:.3f}" for p, v in mean_g.items()),
              "aprendizes: " + "; ".join(f"{k} {v:.3f}" for k, v in mean_l.items()),
              f"melhor aprendiz online por tipo, media: {float(C[[f'g_{k}' for k in LEARN]].max(axis=1).mean()):.3f}", "",
              C.round(3).to_string(index=False), "",
              "politicas estaticas que ganham do aleatorio alem da margem (de 7 tipos): " + "; ".join(f"{p} {int(beyond(FA[FA.id.str.split('-').str[1] == p]).sum())}" for p in pols),
              "aprendizes que batem a estatica alem da margem (B'; de 7 tipos): " + "; ".join(f"{k}->{p}: {int(beyond(FB[(FB.id.str.split('-').str[1] == k) & (FB.id.str.split('-').str[2] == p)]).sum())}" for p in ("G0", "LOTO_G0", "Gtn") for k in learners), "",
              "por tamanho (G0 e LOSO_G0 contra o aleatorio):", FS[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), ""]
        summ[sp] = dict(C=C, W=W, FA=FA, FB=FB, FS=FS, mean_g=mean_g, mean_l=mean_l, pols=pols, learners=learners)
    # ---- previsoes (B30)
    s = summ["B30"]
    C, FA, FB, FS, W = s["C"], s["FA"], s["FB"], s["FS"], s["W"]
    cnt = lambda p: int(beyond(FA[FA.id.str.split("-").str[1] == p]).sum())
    k1 = cnt("G0")
    k2 = s["mean_g"]["G0"] / s["mean_g"]["Gtn"]
    pairs = sum(int(beyond(FB[(FB.id.str.split("-").str[1] == k) & (FB.id.str.split("-").str[2] == "G0")]).sum()) for k in LEARN)
    mg = s["mean_g"]
    k4 = mg["Gtn"] >= mg["Gt"] >= mg["G0"] - 0.05 and mg["Gtn"] >= mg["Gn"] >= mg["G0"] - 0.05
    k5 = cnt("LOTO_G0")
    # K6: o melhor aprendiz online de cada tipo contra LOTO_G0
    rows6 = []
    for t in TYPES:
        x = W[W["type"] == t]
        kb = max(LEARN, key=lambda k: float(-rel(x[k], x.rand).mean()))
        b = test(x[kb], x["LOTO_G0"]); b.update(id=f"K6-{kb}-{t}"); rows6.append(b)
    F6 = family(rows6)
    k6 = int(beyond(F6).sum())
    k7 = int(beyond(FS[FS.id.str.startswith("A-LOSO_G0")]).sum())
    k8v = float((C.g_HCF - C.g_Gtn).mean())
    rows9 = []
    for t in TYPES:
        x = W[W["type"] == t]
        b = test(x["nn"], x["G0"]); b.update(id=f"K9-nn-{t}"); rows9.append(b)
    F9 = family(rows9)
    k9 = int(beyond(F9).sum())
    tests_all += [F6, F9]
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- previsoes (B30) ---",
          f"(K1) tipos em que G0 ganha do aleatorio alem da margem: {k1} de 7 (previsto >= 5): {yn(k1 >= 5)}",
          f"(K2) ganho medio G0 / Gtn: {k2:.3f} (previsto >= 0,50): {yn(k2 >= 0.5)}",
          f"(K3) pares (aprendiz, tipo) em que um aprendiz online bate G0 alem da margem: {pairs} de 49 (previsto <= 3): {yn(pairs <= 3)}",
          f"(K4) ordem de granularidade: Gtn {mg['Gtn']:.3f} >= Gt {mg['Gt']:.3f} >= G0-0,05 ({mg['G0'] - 0.05:.3f}) e Gtn >= Gn {mg['Gn']:.3f} >= G0-0,05: {yn(k4)}",
          f"(K5) tipos em que LOTO-G0 ganha do aleatorio alem da margem: {k5} de 7 (previsto >= 5): {yn(k5 >= 5)}",
          f"(K6) tipos em que o melhor aprendiz online bate LOTO-G0 alem da margem: {k6} de 7 (previsto <= 2): {yn(k6 <= 2)}",
          f"(K7) tamanhos em que LOSO-G0 ganha do aleatorio alem da margem: {k7} de 4 (previsto >= 3): {yn(k7 >= 3)}",
          f"(K8) folga estatica por classe com validacao cruzada no teste menos o ganho de Gtn: {k8v:.3f} (previsto <= 0,30): {yn(k8v <= 0.30)}",
          f"(K9) tipos em que a politica neural bate G0 alem da margem: {k9} de 7 (previsto <= 2): {yn(k9 <= 2)}"]
    pred = [k1 >= 5, k2 >= 0.5, pairs <= 3, k4, k5 >= 5, k6 <= 2, k7 >= 3, k8v <= 0.30, k9 <= 2]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada266_granularity.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat(cells_all).to_csv(R / "entrada266_granularity_cells.csv", index=False)
    pd.concat(tests_all).to_csv(R / "entrada266_granularity_tests.csv", index=False)


if __name__ == "__main__":
    main()
