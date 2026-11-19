"""
analysis/report_entrada282_cvrp.py
==================================
Entrada 282 (Ciclo 50): O PROTOCOLO, SEM ALTERACAO, NUMA IMPLEMENTACAO DE TERCEIROS (roteiro de 16/11/2026, item P1). ALNS para o CVRP com a biblioteca `alns` (Wouda e Lan, JOSS 2023, MIT): o laco, o criterio de aceitacao (RecordToRecordTravel.autofit) e todos os seletores
(RandomSelect, RouletteWheel, SegmentedRouletteWheel, AlphaUCB, MABSelector com UCB1 e epsilon-guloso) sao da biblioteca, com os valores do seu exemplo de CVRP e sem ajuste; so' o problema e os 8 x 2 operadores (16 acoes heterogeneas) sao nossos (pmsp/cvrp_alns.py). T = 1.000 iteracoes, 3 sementes.
8 classes sinteticas (n em {100, 200} x clientes aleatorios R ou agrupados C x 6 ou 16 clientes por rota; 10 instancias de ajuste e 20 de teste) e as 22 instancias X publicadas com n <= 200 (CVRPLIB; melhores solucoes conhecidas). Metodos: random, os cinco aprendizes, os mesmos com a LISTA DE OPERADORES PERMUTADA
(sufixo _perm), as 16 acoes fixas. A validada da classe, a global (G0) e a por (n, clientes por rota) (Gnr) saem so' do ajuste (a de menor razao media custo / menor custo da instancia). Unidade = media das 3 sementes por instancia; ganho = -d sobre o aleatorio (d = 100 (A - B) / B, em % do custo);
testes como nas Entradas 180-281 (Wilcoxon pareado bilateral, IC90 bootstrap, Holm sobre as 8 classes por metodo, equivalencia a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente). Instancias X: uma so' celula de 22 instancias; a validada nao existe por classe e usa-se a Gnr (e a G0).

PRE-REGISTRO (congelado ANTES de rodar qualquer execucao do CVRP, ajuste inclusive; hashes em results/entrada282_prereg_hashes.txt; so' uma execucao de teste em uma instancia X (n = 100) e de fumaca com tres aprendizes foi vista antes, para depurar o codigo; as previsoes dos tempos usam o que ela mostrou):
(U1) A acao validada da classe ganha do aleatorio alem da margem em pelo menos 6 das 8 classes. (75%)
(U2) Pelo menos um aprendiz da biblioteca ganha do aleatorio alem da margem em pelo menos 6 das 8 classes. (65%)
(U3) O melhor aprendiz de cada classe recupera no maximo 50% do ganho da validada em pelo menos 6 das 8 classes. (60%)
(U4) Nenhum aprendiz bate a validada alem da margem em nenhum dos 40 pares (5 aprendizes x 8 classes). (65%)
(U5) O AlphaUCB com a lista permutada difere do sem permutacao alem da margem em pelo menos 2 das 8 classes (sensibilidade a ordem de um seletor de terceiros). (45%)
(U6) O RouletteWheel com a lista permutada difere do sem permutacao alem da margem em no maximo 1 das 8 classes. (75%)
(U7) Nas 22 instancias X, a Gnr ganha do aleatorio alem da margem. (60%)
(U8) Nas 22 instancias X, nenhum aprendiz bate a Gnr alem da margem. (70%)
(U9) O tempo medio por execucao do RouletteWheel, do SegmentedRouletteWheel e do AlphaUCB fica a +-25% do do aleatorio e os dois MABSelector ficam pelo menos 25% acima. (70%)
Uso: python -m analysis.report_entrada282_cvrp
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
LEARN = ["rw", "srw", "aucb", "mab_ucb1", "mab_eps"]
pd.set_option("display.width", 230)


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def per_instance(d):
    return d.groupby(["cls", "rep", "method"]).cost.mean().unstack("method").reset_index()


def main():
    ch = json.loads((R / "cvrp_choice_entrada282.json").read_text())
    S = read("cvrp_test_s*of*_entrada282.csv")
    X = read("cvrp_x_s*of*_entrada282.csv")
    assert len(S) == 160 * 27 * 3, len(S)
    assert len(X) == 22 * 27 * 3, len(X)
    W = per_instance(S)
    arm_cols = [f"arm{a}" for a in range(16)]
    W["val"] = [float(r[f"arm{ch['val'][r.cls]}"]) for _, r in W.iterrows()]
    W["g0"] = W[f"arm{ch['g0']}"]
    W["n"] = W.cls.str.split("_n").str[1].astype(int)
    W["rr"] = W.cls.str.split("_").str[0].str[1:].astype(int)
    W["gnr"] = [float(r[f"arm{ch['gnr'][str(r.n) + '|' + str(r.rr)]}"]) for _, r in W.iterrows()]
    classes = sorted(W.cls.unique())
    L = ["=" * 110, "O PROTOCOLO NUMA IMPLEMENTACAO DE TERCEIROS: ALNS PARA O CVRP COM A BIBLIOTECA `alns` (Entrada 282)", "=" * 110,
         f"escolhas do ajuste: validada por classe {ch['val']}; global (G0) acao {ch['g0']}; por (n, clientes por rota) {ch['gnr']}", ""]
    # ---- celulas sinteticas
    cells, TA, TB, TBp, TP = [], [], [], [], []
    for c in classes:
        x = W[W.cls == c]
        row = dict(cls=c, g_val=float(-rel(x.val, x.random).mean()), g_g0=float(-rel(x.g0, x.random).mean()), g_gnr=float(-rel(x.gnr, x.random).mean()))
        # folga cross-fitted: acao escolhida em duas sementes e avaliada na terceira, entre as 16 fixas
        sub = S[(S.cls == c) & S.method.isin(arm_cols)]
        vals = []
        for s in range(3):
            tr, ev = sub[sub.seed_i != s], sub[sub.seed_i == s]
            best = tr.groupby("method").cost.mean().idxmin()
            vals.append(ev[ev.method == best].set_index("rep").cost)
        hcf = pd.concat(vals, axis=1).mean(axis=1)
        row["g_hcf"] = float(-rel(hcf.reindex(x.rep).to_numpy(), x.random.to_numpy()).mean())
        for m in LEARN + [f"{k}_perm" for k in LEARN]:
            row[f"g_{m}"] = float(-rel(x[m], x.random).mean())
        cells.append(row)
        for m in ["val"] + LEARN + [f"{k}_perm" for k in LEARN]:
            a = test(x[m], x.random); a.update(id=f"A-{m}-{c}"); TA.append(a)
        for m in LEARN:
            b = test(x.val, x[m]); b.update(id=f"B-{m}-{c}"); TB.append(b)
            bp = test(x[m], x.val); bp.update(id=f"Bp-{m}-{c}"); TBp.append(bp)
            p = test(x[f"{m}_perm"], x[m]); p.update(id=f"P-{m}-{c}"); TP.append(p)
    C = pd.DataFrame(cells)
    fam = lambda rows: pd.concat([family([r for r in rows if r["id"].split("-")[1] == m]) for m in sorted({r["id"].split("-")[1] for r in rows})], ignore_index=True)
    FA, FB, FBp, FP = fam(TA), fam(TB), fam(TBp), fam(TP)
    vid = lambda F_, m: F_[F_.id.str.split("-").str[1] == m]
    L += ["--- ganho sobre o aleatorio por classe (pontos, % do custo; positivo = melhor) ---", C.round(3).to_string(index=False), "",
          "classes em que o metodo ganha do aleatorio alem da margem (de 8): " + "; ".join(f"{m} {int(beyond(vid(FA, m)).sum())}" for m in ["val"] + LEARN + [f"{k}_perm" for k in LEARN]),
          "classes em que a validada bate o aprendiz alem da margem (B): " + "; ".join(f"{m} {int(beyond(vid(FB, m)).sum())}" for m in LEARN),
          "classes em que o aprendiz bate a validada alem da margem (B'): " + "; ".join(f"{m} {int(beyond(vid(FBp, m)).sum())}" for m in LEARN),
          "classes em que a versao permutada DIFERE da original alem da margem (rejeita e nao equivalente): " + "; ".join(f"{m} {int((vid(FP, m).reject & ~vid(FP, m)['equivalente_0.05']).sum())}" for m in LEARN), ""]
    # ---- tempo e parcela da primeira operacao
    sec = S.groupby("method").secs.mean()
    first = S[S.method.isin(LEARN + [f"{k}_perm" for k in LEARN] + ["random"])].groupby("method")[["share_d0", "share_r0"]].mean()
    T_ = pd.DataFrame(dict(secs=sec.reindex(first.index), ratio_to_random=sec.reindex(first.index) / sec["random"])).join(first)
    L += ["--- segundos por execucao (razao ao aleatorio) e parcela das iteracoes nos operadores de destruicao e de reparo LISTADOS PRIMEIRO (aleatorio: 1/8 = 0,125 e 1/2) ---", T_.round(3).to_string(), ""]
    # ---- instancias X
    Wx = per_instance(X.rename(columns={"cls": "cls"}))
    bks = X.drop_duplicates("cls").set_index("cls").bks
    Wx["bks"] = Wx.cls.map(bks)
    Wx["n_"] = Wx.cls.str.extract(r"X-n(\d+)-k")[0].astype(int) - 1
    kk = Wx.cls.str.extract(r"-k(\d+)")[0].astype(int)
    Wx["rr"] = np.where((Wx.n_ / kk) <= 10, 6, 16)
    Wx["nb"] = np.where(Wx.n_ <= 150, 100, 200)
    Wx["gnr"] = [float(r[f"arm{ch['gnr'][str(r.nb) + '|' + str(r.rr)]}"]) for _, r in Wx.iterrows()]
    Wx["g0"] = Wx[f"arm{ch['g0']}"]
    gap = lambda col: float((100 * (Wx[col] / Wx.bks - 1)).mean())
    tx = []
    for m in ["gnr", "g0"] + LEARN:
        a = test(Wx[m], Wx.random); a.update(id=f"XA-{m}"); tx.append(a)
    FXA = family(tx)
    tb = []
    for m in LEARN:
        b = test(Wx[m], Wx.gnr); b.update(id=f"XB-{m}"); tb.append(b)
    FXB = family(tb)
    L += [f"--- 22 instancias X (n <= 200): lacuna media (%) aos melhores conhecidos: aleatorio {gap('random'):.2f}; Gnr {gap('gnr'):.2f}; G0 {gap('g0'):.2f}; " + "; ".join(f"{m} {gap(m):.2f}" for m in LEARN),
          "ganho sobre o aleatorio e teste (XA; Holm sobre 7 metodos):", FXA[["id", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False),
          "aprendiz contra a Gnr (XB; confirmada = o aprendiz e' melhor; Holm sobre 5):", FXB[["id", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), ""]
    # ---- previsoes
    u1 = int(beyond(vid(FA, "val")).sum()) >= 6
    per_l = {m: int(beyond(vid(FA, m)).sum()) for m in LEARN}
    u2 = max(per_l.values()) >= 6
    best_ratio = []
    for c in classes:
        r = C[C.cls == c].iloc[0]
        best_ratio.append(max(r[f"g_{m}"] for m in LEARN) / r.g_val if r.g_val > 0 else np.inf)
    u3n = sum(1 for v in best_ratio if v <= 0.5)
    u3 = u3n >= 6
    u4n = int(beyond(FBp).sum())
    u4 = u4n == 0
    pdiff = lambda m: int((vid(FP, m).reject & ~vid(FP, m)["equivalente_0.05"]).sum())
    u5, u6 = pdiff("aucb") >= 2, pdiff("rw") <= 1
    u7 = bool(beyond(FXA[FXA.id == "XA-gnr"]).iloc[0])
    u8 = int(beyond(FXB).sum()) == 0
    rt = T_.ratio_to_random
    u9 = all(0.75 <= rt[m] <= 1.25 for m in ["rw", "srw", "aucb"]) and all(rt[m] >= 1.25 for m in ["mab_ucb1", "mab_eps"])
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- previsoes ---",
          f"(U1) classes em que a validada ganha alem da margem: {int(beyond(vid(FA, 'val')).sum())} de 8 (previsto >= 6): {yn(u1)}",
          f"(U2) classes com ganho alem da margem por aprendiz: {per_l} (previsto o maximo >= 6): {yn(u2)}",
          f"(U3) classes em que o melhor aprendiz recupera no maximo 50% da validada: {u3n} de 8 (previsto >= 6): {yn(u3)}",
          f"(U4) pares (aprendiz, classe) em que o aprendiz bate a validada alem da margem: {u4n} de 40 (previsto 0): {yn(u4)}",
          f"(U5) AlphaUCB permutado difere em {pdiff('aucb')} de 8 classes (previsto >= 2): {yn(u5)}",
          f"(U6) RouletteWheel permutado difere em {pdiff('rw')} de 8 classes (previsto <= 1): {yn(u6)}",
          f"(U7) X: a Gnr ganha do aleatorio alem da margem: {yn(u7)}",
          f"(U8) X: aprendizes que batem a Gnr alem da margem: {int(beyond(FXB).sum())} de 5 (previsto 0): {yn(u8)}",
          f"(U9) razoes de tempo ao aleatorio: " + "; ".join(f"{m} {rt[m]:.2f}" for m in LEARN) + f" (previsto rw, srw, aucb em [0,75; 1,25] e os MAB >= 1,25): {yn(u9)}"]
    pred = [u1, u2, u3, u4, u5, u6, u7, u8, u9]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada282_cvrp.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada282_cvrp_cells.csv", index=False)
    pd.concat([FA, FB, FBp, FP, FXA, FXB]).to_csv(R / "entrada282_cvrp_tests.csv", index=False)
    T_.to_csv(R / "entrada282_cvrp_time.csv")


if __name__ == "__main__":
    main()
