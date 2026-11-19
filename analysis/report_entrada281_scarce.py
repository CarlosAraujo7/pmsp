"""
analysis/report_entrada281_scarce.py
====================================
Entrada 281 (Ciclo 49): ONDE A APRENDIZAGEM ONLINE PAGA: O REGIME DE POUCOS DADOS DE AJUSTE (roteiro de 16/11/2026, item P2). Com k = 1, 2, 3, 5 das 10 instancias de ajuste de cada classe (4 sorteios do subconjunto por k), a acao estatica, o Q-Learning otimista aquecido com a priori
dos mesmos k dados e o controle de taxa de aprendizagem zero (Entrada 272) usam exatamente os mesmos dados (experiment/run_cycle49_scarce_entrada281.py). Referencia k = 10: a validada (Entrada 266), o aquecido 'full' (Entrada 271) e o controle 'full' (Entrada 272). Benchmark de Fanjul-Peyro e Ruiz,
B30, 700 instancias de teste, T = 1.000, 3 sementes. Unidade = media das sementes por instancia e, por k, tambem media dos 4 sorteios (a instancia continua a unidade; a variacao entre sorteios e' reportada a parte); ganho = -d sobre o aleatorio (pontos); celulas = os 7 tipos (100 instancias);
testes como nas Entradas 180-280 (Wilcoxon pareado bilateral, IC90 bootstrap, Holm sobre os 7 tipos por k e por comparacao, equivalencia a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente).

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada281_prereg_hashes.txt; a fumaca usou instancias de AJUSTE; nenhum resultado do teste das variantes foi visto; as 'previsoes' abaixo foram escritas conhecendo as Entradas 271, 272 e 277):
(T1) Com k = 1 o ganho medio do aprendiz aquecido (7 tipos) e' MENOR que o da acao estatica escolhida com o mesmo dado. (60%)
(T2) Com k = 1 o aprendiz aquecido ganha MAIS que o controle sem aprendizagem (aquecido - congelado > 0). (55%)
(T3) A diferenca (aquecido - congelado) com k = 1 excede a de k = 5 em pelo menos 0,05 ponto (a aprendizagem corrige mais quanto mais ruidosa a priori). (60%)
(T4) Com k = 1 o aquecido bate o Q-Learning otimista frio alem da margem em pelo menos 6 dos 7 tipos. (85%)
(T5) Com k = 1 o aquecido bate a acao estatica alem da margem em pelo menos 1 dos 7 tipos. (40%)
(T6) Com k = 1 o ganho medio do aquecido e' pelo menos 60% do ganho da validada (k = 10). (60%)
(T7) O desvio-padrao, entre os 4 sorteios, do ganho medio com k = 1 e' MAIOR para a acao estatica que para o aprendiz aquecido (o aprendiz e' menos sensivel aos poucos dados). (70%)
Uso: python -m analysis.report_entrada281_scarce
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
TYPES = ["U1a100", "U10a100", "U100a120", "U100a200", "U1000a1100", "JobsCorre", "MaqCorre"]
KS = [1, 2, 3, 5]
D = 4
KEY = ["type", "instance"]
pd.set_option("display.width", 230)


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def main():
    arms = read("arms_test_B30_s*of*_entrada266.csv")
    arms["cls"] = arms["type"] + "|" + arms.n.astype(str)
    WA = arms.groupby(KEY + ["arm"]).makespan.mean().unstack("arm")                      # [(type, instance), arm]
    meta = arms.drop_duplicates(KEY)[KEY + ["cls"]].set_index(KEY)
    fr = read("fr_test_B30_s*of4_entrada256.csv")
    base = fr.groupby(KEY + ["method"]).makespan.mean().unstack("method")                  # rand, qo, val, ...
    sc = read("scarce_test_s*of12_entrada281.csv")
    assert len(sc) == 700 * 3 * 32, len(sc)
    new = sc.groupby(KEY + ["method"]).makespan.mean().unstack("method")
    w10 = read("warmqo_test_s*of12_entrada271.csv")
    w10 = w10[w10.method == "qo_warm_full"].groupby(KEY).makespan.mean().rename("w")
    f10 = read("frozenqo_test_s*of12_entrada272.csv")
    f10 = f10[f10.method == "qo_frozen_full"].groupby(KEY).makespan.mean().rename("f")
    pri = json.loads((R / "scarce_prior_entrada281.json").read_text())
    idx = base.index
    cls = meta.loc[idx, "cls"].to_numpy()
    typ = idx.get_level_values("type").to_numpy()
    rand = base["rand"].to_numpy()
    ref = pd.DataFrame(index=idx)
    ref["static_10"] = base["val"].to_numpy()
    ref["warm_10"], ref["frozen_10"] = w10.reindex(idx).to_numpy(), f10.reindex(idx).to_numpy()
    gain = lambda x: float(np.mean([-rel(np.asarray(x)[typ == t], rand[typ == t]).mean() for t in TYPES]))
    L = ["=" * 110, "ONDE A APRENDIZAGEM ONLINE PAGA: O REGIME DE POUCOS DADOS DE AJUSTE (Entrada 281)", "=" * 110,
         f"ganho medio (7 tipos; pontos) da validada (k = 10) {gain(ref.static_10):.3f}; do aquecido 'full' {gain(ref.warm_10):.3f}; do congelado 'full' {gain(ref.frozen_10):.3f}; do otimista frio {gain(base.qo):.3f}", ""]
    rows, tests_all, per = [], [], {}
    for k in KS:
        S = np.zeros((len(idx), D)); W = np.zeros((len(idx), D)); Fz = np.zeros((len(idx), D))
        for d in range(D):
            pa = np.array([WA.loc[i, pri[f"{k}|{d}"][c]["static"]] for i, c in zip(idx, cls)])
            S[:, d] = pa
            W[:, d] = new.loc[idx, f"w{k}_{d}"].to_numpy()
            Fz[:, d] = new.loc[idx, f"f{k}_{d}"].to_numpy()
        Sm, Wm, Fm = S.mean(axis=1), W.mean(axis=1), Fz.mean(axis=1)
        per[k] = dict(static=[gain(S[:, d]) for d in range(D)], warm=[gain(W[:, d]) for d in range(D)], frozen=[gain(Fz[:, d]) for d in range(D)])
        TW, TS, TF, TC = [], [], [], []
        for t in TYPES:
            m_ = typ == t
            a = test(Wm[m_], Sm[m_]); a.update(id=f"WS-{k}-{t}"); TW.append(a)
            b = test(Sm[m_], Wm[m_]); b.update(id=f"SW-{k}-{t}"); TS.append(b)
            c = test(Wm[m_], Fm[m_]); c.update(id=f"WF-{k}-{t}"); TF.append(c)
            e = test(Wm[m_], base.qo.to_numpy()[m_]); e.update(id=f"WC-{k}-{t}"); TC.append(e)
        FW, FS, FF, FC = family(TW), family(TS), family(TF), family(TC)
        tests_all += [FW, FS, FF, FC]
        gs, gw, gf = gain(Sm), gain(Wm), gain(Fm)
        rows.append(dict(k=k, static=gs, warm=gw, frozen=gf, cold=gain(base.qo), warm_minus_static=gw - gs, warm_minus_frozen=gw - gf, warm_beats_static=int(beyond(FW).sum()), static_beats_warm=int(beyond(FS).sum()),
                         warm_beats_frozen=int(beyond(FF).sum()), warm_beats_cold=int(beyond(FC).sum()), sd_static=float(np.std(per[k]["static"], ddof=1)), sd_warm=float(np.std(per[k]["warm"], ddof=1)), sd_frozen=float(np.std(per[k]["frozen"], ddof=1))))
    # k = 10 (referencia)
    typ_ = typ
    FW10 = family([dict(test(ref.warm_10.to_numpy()[typ_ == t], ref.static_10.to_numpy()[typ_ == t]), id=f"WS-10-{t}") for t in TYPES])
    FS10 = family([dict(test(ref.static_10.to_numpy()[typ_ == t], ref.warm_10.to_numpy()[typ_ == t]), id=f"SW-10-{t}") for t in TYPES])
    FF10 = family([dict(test(ref.warm_10.to_numpy()[typ_ == t], ref.frozen_10.to_numpy()[typ_ == t]), id=f"WF-10-{t}") for t in TYPES])
    FC10 = family([dict(test(ref.warm_10.to_numpy()[typ_ == t], base.qo.to_numpy()[typ_ == t]), id=f"WC-10-{t}") for t in TYPES])
    gs, gw, gf = gain(ref.static_10), gain(ref.warm_10), gain(ref.frozen_10)
    rows.append(dict(k=10, static=gs, warm=gw, frozen=gf, cold=gain(base.qo), warm_minus_static=gw - gs, warm_minus_frozen=gw - gf, warm_beats_static=int(beyond(FW10).sum()), static_beats_warm=int(beyond(FS10).sum()),
                     warm_beats_frozen=int(beyond(FF10).sum()), warm_beats_cold=int(beyond(FC10).sum()), sd_static=0.0, sd_warm=0.0, sd_frozen=0.0))
    T = pd.DataFrame(rows)
    L += ["--- por k (mesmos k dados para os tres competidores; k = 10: Entradas 266, 271, 272). Colunas 'x_beats_y' = tipos (de 7) em que x bate y alem da margem; sd_* = desvio-padrao entre os 4 sorteios do ganho medio ---", T.round(3).to_string(index=False), ""]
    L += ["--- ganhos medios por sorteio (static / warm / frozen) ---"] + [f"k = {k}: " + "; ".join(f"d{d}: {per[k]['static'][d]:.3f} / {per[k]['warm'][d]:.3f} / {per[k]['frozen'][d]:.3f}" for d in range(D)) for k in KS] + [""]
    r1 = T[T.k == 1].iloc[0]; r5 = T[T.k == 5].iloc[0]; r10 = T[T.k == 10].iloc[0]
    t1 = r1.warm < r1.static
    t2 = r1.warm_minus_frozen > 0
    t3 = r1.warm_minus_frozen - r5.warm_minus_frozen >= 0.05
    t4 = int(r1.warm_beats_cold) >= 6
    t5 = int(r1.warm_beats_static) >= 1
    t6 = r1.warm >= 0.6 * r10.static
    t7 = r1.sd_static > r1.sd_warm
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- previsoes ---",
          f"(T1) k = 1: ganho do aquecido {r1.warm:.3f} contra o da estatica {r1.static:.3f} (previsto aquecido < estatica): {yn(t1)}",
          f"(T2) k = 1: aquecido - congelado = {r1.warm_minus_frozen:+.3f} (previsto > 0): {yn(t2)}",
          f"(T3) (aquecido - congelado) em k = 1 menos o de k = 5: {r1.warm_minus_frozen - r5.warm_minus_frozen:+.3f} (previsto >= 0,05): {yn(t3)}",
          f"(T4) k = 1: tipos em que o aquecido bate o otimista frio alem da margem: {int(r1.warm_beats_cold)} de 7 (previsto >= 6): {yn(t4)}",
          f"(T5) k = 1: tipos em que o aquecido bate a estatica alem da margem: {int(r1.warm_beats_static)} de 7 (previsto >= 1): {yn(t5)}",
          f"(T6) k = 1: ganho do aquecido {r1.warm:.3f} contra 60% da validada ({0.6 * r10.static:.3f}) (previsto >=): {yn(t6)}",
          f"(T7) k = 1: sd entre sorteios da estatica {r1.sd_static:.3f} contra a do aquecido {r1.sd_warm:.3f} (previsto estatica > aquecido): {yn(t7)}"]
    pred = [t1, t2, t3, t4, t5, t6, t7]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada281_scarce.txt").write_text(text + "\n", encoding="utf-8")
    T.to_csv(R / "entrada281_scarce_by_k.csv", index=False)
    pd.concat(tests_all + [FW10, FS10, FF10, FC10]).to_csv(R / "entrada281_scarce_tests.csv", index=False)


if __name__ == "__main__":
    main()
