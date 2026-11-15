"""
analysis/report_entrada272_frozenqo.py
=======================================
Entrada 272 (Ciclo 44): CONTROLE da Entrada 271. O ganho do Q-Learning otimista aquecido por uma priori offline (qo_warm_*) vem da aprendizagem durante a execucao ou so' da priori? O controle (experiment/run_cycle44_frozenqo_entrada272.py) usa as
mesmas prioris e as mesmas instancias mas com a taxa de aprendizagem ZERO (qo_frozen_*: a tabela Q nao muda; acao de maior Q0 com probabilidade 0,95). Benchmark de Fanjul-Peyro e Ruiz, B30, 700 instancias de teste, T = 1.000, 3 sementes. Comparadores dos arquivos
das Entradas 256, 266 e 271: o aleatorio, o qo aquecido, a LOSO-Gt e a validada por classe (Gtn). Unidade = media das 3 sementes; celulas = 7 tipos; ganho = -d sobre o aleatorio (pontos); testes como nas Entradas 180-271 (Wilcoxon pareado bilateral,
IC90 bootstrap, equivalencia a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente a zero); familias (Holm sobre os 7 tipos): A (congelado contra o aleatorio), L (aquecido contra congelado: positivo/confirmada = a aprendizagem ajuda).

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada272_prereg_hashes.txt; nenhum resultado de teste dos congelados visto; os dos aquecidos foram vistos na Entrada 271)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(F1) O qo_warm_loso bate o qo_frozen_loso alem da margem (L) em pelo menos 4 dos 7 tipos (a aprendizagem acrescenta valor quando a priori e' transferida). (60%)
(F2) O ganho medio do qo_frozen_loso fica a no maximo 0,3 ponto do da LOSO-Gt (0,376): sem aprender, a priori transferida vale o que a acao estatica transferida vale. (65%)
(F3) O ganho medio do qo_warm_loso supera o do qo_frozen_loso em pelo menos 0,3 ponto. (60%)
(F4) O ganho medio do qo_frozen_full e' pelo menos o do qo_warm_full (com a priori da classe inteira, aprender nao ajuda). (65%)
(F5) O ganho medio do qo_frozen_full e' de pelo menos 90% do da Gtn. (70%)
Uso: python -m analysis.report_entrada272_frozenqo
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada266_granularity import NSIZES, TYPES, pick, read, tuning  # noqa: E402

R = ROOT / "results"
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def main():
    key = ["type", "instance"]
    arms = read("arms_test_B30_s*of*_entrada266.csv")
    WA = arms.groupby(key + ["arm"]).makespan.mean().unstack("arm")
    WA.columns = [f"arm{c}" for c in WA.columns]
    meta = arms.drop_duplicates(key)[key + ["n"]]
    T_ = tuning("B30")
    fr = read("fr_test_B30_s*of4_entrada256.csv")
    W = fr.groupby(key + ["method"]).makespan.mean().unstack("method").reset_index().merge(meta, on=key)
    for pat in ("warmqo_test_s*of*_entrada271.csv", "frozenqo_test_s*of*_entrada272.csv"):
        W = W.merge(read(pat).groupby(key + ["method"]).makespan.mean().unstack("method").reset_index(), on=key)
    W = W.merge(WA.reset_index(), on=key)
    assert len(W) == 700, len(W)
    losoT = {n: pick(T_[T_.n != n], "type") for n in NSIZES}
    gtn = pick(T_, "cls")
    W["LOSO_Gt"] = [float(r[f"arm{losoT[r.n][r['type']]}"]) for _, r in W.iterrows()]
    W["Gtn"] = [float(r[f"arm{gtn[r['type'] + '|' + str(int(r.n))]}"]) for _, r in W.iterrows()]
    meths = ["qo", "qo_warm_loso", "qo_frozen_loso", "qo_warm_full", "qo_frozen_full", "LOSO_Gt", "Gtn"]
    rows, TA, TL = [], [], []
    for t in TYPES:
        x = W[W["type"] == t]
        rows.append(dict(type=t, **{f"g_{m}": float(-rel(x[m], x.rand).mean()) for m in meths}))
        for v in ("loso", "full"):
            a = test(x[f"qo_frozen_{v}"], x.rand); a.update(id=f"A-qo_frozen_{v}-{t}"); TA.append(a)
            b = test(x[f"qo_warm_{v}"], x[f"qo_frozen_{v}"]); b.update(id=f"L-{v}-{t}"); TL.append(b)
    C = pd.DataFrame(rows)
    FA = pd.concat([family([r for r in TA if r["id"].split("-")[1] == f"qo_frozen_{v}"]) for v in ("loso", "full")], ignore_index=True)
    FL = pd.concat([family([r for r in TL if r["id"].split("-")[1] == v]) for v in ("loso", "full")], ignore_index=True)
    nl = lambda v: int(beyond(FL[FL.id.str.split("-").str[1] == v]).sum())
    mg = {m: float(C[f"g_{m}"].mean()) for m in meths}
    f1 = nl("loso")
    f2d = abs(mg["qo_frozen_loso"] - mg["LOSO_Gt"])
    f3d = mg["qo_warm_loso"] - mg["qo_frozen_loso"]
    f4 = mg["qo_frozen_full"] >= mg["qo_warm_full"]
    f5v = mg["qo_frozen_full"] / mg["Gtn"]
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "CONTROLE: A PRIORI SEM APRENDIZAGEM (Entrada 272)", "=" * 110, "--- ganhos sobre o aleatorio por tipo (pontos) ---", C.round(3).to_string(index=False), "",
         "ganho medio nos 7 tipos: " + "; ".join(f"{m} {mg[m]:.3f}" for m in meths),
         "tipos em que o congelado ganha do aleatorio alem da margem: " + "; ".join(f"qo_frozen_{v} {int(beyond(FA[FA.id.str.split('-').str[1] == f'qo_frozen_{v}']).sum())}" for v in ("loso", "full")),
         "tipos em que o aquecido bate o congelado alem da margem (L): " + "; ".join(f"{v} {nl(v)}" for v in ("loso", "full")), "",
         "--- previsoes ---",
         f"(F1) tipos em que o qo_warm_loso bate o qo_frozen_loso alem da margem: {f1} de 7 (previsto >= 4): {yn(f1 >= 4)}",
         f"(F2) |ganho do qo_frozen_loso - ganho da LOSO-Gt| = {f2d:.3f} (previsto <= 0,3): {yn(f2d <= 0.3)}",
         f"(F3) ganho do qo_warm_loso - ganho do qo_frozen_loso = {f3d:.3f} (previsto >= 0,3): {yn(f3d >= 0.3)}",
         f"(F4) qo_frozen_full {mg['qo_frozen_full']:.3f} >= qo_warm_full {mg['qo_warm_full']:.3f}: {yn(f4)}",
         f"(F5) qo_frozen_full / Gtn = {f5v:.3f} (previsto >= 0,90): {yn(f5v >= 0.9)}"]
    pred = [f1 >= 4, f2d <= 0.3, f3d >= 0.3, f4, f5v >= 0.9]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada272_frozenqo.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada272_frozenqo_cells.csv", index=False)
    pd.concat([FA, FL]).to_csv(R / "entrada272_frozenqo_tests.csv", index=False)


if __name__ == "__main__":
    main()
