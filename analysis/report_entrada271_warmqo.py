"""
analysis/report_entrada271_warmqo.py
=====================================
Entrada 271 (Ciclo 43): o conhecimento offline pode RESGATAR o aprendiz online? Q-Learning otimista (qo) com a tabela Q inicial aquecida por uma priori offline (experiment/run_cycle43_warmqo_entrada271.py): 'qo_warm_loso' (a priori do tipo usando so'
os dados de ajuste dos OUTROS tamanhos: transferencia a um tamanho nao visto) e 'qo_warm_full' (a priori da classe inteira), no benchmark de Fanjul-Peyro e Ruiz (B30; 700 instancias de teste; T = 1.000; 3 sementes). Comparadores, todos dos arquivos
das Entradas 256 e 266: o aleatorio, o qo frio, a acao estatica LOSO-Gt (por tipo, escolhida sem os dados do tamanho testado) e a validada por classe (Gtn). Unidade = media das 3 sementes por instancia; celulas = 7 tipos (100 instancias cada) e 4 tamanhos (175
cada, os sete tipos agrupados); ganho = -d sobre o aleatorio (pontos); testes como nas Entradas 180-270 (Wilcoxon pareado bilateral, IC90 bootstrap, equivalencia a 0 em +-0,05; 'alem da margem' = confirmada e nao equivalente a zero). Familias (Holm sobre os
7 tipos ou os 4 tamanhos, por comparacao): A (metodo contra o aleatorio), E (aquecido contra o qo frio), Bs (aquecido contra a estatica; positivo/confirmada = o aquecido ganha), Bp (a estatica contra o aquecido).

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada271_prereg_hashes.txt; as prioris vem so' das instancias de ajuste; nenhum resultado de teste dos aquecidos visto)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(W1) O qo_warm_loso bate o qo frio alem da margem (E) em pelo menos 5 dos 7 tipos. (70%)
(W2) O qo_warm_loso ganha do aleatorio alem da margem (A) em pelo menos 5 dos 7 tipos. (65%)
(W3) Nos tamanhos das pontas (n = 100 e n = 1000), o qo_warm_loso bate a LOSO-Gt alem da margem nos dois. (50%)
(W4) Nos tamanhos interiores (200 e 500), a LOSO-Gt bate o qo_warm_loso alem da margem em no maximo um. (50%)
(W5) O qo_warm_full nao bate a validada por classe (Gtn) alem da margem em nenhum dos 7 tipos. (80%)
(W6) O ganho medio do qo_warm_full e' pelo menos 0,3 ponto menor que o da Gtn. (75%)
(W7) O ganho medio do qo_warm_full nos 7 tipos e' de pelo menos 1,0 ponto. (55%)
Uso: python -m analysis.report_entrada271_warmqo
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
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
    wq = read("warmqo_test_s*of*_entrada271.csv").groupby(key + ["method"]).makespan.mean().unstack("method").reset_index()
    W = W.merge(wq, on=key).merge(WA.reset_index(), on=key)
    assert len(W) == 700, len(W)
    losoT = {n: pick(T_[T_.n != n], "type") for n in NSIZES}
    gtn = pick(T_, "cls")
    W["LOSO_Gt"] = [float(r[f"arm{losoT[r.n][r['type']]}"]) for _, r in W.iterrows()]
    W["Gtn"] = [float(r[f"arm{gtn[r['type'] + '|' + str(int(r.n))]}"]) for _, r in W.iterrows()]
    meths = ["qo", "qo_warm_loso", "qo_warm_full", "LOSO_Gt", "Gtn"]
    rows, TA, TE, TS, TP, TSZ = [], [], [], [], [], []
    for t in TYPES:
        x = W[W["type"] == t]
        rows.append(dict(type=t, **{f"g_{m}": float(-rel(x[m], x.rand).mean()) for m in meths}))
        for m in ("qo_warm_loso", "qo_warm_full"):
            a = test(x[m], x.rand); a.update(id=f"A-{m}-{t}"); TA.append(a)
            e = test(x[m], x.qo); e.update(id=f"E-{m}-{t}"); TE.append(e)
        b = test(x.qo_warm_full, x.Gtn); b.update(id=f"Bs-qo_warm_full-Gtn-{t}"); TS.append(b)
        c = test(x.Gtn, x.qo_warm_full); c.update(id=f"Bp-Gtn-qo_warm_full-{t}"); TP.append(c)
    C = pd.DataFrame(rows)
    srows = []
    for n in NSIZES:
        x = W[W.n == n]
        srows.append(dict(n=n, **{f"g_{m}": float(-rel(x[m], x.rand).mean()) for m in meths}))
        a = test(x.qo_warm_loso, x.LOSO_Gt); a.update(id=f"Bs-qo_warm_loso-LOSO_Gt-n{n}"); TSZ.append(a)
        b = test(x.LOSO_Gt, x.qo_warm_loso); b.update(id=f"Bp-LOSO_Gt-qo_warm_loso-n{n}"); TSZ.append(b)
    S = pd.DataFrame(srows)
    fam = lambda rs, tag: pd.concat([family([r for r in rs if r["id"].split("-")[1] == m]) for m in tag], ignore_index=True)
    FA, FE = fam(TA, ["qo_warm_loso", "qo_warm_full"]), fam(TE, ["qo_warm_loso", "qo_warm_full"])
    FS, FP = family(TS), family(TP)
    FZs = family([r for r in TSZ if r["id"].startswith("Bs-")])
    FZp = family([r for r in TSZ if r["id"].startswith("Bp-")])
    nb = lambda F_, m: int(beyond(F_[F_.id.str.split("-").str[1] == m]).sum())
    w1, w2 = nb(FE, "qo_warm_loso"), nb(FA, "qo_warm_loso")
    ext = [n for n in (100, 1000)]
    w3 = int(sum(bool(beyond(FZs[FZs.id.str.endswith(f"-n{n}")]).iloc[0]) for n in ext))
    w4 = int(sum(bool(beyond(FZp[FZp.id.str.endswith(f"-n{n}")]).iloc[0]) for n in (200, 500)))
    w5 = int(beyond(FS).sum())
    gap = float(C.g_Gtn.mean() - C.g_qo_warm_full.mean())
    w7v = float(C.g_qo_warm_full.mean())
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "Q-LEARNING OTIMISTA AQUECIDO POR UMA PRIORI OFFLINE (Entrada 271)", "=" * 110,
         "--- ganhos sobre o aleatorio por tipo (pontos) ---", C.round(3).to_string(index=False), "", "--- por tamanho (sete tipos agrupados; 175 instancias) ---", S.round(3).to_string(index=False), "",
         "ganho medio nos 7 tipos: " + "; ".join(f"{m} {C[f'g_{m}'].mean():.3f}" for m in meths),
         "tipos em que o aquecido ganha do aleatorio alem da margem: " + "; ".join(f"{m} {nb(FA, m)}" for m in ("qo_warm_loso", "qo_warm_full")),
         "tipos em que o aquecido bate o qo frio alem da margem: " + "; ".join(f"{m} {nb(FE, m)}" for m in ("qo_warm_loso", "qo_warm_full")), "",
         "--- aquecido (LOSO) contra a LOSO-Gt por tamanho: Bs (o aquecido ganha) e Bp (a estatica ganha) ---", FZs[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), "",
         FZp[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), "",
         "--- previsoes ---",
         f"(W1) tipos em que o qo_warm_loso bate o qo frio alem da margem: {w1} de 7 (previsto >= 5): {yn(w1 >= 5)}",
         f"(W2) tipos em que o qo_warm_loso ganha do aleatorio alem da margem: {w2} de 7 (previsto >= 5): {yn(w2 >= 5)}",
         f"(W3) tamanhos das pontas em que o qo_warm_loso bate a LOSO-Gt alem da margem: {w3} de 2 (previsto 2): {yn(w3 == 2)}",
         f"(W4) tamanhos interiores em que a LOSO-Gt bate o qo_warm_loso alem da margem: {w4} de 2 (previsto <= 1): {yn(w4 <= 1)}",
         f"(W5) tipos em que o qo_warm_full bate a Gtn alem da margem: {w5} de 7 (previsto 0): {yn(w5 == 0)}",
         f"(W6) ganho medio da Gtn menos o do qo_warm_full: {gap:.3f} (previsto >= 0,3): {yn(gap >= 0.3)}",
         f"(W7) ganho medio do qo_warm_full: {w7v:.3f} (previsto >= 1,0): {yn(w7v >= 1.0)}"]
    pred = [w1 >= 5, w2 >= 5, w3 == 2, w4 <= 1, w5 == 0, gap >= 0.3, w7v >= 1.0]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada271_warmqo.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada271_warmqo_cells.csv", index=False)
    S.to_csv(R / "entrada271_warmqo_sizes.csv", index=False)
    pd.concat([FA, FE, FS, FP, FZs, FZp]).to_csv(R / "entrada271_warmqo_tests.csv", index=False)


if __name__ == "__main__":
    main()
