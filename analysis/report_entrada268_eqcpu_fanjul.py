"""
analysis/report_entrada268_eqcpu_fanjul.py
===========================================
Entrada 268 (Ciclo 41): o protocolo a CPU IGUAL no benchmark de Fanjul-Peyro e Ruiz (espaco B30; 700 instancias de teste; 7 tipos x 100; 3 sementes). A Entrada 256 comparou a iteracoes iguais (T = 1.000); aqui cada um dos sete aprendizes
(T = 1.000) e' comparado a CPU igual com o aleatorio (T_k = max(50, round(1000 K_k))) e com a acao VALIDADA (T = max(50, round(1000 K_k / K_val))), onde K = (CPU de 1.000 iteracoes do metodo) / (CPU do aleatorio), por classe (tipo, n), medido em
sequencia em duas instancias de ajuste (results/eqcpu_K_entrada268.json). Efeito de decisao = ganho a iteracoes iguais (Entrada 256); efeito de custo = ganho a CPU igual menos ganho a iteracoes iguais. Familias (Holm sobre os 7 tipos, por aprendiz):
Acpu (aprendiz contra o aleatorio a CPU igual), Bcpu (a validada a CPU igual contra o aprendiz), Bpcpu (o aprendiz contra a validada a CPU igual). Margem +-0,05; 'alem da margem' = confirmada e nao equivalente a zero.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada268_prereg_hashes.txt; so' as instancias de ajuste foram usadas para medir K; nenhum resultado de teste a CPU igual visto; os de iteracoes iguais sao os da Entrada 256)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(E1) A mediana de K do Q-Learning padrao nas 28 classes e' de no maximo 0,85 (ele repete a acao listada primeiro, a mais barata). (75%)
(E2) A mediana de K de cada um de ucb, pg, alns, qo, iq e fr fica em [0,9; 1,15]. (60%)
(E3) A mediana de K da validada (a acao com busca local) e' de pelo menos 1,5. (70%)
(E4) A CPU igual, o Q-Learning padrao ganha do aleatorio alem da margem (Acpu) em no maximo 1 dos 7 tipos. (65%)
(E5) A validada a CPU igual bate os sete aprendizes (Bcpu, confirmada) em pelo menos 45 dos 49 pares (aprendiz, tipo). (70%)
(E6) Nenhum aprendiz bate a validada a CPU igual alem da margem (Bpcpu) em nenhum dos 49 pares. (75%)
(E7) A vantagem media da validada a CPU igual sobre o melhor aprendiz de cada tipo e' de pelo menos 1,0 ponto. (65%)
Uso: python -m analysis.report_entrada268_eqcpu_fanjul
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
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def main():
    K = json.loads((R / "eqcpu_K_entrada268.json").read_text())
    Kd = pd.DataFrame(K).T
    key = ["type", "instance"]
    fr = read("fr_test_B30_s*of4_entrada256.csv")
    W = fr.groupby(key + ["method"]).makespan.mean().unstack("method").reset_index()
    eq = read("eqcpu_test_s*of*_entrada268.csv")
    We = eq.groupby(key + ["method"]).makespan.mean().unstack("method").reset_index()
    W = W.merge(We, on=key)
    assert len(W) == 700, len(W)
    TA, TB, TP, rows = [], [], [], []
    for t in TYPES:
        x = W[W["type"] == t]
        row = dict(type=t)
        for k in LEARN:
            gi = float(-rel(x[k], x.rand).mean())
            gc = float(-rel(x[k], x[f"rand_eq_{k}"]).mean())
            row.update({f"dec_{k}": gi, f"cpu_{k}": gc, f"cost_{k}": gc - gi, f"vgain_{k}": float(-rel(x[f"val_eq_{k}"], x[k]).mean())})
            a = test(x[k], x[f"rand_eq_{k}"]); a.update(id=f"Acpu-{k}-{t}"); TA.append(a)
            b = test(x[f"val_eq_{k}"], x[k]); b.update(id=f"Bcpu-{k}-{t}"); TB.append(b)
            c = test(x[k], x[f"val_eq_{k}"]); c.update(id=f"Bpcpu-{k}-{t}"); TP.append(c)
        rows.append(row)
    C = pd.DataFrame(rows)
    fam = lambda rs: pd.concat([family([r for r in rs if r["id"].split("-")[1] == k]) for k in LEARN], ignore_index=True)
    FA, FB, FP = fam(TA), fam(TB), fam(TP)
    med = {k: float(Kd[k].median()) for k in LEARN + ["val"]}
    e1 = med["ql"] <= 0.85
    e2 = all(0.9 <= med[k] <= 1.15 for k in ["ucb", "pg", "alns", "qo", "iq", "fr"])
    e3 = med["val"] >= 1.5
    ql_a = int(beyond(FA[FA.id.str.split("-").str[1] == "ql"]).sum())
    nb = int(FB.confirmada.sum())
    npb = int(beyond(FP).sum())
    # vantagem media da validada a CPU igual sobre o MELHOR aprendiz de cada tipo = o menor ganho da validada entre os sete aprendizes, media dos tipos
    vadv = float(np.mean([min(C.loc[i, f"vgain_{k}"] for k in LEARN) for i in C.index]))
    yn = lambda b: "SIM" if b else "NAO"
    L = ["=" * 110, "O PROTOCOLO A CPU IGUAL NO BENCHMARK DE FANJUL-PEYRO E RUIZ, B30 (Entrada 268)", "=" * 110,
         "razao de CPU K (mediana nas 28 classes; por tamanho n a mediana nos 7 tipos):", pd.DataFrame({k: [med[k]] for k in LEARN + ["val"]}).round(3).to_string(index=False),
         "K por tamanho n:", Kd.assign(n=[int(i.split('|')[1]) for i in Kd.index]).groupby("n").median().round(3).to_string(), "",
         "ganhos por tipo (pontos): dec = a iteracoes iguais; cpu = a CPU igual; cost = cpu - dec; vgain = ganho da validada a CPU igual sobre o aprendiz", C.round(3).to_string(index=False), "",
         "ganho medio nos 7 tipos: " + "; ".join(f"{k}: dec {C[f'dec_{k}'].mean():.3f}, cpu {C[f'cpu_{k}'].mean():.3f}, custo {C[f'cost_{k}'].mean():.3f}, validada sobre ele {C[f'vgain_{k}'].mean():.3f}" for k in LEARN),
         "tipos em que cada aprendiz ganha do aleatorio a CPU igual alem da margem (Acpu; de 7): " + "; ".join(f"{k} {int(beyond(FA[FA.id.str.split('-').str[1] == k]).sum())}" for k in LEARN),
         f"a validada a CPU igual bate o aprendiz (Bcpu, confirmada): {nb} de 49; aprendizes que a batem alem da margem (Bpcpu): {npb} de 49", "",
         "--- previsoes ---",
         f"(E1) mediana de K do ql padrao: {med['ql']:.3f} (previsto <= 0,85): {yn(e1)}",
         f"(E2) medianas de K de ucb, pg, alns, qo, iq, fr: " + ", ".join(f"{k} {med[k]:.3f}" for k in ["ucb", "pg", "alns", "qo", "iq", "fr"]) + f" (previsto em [0,9; 1,15]): {yn(e2)}",
         f"(E3) mediana de K da validada: {med['val']:.3f} (previsto >= 1,5): {yn(e3)}",
         f"(E4) tipos em que o ql padrao ganha do aleatorio a CPU igual alem da margem: {ql_a} de 7 (previsto <= 1): {yn(ql_a <= 1)}",
         f"(E5) pares em que a validada a CPU igual bate o aprendiz (confirmada): {nb} de 49 (previsto >= 45): {yn(nb >= 45)}",
         f"(E6) pares em que o aprendiz bate a validada a CPU igual alem da margem: {npb} de 49 (previsto 0): {yn(npb == 0)}",
         f"(E7) vantagem media da validada a CPU igual sobre o melhor aprendiz de cada tipo: {vadv:.3f} (previsto >= 1,0): {yn(vadv >= 1.0)}"]
    pred = [e1, e2, e3, ql_a <= 1, nb >= 45, npb == 0, vadv >= 1.0]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada268_eqcpu_fanjul.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada268_eqcpu_fanjul_cells.csv", index=False)
    pd.concat([FA, FB, FP]).to_csv(R / "entrada268_eqcpu_fanjul_tests.csv", index=False)
    Kd.to_csv(R / "entrada268_eqcpu_fanjul_K.csv")


if __name__ == "__main__":
    main()
