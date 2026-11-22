"""
analysis/report_entrada285_fulldescent.py
==========================================
Entrada 285 (Ciclo 52): A ABLACAO DO IG COM A DESCIDA COMPLETA (o ILS classico). A Entrada 284 cruzou a reconstrucao (gulosa ou reinsercao aleatoria) com a descida curta do IG (no maximo duas melhorias); a variante aleatoria com descida curta (R1) nao e' um ILS forte.
Aqui G2 (gulosa + descida completa) e R2 (reinsercao aleatoria + descida completa = o ILS classico para P||Cmax: perturbacao de d tarefas ao acaso, descida ate' o otimo local, aceita se nao piora) completam o desenho. Mesmas 12 classes x 100 instancias de teste
da Entrada 180, S5 com a acao sorteada (sem aprendiz), T = 1.000, 5 sementes. Executor: experiment/run_cycle52_fulldescent_entrada285.py (bracos Ea, Eb, Ec de CPU igual descritos no docstring dele); G0, G1 e R1 a T = 1.000 vem da Entrada 284.
Contrastes (ganho = -d, d = 100 (A - B) / B por instancia, media das 5 sementes; o primeiro nome e' o braco A; Wilcoxon pareado sem empates, IC90 bootstrap, Holm sobre as 12 classes por contraste; 'alem da margem' = confirmado e IC90 nao contido em +-0,05):
  F1  R2 contra R1 (descida completa contra curta, reinsercao aleatoria)          F2  G2 contra G1 (idem, reconstrucao gulosa)
  F3  G2 contra R2 (gulosa contra aleatoria com descida completa)                F3c G2 contra Ea (R2 com a CPU de G2)
  F4  G0 contra R2 (IG simples contra o ILS classico, iteracoes iguais)          F4c Eb contra R2 (o IG simples com a CPU do ILS classico)
  F5c Ec contra R2 (o IG hibrido com a CPU do ILS classico)

ESTATUTO: um piloto em UMA instancia de AJUSTE de cada uma de quatro classes (20x4, 60x20, 150x30, 300x20; nenhum resultado de teste) tinha mostrado a descida completa barata (no maximo 0,11 s por execucao a T = 1.000, 5 a 6 movimentos por descida do R2, nenhuma
descida truncada) e o R2 atras do G0 nas classes 60x20 e 150x30; as previsoes nao sao, portanto, cegas ao piloto.

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada285_prereg_hashes.txt):
(B1) F1: a descida completa melhora a reinsercao aleatoria: ganho medio de R2 sobre R1 >= 0,15 e R2 vence alem da margem em pelo menos 6 das 12 classes. (65%)
(B2) F2: a descida completa muda pouco a reconstrucao gulosa: ganho medio de G2 sobre G1 entre -0,05 e +0,10. (70%)
(B3) F4: o IG simples (G0, 1.000 iteracoes) vence o ILS classico (R2, 1.000 iteracoes) alem da margem em pelo menos 5 das 12 classes. (65%)
(B4) F4c: com a CPU do ILS classico, o IG simples vence-o alem da margem em pelo menos 6 classes e perde em no maximo 1. (65%)
(B5a) F3: G2 vence R2 alem da margem em pelo menos 6 classes a iteracoes iguais. (70%)   (B5b) F3c: o mesmo a CPU igual. (60%)
(B6) K(R2/G0) esta entre 2 e 40 nas 12 classes. (65%)
(B7) nenhuma descida e' truncada pelo tampo de movimentos em nenhuma execucao. (90%)
(B8) a lacuna media ao limite inferior do R2 (media sobre as classes) fica entre a do G0 e a do R1. (75%)
Uso: python -m analysis.report_entrada285_fulldescent
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
from analysis.report_entrada180_agents import family, test  # noqa: E402

R = ROOT / "results"
CONTRASTS = [("F1", "R2", "R1"), ("F2", "G2", "G1"), ("F3", "G2", "R2"), ("F3c", "G2", "Ea"), ("F4", "G0", "R2"), ("F4c", "Eb", "R2"), ("F5c", "Ec", "R2")]
pd.set_option("display.width", 230)


def main():
    D5 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "fulldescent_test_s*of*_entrada285.csv")))], ignore_index=True)
    D4 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "ablation_test_s*of*_entrada284.csv")))], ignore_index=True)
    D4 = D4[D4.arm.isin(["G0", "G1", "R1"])]
    assert not D5.duplicated(["cls", "rep", "arm", "seed_i"]).any()
    W = pd.concat([D5, D4], ignore_index=True).groupby(["cls", "rep", "arm"]).makespan.mean().unstack()
    classes = list(dict.fromkeys(D5.cls))
    gap = (100 * ((pd.concat([D5, D4], ignore_index=True).assign(g=lambda x: (x.makespan - x.lb) / x.lb)).groupby(["arm", "cls"]).g.mean())).groupby("arm").mean()
    L = ["=" * 110, "ABLACAO DO IG COM A DESCIDA COMPLETA, O ILS CLASSICO (Entrada 285)", "=" * 110, f"execucoes novas: {len(D5)}; instancias: {W.dropna().shape[0]}; bracos novos: {sorted(D5.arm.unique())}"]
    L += ["", "--- iteracoes dos bracos de CPU igual (mediana sobre as classes): " + "; ".join(f"{a} {int(D5[D5.arm == a].groupby('cls').T.first().median())}" for a in ("Ea", "Eb", "Ec")),
          "--- lacuna media ao limite inferior (%), media sobre as classes: " + "; ".join(f"{a} {gap[a]:.3f}" for a in ("G0", "G1", "R1", "G2", "R2", "Ea", "Eb", "Ec")),
          f"--- descidas: {int(D5.descents.sum())} no total; truncadas pelo tampo: {int(D5.capped.sum())}; movimentos por descida do R2: {float(D5[D5.arm == 'R2'].moves.sum() / D5[D5.arm == 'R2'].descents.sum()):.2f}; do G2: {float(D5[D5.arm == 'G2'].moves.sum() / D5[D5.arm == 'G2'].descents.sum()):.2f}"]
    K = json.loads((R / "fulldescent_cpu_ratio_entrada285.json").read_text())
    kr = {c: v["R2/G0"] for c, v in K.items()}
    L.append("--- K(R2/G0) por classe: " + "; ".join(f"{c} {v:.1f}" for c, v in kr.items()))
    res = {}
    for cid, a, b in CONTRASTS:
        rows = []
        for c in classes:
            x = W.loc[c]
            t = test(x[a], x[b])
            t["id"] = f"{cid}-{c}"
            rows.append(t)
        F = family(rows)
        eq = (F.ci90_lo > -0.05) & (F.ci90_hi < 0.05)
        res[cid] = dict(contraste=f"{a} contra {b}", ganho_medio=float(-F.mean_d_pct.mean()), ganho_min=float(-F.mean_d_pct.max()), ganho_max=float(-F.mean_d_pct.min()),
                        vence_alem=int((F.confirmada & ~eq).sum()), perde_alem=int((F.reject & (F.mean_d_pct > 0) & ~eq).sum()), paridade=int((F.confirmada & eq).sum()), sem_confirmar=int((~F.reject).sum()))
        F.to_csv(R / f"entrada285_tests_{cid}.csv", index=False)
    T = pd.DataFrame(res).T
    L += ["", "--- contrastes: ganho medio sobre as 12 classes (pontos), classes em que A vence B alem da margem, perde alem da margem, paridade, sem confirmar ---", T.to_string(float_format=lambda v: f"{v:.3f}")]
    T.to_csv(R / "entrada285_contrasts.csv")
    cls_tab = pd.DataFrame({cid: {c: float(-test(W.loc[c][a], W.loc[c][b])["mean_d_pct"]) for c in classes} for cid, a, b in CONTRASTS})
    L += ["", "--- ganho por classe (pontos) ---", cls_tab.to_string(float_format=lambda v: f"{v:.3f}")]
    cls_tab.to_csv(R / "entrada285_by_class.csv")

    yn = lambda b: "SIM" if b else "NAO"
    p = []
    b1 = res["F1"]["ganho_medio"] >= 0.15 and res["F1"]["vence_alem"] >= 6; p.append(b1)
    b2 = -0.05 <= res["F2"]["ganho_medio"] <= 0.10; p.append(b2)
    b3 = res["F4"]["vence_alem"] >= 5; p.append(b3)
    b4 = res["F4c"]["vence_alem"] >= 6 and res["F4c"]["perde_alem"] <= 1; p.append(b4)
    b5a = res["F3"]["vence_alem"] >= 6; p.append(b5a)
    b5b = res["F3c"]["vence_alem"] >= 6; p.append(b5b)
    b6 = all(2 <= v <= 40 for v in kr.values()); p.append(b6)
    b7 = int(D5.capped.sum()) == 0; p.append(b7)
    b8 = gap["G0"] < gap["R2"] < gap["R1"]; p.append(b8)
    L += ["", "--- previsoes ---",
          f"(B1) F1: ganho medio de R2 sobre R1 {res['F1']['ganho_medio']:.3f} (previsto >= 0,15); R2 vence alem da margem em {res['F1']['vence_alem']} de 12 (previsto >= 6): {yn(b1)}",
          f"(B2) F2: ganho medio de G2 sobre G1 {res['F2']['ganho_medio']:.3f} (previsto entre -0,05 e +0,10): {yn(b2)}",
          f"(B3) F4: G0 vence R2 alem da margem em {res['F4']['vence_alem']} de 12 a iteracoes iguais (previsto >= 5): {yn(b3)}",
          f"(B4) F4c: com a CPU do R2, o IG simples vence em {res['F4c']['vence_alem']} e perde em {res['F4c']['perde_alem']} (previsto >= 6 e <= 1): {yn(b4)}",
          f"(B5a) F3: G2 vence R2 em {res['F3']['vence_alem']} de 12 (previsto >= 6): {yn(b5a)}; (B5b) F3c a CPU igual: {res['F3c']['vence_alem']} (previsto >= 6): {yn(b5b)}",
          f"(B6) K(R2/G0) entre {min(kr.values()):.1f} e {max(kr.values()):.1f} (previsto entre 2 e 40 nas 12 classes): {yn(b6)}",
          f"(B7) descidas truncadas: {int(D5.capped.sum())} (previsto 0): {yn(b7)}",
          f"(B8) lacuna media: G0 {gap['G0']:.3f} < R2 {gap['R2']:.3f} < R1 {gap['R1']:.3f}: {yn(b8)}",
          f"previsoes acertadas: {sum(p)} de {len(p)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada285_fulldescent.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
