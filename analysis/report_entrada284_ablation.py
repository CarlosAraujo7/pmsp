"""
analysis/report_entrada284_ablation.py
=======================================
Entrada 284 (Ciclo 51): ABLACAO DOS COMPONENTES DA ESTRUTURA DO IG (reconstrucao gulosa ou aleatoria x descida curta ou nao; escolha uniforme das cinco acoes de tamanho, sem aprendiz; P||Cmax, 12 classes x 100 instancias de teste x 5 sementes,
T = 1.000). Resposta ao ponto P4 da revisao duplo-cega do CEC (18/11/2026): o ILS do artigo e' fraco (perturbacao de 3 movimentos) e nao ha' ablacao do IG. Aqui R1 (reinsercao aleatoria de d tarefas + descida curta + aceita se nao piora) e'
um ILS com perturbacao forte; G0 e' o IG simples do artigo e G1 o IG hibrido. Executor: experiment/run_cycle51_ablation_entrada284.py (descricao dos bracos E1-E5 de CPU igual no docstring dele).
Contrastes (ganho = -d, d = 100 (A - B) / B por instancia, media das 5 sementes; o primeiro nome e' o braco A; teste de Wilcoxon pareado sem empates, IC90 bootstrap, Holm sobre as 12 classes por contraste; 'alem da margem' = confirmado e IC90 nao contido em +-0,05):
  C1  G0 contra R0 (reconstrucao gulosa contra aleatoria, sem descida)       C1c  G0 contra E1 (CPU igual)
  C2  G1 contra R1 (idem, com descida)                                        C2c  G1 contra E2
  C3  G1 contra G0 (efeito da descida com reconstrucao gulosa)                C3c  G1 contra E3
  C4  R1 contra R0 (efeito da descida com reconstrucao aleatoria)             C4c  R1 contra E4
  C5  G0 contra R1 (IG simples contra ILS de perturbacao forte)               C5c  E5 contra R1 (o IG simples com a CPU do ILS forte)

PRE-REGISTRO (congelado ANTES de rodar o executor e este relatorio; hashes em results/entrada284_prereg_hashes.txt):
(A1) C1: a reconstrucao gulosa vence a aleatoria sem descida, alem da margem, em pelo menos 9 das 12 classes e o ganho medio sobre as classes e' >= 0,15. (75%)
(A2) C1c: o mesmo a CPU igual (alem da margem em pelo menos 9 classes; ganho medio >= 0,15). (75%)
(A3) C2: com descida, a gulosa vence a aleatoria alem da margem em pelo menos 8 das 12 classes e o ganho medio e' positivo. (70%)
(A4) O efeito da descida e' maior com a reconstrucao aleatoria que com a gulosa: ganho medio de C4 > ganho medio de C3, a iteracoes iguais. (80%)
(A5) C5c: o IG simples com a CPU do ILS de perturbacao forte vence-o alem da margem em pelo menos 6 das 12 classes. (65%)
(A6) Nos contrastes C1, C2, C3, C4 o sinal do ganho medio sobre as classes e' o mesmo a CPU igual e a iteracoes iguais. (85%)
(A7) O efeito da reconstrucao (ganho medio de C1) e' pelo menos 4 vezes o maior ganho de um aprendiz sobre o aleatorio a iteracoes iguais no mesmo espaco (0,039, a roleta ALNS; Tabela II do artigo). (70%)
Uso: python -m analysis.report_entrada284_ablation
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, test  # noqa: E402

R = ROOT / "results"
CONTRASTS = [("C1", "G0", "R0"), ("C1c", "G0", "E1"), ("C2", "G1", "R1"), ("C2c", "G1", "E2"), ("C3", "G1", "G0"), ("C3c", "G1", "E3"), ("C4", "R1", "R0"), ("C4c", "R1", "E4"), ("C5", "G0", "R1"), ("C5c", "E5", "R1")]
pd.set_option("display.width", 230)


def main():
    files = sorted(glob.glob(str(R / "ablation_test_s*of*_entrada284.csv")))
    D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert not D.duplicated(["cls", "rep", "arm", "seed_i"]).any()
    W = D.groupby(["cls", "rep", "arm"]).makespan.mean().unstack()
    classes = list(dict.fromkeys(D.cls))
    L = ["=" * 110, "ABLACAO DOS COMPONENTES DA ESTRUTURA DO IG (Entrada 284)", "=" * 110, f"execucoes: {len(D)}; instancias: {len(W)}; bracos: {sorted(D.arm.unique())}"]
    L += ["", "--- iteracoes dos bracos de CPU igual (mediana sobre as classes): " + "; ".join(f"{a} {int(D[D.arm == a].groupby('cls').T.first().median())}" for a in ("E1", "E2", "E3", "E4", "E5")),
          "--- makespan medio relativo ao limite inferior (lacuna, %), media sobre as 12 classes: " + "; ".join(
              f"{a} {float((100 * (D[D.arm == a].assign(g=lambda x: (x.makespan - x.lb) / x.lb).groupby('cls').g.mean())).mean()):.3f}" for a in ("G0", "R0", "G1", "R1"))]
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
        beyond_pos = F.confirmada & ~eq
        beyond_neg = F.reject & (F.mean_d_pct > 0) & ~eq
        res[cid] = dict(contraste=f"{a} contra {b}", ganho_medio=float(-F.mean_d_pct.mean()), ganho_min=float(-F.mean_d_pct.max()), ganho_max=float(-F.mean_d_pct.min()),
                        vence_alem=int(beyond_pos.sum()), perde_alem=int(beyond_neg.sum()), paridade=int((F.confirmada & eq).sum()), sem_confirmar=int((~F.reject).sum()))
        F.to_csv(R / f"entrada284_tests_{cid}.csv", index=False)
    T = pd.DataFrame(res).T
    L += ["", "--- contrastes: ganho medio sobre as 12 classes (pontos), classes em que A vence B alem da margem, perde alem da margem, paridade, sem confirmar ---", T.to_string(float_format=lambda v: f"{v:.3f}")]
    T.to_csv(R / "entrada284_contrasts.csv")
    cls_tab = pd.DataFrame({cid: {c: float(-test(W.loc[c][a], W.loc[c][b])["mean_d_pct"]) for c in classes} for cid, a, b in CONTRASTS})
    L += ["", "--- ganho por classe (pontos) ---", cls_tab.to_string(float_format=lambda v: f"{v:.3f}")]
    cls_tab.to_csv(R / "entrada284_by_class.csv")

    yn = lambda b: "SIM" if b else "NAO"
    p = []
    a1 = res["C1"]["vence_alem"] >= 9 and res["C1"]["ganho_medio"] >= 0.15; p.append(a1)
    a2 = res["C1c"]["vence_alem"] >= 9 and res["C1c"]["ganho_medio"] >= 0.15; p.append(a2)
    a3 = res["C2"]["vence_alem"] >= 8 and res["C2"]["ganho_medio"] > 0; p.append(a3)
    a4 = res["C4"]["ganho_medio"] > res["C3"]["ganho_medio"]; p.append(a4)
    a5 = res["C5c"]["vence_alem"] >= 6; p.append(a5)
    a6 = all(np.sign(res[k]["ganho_medio"]) == np.sign(res[k + "c"]["ganho_medio"]) for k in ("C1", "C2", "C3", "C4")); p.append(a6)
    a7 = res["C1"]["ganho_medio"] >= 4 * 0.039; p.append(a7)
    L += ["", "--- previsoes ---",
          f"(A1) C1: vence alem da margem em {res['C1']['vence_alem']} de 12 (previsto >= 9); ganho medio {res['C1']['ganho_medio']:.3f} (previsto >= 0,15): {yn(a1)}",
          f"(A2) C1c: vence alem da margem em {res['C1c']['vence_alem']} de 12; ganho medio {res['C1c']['ganho_medio']:.3f}: {yn(a2)}",
          f"(A3) C2: vence alem da margem em {res['C2']['vence_alem']} de 12 (previsto >= 8); ganho medio {res['C2']['ganho_medio']:.3f} (previsto > 0): {yn(a3)}",
          f"(A4) ganho medio de C4 {res['C4']['ganho_medio']:.3f} contra C3 {res['C3']['ganho_medio']:.3f} (previsto C4 > C3): {yn(a4)}",
          f"(A5) C5c: o IG simples vence o ILS forte alem da margem em {res['C5c']['vence_alem']} de 12 (previsto >= 6): {yn(a5)}",
          f"(A6) mesmo sinal a CPU igual e a iteracoes iguais em C1-C4: {yn(a6)}",
          f"(A7) ganho medio de C1 {res['C1']['ganho_medio']:.3f} contra 4 x 0,039 = {4 * 0.039:.3f}: {yn(a7)}",
          f"previsoes acertadas: {sum(p)} de {len(p)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada284_ablation.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
