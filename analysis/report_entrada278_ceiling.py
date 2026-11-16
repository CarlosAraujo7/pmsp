"""
analysis/report_entrada278_ceiling.py
=====================================
Entrada 278 (Ciclo 47): O TETO DAS CLASSES SINTETICAS DE P||Cmax E A SENSIBILIDADE A MARGEM. Resposta a revisao cega de 15/11/2026 (Revisor 1): a margem de +-0,05 e' da ordem da folga cruzada inteira das classes sinteticas (0,043-0,063), de modo que a paridade com o aleatorio ali seria quase
garantida; pediu-se (a) a lacuna media ao limite inferior e a fracao de instancias em que todos os bracos o atingem, e (b) os resultados com margens de 0,01 a 0,10 com a CLASSE como unidade. So' dados que ja existem (results/agents_S5_entrada180.csv e agents_S15_entrada180.csv: 12 classes x 100 instancias,
5 sementes para o aleatorio, os tres aprendizes originais (Q-Learning, UCB1, gradiente de politica) e a validada; os bracos fixos, 1 semente). Aprendizes originais apenas: os demais aprendizes sinteticos (roleta ALNS etc.) estao em outros arquivos e nao entram aqui.
Unidade = media das sementes por instancia; ganho = -d sobre o aleatorio a iteracoes iguais; testes como nas Entradas 180-278 (Wilcoxon pareado, IC90 bootstrap, Holm sobre as 12 classes por espaco e aprendiz); 'alem da margem m' = confirmada e IC90 nao contido em +-m.

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hashes em results/entrada278_prereg_hashes.txt):
(Q1) A lacuna media da escolha aleatoria ao limite inferior e' menor que 0,30% em S5 e em S15. (65%)
(Q2) A escolha aleatoria atinge o limite inferior em pelo menos 50% das execucoes em S5 e em S15. (60%)
(Q3) Em pelo menos 30% das instancias TODOS os bracos fixos atingem o limite inferior, em S5 e em S15. (55%)
(Q4) Com margem 0,01, o numero de pares (aprendiz, classe, espaco) em que o aprendiz ganha do aleatorio alem da margem e' de no maximo 6 dos 72. (60%)
(Q5) Com margem 0,10, esse numero e' zero. (90%)
(Q6) A validada ganha do aleatorio alem da margem em pelo menos 12 das 24 celulas (classe, espaco) com margem 0,05 e em no maximo 8 com margem 0,10. (65%)
Uso: python -m analysis.report_entrada278_ceiling
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
LEARN = ["ql", "ucb", "pg"]
MARGINS = [0.01, 0.02, 0.05, 0.10]
pd.set_option("display.width", 230)


def main():
    L = ["=" * 110, "O TETO DAS CLASSES SINTETICAS E A SENSIBILIDADE A MARGEM (Entrada 278)", "=" * 110]
    cnt = {m: {"learn": 0, "val": 0} for m in MARGINS}
    q1, q2, q3 = {}, {}, {}
    for sp in ("S5", "S15"):
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        r = d[d.method == "rand"]
        v = d[d.method == "val"]
        arms = d[d.method.str.startswith("arm")]
        gap_r = float((100 * (r.makespan / r.lb - 1)).mean())
        gap_v = float((100 * (v.makespan / v.lb - 1)).mean())
        hit_r = float((r.makespan == r.lb).mean())
        hit_v = float((v.makespan == v.lb).mean())
        piv = arms.pivot_table(index=["cls", "rep"], columns="method", values="makespan")
        lbi = arms.drop_duplicates(["cls", "rep"]).set_index(["cls", "rep"]).lb
        all_hit = float((piv.eq(lbi, axis=0)).all(axis=1).mean())
        any_hit = float((piv.eq(lbi, axis=0)).any(axis=1).mean())
        q1[sp], q2[sp], q3[sp] = gap_r, hit_r, all_hit
        L += [f"--- {sp}: teto ---",
              f"lacuna media ao limite inferior: aleatorio {gap_r:.3f}%; validada {gap_v:.3f}%; fracao das execucoes que atingem o limite inferior: aleatorio {hit_r:.3f}; validada {hit_v:.3f}",
              f"fracao das instancias em que TODOS os {piv.shape[1]} bracos fixos atingem o limite inferior: {all_hit:.3f}; em que ALGUM braco o atinge: {any_hit:.3f}", ""]
        W = d[d.method.isin(["rand", "val"] + LEARN)].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
        classes = sorted(W.cls.unique())
        TT = {k: [] for k in LEARN + ["val"]}
        for c in classes:
            x = W[W.cls == c]
            for k in LEARN + ["val"]:
                t = test(x[k], x.rand)
                t.update(id=f"{k}-{c}-{sp}")
                TT[k].append(t)
        rows = []
        for k in LEARN + ["val"]:
            F_ = family(TT[k])
            for m in MARGINS:
                eqm = (F_.ci90_lo > -m) & (F_.ci90_hi < m)
                b = int((F_.confirmada & ~eqm).sum())
                rows.append(dict(space=sp, method=k, margin=m, beyond=b, of=len(F_), mean_gain=-float(F_.mean_d_pct.mean())))
                cnt[m]["val" if k == "val" else "learn"] += b
        T_ = pd.DataFrame(rows)
        L += [f"--- {sp}: classes (de 12) em que o metodo ganha do aleatorio alem da margem m (a margem do manuscrito e' 0,05) ---",
              T_.pivot_table(index="method", columns="margin", values="beyond").astype(int).to_string(), f"ganho medio (pontos) sobre as 12 classes: " + "; ".join(f"{k} {T_[T_.method == k].mean_gain.iloc[0]:.3f}" for k in LEARN + ["val"]), ""]
    L += ["--- total (S5 + S15): pares (aprendiz, classe, espaco) em que um aprendiz ganha alem da margem, de 72; celulas (classe, espaco) em que a validada ganha alem da margem, de 24 ---",
          pd.DataFrame({m: [cnt[m]["learn"], cnt[m]["val"]] for m in MARGINS}, index=["aprendizes (de 72)", "validada (de 24)"]).to_string(), ""]
    q4, q5 = cnt[0.01]["learn"], cnt[0.10]["learn"]
    q6a, q6b = cnt[0.05]["val"], cnt[0.10]["val"]
    yn = lambda b: "SIM" if b else "NAO"
    L += ["--- previsoes ---",
          f"(Q1) lacuna media do aleatorio ao limite inferior: S5 {q1['S5']:.3f}%, S15 {q1['S15']:.3f}% (previsto < 0,30% nos dois): {yn(q1['S5'] < 0.30 and q1['S15'] < 0.30)}",
          f"(Q2) fracao das execucoes do aleatorio que atingem o limite inferior: S5 {q2['S5']:.3f}, S15 {q2['S15']:.3f} (previsto >= 0,50 nos dois): {yn(q2['S5'] >= 0.5 and q2['S15'] >= 0.5)}",
          f"(Q3) fracao das instancias em que todos os bracos atingem o limite inferior: S5 {q3['S5']:.3f}, S15 {q3['S15']:.3f} (previsto >= 0,30 nos dois): {yn(q3['S5'] >= 0.3 and q3['S15'] >= 0.3)}",
          f"(Q4) pares com ganho de aprendiz alem da margem 0,01: {q4} de 72 (previsto <= 6): {yn(q4 <= 6)}",
          f"(Q5) pares com ganho de aprendiz alem da margem 0,10: {q5} (previsto 0): {yn(q5 == 0)}",
          f"(Q6) celulas em que a validada ganha alem da margem: {q6a} de 24 com 0,05 (previsto >= 12) e {q6b} de 24 com 0,10 (previsto <= 8): {yn(q6a >= 12 and q6b <= 8)}"]
    pred = [q1['S5'] < 0.30 and q1['S15'] < 0.30, q2['S5'] >= 0.5 and q2['S15'] >= 0.5, q3['S5'] >= 0.3 and q3['S15'] >= 0.3, q4 <= 6, q5 == 0, q6a >= 12 and q6b <= 8]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada278_ceiling.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
