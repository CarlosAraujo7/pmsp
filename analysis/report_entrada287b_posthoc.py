"""
analysis/report_entrada287b_posthoc.py (Entrada 287b, Ciclo 54) -- POS HOC
============================================================================
O modelo de bandido oblivio da Entrada 287 falhou no portao (2 de 11 previsoes). Esta analise e' POS HOC (nao ha' previsoes; escrita depois de ver o resultado) e le so' results/entrada287_cells.csv. Pergunta: o que explica a falha?
 (i)  maldicao do vencedor ou depleção? A vantagem da melhor acao estatica no bandido, com validacao cruzada (T Delta_cf, escolhida numa metade das instancias e medida na outra), contra o ganho observado da validada.
      Se a razao ficar perto de 1, o excesso do gemeo era maldicao do vencedor; se continuar muito acima de 1, a recompensa nao e' independente da trajetoria (a melhora total e' finita: o ganho de uma acao fixa SATURA).
 (ii) a captura observada do rt contra a INFORMACAO disponivel: recompensas positivas por acao por execucao, numero de acoes K, taxa de recompensa positiva; e a captura por terco de cada uma.
Uso: python -m analysis.report_entrada287b_posthoc
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
T = 1000


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(sstats.spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 3 else float("nan")


def main():
    D = pd.read_csv(R / "entrada287_cells.csv")
    M = D[D.g_val >= 0.05].copy()
    L = ["=" * 110, "POS HOC: POR QUE O BANDIDO OBLIVIO FALHOU, E A CAPTURA CONTRA A INFORMACAO (Entrada 287b)", "=" * 110]
    D["adv_cf"] = T * D.delta_cf                       # vantagem estatica com validacao cruzada, pontos do limite inferior por execucao
    pos = D[D.adv_cf > 0]
    L += [f"(i) vantagem estatica cruzada do bandido (T Delta_cf) contra o ganho observado da validada (celulas com Delta_cf > 0: {len(pos)} de {len(D)}):",
          f"    mediana de T Delta_cf / g_val (g_val >= 0,05): {float((D.adv_cf / D.g_val)[(D.g_val >= 0.05) & (D.adv_cf > 0)].median()):.3f}",
          f"    mediana de tw_oracle / g_val (em amostra, sem validacao cruzada, g_val >= 0,05): {float((D.tw_oracle / D.g_val)[D.g_val >= 0.05].median()):.3f}",
          f"    Spearman(T Delta_cf, g_val) nas 69: {sp(D.adv_cf, D.g_val):+.3f}; mediana da razao por colecao:"]
    for coll, g in D[(D.g_val >= 0.05) & (D.adv_cf > 0)].groupby("coll"):
        L.append(f"      {coll}: cruzada {float((g.adv_cf / g.g_val).median()):.3f} (n = {len(g)}); em amostra {float((g.tw_oracle / g.g_val).median()):.3f}")
    L += ["", f"(ii) captura observada do rt (celulas com g_val >= 0,05; n = {len(M)}) contra a informacao disponivel:"]
    for col, nm in (("pos_per_arm_run", "recompensas positivas por acao por execucao"), ("K", "numero de acoes K"), ("pos_rate", "taxa de recompensa positiva"), ("pos_per_run", "recompensas positivas por execucao")):
        L.append(f"    Spearman({nm}, captura do rt): {sp(M[col], M.kap_rt):+.3f}")
    for col in ("pos_per_arm_run", "K"):
        q = pd.qcut(M[col].rank(method="first"), 3, labels=["terco baixo", "terco medio", "terco alto"])
        t = M.groupby(q, observed=True).agg(celulas=("cell", "size"), mediana_do_fator=(col, "median"), captura_mediana=("kap_rt", "median"), g_val_mediano=("g_val", "median"))
        L += ["", f"    captura mediana por terco de {col}:", t.round(4).to_string()]
    text = "\n".join(L)
    print(text)
    (R / "entrada287b_posthoc.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
