"""
analysis/report_entrada276_global.py
=====================================
Entrada 276 (Ciclo 46, fecho): o CONTROLE GLOBAL de falsas descobertas (Benjamini-Yekutieli, dependencia arbitraria, 0,05) estendido as baterias das Entradas 274 e 275 (a replica, no ILS do TSP sintetico e nas instancias da TSPLIB, do achado
do Q-Learning otimista aquecido por uma priori offline), alem das ate' a Entrada 272 (Entradas 247, 265, 269 e 273). Usa o coletor da Entrada 269 (que aceita os sufixos L e b e ignora as leituras pos hoc) e o resumo da Entrada 247; o p bruto
(Wilcoxon, bilateral) de cada teste registrado entra em Holm e em BY sobre o conjunto TODO; um teste 'confirmada' e' o que cumpriu a regra de decisao da sua bateria.

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hash em results/entrada276_prereg_hashes.txt)
----------------------------------------------------------------------------------------------------
(Q1) O conjunto ate' a Entrada 272, recoletado hoje, tem exatamente 3.318 testes e 1.550 'confirmada', dos quais 1.486 sobrevivem ao BY do conjunto proprio (os numeros da Entrada 273). (certo, pelo arquivo salvo)
(Q2) Entre os 'confirmada' das Entradas 274 e 275, pelo menos 90% sobrevivem ao BY global calculado sobre o conjunto completo. (75%)
(Q3) No conjunto completo (ate' a Entrada 275), pelo menos 95% dos 'confirmada' sobrevivem ao BY global. (70%)
Uso: python -m analysis.report_entrada276_global
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada215_robustness import by_adj  # noqa: E402
from analysis.report_entrada247_global import summarize  # noqa: E402
from analysis.report_entrada269_global import collect  # noqa: E402

R = ROOT / "results"


def main():
    G = collect()
    old = G[G.entrada <= 272].reset_index(drop=True)
    L = ["=" * 110, "CONTROLE GLOBAL ESTENDIDO AS ENTRADAS 274 E 275 (Entrada 276)", "=" * 110,
         f"arquivos com testes registrados: {G.file.nunique()}; testes: {len(G)}; linhas repetidas (arquivo, id): {int(G.duplicated(['file', 'id']).sum())}"]
    t_old, _, _ = summarize("ate a Entrada 272", old)
    t_all, holm, by = summarize("todas (ate a Entrada 275)", G)
    G["p_holm_global"], G["p_by_global"] = holm, by
    new = G[G.entrada >= 274]
    nconf = int(new.confirmada.sum())
    nby = int((new[new.confirmada].p_by_global < 0.05).sum())
    old_by = by_adj(old.p_raw)
    n_old_conf = int(old.confirmada.sum())
    n_old_by = int((old_by[old.confirmada.to_numpy()] < 0.05).sum())
    q1 = (len(old) == 3318 and n_old_conf == 1550 and n_old_by == 1486)
    q2 = nby / max(nconf, 1)
    q3 = float((G[G.confirmada].p_by_global < 0.05).mean())
    L += ["", t_old, t_all,
          f"entradas 274-275: {len(new)} testes em {new.file.nunique()} arquivos; {nconf} 'confirmada'; sobrevivem ao BY global (calculado sobre todos) {nby} ({q2:.3f})", "",
          f"(Q1) o conjunto ate' a Entrada 272: {len(old)} testes, {n_old_conf} 'confirmada', {n_old_by} sobrevivem ao BY do conjunto proprio (previsto 3.318 / 1.550 / 1.486): {'SIM' if q1 else 'NAO'}",
          f"(Q2) >= 0,90 entre os 'confirmada' das Entradas 274-275: {q2:.3f} -> {'SIM' if q2 >= 0.90 else 'NAO'}",
          f"(Q3) >= 0,95 no conjunto completo: {q3:.3f} -> {'SIM' if q3 >= 0.95 else 'NAO'}",
          f"previsoes acertadas: {sum([q1, q2 >= 0.90, q3 >= 0.95])} de 3"]
    drop = G[G.confirmada & (G.p_by_global >= 0.05)][["file", "id", "mean_d_pct", "p_raw", "p_by_global"]]
    if len(drop):
        L += ["", "testes 'confirmada' que NAO sobrevivem ao BY global (conjunto completo):", drop.to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada276_global.txt").write_text(text + "\n", encoding="utf-8")
    G.to_csv(R / "entrada276_global_tests.csv", index=False)


if __name__ == "__main__":
    main()
