"""
analysis/report_entrada273_global.py
=====================================
Entrada 273 (Ciclo 44, fecho): o CONTROLE GLOBAL de falsas descobertas (Benjamini-Yekutieli, dependencia arbitraria, 0,05) estendido as baterias das Entradas 271 e 272 (Q-Learning otimista aquecido por uma priori offline e o controle sem aprendizagem),
alem das ate' a Entrada 270 (Entradas 247, 265 e 269). Usa o coletor da Entrada 269 (que aceita os sufixos L e b e ignora as leituras pos hoc) e o resumo da Entrada 247; o p bruto (Wilcoxon, bilateral) de cada teste registrado entra em Holm e em BY sobre o
conjunto TODO; um teste 'confirmada' e' o que cumpriu a regra de decisao da sua bateria.

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hash em results/entrada273_prereg_hashes.txt)
----------------------------------------------------------------------------------------------------
(J1) O conjunto ate' a Entrada 270, recoletado hoje, tem exatamente 3.240 testes e 1.497 'confirmada', dos quais 1.435 sobrevivem ao BY do conjunto proprio (os numeros da Entrada 269). (certo, pelo arquivo salvo)
(J2) Entre os 'confirmada' das Entradas 271 e 272, pelo menos 90% sobrevivem ao BY global calculado sobre o conjunto completo. (75%)
(J3) No conjunto completo (ate' a Entrada 272), pelo menos 95% dos 'confirmada' sobrevivem ao BY global. (75%)
Uso: python -m analysis.report_entrada273_global
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
    old = G[G.entrada <= 270].reset_index(drop=True)
    L = ["=" * 110, "CONTROLE GLOBAL ESTENDIDO AS ENTRADAS 271 E 272 (Entrada 273)", "=" * 110,
         f"arquivos com testes registrados: {G.file.nunique()}; testes: {len(G)}; linhas repetidas (arquivo, id): {int(G.duplicated(['file', 'id']).sum())}"]
    t_old, _, _ = summarize("ate a Entrada 270", old)
    t_all, holm, by = summarize("todas (ate a Entrada 272)", G)
    G["p_holm_global"], G["p_by_global"] = holm, by
    new = G[G.entrada >= 271]
    nconf = int(new.confirmada.sum())
    nby = int((new[new.confirmada].p_by_global < 0.05).sum())
    old_by = by_adj(old.p_raw)
    n_old_conf = int(old.confirmada.sum())
    n_old_by = int((old_by[old.confirmada.to_numpy()] < 0.05).sum())
    j1 = (len(old) == 3240 and n_old_conf == 1497 and n_old_by == 1435)
    j2 = nby / max(nconf, 1)
    j3 = float((G[G.confirmada].p_by_global < 0.05).mean())
    L += ["", t_old, t_all,
          f"entradas 271-272: {len(new)} testes em {new.file.nunique()} arquivos; {nconf} 'confirmada'; sobrevivem ao BY global (calculado sobre todos) {nby} ({j2:.3f})", "",
          f"(J1) o conjunto ate' a Entrada 270: {len(old)} testes, {n_old_conf} 'confirmada', {n_old_by} sobrevivem ao BY do conjunto proprio (previsto 3.240 / 1.497 / 1.435): {'SIM' if j1 else 'NAO'}",
          f"(J2) >= 0,90 entre os 'confirmada' das Entradas 271-272: {j2:.3f} -> {'SIM' if j2 >= 0.90 else 'NAO'}",
          f"(J3) >= 0,95 no conjunto completo: {j3:.3f} -> {'SIM' if j3 >= 0.95 else 'NAO'}",
          f"previsoes acertadas: {sum([j1, j2 >= 0.90, j3 >= 0.95])} de 3"]
    text = "\n".join(L)
    print(text)
    (R / "entrada273_global.txt").write_text(text + "\n", encoding="utf-8")
    G.to_csv(R / "entrada273_global_tests.csv", index=False)


if __name__ == "__main__":
    main()
