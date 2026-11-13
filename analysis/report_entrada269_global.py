"""
analysis/report_entrada269_global.py
=====================================
Entrada 269 (Ciclo 42, fecho): o CONTROLE GLOBAL de falsas descobertas (Benjamini-Yekutieli, dependencia arbitraria, 0,05) estendido as baterias das Entradas 266-270 (valor do conhecimento offline por granularidade, o segundo dominio do TSP, o protocolo
a CPU igual no benchmark de Fanjul-Peyro e Ruiz, o TSP em instancias da TSPLIB), alem das de 256-264 da Entrada 265. Usa o coletor da Entrada 265 (que aceita o sufixo L) e o resumo da Entrada 247; o p bruto (Wilcoxon, bilateral) de cada teste registrado entra em Holm e em BY sobre o
conjunto TODO; um teste 'confirmada' e' o que cumpriu a regra de decisao da sua bateria. A leitura pos hoc da Entrada 266b (analysis/report_entrada266b_posthoc.py) NAO e' registrada e NAO entra nos testes contados (seu arquivo se chama entrada266b_posthoc_tests.csv e e' excluido).

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hash em results/entrada269_prereg_hashes.txt)
----------------------------------------------------------------------------------------------------
(H1) O conjunto ate' a Entrada 264, recoletado hoje, tem exatamente 1.844 testes e 1.048 'confirmada', dos quais 1.020 sobrevivem ao BY do conjunto proprio (os numeros da Entrada 265). (certo, pelo arquivo salvo)
(H2) Entre os 'confirmada' das Entradas 266-270, pelo menos 90% sobrevivem ao BY global calculado sobre o conjunto completo. (75%)
(H3) No conjunto completo (ate' a Entrada 270), pelo menos 95% dos 'confirmada' sobrevivem ao BY global. (75%)
Uso: python -m analysis.report_entrada269_global
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada215_robustness import by_adj  # noqa: E402
from analysis.report_entrada247_global import EXCLUDE, summarize  # noqa: E402

R = ROOT / "results"
EXCLUDE2 = EXCLUDE + ("posthoc",)


def collect():
    """Como a coleta das Entradas 247 e 265 (aceita os sufixos L e b), sem os arquivos pos hoc."""
    import glob
    import re

    import pandas as pd
    files = [f for f in sorted(glob.glob(str(R / "entrada*_*.csv"))) if not any(k in Path(f).name for k in EXCLUDE2)]
    parts = []
    for f in files:
        try:
            x = pd.read_csv(f)
        except Exception:
            continue
        if {"p_raw", "mean_d_pct", "id"} <= set(x.columns):
            x = x[["id", "p_raw", "mean_d_pct"] + (["confirmada"] if "confirmada" in x.columns else [])].copy()
            x["file"] = Path(f).name
            x["entrada"] = int(re.search(r"entrada(\d+)[bL]?_", Path(f).name).group(1))
            parts.append(x)
    G = pd.concat(parts, ignore_index=True)
    G = G[G.p_raw.notna()].reset_index(drop=True)
    if "confirmada" not in G.columns:
        G["confirmada"] = False
    G["confirmada"] = G.confirmada.fillna(False).astype(bool)
    return G


def main():
    G = collect()
    old = G[G.entrada <= 264].reset_index(drop=True)
    L = ["=" * 110, "CONTROLE GLOBAL ESTENDIDO AS ENTRADAS 266-270 (Entrada 269)", "=" * 110,
         f"arquivos com testes registrados: {G.file.nunique()}; testes: {len(G)}; linhas repetidas (arquivo, id): {int(G.duplicated(['file', 'id']).sum())}"]
    t_old, _, _ = summarize("ate a Entrada 264", old)
    t_all, holm, by = summarize("todas (ate a Entrada 270)", G)
    G["p_holm_global"], G["p_by_global"] = holm, by
    new = G[G.entrada >= 266]
    nconf = int(new.confirmada.sum())
    nby = int((new[new.confirmada].p_by_global < 0.05).sum())
    old_by = by_adj(old.p_raw)
    n_old_conf = int(old.confirmada.sum())
    n_old_by = int((old_by[old.confirmada.to_numpy()] < 0.05).sum())
    h1 = (len(old) == 1844 and n_old_conf == 1048 and n_old_by == 1020)
    h2 = nby / max(nconf, 1)
    h3 = float((G[G.confirmada].p_by_global < 0.05).mean())
    L += ["", t_old, t_all,
          f"entradas 266-270: {len(new)} testes em {new.file.nunique()} arquivos; {nconf} 'confirmada'; sobrevivem ao BY global (calculado sobre todos) {nby} ({h2:.3f})", "",
          f"(H1) o conjunto ate' a Entrada 264: {len(old)} testes, {n_old_conf} 'confirmada', {n_old_by} sobrevivem ao BY do conjunto proprio (previsto 1.844 / 1.048 / 1.020): {'SIM' if h1 else 'NAO'}",
          f"(H2) >= 0,90 entre os 'confirmada' das Entradas 266-270: {h2:.3f} -> {'SIM' if h2 >= 0.90 else 'NAO'}",
          f"(H3) >= 0,95 no conjunto completo: {h3:.3f} -> {'SIM' if h3 >= 0.95 else 'NAO'}",
          f"previsoes acertadas: {sum([h1, h2 >= 0.90, h3 >= 0.95])} de 3"]
    drop = G[G.confirmada & (G.p_by_global >= 0.05)][["file", "id", "mean_d_pct", "p_raw", "p_by_global"]]
    if len(drop):
        L += ["", "testes 'confirmada' que NAO sobrevivem ao BY global (conjunto completo):", drop.to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada269_global.txt").write_text(text + "\n", encoding="utf-8")
    G.to_csv(R / "entrada269_global_tests.csv", index=False)


if __name__ == "__main__":
    main()
