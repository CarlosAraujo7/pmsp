"""
analysis/report_entrada265_global.py
=====================================
Entrada 265 (Ciclo 38): o CONTROLE GLOBAL de falsas descobertas (Benjamini-Yekutieli, dependencia arbitraria, 0,05) estendido as baterias das Entradas 256-264 (benchmark padrao de Fanjul-Peyro e Ruiz, mecanismo da ordem das acoes,
Q-Learning invariante a ordem, horizonte longo, politica neural, instancias publicadas held-out). Usa o coletor e o resumo da Entrada 247 (`analysis/report_entrada247_global.py`), sem reescrever a saida dela; o p bruto de cada teste registrado
(Wilcoxon, bilateral) entra em Holm e em BY sobre o conjunto TODO; um teste 'confirmada' e' o que cumpriu a regra de decisao da sua bateria (nas tabelas A/B/E/D novas, a regra e' a da propria bateria).

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hash em results/entrada265_prereg_hashes.txt)
----------------------------------------------------------------------------------------------------
(G1) O conjunto ate' a Entrada 255, recoletado hoje, tem exatamente 1.169 testes e 685 'confirmada', dos quais 663 sobrevivem ao BY global (os numeros do artigo). (certo, pelo arquivo salvo da Entrada 247)
(G2) Entre os 'confirmada' das Entradas 256-264, pelo menos 90% sobrevivem ao BY global calculado sobre o conjunto completo. (75%)
(G3) No conjunto completo (ate' a Entrada 264), pelo menos 95% dos 'confirmada' sobrevivem ao BY global. (70%)
Uso: python -m analysis.report_entrada265_global
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada215_robustness import by_adj, holm_adj  # noqa: E402
from analysis.report_entrada247_global import EXCLUDE, summarize  # noqa: E402

R = ROOT / "results"


def collect():
    """Como `collect` da Entrada 247, mas aceita o sufixo L do horizonte longo (entrada258L_...)."""
    import glob
    import re
    files = [f for f in sorted(glob.glob(str(R / "entrada*_*.csv"))) if not any(k in Path(f).name for k in EXCLUDE)]
    parts = []
    for f in files:
        try:
            x = pd.read_csv(f)
        except Exception:
            continue
        if {"p_raw", "mean_d_pct", "id"} <= set(x.columns):
            x = x[["id", "p_raw", "mean_d_pct"] + (["confirmada"] if "confirmada" in x.columns else [])].copy()
            x["file"] = Path(f).name
            x["entrada"] = int(re.search(r"entrada(\d+)L?_", Path(f).name).group(1))
            parts.append(x)
    G = pd.concat(parts, ignore_index=True)
    G = G[G.p_raw.notna()].reset_index(drop=True)
    if "confirmada" not in G.columns:
        G["confirmada"] = False
    G["confirmada"] = G.confirmada.fillna(False).astype(bool)
    return G


def main():
    G = collect()
    old = G[G.entrada <= 255].reset_index(drop=True)
    L = ["=" * 110, "CONTROLE GLOBAL ESTENDIDO AS ENTRADAS 256-264 (Entrada 265)", "=" * 110,
         f"arquivos com testes registrados: {G.file.nunique()}; testes: {len(G)}; linhas repetidas (arquivo, id): {int(G.duplicated(['file', 'id']).sum())}"]
    t_old, _, _ = summarize("ate a Entrada 255 (o conjunto do artigo)", old)
    t_all, holm, by = summarize("todas (ate a Entrada 264)", G)
    G["p_holm_global"], G["p_by_global"] = holm, by
    new = G[G.entrada >= 256]
    nconf = int(new.confirmada.sum())
    nby = int((new[new.confirmada].p_by_global < 0.05).sum())
    old_by = by_adj(old.p_raw)
    n_old_conf = int(old.confirmada.sum())
    n_old_by = int((old_by[old.confirmada.to_numpy()] < 0.05).sum())
    g1 = (len(old) == 1169 and n_old_conf == 685 and n_old_by == 663)
    g2 = nby / max(nconf, 1)
    g3 = float((G[G.confirmada].p_by_global < 0.05).mean())
    L += ["", t_old, t_all,
          f"entradas 256-264: {len(new)} testes em {new.file.nunique()} arquivos; {nconf} 'confirmada'; sobrevivem ao BY global (calculado sobre todos) {nby} ({g2:.3f})", "",
          f"(G1) o conjunto ate' a Entrada 255: {len(old)} testes, {n_old_conf} 'confirmada', {n_old_by} sobrevivem ao BY do conjunto proprio (previsto 1.169 / 685 / 663): {'SIM' if g1 else 'NAO'}",
          f"(G2) >= 0,90 entre os 'confirmada' das Entradas 256-264: {g2:.3f} -> {'SIM' if g2 >= 0.90 else 'NAO'}",
          f"(G3) >= 0,95 no conjunto completo: {g3:.3f} -> {'SIM' if g3 >= 0.95 else 'NAO'}",
          f"previsoes acertadas: {sum([g1, g2 >= 0.90, g3 >= 0.95])} de 3"]
    drop = G[G.confirmada & (G.p_by_global >= 0.05)][["file", "id", "mean_d_pct", "p_raw", "p_by_global"]]
    if len(drop):
        L += ["", "testes 'confirmada' que NAO sobrevivem ao BY global (conjunto completo):", drop.to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada265_global.txt").write_text(text + "\n", encoding="utf-8")
    G.to_csv(R / "entrada265_global_tests.csv", index=False)


if __name__ == "__main__":
    main()
