"""
analysis/report_entrada247_global.py
=====================================
Ciclo 25: CONTROLE GLOBAL de falsas descobertas sobre TODAS as baterias registradas (Entradas 180-249), corrigindo a contagem do controle global da Entrada 215.

Por que corrigir: `analysis/report_entrada215_robustness.py` grava `results/entrada215_robustness_global.csv` no mesmo diretorio e no mesmo padrao `entrada*_*.csv` que ele proprio le; ao ser reexecutado, o proprio arquivo de saida entrou de novo no conjunto (731 linhas), e a contagem do artigo ('1.562 testes, 855 confirmados, 829 sobrevivem') ficou DUPLICADA. A contagem correta dos testes registrados ate a bateria do algoritmo genetico (Entrada 220) e' 1.562 - 731 = 831. Este relatorio exclui todo arquivo de saida de controle global e confere a identidade (arquivo, id) de cada teste.

Metodo: o p bruto (Wilcoxon, bilateral) de cada teste registrado entra em Holm e em Benjamini-Yekutieli (dependencia arbitraria) a 0,05; um teste 'confirmada' e' o que cumpriu a regra de decisao registrada da sua bateria. Conta-se quantos dos 'confirmada' sobrevivem. E' uma analise de robustez descritiva, sem hipotese nova sobre os dados.

PRE-REGISTRO (congelado ANTES de rodar este relatorio; hash em results/entrada247_prereg_hashes.txt)
----------------------------------------------------------------------------------------------------
(K1) No conjunto original corrigido (Entradas ate 220, o do artigo: 831 testes), >= 95% dos testes 'confirmada' sobrevivem ao BY global. (90%)
(K2) No conjunto completo (Entradas ate 244), >= 95% dos 'confirmada' sobrevivem ao BY global. (80%)
(K3) Entre os 'confirmada' das Entradas 223-244 (otimista, R||Cmax, cronograma, desenhos publicados, irace), >= 90% sobrevivem ao BY global. (65%)
(K4) Conferencia da contagem: o conjunto ate a Entrada 220, sem o arquivo de saida da Entrada 215, tem exatamente 831 testes. (certo, pelo arquivo salvo)
Uso: python -m analysis.report_entrada247_global
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada215_robustness import by_adj, holm_adj  # noqa: E402

R = ROOT / "results"
EXCLUDE = ("cells", "k.csv", "headroom", "family", "summary", "power", "plant", "by_family", "desc", "pooled", "corr", "by_set", "global")


def collect():
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
            x["entrada"] = int(re.search(r"entrada(\d+)_", Path(f).name).group(1))
            parts.append(x)
    G = pd.concat(parts, ignore_index=True)
    G = G[G.p_raw.notna()].reset_index(drop=True)
    if "confirmada" not in G.columns:
        G["confirmada"] = False
    G["confirmada"] = G.confirmada.fillna(False).astype(bool)
    return G


def summarize(label, G):
    p = G.p_raw.to_numpy(float)
    holm, by = holm_adj(p), by_adj(p)
    conf = G.confirmada.to_numpy()
    nc = int(conf.sum())
    sh, sb = int((holm[conf] < 0.05).sum()), int((by[conf] < 0.05).sum())
    return (f"{label}: {len(G)} testes em {G.file.nunique()} arquivos; {nc} 'confirmada'; sobrevivem a Holm global {sh} ({sh / max(nc, 1):.3f}) "
            f"e a Benjamini-Yekutieli global {sb} ({sb / max(nc, 1):.3f})"), holm, by


def main():
    G = collect()
    dup = int(G.duplicated(["file", "id"]).sum())
    L = ["=" * 110, "CONTROLE GLOBAL SOBRE TODAS AS BATERIAS REGISTRADAS (Entrada 247)", "=" * 110,
         f"arquivos com testes registrados: {G.file.nunique()}; testes: {len(G)}; linhas repetidas (arquivo, id): {dup}"]
    base = G[G.entrada <= 220].reset_index(drop=True)
    new = G[G.entrada >= 223]
    t0, _, _ = summarize("ate a Entrada 220 (algoritmo genetico), corrigido", base)
    t1, holm, by = summarize("todas (ate a Entrada 249)", G)
    G["p_holm_global"], G["p_by_global"] = holm, by
    nn = G[G.entrada >= 223]
    nconf = int(nn.confirmada.sum())
    nby = int((nn[nn.confirmada].p_by_global < 0.05).sum())
    L += ["", t0, t1,
          f"entradas 223-244: {len(nn)} testes em {nn.file.nunique()} arquivos; {nconf} 'confirmada'; sobrevivem ao BY global (calculado sobre todos) {nby} ({nby / max(nconf, 1):.3f})"]
    K1 = float((by_adj(base.p_raw)[base.confirmada.to_numpy()] < 0.05).mean())
    K2 = float((G[G.confirmada].p_by_global < 0.05).mean())
    K3 = nby / max(nconf, 1)
    L += ["", f"K1 (>= 0,95, conjunto corrigido ate 220): {K1:.3f} -> {'SIM' if K1 >= 0.95 else 'NAO'}",
          f"K2 (>= 0,95, conjunto completo): {K2:.3f} -> {'SIM' if K2 >= 0.95 else 'NAO'}",
          f"K3 (>= 0,90, entradas 223-244): {K3:.3f} -> {'SIM' if K3 >= 0.90 else 'NAO'}",
          f"K4 (831 testes ate a Entrada 220): {len(base)} -> {'SIM' if len(base) == 831 else 'NAO'}"]
    drop = G[G.confirmada & (G.p_by_global >= 0.05)][["file", "id", "mean_d_pct", "p_raw", "p_by_global"]]
    if len(drop):
        L += ["", "testes 'confirmada' que NAO sobrevivem ao BY global (conjunto completo):", drop.to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada247_global.txt").write_text(text + "\n", encoding="utf-8")
    G.to_csv(R / "entrada247_global_tests.csv", index=False)


if __name__ == "__main__":
    main()
