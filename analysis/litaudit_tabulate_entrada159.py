"""
analysis/litaudit_tabulate_entrada159.py
==========================================
Tabula a auditoria de literatura da Entrada 159 (protocolo: analysis/litaudit_codebook_entrada159.md). Os codigos abaixo foram atribuidos lendo o texto completo (versao HTML do arXiv) de cada artigo incluido;
a leitura foi feita com a ferramenta de recuperacao de texto do assistente (resumo por pergunta com citacoes literais); casos duvidosos foram conferidos com uma segunda consulta (ex.: 2207.03876, parte 2).
Saida: results/litaudit_entrada159.csv e results/entrada159_litaudit.txt (contagens e comparacao com as previsoes registradas antes de ler).
Emendas (todas ANTES de qualquer artigo ser lido, exceto E2 que e' posterior a triagem por resumo): E1 -- as 8 consultas foram traduzidas para a sintaxe da busca do arXiv (as formas estritas devolveram 0-3 resultados) e
4 consultas adicionais (q9-q12) foram acrescentadas para ampliar a amostra; E2 -- na etapa de texto completo, 2401.05350 foi excluido (texto completo inacessivel: a versao HTML era um modelo vazio e a extracao de PDF expirou) e
2601.11189 (seletor de regras de despacho, sem busca iterativa) e 2310.09686 (hospedeiro: geracao de colunas exata) foram excluidos por nao terem metaheuristica hospedeira (criterio de inclusao i).
Uso: python -m analysis.litaudit_tabulate_entrada159
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
COLS = ["id", "year", "problem", "learner", "host", "random_control", "budget_basis", "learner_overhead_reported", "offline_tuned_static", "tune_test_split", "stat_test", "multiplicity",
        "ablation_policy", "code_or_data", "claim_policy_helps", "note"]
INCLUDED = [
    ("2607.02315", 2026, "1D bin packing", "tabular Q-learning", "hybrid grouping GA", "N", "ITER", "N", "U", "U", "N", "NA", "Y", "U", "MIXED", "ablation vs fixed-probability selection (not random); learner cost not separated"),
    ("2408.14864", 2024, "permutation flowshop", "tabular Q-learning", "iterated greedy / ILS", "Y", "CPU", "Y", "U", "U", "Y", "U", "Y", "Y", "Y", "RIG = random selection; Wilcoxon; overhead discussed analytically"),
    ("2202.12866", 2022, "mining complexes (stochastic IP)", "actor-critic RL", "multi-neighbourhood SA", "N", "MIXED", "N", "N", "N", "N", "NA", "Y", "U", "MIXED", "lambda_RL = 0 ablation; same instances for training and test in stages 1-2"),
    ("2506.02746", 2025, "pod repositioning", "PPO", "ALNS", "N", "U", "Y", "Y", "N", "U", "NA", "N", "U", "Y", "ALNS tuned by random search on the small instance; training cost reported"),
    ("2211.00759", 2022, "OPSWTW (and CVRP, TSP, mTSP)", "PPO", "ALNS", "N", "ITER", "Y", "Y", "Y", "Y", "U", "N", "Y", "Y", "ALNS-BO tuned with SMAC3 on separate instances; significance test unnamed"),
    ("2505.23098", 2025, "VRP with multiple time windows", "PPO transformer policy", "adaptive VNS", "Y", "MIXED", "N", "U", "Y", "N", "NA", "Y", "U", "Y", "RVNS random policy within the same VNS"),
    ("2305.02805", 2023, "CVRP", "none (LOC modifier on PM/AP/DRL selectors)", "local search with 17 operators", "N", "ITER", "Y", "N", "U", "Y", "U", "N", "U", "Y", "Wilcoxon; parameters of PM/AP 'arbitrarily set'; third-party code only"),
    ("2404.10252", 2024, "CVRPTW and real-valued benchmarks", "DRL (DDQN/DQN) + adaptive pursuit", "DE / CVRPTW host", "Y", "U", "Y", "U", "Y", "Y", "U", "Y", "U", "Y", "Random selection compared; overhead qualitative only; budget by reference to prior work"),
    ("2512.19321", 2025, "urban cable routing", "multi-agent DRL (REINFORCE) for position sampling", "VNS", "Y", "CPU", "Y", "N", "U", "Y", "U", "Y", "Y", "Y", "uniform random sampling replaces DRL-guided sampling (positions, not operators)"),
    ("2206.03185", 2022, "capacitated electric VRP", "bandits (eps-greedy, TS, UCB1)", "adaptive SA", "Y", "ITER", "N", "U", "U", "Y", "Y", "Y", "N", "Y", "Friedman + Holm; code not available; baselines taken from the literature"),
    ("2304.04022", 2023, "team formation (0-1 IP)", "tabular Q-learning", "genetic programming", "N", "U", "N", "N", "U", "N", "NA", "N", "U", "Y", "claim confounded with ensemble and surrogate; no ablation of the selector"),
    ("2207.03876", 2022, "TSP, TSPTW, CTSP", "Q-learning / Sarsa / Monte Carlo", "LKH / LKH-3", "N", "MIXED", "N", "N", "U", "N", "NA", "Y", "Y", "Y", "ablation against fixed candidate-set choices (not random); same time/iteration limits for all"),
]
EXCLUDED = [
    ("2401.05350", "full text inaccessible (HTML template; PDF extraction timed out)"),
    ("2601.11189", "no metaheuristic host (dispatching-rule selector in a constructive procedure)"),
    ("2310.09686", "host is exact column generation, not a (meta)heuristic search"),
    ("2304.06675", "learned component is a surrogate model, not a selector (abstract screening)"),
    ("2004.10874", "continuous multi-objective benchmark, not combinatorial (abstract screening)"),
    ("1905.08006", "continuous parameter control in differential evolution (abstract screening)"),
    ("1812.07958", "continuous multi-objective engineering optimisation (abstract screening)"),
]


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def main():
    d = pd.DataFrame(INCLUDED, columns=COLS)
    d.insert(1, "included", "Y")
    ex = pd.DataFrame([(i, "N", r) for i, r in EXCLUDED], columns=["id", "included", "note"])
    pd.concat([d, ex], ignore_index=True).to_csv(R / "litaudit_entrada159.csv", index=False)
    n = len(d)
    c = lambda col, v: int((d[col] == v).sum())
    nt = c("stat_test", "Y")
    L = ["=" * 100, "AUDITORIA DE LITERATURA (Entrada 159): prevalencia dos controles do protocolo", "=" * 100,
         f"artigos incluidos: {n} (arXiv, 2015-2026; texto completo aberto; um codificador; ver emendas no cabecalho do script); excluidos apos triagem/texto completo: {len(EXCLUDED)}",
         "Leitura principal: 'U' conta como 'nao demonstrado'.", ""]
    rows = [
        ("controle aleatorio das mesmas acoes (random_control = Y)", c("random_control", "Y"), n, "<40%"),
        ("orcamento por CPU/tempo nas comparacoes (budget_basis = CPU)", c("budget_basis", "CPU"), n, "<30%"),
        ("custo do aprendiz reportado/discutido (learner_overhead_reported = Y)", c("learner_overhead_reported", "Y"), n, "<40%"),
        ("concorrente estatico ajustado offline (offline_tuned_static = Y)", c("offline_tuned_static", "Y"), n, "<25%"),
        ("instancias de ajuste separadas das de teste (tune_test_split = Y)", c("tune_test_split", "Y"), n, "(sem previsao)"),
        ("teste estatistico (stat_test = Y)", nt, n, "<50%"),
        ("correcao de multiplicidade entre os que testam (multiplicity = Y)", c("multiplicity", "Y"), nt, "<15%"),
        ("ablacao que isola a politica do mecanismo (ablation_policy = Y)", c("ablation_policy", "Y"), n, "<40%"),
        ("codigo ou dados disponiveis (code_or_data = Y)", c("code_or_data", "Y"), n, "(sem previsao)"),
    ]
    for lab, k, m, pred in rows:
        L.append(f"{lab:78s} {pct(k, m):>10s}   previsao registrada: {pred}")
    L.append("")
    L.append("budget_basis: " + ", ".join(f"{v} {int((d.budget_basis == v).sum())}" for v in ["CPU", "ITER", "MIXED", "U"]))
    L.append("claim_policy_helps: " + ", ".join(f"{v} {int((d.claim_policy_helps == v).sum())}" for v in ["Y", "MIXED", "N"]) + "  (nenhum artigo conclui que o componente aprendido nao ajuda)")
    pred_ok = [c("random_control", "Y") / n < 0.40, c("budget_basis", "CPU") / n < 0.30, c("learner_overhead_reported", "Y") / n < 0.40, c("offline_tuned_static", "Y") / n < 0.25,
               nt / n < 0.50, (c("multiplicity", "Y") / nt if nt else 0) < 0.15, c("ablation_policy", "Y") / n < 0.40]
    L.append(f"previsoes registradas antes de ler que se confirmaram: {sum(pred_ok)} de 7 (orcamento por CPU e ajuste offline, sim; controle aleatorio, custo do aprendiz, teste estatistico, multiplicidade e ablacao, nao: "
             "os controles baratos sao mais comuns do que eu previ; os controles de ORCAMENTO, de AJUSTE OFFLINE, de SEPARACAO ajuste/teste e de MULTIPLICIDADE sao raros)")
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(d.id.values, size=8, replace=False).tolist())
    L.append("\nArtigos sorteados (semente 0) para dupla codificacao por Carlos: " + ", ".join(pick))
    L.append("Limites: n pequeno; so' arXiv (texto aberto); um codificador (assistente de IA) com leitura por recuperacao de texto; sem inter-codificacao ate' Carlos revisar a amostra.")
    text = "\n".join(L)
    print(text)
    (R / "entrada159_litaudit.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
