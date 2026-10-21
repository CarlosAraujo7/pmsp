"""
analysis/litaudit_tabulate_round2_entrada172.py
================================================
Auditoria de literatura, RODADA 2 (Entrada 172; emenda em analysis/litaudit_codebook_round2_entrada172.md, congelada e com hash antes de ler qualquer artigo novo) e amostra AGRUPADA (rodada 1 + rodada 2).
Os codigos foram atribuidos lendo o texto completo de cada artigo (HTML do arXiv por recuperacao de texto com uma pergunta por item do livro de codigos; para 2506.08608 e 2311.05937, sem HTML, o PDF foi baixado e lido com pdftotext;
para 2501.10243, a segunda metade do texto foi lida com uma segunda consulta). Regras de codificacao iguais as da rodada 1: 'U' = nao determinavel pelo texto (conta como 'nao demonstrado'); multiplicity = 'U' para quem faz teste e nao menciona correcao,
'NA' para quem nao testa; ablation_policy = Y quando ha variante do MESMO hospedeiro sem a politica aprendida (aleatoria, rodizio, selecao fixa ou aprendizado desligado); budget_basis = CPU para limite de tempo comum a todos os metodos.
Saida: results/litaudit_round2_entrada172.csv, results/litaudit_pooled_entrada172.csv e results/entrada172_litaudit.txt (contagens, previsoes registradas antes de ler e a leitura pre-fixada).
Uso: python -m analysis.litaudit_tabulate_round2_entrada172
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.litaudit_tabulate_entrada159 import COLS, INCLUDED as ROUND1

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
ROUND2 = [
    ("2601.07948", 2026, "TSP, PDPTW, car sequencing", "eps-greedy/UCB bandits, DDQN, PPO choosing the move operator", "constraint-based local search (OscaR-CBLS)", "Y", "CPU", "Y", "N", "Y", "N", "NA", "Y", "N", "MIXED",
     "Random and round-robin selectors on the same host and time limit; irace tunes the learners only; 50% of instances reserved for tuning; 'no single method dominates'; overhead discussed, not quantified"),
    ("2312.16767", 2023, "multi-agent path finding", "bi-level bandit (roulette, UCB1, Thompson) choosing destroy heuristic and neighbourhood size", "MAPF-LNS", "Y", "CPU", "Y", "Y", "U", "N", "NA", "Y", "Y", "Y",
     "BALANCE (Random) is the random control; empirically best fixed choice by grid search over the 15 combinations (tuning split not stated); overhead 'negligible', not measured; 95% CI bands, no test"),
    ("2412.14382", 2024, "MIP (D-MIPLIB, MIPLIB2017-Hard)", "bandit choosing destroy/repair pair", "ALNS on SCIP/Gurobi", "N", "U", "N", "N", "N", "N", "NA", "Y", "Y", "Y",
     "with/without bandit and single-neighbourhood LNS compared; no random-selection control; means and SDs only"),
    ("2508.06736", 2025, "MIP (auctions, set cover, independent set, real-world)", "bandit (eps-greedy, softmax, Thompson) choosing destroy operator", "parallel ALNS (ParBalans)", "N", "U", "N", "N", "N", "N", "NA", "N", "Y", "N",
     "only comparison is against Gurobi at a common time limit (no non-learning variant); claims the gain of parallelism, not of the learned selector"),
    ("2509.15828", 2025, "ILPs (MIS, CA, MVC, SC)", "PPO policy predicting the neighbourhood size", "large neighbourhood search", "Y", "CPU", "N", "N", "Y", "N", "NA", "Y", "N", "Y",
     "fixed 30/50/70% sizes and random size distributions on the same host (Table 3); separate train/validation/test instances; mean and SD over 5 runs, no test; no code"),
    ("2004.00422", 2020, "ILPs (MVC, max-cut, auctions, path planning)", "learned decomposition policy (behaviour cloning, forward training, REINFORCE)", "decomposition-based LNS", "Y", "U", "N", "U", "Y", "N", "NA", "Y", "U", "MIXED",
     "Random-LNS shares the framework; wall-clock limits; 100/10/50 train/validation/test; imitation beats random in most cases, RL-LNS does not clearly"),
    ("2506.08608", 2025, "steelmaking-continuous casting scheduling", "Q-learning cooperating inside local-search frameworks (neighbourhood operators)", "hierarchical cooperative local search (HierC_Q)", "N", "CPU", "N", "U", "U", "Y", "Y", "U", "Y", "Y",
     "equal time limit for all algorithms; ANOVA/Tukey HSD intervals; ablations and parameter calibration only in the supplementary material (not read); instances and code on GitHub"),
    ("2403.12389", 2024, "minmax multiple TSP", "multi-armed bandit choosing removal and insertion operators", "iterated local search (MILS)", "Y", "U", "N", "N", "N", "Y", "U", "Y", "Y", "Y",
     "MILS2 = random operator selection, MILS1 = roulette; Wilcoxon, no correction mentioned; irace on 8 instances that belong to the 77 reported; budget of the variants not stated"),
    ("2501.10243", 2025, "integrated operating room scheduling", "Q-learning controlling BRKGA parameters", "biased random-key GA (BRKGA-QL)", "N", "U", "N", "N", "U", "Y", "U", "N", "Y", "MIXED",
     "SA and ILS tuned offline are other algorithms; no BRKGA without Q-learning; tests without correction; budget equality not stated; 'no single method outperformed the others in all cases'"),
    ("2311.05937", 2023, "permutation flowshop (Taillard)", "online/offline DRL choosing the parent-selection operator, rates and mutation", "genetic algorithm", "N", "U", "Y", "N", "U", "N", "NA", "Y", "U", "Y",
     "standard GA with fixed roulette parameters is the learning-free variant; population/iteration settings differ between methods; times compared (CPU/GPU); no test; tuning instances not separated"),
    ("2206.05694", 2022, "electromagnetic detection satellite scheduling", "Q-learning choosing crossover/mutation operators (15 actions)", "genetic algorithm (RL-GA)", "N", "ITER", "Y", "N", "N", "Y", "U", "N", "N", "Y",
     "same evaluation limit (5000) for all; Wilcoxon rank-sum without correction; ablation removes elite retention, not the policy; CPU times reported"),
]
EXCLUDED2 = [
    ("2404.03101", "cooperative multi-agent RL training, not a metaheuristic with a learned selector (abstract screening)"),
    ("2411.09068", "end-to-end RL for network design, no iterative-search host (abstract screening)"),
    ("2205.00772", "neural construction heuristic used as the repair operator, no selection (abstract screening)"),
    ("2007.05971", "probability learning guides the perturbation, no operator/strategy/parameter selection (abstract screening; liberal criterion not applied to avoid inflating the share of weakly controlled papers)"),
    ("2207.03244", "supervised predictor of permutation quality, not a selector (abstract screening)"),
    ("2409.02697", "neural local-search policy without operator selection (abstract screening)"),
    ("2607.22036", "runtime analysis (theory)"),
    ("2605.28844", "continuous optimisation (abstract screening)"),
    ("2508.16915", "hyper-parameter optimisation of a neural network, not combinatorial (abstract screening)"),
    ("2002.07443", "Monte Carlo hyper-heuristic without a learned component (abstract screening)"),
]


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def main():
    d2 = pd.DataFrame(ROUND2, columns=COLS)
    d2.insert(1, "included", "Y")
    ex = pd.DataFrame([(i, "N", r) for i, r in EXCLUDED2], columns=["id", "included", "note"])
    pd.concat([d2, ex], ignore_index=True).to_csv(R / "litaudit_round2_entrada172.csv", index=False)
    d1 = pd.DataFrame(ROUND1, columns=COLS)
    d1.insert(1, "included", "Y")
    d1.insert(2, "round", 1)
    d2 = d2.assign(round=2)[list(d1.columns)]
    d = pd.concat([d1, d2], ignore_index=True)
    d.to_csv(R / "litaudit_pooled_entrada172.csv", index=False)
    n = len(d)
    c = lambda col, v, x=d: int((x[col] == v).sum())
    L = ["=" * 100, "AUDITORIA DE LITERATURA, RODADA 2 E AMOSTRA AGRUPADA (Entrada 172)", "=" * 100,
         f"incluidos: rodada 1 = {len(d1)}, rodada 2 = {len(d2)}, agrupada = {n}; excluidos na rodada 2: {len(EXCLUDED2)} (lista em results/litaudit_round2_entrada172.csv)", ""]
    for lab, x in (("RODADA 2 (11 novos)", d2), ("AGRUPADA", d)):
        m = len(x)
        nt = c("stat_test", "Y", x)
        L.append(f"--- {lab} ---")
        for name, k, tot in [("random_control = Y", c("random_control", "Y", x), m), ("budget_basis = CPU", c("budget_basis", "CPU", x), m), ("learner_overhead_reported = Y", c("learner_overhead_reported", "Y", x), m),
                             ("offline_tuned_static = Y", c("offline_tuned_static", "Y", x), m), ("tune_test_split = Y", c("tune_test_split", "Y", x), m), ("stat_test = Y", nt, m),
                             ("multiplicity = Y (entre os que testam)", c("multiplicity", "Y", x), nt), ("ablation_policy = Y", c("ablation_policy", "Y", x), m), ("code_or_data = Y", c("code_or_data", "Y", x), m)]:
            L.append(f"{name:42s} {pct(k, tot):>10s}")
        L.append("budget_basis: " + ", ".join(f"{v} {c('budget_basis', v, x)}" for v in ["CPU", "ITER", "MIXED", "U"]))
        L.append("claim_policy_helps: " + ", ".join(f"{v} {c('claim_policy_helps', v, x)}" for v in ["Y", "MIXED", "N"]))
        L.append("")
    nt = c("stat_test", "Y")
    claim = c("claim_policy_helps", "Y") + c("claim_policy_helps", "MIXED")
    checks = [("random_control = Y em 25% a 55%", 0.25 <= c("random_control", "Y") / n <= 0.55), ("budget_basis = CPU em menos de 25%", c("budget_basis", "CPU") / n < 0.25),
              ("offline_tuned_static = Y em menos de 25%", c("offline_tuned_static", "Y") / n < 0.25), ("tune_test_split = Y em menos de 40%", c("tune_test_split", "Y") / n < 0.40),
              ("multiplicity = Y em menos de 20% dos que testam", c("multiplicity", "Y") / nt < 0.20), ("ablation_policy = Y em 40% a 75%", 0.40 <= c("ablation_policy", "Y") / n <= 0.75),
              ("claim_policy_helps = Y ou MIXED em pelo menos 90%", claim / n >= 0.90)]
    L.append("--- previsoes registradas antes de ler (emenda da Entrada 172) ---")
    for lab, ok in checks:
        L.append(f"{lab:56s} {'ACERTOU' if ok else 'ERROU'}")
    L.append(f"acertos: {sum(ok for _, ok in checks)} de {len(checks)}")
    cpu, off = c("budget_basis", "CPU") / n, c("offline_tuned_static", "Y") / n
    L.append(f"\nLeitura pre-fixada: 'os controles que tornam a comparacao justa ao orcamento sao raros' CONFIRMADA se budget_basis = CPU e offline_tuned_static = Y ficam ambos abaixo de 33%: "
             f"CPU {100 * cpu:.0f}%, offline {100 * off:.0f}% -> {'CONFIRMADA' if cpu < 0.33 and off < 0.33 else 'NAO CONFIRMADA (enfraquecer o texto)'}")
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(d.id.values, size=8, replace=False).tolist())
    L.append("\nArtigos sorteados (semente 0, sobre a amostra agrupada de " + str(n) + ") para dupla codificacao por Carlos: " + ", ".join(pick) + "  (substitui a lista de 8 da rodada 1, que foi sorteada sobre 12)")
    L.append("Limites: so' arXiv (texto aberto); um codificador (assistente de IA) com leitura por recuperacao de texto; os artigos de 2506.08608 (parte suplementar nao lida) e 2501.10243 (parte do texto) tem celulas 'U' por leitura incompleta; sem inter-codificacao ate' Carlos revisar a amostra.")
    text = "\n".join(L)
    print(text)
    (R / "entrada172_litaudit.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
