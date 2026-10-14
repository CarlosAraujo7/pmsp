# Auditoria no papel de revisor — fim do Ciclo 3 (10/10/2026)

Regra mestre: plano -> execução -> auditoria como revisor -> novo plano (`LEI_GERAL_PESQUISA.md` II.13). Evidência do Ciclo 3 (Entradas 121–123): lei do orçamento (não confirmada; a folga
cresce com T apertado mas o aprendiz não a usa), ações d com T=5000 e pré-treino (não resolvem; o ajuste fixo por validação vence), R||Cmax a CPU igual (mecanismo −2,2%), IG ajustado +
semeado (8/8: 74,7% ótimo provado, melhora MULTIFIT/LDM em todos os conjuntos).

## Fragilidades que um revisor do EJOR apontaria agora

| # | Fragilidade | Gravidade | Classe | Ação (Ciclo 4) |
|---|---|---|---|---|
| G1 | A "lei da folga" (a política acompanha a folga: R||Cmax, Spearman 0,68 entre 20 células; razão política/folga 0,38) é **exploratória**, descoberta nos mesmos dados | Alta | corrigível | Confirmação pré-especificada em **instâncias novas** de R||Cmax (seeds novos, 20 células x 20 reps) com os braços fixos e o Q-Learning: Spearman >= 0,5 e razão agregada entre 0,2 e 0,6 |
| G2 | No R||Cmax o UCB1 não capta a folga (+2e-7%) mas o Q-Learning capta (−0,145%): **de onde vem o ganho — estado ou TD?** A ablação 2x2 do P||Cmax não vale para o R | Alta | corrigível | Ablação no R: `ql_flat` (sem estado), `ucb_ctx` (estado sem TD), `ql` e `random`, 400 instâncias (ou 160), hipóteses pré-especificadas |
| G3 | "O mecanismo destruir-reconstruir ganha": no P||Cmax o IG (sem busca local) bate o ILS; no R||Cmax o IG **puro** perde para o ILS (+1,4%) e só a **combinação** IG+LS ganha (−5,4% sobre o IG puro). A comparação IG vs ILS mistura destruição/reconstrução com busca local | Alta | corrigível | No P||Cmax, IG+LS (reconstrução + descida com os operadores do ILS): IG+LS vs IG vs ILS nas 12 classes; reescrever a tese como "a estrutura de busca (destruir-reconstruir, com ou sem busca local)" |
| G4 | O manuscrito vai crescer muito (≥ 10 frentes novas): risco de falta de espinha dorsal | Média | corrigível (escrita) | Mapa de evidências: 1 tese, 5 afirmações, para cada uma a(s) frente(s) que a sustentam e as que a refutam; detalhes no suplemento |
| G5 | Muitos testes pré-especificados (> 70 hipóteses): falta quadro global (confirmadas/não) | Média | corrigível (texto) | Quadro único no suplemento com todas as famílias e vereditos, incluindo as não confirmadas (lei do orçamento, S1 da Entrada 109, robustez a hiperparâmetros, RB1, D1/D3) |
| G6 | R||Cmax: heurísticas não são estado da arte e o MILP-60s ganha (+0,95% a T=5000); instâncias sintéticas, não o benchmark de Fanjul-Peyro & Ruiz | Média | limitação / depende de Carlos | Declarar; pedir autorização para baixar o benchmark padrão de R||Cmax (arquivo de terceiro) ou citar como trabalho futuro |
| G7 | Aprendizado só tabular; uma recompensa densa; um pré-treino simples | Média | limitação | Declarar; a referência "melhor braço" (ajuste por validação) delimita o que um aprendiz poderia ganhar |
| G8 | Margem de equivalência ±0,05% arbitrária | Baixa | texto | Justificar e mostrar sensibilidade ±0,10% |
| G9 | Reprodutibilidade: scripts das Entradas 114–123 e hashes precisam constar no README; depósito/DOI ainda não existe (depende de Carlos) | Média | corrigível | README, `audit_entrada*_final_claims.py` ampliado, checklist |

## Plano do Ciclo 4 (pré-especificar antes de rodar)
1. **R||Cmax ablação de componentes + confirmação da lei da folga em instâncias novas** (G1, G2).
2. **P||Cmax IG com busca local (G3).**
3. Depois da execução: reescrever o manuscrito (espinha dorsal, mapa de evidências, quadro global) e ampliar a auditoria numérica (G4, G5, G8, G9).
Critério de parada: nenhuma fragilidade "corrigível" de gravidade alta restante.
