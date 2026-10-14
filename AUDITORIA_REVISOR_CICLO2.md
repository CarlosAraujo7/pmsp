# Auditoria no papel de revisor — fim do Ciclo 2 (09/10/2026)

Regra mestre (Carlos, 08/10/2026): plano -> execução -> auditoria como revisor -> novo plano, até ficar sólido (`LEI_GERAL_PESQUISA.md` II.13).
Evidência do Ciclo 2 (Entradas 114–119 do diário): seleção de d com folga (A1 sim / A2 não), 2º teste confirmatório externo (replicado 3/3; estado informativo não agrega),
variância entre seeds nas 12 classes (política = −0,011%, estritamente equivalente), ILS semeado (IG+ vs ILS+ dependente da família), R||Cmax (mecanismo −4,2%, política IG+LS
−0,145%, política ILS +0,96% pior, MILP-60s melhor que o IG até n=500).

## Fragilidades que um revisor do EJOR apontaria agora

| # | Fragilidade | Gravidade | Classe | Ação (Ciclo 3) |
|---|---|---|---|---|
| F1 | **Narrativa dividida**: no P||Cmax a política ≈ 0 (0,011%), no R||Cmax vale −0,145% (até −0,7% por célula, capturando 55–90% da folga). "Mecanismo, não política" vira "depende do problema", e o texto atual não explica a diferença | Alta | corrigível (experimento) | Hipótese: a política só vale quando há folga E o orçamento é **vinculante** (no P||Cmax a busca satura em T=1000). Teste pré-especificado variando o orçamento (P: T = 100, 250, 1000; R: T = 1000, 5000) |
| F2 | O nulo da política no P||Cmax foi testado só com Q-Learning tabular e UCB1; mesmo com folga em d (A1) nenhum capturou. Um revisor pede: mais orçamento, pré-treino com ações de d, outro aprendiz | Alta | corrigível | T=5000 com ações de d; pré-treino cruzado com ações de d; referência "identificação do melhor braço" (limite superior do que um aprendiz online poderia capturar) |
| F3 | **R||Cmax**: IG+LS usa ~8x a CPU do ILS (5,3 s vs 0,6 s por execução), logo M1 (−4,2%) não é a CPU igual | Alta | corrigível | ILS com orçamento de CPU equivalente (T ≈ 8000), pré-especificado |
| F4 | R||Cmax: o MILP-60s ganha do IG+LS (+1,6%) em 115/160 instâncias; heurísticas não são estado da arte — o revisor dirá "estudo de heurísticas fracas" | Média | parcialmente corrigível | Mostrar como o IG+LS evolui com T (5000) frente ao MILP; declarar que o objetivo é o protocolo de atribuição, não competir com HiGHS; instâncias-padrão de Fanjul-Peyro & Ruiz exigem download (pedir autorização) |
| F5 | O d padrão (0,3 n) nunca foi ajustado no resultado principal; agora há d por validação (−0,19% sobre o padrão), mas **não** se sabe se o IG ajustado + semeado supera MULTIFIT/LDM em n >= 75 | Alta | corrigível (sem custo alto) | `igd_cv_seed`: d da validação + partida pela melhor construtiva vs MULTIFIT, LDM, ig_ql_seed (sintético e subamostra externa) |
| F6 | Muitos testes no artigo todo (>60 hipóteses) — "forking paths". Cada família tem Holm, mas falta o quadro global | Média | corrigível (texto) | Tabela única de todas as famílias pré-especificadas: hipótese, regra, veredito, magnitude (suplemento S4.7 ampliado) e contagem de não-confirmações |
| F7 | A margem de equivalência ±0,05% é arbitrária | Baixa | corrigível (texto) | Justificar (1/6 do menor efeito de mecanismo 0,3%) e mostrar sensibilidade ±0,10% |
| F8 | Estado informativo: limiares pooled; sem efeito. Ok, mas o `rel` só foi testado em IG | Baixa | limitação | Declarar |
| F9 | Q-Learning só tabular, 1 recompensa densa testada | Média | limitação + 1 teste | A referência "melhor braço" (F2) delimita o ganho máximo; declarar função-aproximação como fora de escopo |
| F10 | Reprodutibilidade: scripts novos (Entradas 114–119) não estão no README; depósito/DOI ainda não existe | Média | corrigível | Atualizar README, hashes, checklist |
| F11 | Instâncias reais/benchmark de R||Cmax ausentes; as de P||Cmax são do benchmark público (OK) | Média | limitação / depende de Carlos | Pedir autorização para baixar o benchmark de R||Cmax (Fanjul-Peyro & Ruiz) ou declarar |

## Plano do Ciclo 3 (pré-especificar antes de rodar; um diário e hashes por frente)
1. **Lei do orçamento (F1):** P||Cmax em T = 100 e 250 (12 classes, 100 reps; ig_random, ig_ql, ig_fix0..2) e R||Cmax em T = 5000 (reps 0-7 das 20 células); hipóteses sobre a variação do efeito da política e da folga com T.
2. **Seleção de d com mais orçamento e pré-treino (F2):** igd_ql, igd_random, d fixo em T = 5000 (6 classes x 50 reps) e igd_ql pré-treinado por classe; referência de melhor braço.
3. **R||Cmax CPU igual (F3)** e **evolução frente ao MILP (F4):** ILS com T equivalente à CPU do IG+LS; IG+LS-QL em T=5000 vs MILP-60s.
4. **IG ajustado + semeado (F5):** `igd_cv_seed` vs MULTIFIT/LDM/ig_ql_seed.
5. **Quadro global de hipóteses (F6)**, justificativa da margem (F7), README/hashes (F10), checklist (F11).
Critério de parada do ciclo: nenhuma fragilidade "corrigível" de gravidade alta remanescente; as limitações restantes ficam declaradas.
