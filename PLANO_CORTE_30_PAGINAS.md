# Plano de corte para 30 páginas e reposicionamento (11/10/2026)

Estado atual: corpo principal **46 páginas** (seções 1–7 até a p. 43; administrativo e referências p. 44–46), suplemento **18 páginas**. Tudo que cortar pode ir para o suplemento (a tabela de pré-registro `s_prereg_all` já lista todos os testes).

## 1. Reposicionamento recomendado (tese em uma frase)

> **Os ganhos atribuídos à política aprendida em heurísticas guiadas por aprendizado são, em regra, ganhos da estrutura de busca e do orçamento; um protocolo de atribuição a CPU igual, com braço validado offline, os revela em três problemas (máquinas paralelas idênticas e não relacionadas, flowshop de permutação) e leva a solvers sem aprendizado fortes.**

Quatro contribuições, em ordem de força:
1. **Protocolo** (CPU igual calibrada em sequência, controle aleatório pareado, braço validado offline, folga do oráculo, limite certificado, equivalência, pré-registro com hashes) e a demonstração de que ele **muda conclusões** (o ganho de 0,150% da busca local vira 0,010% a CPU igual; a política custa 1,4–1,7× a CPU em R||Cmax).
2. **Evidência em 3 problemas / 7 ambientes** (Tabela `f35_synthesis`): ingrediente ativo problema-dependente (destruir-reconstruir; combinação com descida; busca local), validação offline > Q-Learning online, "folga necessária, não suficiente" (e a lei da folga **refutada prospectivamente** para o tamanho de destruição).
3. **Solvers sem aprendizado fortes**: receita de P||Cmax (97,5% de ótimos provados em T = 64.000) e LP-IG + MILP restrito em R||Cmax (0,25% do melhor conhecido do benchmark padrão; CPLEX 2 h: 0,20%).
4. **Reprodutibilidade**: código, hashes, diário, auditorias numéricas automáticas (214 conferências), benchmark padrão de terceiros.

Títulos candidatos (a escolher; o atual é específico a máquinas paralelas):
- *Mechanism or policy? Budget-fair attribution of the gains of learning-guided metaheuristics in scheduling* (mais próximo do atual; recomendo)
- *When does learning help metaheuristics? Equal-CPU attribution evidence from three scheduling problems*
- *Learning-free beats learning-guided: attribution and strong solvers for parallel machine and flowshop scheduling*

## 2. Alocação sugerida de páginas (meta: 30 incluindo referências e texto administrativo)

| Seção atual | Páginas hoje | Meta | O que fazer |
|---|---|---|---|
| 1 Introdução | 2 | 2,5 | Reescrever em torno da tese e das 4 contribuições (acima); tirar o parágrafo longo de diferenciação do QIG para o suplemento/Trabalhos relacionados |
| 2 Trabalhos relacionados | 2 | 1,5 | Condensar 2.1–2.3; manter 2.4 (IG e adaptação) |
| 3 Problema + 4 Métodos | 4 | 3 | Manter o essencial; algoritmo do IG e a tabela de estratégias; ILS e estado -> suplemento |
| 5.1 Setup | 3 | 1,5 | Tabela de réplicas e inventário de métodos -> suplemento |
| 5.2 Headline (ig_ql vs ILS/LPT) | 2 | 1 | Uma tabela resumo; o resto -> suplemento |
| 5.3 Baselines publicados (SA/BCO/MULTIFIT) | 1,5 | 0,7 | Frase + tabela f7 -> suplemento |
| 5.4 Baselines fortes + exato | 3 | 1,5 | Manter o limite certificado e a Tabela f12 resumida |
| **5.5 Mecanismo vs política** | 5 | 4 | **Núcleo**: ablação, estado, folga, orçamento; cortar o que for repetido |
| **5.6 Receita (inclui orçamento e CPU igual)** | 2 | 2,5 | **Núcleo**: receita, busca local a CPU igual, curva de orçamento (Tabela do suplemento sobe um resumo) |
| 5.7 Validação externa (P||Cmax) | 3 | 1,5 | Uma tabela por teste confirmatório; detalhe -> suplemento |
| 5.8 Limites estruturais; 5.9 Despacho | 2 | 0 (suplemento) | Mover inteiro (já tem referência cruzada no texto) |
| **5.10 Máquinas não relacionadas** | 7 | 4 | **Núcleo**: decomposição, LP-IG + MILP restrito, benchmark padrão, política sobre d (refutação); 1000x50 e validação a 1000x50 -> suplemento |
| **5.11 Terceiro problema + síntese** | 2 | 2 | **Núcleo**: manter inteira (Tabela f35 é a figura-resumo) |
| 5.12 Companion ils_ql | 0,3 | 0 | Suplemento |
| 6 Discussão | 3 | 2,5 | Reorganizar em: onde vem o ganho / quando a política pode ajudar / o que funciona na prática / limites |
| 7 Conclusões | 1 | 0,7 | |
| Administrativo + referências | 3 | 3 | Referências ~2,5 pp com o `elsarticle-harv` |

Resultado esperado: ~29–30 páginas. Nada do que sai deixa de estar citado: o suplemento mantém tabelas completas e a tabela de pré-registro.

## 3. Lacunas que um revisor ainda pode apontar (e o que já temos)

| Crítica provável | Resposta no trabalho |
|---|---|
| "O aprendiz é tabular e fraco" | Q-Learning, UCB1 normalizado, UCB1 por estado, Q-Learning sem estado, pré-treino offline (por (tipo, n)), recompensa densa; **limitação declarada** (sem rede neural); a refutação da lei mostra que o problema é a janela de aprendizado, não o tipo de tabela |
| "Os solvers não são o estado da arte" | R||Cmax: 0,25% do melhor conhecido do benchmark padrão contra CPLEX 2 h (0,20%); exatos ainda melhores (Gurobi/LR ~0,1%); P||Cmax: 97,5% de ótimos provados em 6 classes, métodos exatos resolvem o resto |
| "Muitas análises, risco de falso positivo" | Pré-registro com hashes, famílias com Holm, previsões registradas (várias erradas) e relato de **todas** as não-confirmações |
| "Instâncias sintéticas" | Benchmark padrão de R||Cmax (1.400 instâncias; 700 teste + 420 reserva) e 6.780 + 4.160 + 2.690 instâncias publicadas de P||Cmax; no flowshop, instâncias estilo Taillard geradas (os arquivos de Taillard não foram baixados: **pedir autorização** se quiser usá-los) |
| "Pré-registro não público" | Depositar `diario_pesquisa.md` e `results/entrada*_prereg_hashes.txt` em Zenodo/OSF na submissão |

## 4. Pendências só suas

Coautoria; palavra-chave oficial do EJOR; financiamento; URL/DOI e primeiro commit; depósito do pré-registro; escolher o título; ler o texto novo; (opcional) autorizar o download das instâncias de Taillard do flowshop para substituir as geradas.
