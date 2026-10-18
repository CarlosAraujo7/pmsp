# Auditoria de literatura: protocolo e livro de códigos (Entrada 159, congelado antes de ler qualquer artigo)

**Pergunta.** Entre os artigos que propõem um seletor/controlador aprendido (aprendizado por reforço, bandits, preditores) de operadores, estratégias ou parâmetros dentro de uma metaheurística para otimização combinatória, quantos incluem os controles do protocolo de atribuição justo ao orçamento? Objetivo: dar evidência (e não suposição) à Tabela "prática comum × orçamento justo" do artigo.

**População e amostragem (fixas antes da leitura).**
1. Fonte: arXiv (texto completo aberto), via a ferramenta de busca do arXiv; ano 2015–2026.
2. Consultas (as 8 abaixo, todas executadas; primeiros 12 resultados de cada, na ordem de relevância devolvida): (q1) "Q-learning operator selection metaheuristic"; (q2) "reinforcement learning adaptive operator selection iterated local search";
   (q3) "reinforcement learning hyper-heuristic scheduling"; (q4) "learning-based parameter control iterated greedy"; (q5) "multi-armed bandit operator selection genetic algorithm combinatorial optimization";
   (q6) "deep reinforcement learning operator selection adaptive large neighborhood search"; (q7) "reinforcement learning guided metaheuristic vehicle routing operator selection"; (q8) "learnheuristic reinforcement learning metaheuristic".
3. Triagem por título e resumo. **Inclusão:** (i) propõe um componente aprendido que escolhe operadores, estratégias, parâmetros ou heurísticas *durante a busca* de uma metaheurística/heurística iterativa para um problema de otimização combinatória; (ii) reporta uma comparação computacional.
   **Exclusão:** aprendizado de construtores neurais de ponta a ponta sem busca iterativa; trabalho puramente teórico; revisões/surveys; texto completo indisponível.
4. Se houver mais de 30 incluídos, ficam os 30 primeiros na ordem (consulta, posição). A ordem de leitura é a de inclusão. Nenhum artigo é substituído depois de lido.
5. Cada artigo é codificado a partir do texto completo, sem consultar resultados de outros artigos; evidência (seção/tabela) anotada por célula em `results/litaudit_entrada159.csv`.

**Livro de códigos (valores: Y = sim, N = não, U = não determinável pelo texto).**
- `random_control`: há comparação com seleção uniformemente aleatória das *mesmas* ações, no mesmo orçamento? (Variante sem aprendizado que escolhe ao acaso; não vale um baseline sem o mecanismo.)
- `budget_basis`: base do orçamento nas comparações com variantes sem aprendizado: `CPU` (tempo igual ou limite de tempo comum), `ITER` (mesmo número de iterações/avaliações), `MIXED`, `U`.
- `learner_overhead_reported`: o custo de CPU do componente aprendido é reportado ou considerado? (Y/N)
- `offline_tuned_static`: há concorrente estático (parâmetro/operador fixo) ajustado offline em instâncias de ajuste (grade, irace, etc.)? (Y/N)
- `tune_test_split`: instâncias de ajuste separadas das de teste? (Y/N/U)
- `stat_test`: teste estatístico pareado ou não paramétrico? (Y/N); `multiplicity`: correção de multiplicidade quando há várias comparações? (Y/N/NA)
- `ablation_policy`: ablação que isola o efeito da política do efeito do mecanismo hospedeiro? (Y/N)
- `code_or_data`: código ou dados disponíveis? (Y/N)
- `claim_policy_helps`: o artigo conclui que o componente aprendido melhora o resultado? (Y/N/MIXED)
- `problem`, `learner`, `host` (descrição curta).

**Regras de codificação.** Conservadora: na dúvida, `U` (conta como "não demonstrado"); quando um dos itens aparece só em tabela suplementar não acessível, `U`. Um controle aleatório só conta se o texto o descreve como seleção uniforme entre as mesmas ações. Relatar contagens com `U` separado; a leitura principal trata `U` como "não demonstrado".

**Limites declarados antes.** Um único codificador (o assistente de IA); amostra restrita a artigos com texto aberto no arXiv (viés a favor de ciência aberta e de áreas de IA/ML); sem inter-codificação. Para reduzir o limite, 8 artigos escolhidos ao acaso (semente 0) serão listados no relatório para dupla codificação por Carlos.

**Previsão registrada antes de ler (para ser julgada depois).** Entre os incluídos: `random_control` = Y em menos de 40%; `budget_basis` = CPU em menos de 30%; `offline_tuned_static` = Y em menos de 25%; `stat_test` = Y em menos de 50%; `multiplicity` = Y em menos de 15% dos que têm teste; `ablation_policy` = Y em menos de 40%; `learner_overhead_reported` = Y em menos de 40%.
