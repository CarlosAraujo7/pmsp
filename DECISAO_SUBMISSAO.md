# Decisão de submissão e limite da abordagem (14/10/2026)

Documento de decisão para Carlos. Fatos marcados **[verificado]** foram conferidos em fontes públicas em 13–14/10/2026; **[não confirmado]** precisa ser conferido no guia oficial antes de submeter.

## 1. Já chegamos ao limite desta abordagem? Sim, com uma ressalva

**Sim, para a linha "método novo baseado em aprendizado".** Tentamos, em ordem: Q-Learning sobre operadores (ILS), sobre estratégias de destruição (IG), sobre o tamanho de destruição, estado recalibrado, recompensa densa, pré-treino por classe, UCB1 (global e por estado), Q-Learning sem estado, Thompson (com e sem desconto) e torneio de eliminação, em 3 problemas e 7 ambientes. Nenhum aprendiz bate a escolha offline da configuração; o que ganha é a estrutura de busca e o orçamento. A lei da folga ("a política ganha onde há folga") valeu para estratégias (ρ = 0,59 e 0,74) e **falhou prospectivamente** para o tamanho de destruição (ρ = −0,37). Continuar nessa linha sem mudar de classe de aprendiz (redes neurais, ações mais ricas, outro problema) não produz resultado positivo com o orçamento e a janela de aprendizado atuais; isso é um resultado, não uma falha de execução.

**Ressalva: ainda há método novo modesto**, mas sem aprendizado: IG com tamanho de destruição validado e partida forte (97,5% de ótimos provados em T = 64.000, P||Cmax sintético) e LP-IG + MILP restrito (0,25–0,30% do melhor conhecido do benchmark padrão de R||Cmax; CPLEX 2 h: 0,18–0,20%; Gurobi/exato de terceiros: ~0,07–0,1%). A novidade algorítmica é pequena (arredondamento de LP e MILP restrito são ideias conhecidas) e está declarada como tal. O valor do trabalho está em **protocolo + evidência negativa/atributiva + solvers sem aprendizado auditáveis**.

O que ainda pode aumentar a força, fora do escopo desta rodada: (a) instâncias clássicas de Taillard para o flowshop (precisa da sua autorização de download); (b) um aprendiz de outra classe (rede neural/política com contexto da instância) como trabalho futuro com folga pré-especificada; (c) pré-registro público (Zenodo/OSF); (d) revisão por um coautor.

## 2. Onde submeter: recomendação

| Ordem | Destino | Por quê | Riscos |
|---|---|---|---|
| **1 (recomendado)** | **Journal of Heuristics** (Springer) | A política do periódico exige que os métodos sejam testados com os princípios de Hooker (experimentos controlados que isolam o efeito de cada componente, com hipóteses e significância estatística), que é exatamente o protocolo do artigo **[verificado]**. Escopo: busca heurística, metaheurísticas, hiper-heurísticas. Sem limite de páginas conhecido que nos force a cortar além das 24 atuais **[não confirmado]**. | Fator de impacto modesto (~1,4 em 2024; fontes divergem) **[verificado]**. APC em acesso aberto híbrido ~US$ 2.890 **[não confirmado]** (publicar fechado é gratuito). Gabarito Springer Nature exige reformatar. |
| 2 | **Computers & Operations Research** (Elsevier) | Escopo cobre escalonamento e métodos aproximados; aceita estudos computacionais e comparações rigorosas; mesmo `elsarticle` que já usamos. | Pede contribuição de método ou aplicação mais clara; a tese "o aprendizado não paga" pode ser lida como resultado negativo. |
| 3 | **INFORMS Journal on Computing** | Área de "heuristic search & learning"; valoriza software e dados reprodutíveis (repositório congelado no GitHub, que já temos quase pronto). | Barra alta de novidade; exige o pacote de software conforme a política do IJOC; revisão longa. |
| 4 (aspiracional) | **EJOR** | Plano original; área "Computational Intelligence and Information Management". | Contribuição majoritariamente metodológica/negativa; o EJOR cobra inovação de método ou insight de gestão; chance baixa a moderada. Se tentar, é a opção de maior prestígio, e a versão curta já está em `elsarticle`. |
| Conferência (opcional, em paralelo) | **EvoStar/EvoCOP 2027** (prazo 1/11/2026 **[a conferir]**), CEC 2027 (Edimburgo, 25–28/7/2027; prazo **[não confirmado]**) | Feedback rápido de comunidade; EvoCOP é o fórum natural de metaheurísticas para otimização combinatória. | Exige versão de ~16 páginas (LNCS); publicar versão de conferência antes do periódico exige extensão substancial (em geral > 30% de conteúdo novo) e declaração. Não recomendo dividir o esforço agora. |

**Recomendação:** submeter ao **Journal of Heuristics**; se rejeitado por escopo, COR; EJOR/IJOC só se você preferir apostar em prestígio, aceitando o risco. A ordem pode ser invertida sem retrabalho grande: o texto está em `paper/short/*.tex` e só o gabarito muda.

## 3. O que falta antes de submeter (checklist objetivo)

1. **Você decide:** periódico; coautoria (ex.: orientador) e carta; título final (hoje: *Mechanism or policy? Budget-fair attribution of the gains of learning-guided metaheuristics in scheduling*; alternativas em `PLANO_CORTE_30_PAGINAS.md`); palavra-chave oficial; financiamento; declaração de uso de IA (já escrita, leia e ajuste).
2. **Repositório e DOI:** primeiro commit (nenhum foi feito), URL do GitHub, depósito do diário e dos hashes (`diario_pesquisa.md`, `results/entrada*_prereg_hashes.txt`) em Zenodo/OSF; preencher o placeholder em *Data and Code Availability*.
3. **Referências:** as 8 novas de `paper/refs.bib` (Entrada 158) foram em parte conferidas; confira as restantes no DOI antes de submeter. Rode `bash paper/build.sh` e confira se não há citações indefinidas.
4. **Reformatar** para o gabarito do periódico escolhido (Springer Nature `sn-jnl` para JoH). O conteúdo não muda.
5. **Ler o texto novo** (resumo, introdução, discussão): foi escrito por mim a partir dos resultados auditados; as frases de posicionamento são sua decisão.
6. ~~Instâncias clássicas de Taillard do flowshop~~: **feito em 18/10/2026** (regeneradas do gerador publicado; replicação com 40 instâncias, o achado se replica).

## 4. Estado do arquivo para submissão

- Artigo principal: `paper/learnheuristic_pmsp.pdf` (25 páginas com referências; limite de 30).
- Suplemento: `paper/supplementary_material.pdf` (44 páginas, inclui o material movido, a auditoria de literatura, a replicação em Taillard e a tabela de pré-registro).
- Relatório técnico (versão longa): `paper/learnheuristic_pmsp_long_v46pp.pdf` (46 páginas), com tudo que existia antes do corte.
- Auditorias: 342 testes; 63/63, 59/59, 21/21, 225/225 e 45/45 conferências numéricas; auditoria de proveniência numérica da versão curta (`analysis/audit_short_numbers.py`).

## 5. Atualização de 18/10/2026 (Ciclo 12, Entradas 159-161)

Nova nota crítica: **~70/100** (faixa 66–74; era ~67 na mesma escala; a estimativa de ~72 dada logo após o ciclo era otimista e foi corrigida com uma tabela ponderada, ver a conversa de 18/10/2026). O que mudou desde a nota de 67:
- **Crítica "o aprendiz é um espantalho":** agora há um bandit contextual linear (LinUCB) e um seletor treinado offline condicionado à instância, no espaço do tamanho de destruição de R||Cmax (700 instâncias). Resultado: nenhum muda a conclusão (LinUCB capta 19% do ganho validado; o seletor offline capta 50% mas perde para a validação por célula por 0,105%, p = 3e-9). Limite: uma classe adicional num ambiente; redes neurais não testadas.
- **Crítica "flowshop em instâncias geradas":** replicação em 40 instâncias clássicas de Taillard (ta041–090 em quatro tamanhos), tudo congelado; o achado do flowshop se replica (IG ≈ ILS a CPU igual; busca local −0,28%; política não ajuda; validação −0,17%, p_Holm = 0,064 com N = 40).
- **Crítica "a prática comum é um espantalho":** auditoria de literatura (12 artigos de arXiv, protocolo congelado antes de ler): controles baratos são mais comuns do que eu previ (aleatório 5/12; ablação 8/12), mas CPU igual (2/12), concorrente offline (2/12), separação ajuste/teste (3/12) e multiplicidade (1 de 6 que testam) são raros; todos os 12 concluem que o aprendizado ajuda. Limites: n = 12, só arXiv, um codificador (IA).
- **Referências:** as 8 novas foram conferidas na web.
- Por que não mais: o teto continua sendo novidade (protocolo reúne boas práticas conhecidas), autor único, sem revisão externa e sem DOI do pré-registro (use `DEPOSITO_ZENODO_PASSO_A_PASSO.md`, ~+3 pontos), e sem dupla codificação da auditoria (Carlos, 8 artigos sorteados listados em `results/entrada159_litaudit.txt`).
