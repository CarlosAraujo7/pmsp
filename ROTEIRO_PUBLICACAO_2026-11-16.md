# Roteiro de publicação e de melhorias (2026-11-16)

Pedido de Carlos: não focar mais no EJOR; analisar outros locais e os caminhos de melhoria, a critério do assistente. Este documento é uma análise e uma recomendação, não uma decisão: nada abaixo foi executado, commitado ou submetido.

## 1. O que o trabalho é hoje (depois das correções de 16/11)

Três resultados que sobrevivem à revisão cega e às Entradas 277-280:

1. **Protocolo de atribuição com orçamento justo** (controle aleatório a orçamento igual, decomposição decisão/custo a CPU igual, ação estática validada, folga com validação cruzada, margem de equivalência). Contribuição metodológica; cada ingrediente é prática conhecida, a combinação e as ferramentas são o que há de novo.
2. **A regra de desempate pode fabricar "aprendizagem"**: o Q-Learning padrão (tabela zerada, primeiro máximo) repete a primeira ação da lista; com desempate aleatório por decisão ele vira um aprendiz modesto (além da margem em 32 de 85 células, 12-18% do ganho da validada, acompanha a folga com Spearman 0,81). A permutação por execução, que usamos primeiro, **não** é um controle válido. É o resultado mais citável e o mais enxuto.
3. **O conhecimento offline paga quando amortizado**: o ajuste custa 11,4 vezes o teste da classe, uma instância de ajuste por classe dá 81% do ganho, e com o ajuste cobrado o aleatório a CPU total igual vence para classes com menos de cerca de 72 a 130 instâncias.

O que o trabalho **não** mostra: que aprendizagem online "não paga" em geral. Dois domínios, aprendizes tabulares e de bandido, 10³ a 10⁴ iterações, ações que são em grande parte uma configuração de um parâmetro.

## 2. Mapa de locais (fatos conferidos em 16/11/2026; o que não consegui confirmar está marcado)

| Local | Por que combina | Fatos e cautelas |
|---|---|---|
| **IEEE CEC 2027** (Edimburgo, 25-28/jul/2027) | O artigo de 8 páginas já existe e está corrigido. | O PDF oficial da chamada traz prazo de artigos em **15/mar/2027**; listagens secundárias dizem 28/fev. Confirmar em ieeecec.org/2027. |
| **GECCO 2027** | Trilhas de otimização combinatória/metaheurísticas; 8 páginas. | Chamada de 2027 ainda não encontrada; em 2026 o prazo foi 26/jan. Sobrepõe-se ao CEC: escolher um dos dois para o mesmo conteúdo. |
| **EvoStar 2027** (EvoCOP, Mainz, 31/mar-2/abr/2027) | Tema de otimização combinatória, LNCS. | Prazo era 1/nov/2026, com uma publicação dizendo prorrogado até 16/nov/2026 (hoje): na prática inviável. |
| **LION 21** (2027) | "Learning and Intelligent Optimization" é exatamente o tema. | Chamada de 2027 não encontrada. A edição 19 aceitou artigos longos de 12-15 páginas LNCS e convidou versões estendidas a um número especial. Verificar o site de Hamburgo. |
| **INFORMS Journal on Computing** | Pede desenho experimental sólido e reprodutibilidade; o repositório é liberado como condição de aceitação para trabalhos computacionais. Nosso pacote (scripts que recalculam cada número) é um ponto forte. | Barra alta e exige contribuição metodológica clara: viável depois das melhorias P1 e P2 abaixo, não antes. |
| **ACM TELO** | Escopo inclui a interface otimização-aprendizagem; tem conselho de reprodutibilidade com selos. | Política sobre resultados negativos não confirmada. |
| **Journal of Heuristics** | Casa histórica da metodologia experimental para heurísticas (Barr et al. 1995 e Rardin e Uzsoy 2001 saíram lá). | Não consegui abrir a página atual de escopo e submissão. |
| **Computers & Operations Research** | Aceita metodologia e aplicações de PO. | Política de reprodutibilidade não confirmada. |
| Não verificados | Swarm and Evolutionary Computation, Applied Soft Computing, International Transactions in Operational Research, Journal of Scheduling, EURO Journal on Computational Optimization, Operations Research Perspectives, Evolutionary Computation (MIT). | Checar escopo, tempo de revisão e taxa de publicação antes de decidir. |

Não verifiquei fatores de impacto nem a classificação Qualis; o Qualis do PPGCC/UECE deve ser conferido na Plataforma Sucupira.

**Cuidados de submissão:** o artigo do CEC e o artigo de periódico se sobrepõem muito; uma versão de periódico precisa de material novo substancial e de declaração da relação com a versão de conferência (checar o copyright da IEEE). Não submeter o mesmo conteúdo a dois locais ao mesmo tempo.

## 3. Caminhos de melhoria, por valor sobre custo

**P1. Validação com implementação de terceiros (maior valor).** A biblioteca `alns` (Wouda e Lan, JOSS 2023, licença MIT) traz seletores `RandomSelect`, `RouletteWheel`, `SegmentedRouletteWheel`, `AlphaUCB` e `MABSelector`, vários critérios de aceitação e exemplos de CVRP, flowshop de permutação e TSP. Rodar o protocolo nela, sem mudar nada: `RandomSelect` como controle aleatório, um operador fixo como ação estática, os demais como aprendizes. Isso responde de uma vez às críticas de "aprendizes de palha", de hospedeiros pouco representativos (aqui há operadores de destruição e reparação heterogêneos, a arena clássica da seleção adaptativa) e de "projetos publicados só no estilo de". O resultado pode ir em qualquer direção, e isso é desejável: é a resposta honesta a "onde a aprendizagem online paga".

**P2. O regime de poucos dados de ajuste (barato, com código já existente).** Comparar o aprendiz aquecido com a priori de k instâncias de ajuste contra a ação estática escolhida com as mesmas k (k = 1, 2, 3, 5). É a pergunta central do título em forma mensurável: a aprendizagem online provavelmente paga como corretora quando a priori é ruidosa, e isso ainda não foi medido. Horas de computação, não dias.

**P3. Reestruturar em dois trabalhos.** (A) Protocolo, artefato do desempate, lista de verificação e custo do conhecimento offline: artigo metodológico curto. (B) Conhecimento offline contra aprendizagem online com aquecimento e P2: artigo de PO. Cortar a maquinaria estatística do corpo principal (efeitos com IC por classe; o controle global de 3.578 testes vai para o suplemento), rótulos de três níveis (registrado, pós-hoc confirmado, exploratório) e resumo de até 250 palavras.

**P4. Registro com carimbo de terceiros, antes de rodar P1 e P2.** Registrar no OSF as previsões e os desenhos de P1 e P2 antes de executar. É barato e responde à crítica de que o pré-registro era só um diário privado. (Parte sua.)

**P5. Opcionais.** Horizonte T ≥ 10⁵ no benchmark padrão; contato com os autores de Karimi-Mamaghan et al. e Missaoui e O'Sullivan para obter o código e replicar os projetos originais; um segundo hospedeiro de roteamento além de P1.

**P6. Antes de qualquer submissão:** segunda revisão cega do texto corrigido, de preferência por pessoas (orientador, colegas do PPGCC), já que as revisões por modelo compartilham vieses.

## 4. Sequência recomendada

1. Decisões suas: orientador e coautoria; registro no OSF.
2. P4 e P2 (dias), depois P1 (uma a duas semanas de trabalho), cada um com previsões e hashes antes de rodar.
3. P3: dois manuscritos. O artigo do CEC segue para a data oficial (15/mar/2027 pela chamada) como degrau de conferência, sem esperar os periódicos.
4. Artigo A: Journal of Heuristics ou ACM TELO; INFORMS JoC se P1 trouxer evidência em hospedeiro heterogêneo. Artigo B: Computers & Operations Research, Journal of Scheduling ou ITOR.

## 5. Incertezas

- Nenhuma probabilidade de aceite foi estimada; as notas calibradas (50-60) eram para o EJOR e não se transferem.
- P1 pode mostrar que, em portfólios heterogêneos, os aprendizes pagam mais do que em nossos hospedeiros; isso mudaria o título e a mensagem, para melhor.
- Os prazos de CEC, GECCO e LION precisam ser confirmados nos sites oficiais.

Fontes consultadas: [ALNS (JOSS 2023)](https://joss.theoj.org/papers/10.21105/joss.05028), [repositório ALNS](https://github.com/N-Wouda/ALNS), [chamada do CEC 2027](https://confcats-siteplex.s3.amazonaws.com/ieeecec/files/cec27-cfp_web-06.pdf), [EvoStar 2027](https://www.euro-online.org/bb/topic/35347), [GECCO 2027](https://gecco-2027.sigevo.org/Call-for-Papers), [política de software do IJOC](https://pubsonline.informs.org/page/ijoc/softwarepolicy), [escopo do ACM TELO](https://www.myhuiban.com/journal/1182.md).
