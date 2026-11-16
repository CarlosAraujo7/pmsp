# Revisão cega independente do manuscrito (2026-11-15)

Manuscrito revisado: `journal/main_short.pdf` (36 pp), com `journal/main.pdf` (69 pp) como material suplementar.
Dois revisores de contexto limpo (sem acesso ao diário, à memória, às auditorias nem às notas de autoavaliação), restritos à pasta `journal/`, com lentes diferentes. Nada foi alterado no manuscrito por causa destas revisões; este arquivo é só a síntese e a triagem.

## Veredictos

| Revisor | Lente | Recomendação | Nota calibrada (0-100) |
|---|---|---|---|
| 1 | metodologia e estatística | revisão maior | **60** |
| 2 | pesquisa operacional, novidade, apresentação | rejeitar na forma atual, incentivando reenvio reestruturado | **50** |

Média 55. A estimativa interna anterior para o EJOR (63-69) estava otimista em 8 a 14 pontos. Nova estimativa calibrada: **EJOR ~50-60**. A meta de 85 está longe. O Revisor 1 não vê falha fatal única, mas diz que os itens M1, M2 e M4 abaixo seriam fatais para as afirmações do título e do resumo se ficassem sem tratamento.

## Pontos em que os dois concordam (maior peso)

| # | Crítica | Conferência minha | Tipo de correção |
|---|---|---|---|
| A | O controle de ordem (`ql_perm`) é um aprendiz pior que o aleatório (-0,30 a -0,06 em Tabela 2). A permutação é uma por execução, não desempate aleatório a cada decisão (`sec_order.tex:9`), então mede "repetir uma ação aleatória". "0 de 85 células" prova que Q-Learning com tabela zerada e primeiro máximo é um mau aprendiz, não que aprender não paga. UCB1, PG e roleta nunca rodaram com ordem permutada (limitação v, `s_discussion.tex:28`). | Procede. A limitação já está declarada, mas o resumo generaliza. | **Experimento novo** (Q-Learning com Q0 acima da recompensa máxima e desempate aleatório por decisão; UCB1/PG/roleta permutados) + texto |
| B | A "regra de folga" é quase vazia: acerto de 78% não excede significativamente a taxa-base de 68% de sempre prever "não ganha"; precisão balanceada 0,38 no flowshop; Thompson ganha no GA com folga cruzada -0,073. A versão curta cortou esses qualificadores. | Procede para a versão curta. O texto completo (`sec_results.tex:65-67`) tem os qualificadores; `s_results.tex:40` não. Também confunde "60 células" com "71+79 = 150" sem explicar. | **Texto** (restaurar qualificadores, explicar 60 vs 150, tirar a regra do resumo ou rebaixá-la a observação) |
| C | "Ação estática validada vence todo aprendiz" compara informação, não aprendizagem. A ação é escolhida com 30 ações x instâncias de ajuste x sementes; o aprendiz tem uma execução de 1000 iterações; o treino (115-138 mil execuções) não é cobrado. Falta CPU total (offline + online) e curva de aprendizagem no número de instâncias de ajuste; falta ligar a teoria de arrependimento (melhor braço fixo vence com T pequeno). | Procede em parte. O mapa de granularidade e o CPU igual já atacam isso, mas não há custo total. | **Análise com dados existentes** (curva em k instâncias de ajuste, custo total) + texto |
| D | O pré-registro não é verificável por terceiros (diário privado, md5 próprio, sem carimbo externo). 349/435 mede processo, não verdade; muitas previsões são fáceis. Achados pós-hoc (regime de tamanho não visto, regra de folga, diagnóstico de ordem, mistura validada, Q otimista) aparecem no resumo junto aos registrados. | Procede. | **Texto** (rótulo de três níveis: registrado / pós-hoc confirmado em instâncias novas / exploratório; tirar contagens do resumo) + **sua parte manual** (depósito no OSF/Zenodo com carimbo) |
| E | Literatura ausente. | **Confirmado ausente**: Turkeš, Sörensen, Hvattum (2021, EJOR 292), meta-análise do efeito da adaptatividade em ALNS (o precedente mais próximo); HyFlex/CHeSC; Sobotka et al. (2025); Alanazi e Lehre (2014). Também citados pelos revisores, ainda a verificar: Cowling et al., Aleti e Moser, Adriaensen et al. (2022, já na bib), Rashid et al. (ICLR 2020) sobre inicialização otimista. | **Texto** (verificar cada referência antes de citar; não reproduzir números dos revisores sem checar) |
| F | A afirmação "mesma assimetria no TSP" é exagerada. | Procede. Na TSPLIB o efeito líquido da aprendizagem é **negativo**: aquecido 0,560 contra congelado com a mesma priori 0,626 (synn), 0,582 contra 0,677 (syn0). Na priori realista (loso) a aprendizagem ajuda em 0 células. O ganho positivo vem da priori invertida de propósito (+0,72) e da faixa pequena (+0,07, 15 instâncias, sem teste). | **Texto** (dar o efeito líquido no manuscrito e no resumo; rebaixar a leitura de "corretor") |
| G | Apresentação: espaço reservado vermelho, rodapé "journal to be chosen", frase "simulated blind reviews", referências a "Entries 274/275" que o leitor não vê, link do GitHub com o nome, resumo de 430-480 palavras, eixo da Fig. 1 cortado, códigos de tabela não definidos (qo, ql_perm, loso, synn, syn0, nn_const). | Conferido: rodapé, frase e placeholder existem (`main_short.tex:28`, `s_discussion.tex:32,41`). | **Texto** |

## Pontos de cada revisor

Revisor 1 (metodologia): erros de multiplicidade assimétrica (Holm nos ganhos, IC 90% sem ajuste na "paridade"); teste misto (Wilcoxon sem empates contra bootstrap da média); a unidade de generalização é o hospedeiro (cerca de 5, um autor); margem de ±0,05 é da ordem da folga cruzada inteira das classes sintéticas (0,043-0,063), então a paridade com o aleatório ali é quase garantida, e faltam a lacuna média ao limite inferior e a fração de instâncias em que todos os braços o atingem; a "atribuição" é uma sequência de ablações dependente de ordem, e a faixa de 28-93% de orçamento é propriedade da implementação; a identidade (1) é contábil; valores diferentes para o mesmo estimando (ganho validado em B30 = 2,24 na Tabela 3, 2,26 no horizonte longo, 2,35 no suplemento; 1,60 contra 1,57; 1,95 contra 1,92) sem dizer a bateria; "60% de 67%" ilegível; a bateria "registrada" de aquecimento pode reutilizar as instâncias que geraram a hipótese.

Revisor 2 (domínio): os aprendizes são "no estilo de" e não replicam as configurações dos artigos publicados (a conclusão vale para o seu IG, não para Karimi-Mamaghan et al.); as ações do IG são quase uma configuração de um parâmetro (tamanho da destruição), onde o valor estático ajustado ganhar é esperado (Eiben e Smit), e não um portfólio heterogêneo de operadores como no ALNS de roteamento; classes sintéticas de P||Cmax têm folga de 0,12 ponto e o flowshop 0,012-0,025, então uma nula ali não contradiz Karimi-Mamaghan; as famílias "publicadas" de R||Cmax são regeneradas a partir da descrição; a auditoria de 23 artigos tem amostragem sem protocolo e um codificador; o título fala em IG mas metade do artigo é TSP, GA e aquecimento; sugere dividir em dois artigos (A: protocolo, ordem, inicialização otimista contra pessimista com reexecução dos projetos publicados; B: conhecimento offline e aquecimento, com horizonte longo e hospedeiro com folga).

## Onde o revisor errou ou é incerto

- **Revisor 2, F1 (aprendizes "famintos de amostras", controle positivo fraco): premissa errada.** O texto (`s_results.tex:32`) escreve só o limiar pré-fixado "pelo menos 1,5 vez o uniforme". O resultado bruto (`results/entrada185_plant.txt`) mostra que a ação plantada é escolhida de 2,2 a 13,6 vezes o uniforme (no espaço de 15 ações, até cerca de 90% das escolhas; mínimo de 2,49 com bônus 5). Corrigir escrevendo o mínimo e a mediana observados. Resta válido que o horizonte longo foi testado em hospedeiro saturado (folga de 0,125% a 0,015%) e que T = 10^4 só existe no espaço mais rico da bateria padrão.
- **Revisor 2, M2: "o IG de Fanjul-Peyro e Ruiz atinge desvios bem menores"** é lembrança não verificada pelo próprio revisor.
- **Números atribuídos pelo Revisor 2 a Turkeš et al. (0,14%, IC 0,07-0,22%)** e a referência Rashid et al. (arXiv 2002.12174) não foram verificados por mim. Verificar antes de qualquer citação.
- **Revisor 1, "necessidade não é necessidade"**: o texto completo já diz que é quase truísmo e dá o contraexemplo do GA; o ponto procede apenas para o resumo, que diz "condição necessária" sem o qualificador "para um seletor que se compromete com uma ação".

## Plano proposto, por custo

1. **Só texto (sem experimento novo):** resumo em até 250 palavras com três afirmações (protocolo, sensibilidade à ordem, offline contra online com a ressalva "informação, não aprendizagem"); rótulos de três níveis; qualificadores da regra de folga de volta; efeito líquido do TSP; mínimo e mediana do controle positivo; referências verificadas e discutidas (Turkeš 2021 em primeiro lugar); números do mesmo estimando reconciliados com a bateria indicada; placeholder, rodapé e referências a "Entries" tratados; códigos de tabela definidos; Fig. 1 corrigida.
2. **Análises com dados já existentes:** curva da ação estática pelo número de instâncias de ajuste e custo total; lacuna média ao limite inferior e fração de instâncias em que todos atingem; sensibilidade da margem (0,01-0,10) com a classe como unidade; correlação folga-ganho para os aprendizes sem travamento (Thompson, Q otimista) onde os dados existem.
3. **Experimentos novos (pré-registrados com hash):** Q-Learning com Q0 acima da recompensa máxima e desempate aleatório por decisão, e UCB1/PG/roleta com ordem permutada, nas 21 células padrão, nas três famílias não relacionadas e no TSP; no aquecimento, mistura top-k e aprendiz frio com mesmo epsilon.
4. **Suas partes manuais:** carimbo independente do registro de previsões (OSF/Zenodo), link anônimo, financiamento, orientador/coautor, e a decisão estratégica de dividir em dois artigos.

## Estado das correcoes (16/11/2026)

Executado o plano recomendado (itens 1, 2 e 3; o item 4 e' seu).

| Critica | O que foi feito | Onde |
|---|---|---|
| A. Controle de ordem | Q-Learning padrao com desempate aleatorio por decisao (`ql_rt`) e os outros aprendizes com lista permutada, em B15, B30 (Entrada 279) e nas 85 celulas (Entrada 280). **O resultado inverteu a leitura anterior**: o `ql_perm` nao era invariante a ordem; o `ql_rt` ganha alem da margem em 32 de 85 celulas e acompanha a folga (Spearman 0,81). | diario 279, 280; `sec_order.tex`, `s_results.tex`, resumo, CEC |
| B. Regra de folga | Qualificadores restaurados na versao curta; 60 contra 150 explicado; correlacao dos aprendizes sem travamento. | `s_results.tex`, `sec_results.tex` |
| C. Informacao contra aprendizagem | Custo do ajuste (11,4x), curva por numero de instancias (1 instancia = 81%), CPU total igual (cruzamento em 72; interpolado 130). | Entrada 277, `sec_granularity.tex` |
| D. Registro | Rotulos de tres niveis NAO feitos como tabela; contagens tiradas do resumo; limitacao mantida; carimbo independente (OSF/Zenodo) e' seu. | resumo, limitacoes |
| E. Literatura | Seis referencias verificadas por busca e um paragrafo em Related Work (Turkes et al. 2021, HyFlex, Sobotka et al. 2025, Alanazi e Lehre 2014, Aleti e Moser 2016, Rashid et al. 2020). Nenhum numero dos revisores foi usado. | `refs_extra.bib`, `s_related.tex`, `sec_related.tex` |
| F. TSP | Efeito liquido negativo na TSPLIB dado (0,56 contra 0,63; 0,58 contra 0,68); leitura da assimetria rebaixada. | `sec_tsp.tex`, `s_external.tex`, resumo |
| G. Apresentacao | Resumo reescrito; titulo; 'Entries' fora da versao curta; controle positivo com o observado (2,2 a 13,7x); frase 'simulated blind reviews' explicada. NAO mexidos: placeholders vermelhos, 'journal to be chosen', link com o nome (sao seus). | varios |
| M5 (R1) teto e margem | Entrada 278: perto do teto, nao no teto; margem 0,01 a 0,10. | `sec_results.tex` |

Resultados que mudam o manuscrito e que voce deve ler primeiro: Entradas 279 e 280 (a reversao do 'invariante a ordem') e a Entrada 277 (o conhecimento offline paga quando amortizado). Previsoes das quatro entradas: 14 de 30 (277: 5/7, 278: 1/6, 279: 2/9, 280: 6/8); contador total 363 de 465 em 65 baterias.

Nao feito: segunda revisao cega da versao corrigida; cortes a 25 paginas (a versao condensada tem 40, a completa 76); regra de tres niveis na tabela de respostas; replicacao das configuracoes dos artigos publicados de AOS; horizonte longo (T >= 10^5) no benchmark padrao.
