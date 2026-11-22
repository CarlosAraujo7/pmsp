# Segunda revisão duplo-cega do artigo do CEC: só conteúdo e novidade (2026-11-22)

Material: `cec/cec2027_anon.pdf` (v7, 8 páginas, md5 565520d9…, commit 08d26f7). Quatro revisores de contexto limpo (modelo `opus`; os revisores 2 e 4 foram relançados porque a primeira tentativa, com `fable`, parou no limite de gasto mensal antes de escrever qualquer coisa). Cada um recebeu só o PDF e um arquivo de instruções numa pasta isolada: sem diário, auditorias, código, memória do projeto nem revisões anteriores. Pedido do Carlos: **ignorar a escrita** e avaliar só conteúdo e novidade; nota 0-100 (65 = weak accept); notas 1-5 sem clareza; **nota teto** depois das melhorias que cada um propõe; propostas de **método, formalismo e evidência** novos. Os relatórios completos, sem edição (em inglês), estão em `revisao_cega_cec_2026-11-22_relatorios/`. Este arquivo é a síntese, a conferência e o plano. **Nada no artigo foi alterado e nada foi executado por causa desta revisão.**

Lentes: 1 computação evolutiva e hiper-heurísticas; 2 metodologia e estatística; 3 escalonamento e pesquisa operacional; 4 teoria de bandidos e aprendizado por reforço.

## Veredictos

| Revisor | Lente | Nota (conteúdo e novidade) | Teto | Relev. | Orig. | Solidez | Aval. | Signif. | Recomendação | Conf. |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | EC, hiper-heurísticas | **56** | 67 | 4 | 2 | 4 | 3 | 2 | borderline, tende a weak reject | 4/5 |
| 2 | metodologia e estatística | **60** | 73 | 4 | 3 | 3 | 4 | 3 | borderline, tende a weak accept | 4/5 |
| 3 | escalonamento e OR | **57** | 70 | 4 | 2 | 3 | 3 | 2 | borderline, tende a weak reject | 4/5 |
| 4 | teoria de bandidos e RL | **60** | 74 | 4 | 2 | 4 | 4 | 3 | borderline, tende a weak accept | 4/5 |
| **Média** | | **58,3** | **71,0** | 4,0 | 2,25 | 3,5 | 3,5 | 2,5 | | |

Leitura:

1. **A nota só de conteúdo (58) fica abaixo do limiar de weak accept (65) e quase igual à da rodada anterior, que incluía a escrita (55-57).** A escrita já não é o que segura a nota; o que segura é conteúdo. Mais polimento de texto não move nada.
2. **Originalidade e significância caíram em relação à rodada anterior (3 e 3, agora 2,25 e 2,5).** Sem a escrita para discutir, os revisores olharam a literatura: cada contribuição vale 2 a 3 de 5 em novidade.
3. **Relevância 4 nos quatro, solidez 3-4 e avaliação 3-4.** O trabalho é cuidadoso e honesto; o que falta é *explicar* por que o aprendizado não paga e *mostrar um regime em que deveria pagar*.
4. **Tetos de 67 a 74 (média 71) valem se as melhorias forem executadas bem; as propostas se sobrepõem, então os pontos não se somam.** Veja a estimativa minha mais abaixo.

## O que os revisores elogiam (preservar)

- Higiene experimental: pareamento, Holm, bootstrap, margem de equivalência, poder (efeito mínimo detectável de 0,006-0,014 pp), controle positivo com bônus plantado (R1, R2, R4).
- A divisão decisão/custo explica as "vitórias" a CPU igual (efeito de decisão −0,003 e −0,004 pp; efeito de custo +0,062 e +0,053) (R1, R3, R4).
- O experimento da ordem como intervenção causal (R2, R3, R4).
- O hospedeiro de terceiros (`alns`) responde a "é o seu código" (os quatro).
- **O R||Cmax com folga real é "a parte mais forte da evidência"** (R1, R2, R3): é onde o nulo não é forçado pelo teto.
- A correção da nossa própria leitura anterior (aprendiz com desempate aleatório) (R4).

## Novidade, contribuição por contribuição (nota 1-5 de cada revisor)

| Contribuição | R1 | R2 | R3 | R4 | Anterioridade apontada pelos revisores (não verificada por mim, salvo Turkeš) | O que sobra de novo |
|---|---|---|---|---|---|---|
| Protocolo de atribuição a orçamento justo | 2 | 2,5 | 2,5 | 2 | aleatório como base (HyFlex/CHeSC, Ochoa 2012; Burke 2013); CPU igual (Johnson 2002; Bartz-Beielstein 2020); estática ajustada como base (Karafotias 2015; literatura de configuração dinâmica); lacuna melhor-única/oráculo (ASlib, Bischl 2016); ablação (Fawcett e Hoos 2016); equivalência (Lakens 2017); poder (Campelo e Takahashi 2019) | divisão decisão/custo via K; oráculo com validação cruzada; empacotamento com controle positivo |
| Estrutura gulosa, não política (IG contra ILS) | 2 | 2 | 2 | 2,5 | Ruiz e Stützle 2007 (IG como ILS com destruição e construção); LPT quase ótimo em P\|\|Cmax (Frenk e Rinnooy Kan 1987) | fatorial limpo 2×3; específico do hospedeiro |
| Aprendizes ≈ aleatório; a estática validada vence | 2 | 3 | 3 | 3 | Bilgin et al. 2006; Özcan et al. 2008; CHeSC 2011; Alanazi e Lehre 2014; Lissovoi et al. 2019/2020; Doerr et al. 2018; Fialho 2010; Da Costa 2008; Karafotias 2015; **Turkeš, Sörensen e Hvattum 2021 (meta-análise do ALNS, já verificado por mim)** | escala (9-10 aprendizes, 69 células), margens e poder, comparação com a estática validada, hospedeiro de terceiros |
| Armadilha da ordem (aprisionamento na primeira ação) | 2 (3 como alerta) | 3 | 3 | 2,5 | Sutton e Barto 2018 (desempate aleatório); inicialização em zero | auditoria causal da ordem; alerta de reprodutibilidade; 16% de captura (exploratório) |

**Consenso: nenhuma contribuição, sozinha, tem novidade de nível CEC; o conjunto é uma consolidação cuidadosa.** O R1 pede que o nulo seja apresentado como replicação ampliada de resultados de hiper-heurísticas, não como descoberta.

## Pontos de solidez, por concordância (conferidos por mim)

| # | Ponto | Quem | Conferência | Correção |
|---|---|---|---|---|
| S1 | **Teto e folga decidem os veredictos sintéticos.** O nulo no P\|\|Cmax é "quase forçado pelo hospedeiro"; a margem de ±0,05 pp pesa 40% da folga cruzada positiva média (0,12 pp); flowshop com T = 1000 e ALNS com lacuna de 2,0-2,5% são hospedeiros curtos ou fracos | 4 de 4 | **Confirmado** na Tabela I: ganho validado de 0,003 a 0,241 pp (0,01 a 0,62 unidades de makespan), 3 das 12 classes com ganho validado ≤ 0,05 pp, folga cruzada com cinco valores negativos (um é −0,002). **Mas** o R\|\|Cmax (folga de 1,5 a 2,8 pp) e o ALNS têm folga real e o resultado se mantém (aprendizes ≤ 18%; 0 de 40) | pôr o R\|\|Cmax e o ALNS no centro, o P\|\|Cmax sintético como âncora de baixa folga; margem relativa; hospedeiro mais forte |
| S2 | **Não há modelo de por que o aprendizado falha, nem diagnóstico de não estacionariedade ou de dependência de estado** | 4 de 4 | **Confirmado**: o `.tex` v7 não contém "phase", "schedule" nem "stationary". **Fato que os revisores não viram:** o diário tem o competidor no tempo (Entradas 238-239): o melhor cronograma de 3 fases por classe (125 combinações, escolhido no ajuste) **empata** com a ação validada no P\|\|Cmax (0,095 contra 0,092) e **perde 0,166** no R\|\|Cmax (era 0,178% melhor no ajuste: maldição do vencedor). A frase foi cortada do v7 por espaço. Isso cobre o tempo ao nível da classe, **não** o estado da instância | restaurar a frase e o número; medir o estado (S2 abaixo, proposta B) |
| S3 | **Razão de captura mal condicionada**: denominador perto de zero ou negativo; "captura de 16%" sem dizer se é razão de somas; sem Fieller; "captura" tem dois denominadores (ganho da validada e folga de uma semente) | 4 de 4 | **Confirmado** (o texto não declara a agregação; os dois usos existem) | estimando primário = diferença (prêmio offline, G_validada − G_aprendiz); razão só onde o denominador for positivo, com IC de Fieller ou bootstrap por classe; inclinação hierárquica |
| S4 | **Acoplamento matemático e pseudo-replicação do 0,78**: os dois ganhos são diferenças do mesmo braço aleatório (mesmas instâncias e sementes); as mesmas 12 classes entram em dois espaços; escalas muito diferentes entre coleções | R2 explícito; R1 e R3 em parte | **Procede em princípio**; a magnitude é desconhecida. O bootstrap por classe trata a pseudo-replicação, não o acoplamento. **Testável sem rodar nada:** os dados brutos têm instância × semente × método (`results/agents_*_entrada180.csv`), então dá para dividir as sementes | correlação com sementes divididas; reportar antes e depois |
| S5 | **"Além da margem" frouxo e mistura de estimandos**: Holm-significativo (Wilcoxon, pseudo-mediana) e IC90 da média "não contido em ±0,05"; um ganho de 0,02 com IC [0,004; 0,06] conta | R2 | **Confirmado** (é a definição do artigo) | regra estrita (limite inferior do IC > margem) e TOST sobre a média, como sensibilidade; as contagens 29/25/19 de 69 devem cair |
| S6 | **Sinal de aprendizado fraco por construção**: 81 estados × 5 ações com T = 1000 dão 2,5 visitas por célula; recompensa esparsa truncada em zero; cerca de 200 puxadas por braço | R1, R2, R3, R4 | **Confirmado** (as 2,5 visitas estão no artigo). Contra-ponto: UCB1, PG, Thompson e DMAB não usam essa tabela, e o controle positivo mostra que detectam um bônus plantado (1,5 a 13,7 vezes). **O que falta é prever quanto *deveria* ser capturado** | modelo de aprendibilidade (proposta A) |
| S7 | **Cobertura e ajuste dos aprendizes**: faltam explore-then-commit, successive halving, UCB robusto (mediana das médias) e UCB de janela deslizante; assimetria de ajuste (600 execuções por classe para a validada contra grade pequena para os aprendizes) | R1, R4 | **Parcial**: o artigo já tem PM, AP, DMAB, FRRMAB, TS, LinUCB, DQN, UCB1, PG, corrida e irace como competidor estático; não tem ETC, halving nem UCB robusto. O ajuste dos aprendizes em grade pequena precisa ser conferido no código antes de afirmar | proposta G |
| S8 | **Generalidade da estrutura gulosa**: uma perturbação, ILS fraco, P\|\|Cmax onde LPT é quase ótimo | 4 de 4 | **Confirmado** (limitação ii). As Entradas 285 e 285b já cobrem o ILS com descida completa e o tamanho de chute oracular; falta um segundo problema e um ILS estruturalmente mais forte | proposta H |
| S9 | **Assimetria de informação**: a validada usa o rótulo da classe e 30 instâncias; o aprendiz começa frio; o seletor offline não paga o treino | R3, R2 | **Parcial**: o artigo cobra o ajuste da validada (11,4 vezes o teste; ponto de equilíbrio de 70 a 130 instâncias) e testa a versão aquecida (1,86 contra 1,92); **não** cobra o treino do seletor ExtraTrees | pools sem rótulo (proposta C-i); linha de custo do seletor |
| S10 | **Registro interno e floresta de caminhos** | os quatro | **Já declarado** (limitação iii; o 16% e o 0,78 são exploratórios). A correção real é o carimbo externo (camada D, do Carlos) e congelar as novas previsões antes de rodar | pré-registro de cada ciclo novo |
| S11 | **Resumo além da evidência**: "o ganho vem de conhecimento offline, não do aprendizado durante a execução" vale no regime testado; "um ILS clássico fica 0,6-0,8 atrás" vale para um hospedeiro; "um detalhe do Q-Learning produz vitórias aparentes" vale para a nossa implementação ([9] não foi verificado) | os quatro | **Procede**: as três ressalvas estão no corpo, não no resumo | qualificar o resumo quando o conteúdo novo entrar |
| S12 | **Posicionamento**: Turkeš et al. 2021, a análise de tempo de execução de hiper-heurísticas (Alanazi e Lehre; Lissovoi et al.; Doerr et al.) e Özcan et al. 2008 | R1, R2, R3 | **Confirmado** para Turkeš (ausente do `.tex` e da bib do CEC; já está em `journal/refs.bib`, verificado no ciclo 47-48). Os demais: citados de memória pelos revisores (o R1 declarou que não usou busca); **verificar antes de usar** | acrescentar ao `make_bib.py` depois de verificar |

## Leituras dos revisores a tratar com cuidado

- "Folga cruzada negativa em quatro classes" (os quatro): a Tabela I tem cinco valores negativos, um deles −0,002 (≈ 0). Trivial.
- R3: "o ALNS tem 16 operadores de destruição e reparo aleatórios, um forte domina": o ganho validado por classe é de 0,6 a 1,1 em 7 de 8 classes, então a ação importa; **se a melhor muda durante a corrida nunca foi medido** (entra na proposta B).
- R3: "HGS-CVRP chega a 0,2-0,5% nas instâncias X": número externo, não verificado.
- R4: "σ²/Δ² deve estar na casa dos milhares e prever captura de 0,1-0,2": **especulação sem dados**; é o que a proposta A mediria.
- R4: "o Q-Learning com γ = 0,9 e 81 estados é escolha estranha": leitura parcial, o artigo tem aprendizes sem estados.
- **Os enunciados candidatos de teoremas dos revisores são conjecturas, não resultados, e alguns estão frouxos.** Por exemplo, o aprisionamento "o aprendiz fica na primeira ação para sempre" não vale em geral: outras ações podem ultrapassar a primeira por exploração. A formulação rigorosa (provavelmente em bandido de um estado, com verificação por simulação) é parte do trabalho.
- Os quatro são modelos de linguagem, com vieses compartilhados entre si e com quem escreveu o artigo; tendem a premiar formalismo. Um revisor humano do CEC pode pesar mais um hospedeiro padrão de EC do que um teorema.

## Propostas consolidadas (as dos quatro, agrupadas por tema)

Legenda: M = método, F = formal, E = evidência. "Quem" lista os revisores que propuseram algo equivalente. Os pontos são os que cada revisor atribuiu; **não se somam** porque as propostas se sobrepõem.

| ID | Proposta | Tipo | Quem | Esforço | Pontos (declarados) | Risco | Estado nos nossos dados |
|---|---|---|---|---|---|---|---|
| **A** | **Modelo de aprendibilidade**: cada célula como bandido de K braços com recompensa esparsa (zero com probabilidade 1−p_a, cauda pesada); índice de aprendibilidade S = T·Δ²/(K·σ²); proposição de que a captura tende a zero abaixo do limiar de detecção; **gêmeo digital** (rodar os aprendizes num bandido reamostrado dos registros) e comparar a captura simulada com a observada | F + M | R4 (F1, M1, M2), R1 (F1), R3 (F3), R2 (F3) | 4-7 dias | +4 a +5 | baixo a médio (o modelo pode falhar por não estacionariedade ou estado; o fracasso informa) | **não há registros por decisão**: os executores devolvem só o final; é preciso um executor instrumentado |
| **B** | **Diagnóstico de não estacionariedade e de estado**: melhor ação por fase (quartis ou decis), oráculo por k fases, oráculo condicionado ao estado, "folga de troca" com validação cruzada | M | R1 (M1), R2 (M1), R3 (M3), R4 (E3, E4) | 3-6 dias | +4 a +5 | baixo (o nulo reforça a tese) | **metade pronta**: cronograma de 3 fases por classe (Entradas 238-239): empate no P\|\|Cmax, perda de 0,166 no R\|\|Cmax. Falta k > 3 ou grade mais fina, o estado e o ALNS |
| **C** | **Regime em que o aprendizado deveria pagar.** (i) pools heterogêneos **sem rótulo de classe**, a partir dos dados existentes; (ii) horizontes longos (T = 10⁴-10⁵) em R\|\|Cmax grande; (iii) **hospedeiro padrão de EC**: evolução diferencial com pool de estratégias (CEC'2017 ou BBOB) ou HyFlex; (iv) hospedeiro desenhado com melhor ação dependente do estado ou da fase | E | R1 (E1, E2), R2 (E1, E4), R3 (E1, E2), R4 (E1, E4) | (i) ~2 dias; (ii) 3 dias; (iii) 1-3 semanas; (iv) 4-8 dias | +3 a +5 | o aprendizado pode ganhar: isso refina a tese e torna o título ("onde o aprendizado paga") respondido | (i) dados por execução existem; (iii) e (iv) são novos |
| **D** | **Estimandos e inferência**: diferença (prêmio offline) como primário; razão de esperanças com Fieller; inclinação hierárquica por célula e por coleção; sementes divididas para quebrar o acoplamento; TOST e margem relativa; **proposição do "sanduíche do oráculo"** (validação cruzada ≤ θ* ≤ uma semente) | F + M | R1 (F2, F4), R2 (F1, M2), R3 (M1, M2), R4 (F2, M3) | 2-3 dias | +2 a +3 | baixo (a correlação pode cair) | reanálise de dados existentes; `analysis/sim_crossfit_estimator.py` já simula o estimador |
| **E** | **Lema do aprisionamento da primeira ação**, previsão da fração observada (55-73%), **teste de invariância à ordem** como diagnóstico reutilizável e **réplica de um desenho publicado** ([9] ou [36]) | F + M + E | R1 (F3, M4), R2 (M3, E3), R3 (F2), R4 (F3) | 2-6 dias | +2 a +4 | baixo ([9] pode quebrar empates de outro modo) | a ordem já está medida; falta a teoria e o teste empacotado |
| **F** | **Atribuição por Shapley / fatorial completo** no lugar da divisão decisão/custo (depende da ordem) e das ablações de um fator por vez; comparação *anytime* (área sob a curva makespan × CPU) em vez de um ponto | F | R2 (F2, M4); R1 (F2) e R4 (F2) pedem as condições de identificação | 3-5 dias + braços novos | +2 a +3 | médio (um só revisor; risco de formalismo em excesso) | as ablações 284, 285, 285b cobrem parte do fatorial (reconstrução × descida); faltam política × orçamento |
| **G** | **Aprendizes recomendados pela teoria e paridade de ajuste**: explore-then-commit, successive halving, UCB robusto, janela deslizante; irace nos hiperparâmetros dos aprendizes; aprendizado entre instâncias | M + E | R4 (M4), R1 (M2, M3), R2 (E2) | ~1 semana | +2 a +3 | baixo (reforça o nulo; se um deles ganhar, vira recomendação construtiva) | o aquecimento entre instâncias já foi testado (Entradas 271-276, 281) |
| **H** | **Segundo problema e ILS mais forte para a estrutura gulosa**: flowshop com reconstrução NEH contra reinserção aleatória; cadeia de ejeção ou VNS de troca múltipla em P\|\|Cmax | E | R1 (E3), R3 (E3), R4 e R2 (Q3) | 4-6 dias | +3 | médio (pode enfraquecer a afirmação) | ablação do P\|\|Cmax feita (284, 285, 285b) |
| **I** | **Posicionamento**: Turkeš 2021, Özcan 2008, Lissovoi, Doerr, Fawcett e Hoos, Campelo e Takahashi, Fréchette (Shapley); apresentar o nulo como replicação ampliada | texto | R1, R2, R3 | linhas (e cortes) | (embutido) | baixo | só citar o que eu verificar |
| J | HyFlex; hospedeiro competitivo (ALNS com qualidade de HGS, VRPTW) | E | R1 (E1), R2 (E4), R3 (E1) | 1-2 semanas | +2 a +5 | médio | baixa prioridade se o DE entrar |

## A versão-alvo (os quatro convergem)

Título de trabalho: **"Quando o aprendizado online de operadores é aprendível? Uma explicação preditiva, testada."**

- O protocolo vira o *instrumento de medida*, não a manchete.
- A manchete é uma **afirmação quantificada e falsificável**: "o ganho de um aprendiz online é limitado pela relação sinal-ruído por decisão (T·Δ²/σ² e variação no tempo ou no estado); em IG e ALNS típicos ela é pequena, então o conhecimento validado offline domina; e há regimes mensuráveis em que isso muda".
- **Mapa de três regimes**: (a) baixa folga ou operadores redundantes (P\|\|Cmax, flowshop): a teoria prevê captura ≈ 0 e os dados concordam; (b) folga real e valores de operadores estacionários (R\|\|Cmax, `alns`): a validada vence e os aprendizes capturam uma fração *previsível*; (c) regime projetado com valores não estacionários, heterogêneos ou dependentes do estado (DE, pools sem rótulo, horizonte longo): onde o aprendizado deveria pagar, **testado contra uma previsão congelada antes de rodar**.
- A armadilha da ordem vira uma seção curta, com lema, teste de permutação e verificação empírica. A estrutura gulosa vira um parágrafo de ablação de apoio.
- Cortes que os revisores sugerem para abrir espaço: a trilha de K = 9,6/1,8/2,0 e as implementações Python-laço/vetorizada/compilada, o hospedeiro AG, os detalhes de flowshop e Taillard, o parágrafo de irace e FRRMAB, a narrativa de "leitura e qualificação" das instâncias publicadas, parte da Tabela IV. O manuscrito longo (sem limite de páginas) guarda o que sair do artigo de 8 páginas.

## Plano em estágios, com portões (nada executado)

Todos os ciclos seguem a regra de sempre: previsões e hashes congelados **antes** de rodar; arquivos congelados não são editados (executores e relatórios novos); auditoria numérica nova; diário atualizado.

**Estágio 0 (dias, só texto, sem rodar nada):** restaurar a frase e o número do cronograma de fases (Entradas 238-239) e acrescentar o posicionamento (proposta I) depois de verificar as referências. Exige cortes de espaço.

**Estágio 1, Ciclo 53 (≈ 1 semana, dados existentes):** proposta D inteira (diferença como estimando primário, inclinação hierárquica, sementes divididas, regra estrita e TOST, margem relativa, sanduíche do oráculo com a simulação que já existe) mais C-i (pools sem rótulo de classe, a partir dos dados por execução). **Portão:** nenhum; é reanálise. O resultado pode reduzir o 0,78 e as contagens, e isso entra no artigo.

**Estágio 2, Ciclo 54 (≈ 1-2 semanas, executor instrumentado):** propostas A, B e E. Registrar, no braço aleatório, (ação, estado, recompensa) por decisão nas classes sintéticas, no R\|\|Cmax e no ALNS; estimar p_a, μ_a e σ_a por braço, por fase e por estado; calcular o índice de aprendibilidade; avaliar os oráculos por fases e por estado com validação cruzada; rodar o gêmeo digital; derivar o lema da ordem com verificação em simulação. **Portão (congelado antes):** se a correlação entre o índice (ou a captura prevista) e a captura observada, com IC por classe, ficar acima de um limiar fixado de antemão, **adota-se o re-enquadramento**; se não, relata-se como limite do modelo (estado, não estacionariedade) e mantém-se o enquadramento atual com as correções.

**Estágio 3, Ciclo 55 (≈ 2-3 semanas, hospedeiro novo):** proposta C-iii (DE com pool de estratégias; CEC'2017 ou BBOB, conforme a disponibilidade e a velocidade das bibliotecas) ou C-iv, com **a previsão do Estágio 2 congelada antes** (teste fora da amostra). Opcional: G (aprendizes da teoria e paridade de ajuste) no mesmo ciclo.

**Estágio 4 (≈ 1 semana):** reescrever o artigo (8 páginas) e o manuscrito longo; auditar; (nova revisão cega só se o Carlos pedir).

Calendário: de 22/11 ao prazo oficial do CEC (15/03/2027; uma listagem de terceiros dá 28/02 para artigos completos) são ≈ 14-15 semanas; os estágios 1 a 4 cabem em 6 a 8.

## Estimativa de nota

- Os revisores: 58,3 hoje (56-60); tetos de 67 a 74 (média 71) se as melhorias forem bem executadas.
- **Minha estimativa cética do conteúdo atual: 57-58**, igual à dos revisores: concordo que a novidade é 2 a 3 de 5 e que o nulo no P\|\|Cmax é quase forçado pelo teto; o que mais me convence é o R\|\|Cmax e o ALNS.
- **Depois dos Estágios 1-2**, se o índice de aprendibilidade prever a captura e o oráculo por fases e por estado confirmar que não há o que adaptar: **62-65**.
- **Depois do Estágio 3**, se um regime onde o aprendizado deveria pagar confirmar a previsão fora da amostra: **66-70**.
- Se o modelo falhar e nenhum regime novo entrar: **≤ 60**.
- Incerteza de ±5; as notas são de modelos de linguagem na mesma escala 0-100 da rodada anterior.

## Incertezas

- Os revisores são modelos de linguagem com vieses compartilhados; o R1 não fez busca na web; as referências que citam não foram verificadas (exceto Turkeš). A revisão por pessoas (orientador, colegas) continua pendente.
- Os tetos supõem execução impecável e as melhorias se sobrepõem.
- O re-enquadramento depende do portão do Estágio 2: se o modelo de bandido não explicar a captura, a manchete preditiva não se sustenta e a opção honesta é manter o enquadramento atual.
- Um hospedeiro de DE é escopo novo (contínuo; implementação nova; funções públicas do CEC'2017 ou BBOB a baixar) e custa páginas.
- Mais análises depois de ver as críticas aumentam o risco de floresta de caminhos; a defesa é congelar previsões e limiares antes de cada rodada e declarar no diário o que foi pós-hoc.

## Decisões que dependem do Carlos

1. Executar os **Estágios 1 e 2** (reanálise e executor instrumentado; sem mudar o escopo)?
2. Incluir o **hospedeiro de DE** (Estágio 3; muda o escopo) ou preferir o hospedeiro desenhado dentro do IG?
3. **Shapley/fatorial completo** (proposta F): sim ou não? (só um revisor pediu; custa braços novos)
4. O **re-enquadramento** do título e do resumo só se decide depois do portão do Estágio 2.
5. Commit deste material (esta síntese e a pasta de relatórios estão **sem commit**).
