# Trajetória do projeto PMSP-Learnheuristic e lições para outros projetos

**Período:** 29/08/2026 a 14/10/2026 (46 dias; 158 entradas no `diario_pesquisa.md`).
**Para quem é:** Carlos, coautores e qualquer chat do Claude Code que herde este projeto ou comece outro semelhante. Os detalhes de cada número estão no diário, citado como "E<n>" (Entrada n). Companheiro operacional: `PLAYBOOK_METODOLOGIA_CLAUDE_CODE.md`.

---

## 1. Em uma página

**O que começamos a fazer:** uma "learnheurística" (Q-Learning escolhendo operadores dentro de um ILS) para minimizar o makespan em máquinas paralelas idênticas (P||Cmax), mirando o EJOR.

**O que descobrimos:** o ganho que se atribui ao aprendizado vem, na prática, da **estrutura de busca** (destruir-reconstruir, busca local) e do **orçamento** (iterações, CPU). Uma política Q-Learning online adiciona de 0,01 a 0,05 ponto percentual em máquinas idênticas, ~0,1% em não relacionadas e nada (ou piora) no flowshop; uma configuração validada **offline** em instâncias separadas vence o agente online nos sete ambientes testados, contra cinco aprendizes. A "lei da folga" (a política ajuda onde há folga entre as ações) valeu para estratégias de destruição e **falhou em teste prospectivo** para o tamanho de destruição: folga é necessária, não suficiente.

**O que sobrou de positivo:** (i) um **protocolo de atribuição justo ao orçamento** (controle aleatório pareado, CPU igual calibrada em sequência, concorrente validado offline, folga do oráculo, limite certificado, margens de equivalência, pré-registro com hashes); (ii) **solvers sem aprendizado fortes** (97,5% de ótimos provados em P||Cmax sintético a T = 64.000; 0,25–0,30% do melhor conhecido no benchmark padrão de R||Cmax, perto do CPLEX de 2 h); (iii) um pacote auditável (338 testes, 368 conferências numéricas automáticas, diário com todas as não-confirmações).

**Onde estamos:** artigo de 25 páginas + suplemento de 44 + relatório técnico longo de 46; decisão de periódico em `DECISAO_SUBMISSAO.md` (recomendação: Journal of Heuristics). Chegamos ao limite da abordagem "método novo com aprendizado"; o que falta é empacotamento, coautoria e submissão.

---

## 2. Linha do tempo por fases

### Fase 0 — Higiene e diagnóstico (29/08; E1–E7)
Aplicamos a `LEI_GERAL_PESQUISA.md` ao projeto herdado. Primeira execução real do pytest revelou drift de API em testes (E4); um backup dentro de `tests/` quebrou a coleta (E5); suíte 127/127 verde na E7. **Lição:** a primeira coisa a fazer num projeto herdado é *rodar* tudo; ninguém sabia que a suíte estava quebrada porque ninguém a tinha executado.

### Fase 1 — Caça ao "ângulo em que ganhamos" (29–31/08; E8–E31)
- Revisão de literatura (E8): achamos trabalho próximo já publicado; a alegação de ineditismo foi revista. Depois da E11–E12, Carlos escolheu **Iterated Greedy + Q-Learning (`ig_ql`)** em vez de um GA como método de comparação.
- Diagnósticos do agente (E14–E16): a "dominância do operador 1" era sobretudo artefato de subtreinamento; a falta de alcançabilidade era estrutural, não de orçamento de treino.
- Granularidade de estado (E17–E18): **sinal favorável com N=30 desapareceu com N=100.** Estado com limiares manuais descalibrados em uma ordem de grandeza.
- Limite estrutural do ILS em n/m alto (E19–E22): o parâmetro errado era o *limiar* de perturbação, não a força (validado, p = 0,0003).
- `ig_ql` bate todos os ILS e LPT de forma consistente (E24–E25), e as ablações (E26–E30; fatorial 2×2 com UCB1, UCB1 por estado, Q-Learning sem estado) mostram que **o mecanismo de busca explica quase tudo e o Q-Learning acrescenta pouco**.
- Baselines publicados reais (E33–E44): MULTIFIT, SA de Li e Perkins, BCO de Davidović et al. O `ig_ql` perde para o MULTIFIT nas classes grandes (padrão de cruzamento por escala). Correção de Holm aplicada retroativamente (E37) derrubou alguns efeitos.
- Um modelo de linguagem externo (Gemini via MCP) **inventou um DOI e a proveniência de instâncias** (E36); pego por conferência com fonte primária.

### Fase 2 — Benchmark externo e dúvidas de ineditismo (01–09/09; E48–E89)
- Instâncias publicadas (Symmetry 2025, repositório GitLab): download autorizado, 6.780 instâncias na amostra final (E49–E68). Dois bugs reais: overflow int64 na família B (E53) e colisão de `inst_id` no resume (E57, achado ao reauditar o CSV).
- Análise externa (E58): a vitória sobre baselines replica, mas **o efeito específico do Q-Learning e a vantagem do mecanismo IG não generalizam uniformemente** (ILS ganha em B, IG em C).
- Tentativas de mecanismo híbrido, racing, portfólio, regra a priori `dup_ratio` (E59–E67). Um "ganho" na família E era artefato de **2 instâncias** (E66→E67).
- "Vocês têm certeza de que é novidade?" (E72): a frase de ineditismo protegia `ils_ql`, mas não `ig_ql`. Carlos conseguiu o texto completo do trabalho mais próximo (Missaoui e O'Sullivan, 2026) diretamente com um dos autores (E78) e a diferenciação foi refeita no nível do mecanismo.
- Reescrita completa do artigo em inglês e migração para `elsarticle` (E76–E89).

### Fase 3 — Auditoria crítica e pré-registro (04–08/10; E90–E113)
Carlos pediu uma avaliação sem filtro (E97): 19 alegações problemáticas (estado degenerado, "variância" mal-rotulada, UCB com recompensa bruta, etc.). Resposta: pré-registros com hashes, **limite inferior certificado** por relaxação do modelo de fluxo em arco (E101), baselines construtivos fortes e `IG` com partida forte (E102–E103), **folga e sinal** (E104–E106), hiperparâmetros (E105), distribuições fora de U[1,100] (E108), variância entre seeds (E109), **CPU igual** (E110–E111), tempos na mesma sessão (E107), versão de 30 páginas (E112) e auditoria numérica (E113).

### Fase 4 — Ciclos de melhoria sob a regra mestra (08–12/10; E114–E157)
Carlos levantou o limite de páginas (E114) e instituiu a **regra mestra** (E115): plano → execução → auditoria como revisor → novo plano. Cada ciclo atacou as fragilidades da auditoria anterior:

| Ciclo | O que se testou | Resultado |
|---|---|---|
| 1–2 (E114–E120) | Folga real (tamanho de destruição); replicação em 2.690 instâncias novas; R||Cmax | A política **não** captura a folga; efeito pequeno replica; estado informativo não agrega |
| 3 (E121–E123) | Receita: IG com d validado + partida forte | Vence 8/8 testes pré-especificados; 74,7% de ótimos provados |
| 4–5 (E124–E132) | IG híbrido (busca local); R||Cmax grande; **CPU igual** | O ganho de 0,150% da busca local vira 0,010% a CPU igual: 93% era orçamento; na receita, o sinal inverte |
| 6–7 (E133–E136) | Política em regime não saturado (R||Cmax 1000×50) | Ganha onde uma estratégia domina; perde onde não; validação offline faz o mesmo ganho |
| 8 (E137–E138) | Benchmark padrão de R||Cmax (1.400 instâncias, download autorizado) | A decomposição replica em dados de terceiros; lei da folga confirmada (ρ = 0,74) |
| 9 (E139–E150) | Curva qualidade×orçamento; LP-IG; LP-IG vs MILP; **teste prospectivo da lei da folga**; LP-IG + MILP restrito | 97,5% de ótimos a T = 64.000; **lei refutada** para o tamanho de destruição (ρ = −0,37); LP-IG + MILP restrito a 0,25% do melhor conhecido |
| 10 (E152–E155) | Terceiro problema (flowshop); aprendizes para janela curta (Thompson, torneio) | Ingrediente ativo é outro (a busca local); nenhum dos 5 aprendizes captura a folga |
| 11 (E156–E157) | Candidatas do MILP restrito por custo reduzido do LP | Significativo em sinal, **desprezível** em tamanho; critério pré-fixado falhou (0,267% > 0,25%) |

### Fase 5 — Reposicionamento e empacotamento (11–14/10; E152, E158)
Com liberdade para reposicionar ("fica a seu critério"), a tese passou a ser **a atribuição** (o protocolo), com os resultados de solver como contribuição de apoio. Pergunta de Carlos: "já chegamos no limite?" Resposta: sim para método novo com aprendizado (`DECISAO_SUBMISSAO.md` §1). Versão de 30 páginas montada (24 pp), suplemento reorganizado, versão longa arquivada, decisão de periódico, este documento e o playbook.

---

### Fase 6 — Ciclo 12: responder às três críticas mais graves (18/10; E159–E161)
Nota crítica de 67/100 apontou: "o aprendiz é um espantalho", "a prática comum é um espantalho" e "flowshop só em instâncias geradas". Respostas: (A) replicação em 40 instâncias **clássicas de Taillard** (regeneradas do gerador publicado, verificadas contra a matriz impressa; o achado se replica); (B) **auditoria de literatura** com protocolo congelado antes de ler (12 artigos de arXiv: controles baratos mais comuns do que o previsto, controles de orçamento, concorrente offline e multiplicidade raros; todos concluem que o aprendizado ajuda; 5 de 7 previsões erradas, relatado); (C) **aprendizes de outra classe** (LinUCB contextual e seletor offline condicionado à instância): nenhum muda a conclusão. Nota estimada depois: ~72. Erros e armadilhas do ciclo: pool de processos caiu no primeiro lançamento do teste do LinUCB (retomado com menos workers), um heredoc com `\\textwidth` virou TAB (corrigido com Edit), o site de Taillard não respondeu (usado o gerador publicado na OR-Library), HTML do arXiv de um artigo era um modelo vazio (excluído por regra).

---

## 3. Becos sem saída (não repita sem motivo novo)

| Tentativa | Onde | Por que não deu |
|---|---|---|
| Consertar o "viés para o operador 1" por treino/teste e ruído | E15–E16 | Era artefato de subtreinamento; o teto é estrutural |
| Estado mais fino (16/81/256 estados) | E17–E18 | Nulo com N=100 |
| Escalar `n_perturb` com o tamanho | E20–E22 | O parâmetro errado era o limiar de perturbação |
| Mecanismo híbrido IG/ILS, racing, portfólio, gatilho de confiança | E59–E65 | Racing decepciona; híbrido preliminar; gatilho caro |
| Features para a família E | E66–E67 | Artefato de 2 instâncias |
| Estado recalibrado (terciles, relativo ao início) | E98–E99 | Não muda nenhuma conclusão; no ILS, piora |
| Recompensa densa; pré-treino por classe | E104–E106 | Densa piora (+0,084%); pré-treino equivale a aleatório |
| Mais orçamento ou menos orçamento para o agente | E114–E121 | A folga cresce; o ganho da política não |
| Thompson (com/sem desconto), torneio de eliminação | E155 | Não superam o sorteio de d |
| Candidatas do MILP por custo reduzido | E156–E157 | Ganho desprezível (−0,03%) |
| Heurísticas puras contra o MILP em R||Cmax | E120, E130 | O MILP vence até 1000×50, exceto com o LP-IG |

---

## 4. Erros nossos e como foram pegos

| Erro | Como apareceu | Correção |
|---|---|---|
| Testes desatualizados e backup em `tests/` | Primeira execução do pytest (E4–E6) | Testes corrigidos; backups fora da coleta |
| Citação/DOI inventado por modelo externo | Conferência contra fonte primária (E36) | Regra: citação entregue por agente = alegação não verificada |
| Overflow int64 na família B | Avisos no log da bateria (E53) | `dtype` adaptativo; 322 linhas corrompidas purgadas |
| Colisão de `inst_id` no resume | Reauditoria do CSV, join por `member` (E57) | Corrigido na origem |
| "Ganho" na família E | Recontagem excluindo o subfolder degenerado (E67) | Retratado no diário e no texto |
| Sinal com N=30 | Ampliação para N=100 (E18) | Regra: tendência com amostra pequena é piloto |
| Ineditismo de `ig_ql` não protegido | Pergunta direta de Carlos (E72) | Busca ativa, texto completo do trabalho mais próximo (E78) |
| Relatórios abortados por erro de tipo | Antes de qualquer saída (E119, E125) | Emenda registrada *antes* de ler resultado |
| "Ressalva" sobre análise não executada | Revisão (E132) | Corrigida no diário (CPU já estava igualada) |
| Rótulo 0,46% vs 0,45%, contagens de auditoria, 0,27% no resumo, "74,7%→97,5%" misturando amostras | Auditorias numéricas (E113, E150–E158) | Textos e `audit_short_numbers.py` |
| Scripts de correção com `\\` em heredoc corromperam texto | Leitura do resultado | Regra: Write/Edit para LaTeX e regex |
| O próprio auditor tratava `\%` como início de comentário | Erro de lógica visível na revisão (E158) | Corrigido antes de valer |

**Previsões registradas antes das execuções que falharam** (e foram reportadas): LP-IG vs MILP (M1), política sobre d (D2–D5), flowshop (P1/P4: o IG não bate o ILS a CPU igual), magnitude e critério total do LP-RC. Registrá-las antes é o que torna crível o resto.

---

## 5. Log de decisões estratégicas (quem decidiu)

| Quando | Decisão | Decidiu |
|---|---|---|
| E3 | Critério Rota A/B dado só pela qualidade da learnheurística | Carlos |
| E9 | Abandonar o template ENIAC/SBC; LaTeX genérico | Carlos |
| E10 | Repaginação extensiva em vez de fases incrementais | Carlos |
| E12 | `ig_ql` no lugar do GA como método central | Carlos |
| E43 | "Síntese A+B" refinada; manter `ig_ql` como manchete | Carlos |
| E46, E51 | Buscar e baixar benchmark público (B+C+E+F) | Carlos (download explícito) |
| E94 | Pré-registro e enquadramento do artigo | Carlos |
| E114 | Levantar o limite de 30 páginas | Carlos |
| E115 | **Regra mestra** (ciclos de auditoria) | Carlos |
| E137 | Baixar o benchmark padrão de R||Cmax | Carlos (download explícito) |
| E152 | Reposicionar em torno da atribuição ("contribuição massiva") | Carlos delegou, eu executei |
| E158 | 30 páginas, periódico, documento da trajetória | Carlos |
| Pendentes | Coautoria, periódico final, título, repositório/DOI, palavra-chave oficial, financiamento | **Carlos** |

---

## 6. Lições

### 6.1 Científicas (transferíveis a qualquer "X guiado por aprendizado")
1. **Controle aleatório e escada de ablação desde o dia 1.** Só construímos o controle (`ig_random`, `ig_ucb`, `ig_ucb_ctx`, `ig_ql_flat`) na E23–E29; o achado central estava a um passo antes.
2. **Atribua ao componente certo.** O quadro "Q-Learning vence" virou "destruir-reconstruir vence" assim que o mesmo mecanismo rodou com seleção aleatória.
3. **Iterações iguais não são orçamento igual.** Medir a CPU por par, em sequência, em máquina ociosa; tempos de execuções paralelas medem a carga da máquina.
4. **O concorrente que falta é o offline.** Uma configuração escolhida em instâncias separadas é o que um praticante usaria; todos os nossos aprendizes online perderam para ela.
5. **Folga necessária, não suficiente.** Uma lei confirmada em dois conjuntos (ρ = 0,59 e 0,74) foi refutada em teste prospectivo (ρ = −0,37). Pré-especifique o teste em *outro* espaço de ações antes de generalizar.
6. **Compare com limite certificado e com terceiros.** Parte da lacuna ao limite clássico é folga do limite (≈38% do GAP na família C); instâncias sintéticas lisonjeiam; só o benchmark padrão e os exatos de terceiros mostram onde estamos (ainda atrás do Gurobi e do exato).
7. **Se o resultado é "o aprendizado não paga", entregue algo que paga.** O pacote só ficou publicável com os solvers sem aprendizado (receita, LP-IG + MILP restrito) ao lado do diagnóstico.
8. **Verifique ineditismo duas vezes, perto do fim também.** O trabalho mais próximo era de 2026.
9. **Efeitos pequenos exigem testes agregados e médias por seed**; testes por classe com N=100 não resolvem 0,01–0,05 ponto. Fixe margens de equivalência antes.

### 6.2 De método
1. **Diário append-only com pré-registro:** hipótese, regra de decisão, hashes (md5) do código e dos dados, previsões — tudo antes de rodar; emendas só antes de ler resultados.
2. **Auditoria como revisor a cada ciclo** (regra mestra): lista de fragilidades, classificadas como corrigíveis (viram o plano do ciclo seguinte) ou limitações declaradas.
3. **Auditorias numéricas automáticas:** scripts que recalculam do CSV os números do texto (`audit_entrada*`), mais `audit_short_numbers.py`, que garante que todo número do texto curto existe numa fonte auditada.
4. **Cortar sem perder:** manter a versão longa como relatório técnico e gerar o material movido por script a partir dela (`make_s_moved.py`), em vez de reescrever à mão.
5. **Backup antes de editar** diário, bibliografia e scripts; nada de apagar sem registro.
6. **Reportar toda não-confirmação** (a tabela de pré-registro lista todas).

### 6.3 Trabalhando com o Claude Code
1. **O contexto acaba; o projeto não pode depender dele.** Foram vários reinícios. O que sobreviveu foi o que estava em arquivo: diário, documentos de plano e auditoria, arquivos de memória, `README`. O resumo automático perde detalhes; a transcrição completa fica no `.jsonl` da sessão e pode ser relida.
2. **Memória persistente enxuta:** perfil do usuário (responder em português), regra mestra, estado do projeto, armadilhas de ferramenta. Atualize o índice no fim de cada ciclo.
3. **Portões de aprovação para o que é irreversível ou estratégico:** downloads, coautoria, título, periódico, mudança de tese. Autonomia para experimentos *já pré-especificados*.
4. **Peça a crítica antes da construção.** As perguntas de Carlos ("quais nossas fraquezas?", "temos algo novo?", "já chegamos no limite?") produziram mais valor que qualquer experimento isolado.
5. **Segundas opiniões de outros modelos são úteis e perigosas:** tudo que vier delas é hipótese até conferir (E36).
6. **Fatie o trabalho em ciclos fechados por uma auditoria**, não por "terminar tudo".

### 6.4 De engenharia (Windows, Python, LaTeX)
- **Não use heredoc do Bash com barras invertidas** (Python/LaTeX/regex): use a ferramenta Write/Edit. Aprendemos isso várias vezes.
- Dois `ProcessPool` ao mesmo tempo no Windows derrubam processos (`WinError 6`); fase de MILP não roda junto de jobs pesados; sem compilador/numba, o flowshop foi implementado só em numpy.
- Calibração de CPU igual: sequencial, máquina ociosa, instâncias separadas.
- `PYTHONIOENCODING=utf-8` para imprimir setas e acentos no console (cp1252 quebra).
- LaTeX: `elsarticle` + `xr` entre artigo e suplemento; compile principal → suplemento → principal; use `bash paper/build.sh`; rótulos não podem se repetir entre os dois documentos.
- O ambiente bloqueia `rm` com variável de shell sem proteção: use caminho literal ou `"${VAR:?}"`.
- Regex que remove comentários LaTeX deve ignorar `\%`.

---

## 7. Se começássemos hoje (roteiro de 10 passos)

1. Rodar a suíte inteira e consertar o que está quebrado (E1–E7).
2. Escolher *duas* perguntas: "o que o aprendizado acrescenta sobre o controle aleatório, a CPU igual e o concorrente offline?" e "onde ficamos contra o estado da arte de terceiros?".
3. Montar o protocolo de atribuição antes de qualquer método novo (controle aleatório, ablações, CPU igual, offline, limite certificado).
4. Escolher o benchmark padrão e pedir autorização de download no primeiro dia.
5. Buscar o trabalho mais próximo (e obter o texto completo) antes de fixar a tese.
6. Pré-registrar cada bateria (hipótese, regra, hashes, previsões); rodar; relatar tudo.
7. Fechar cada ciclo com uma auditoria de revisor; nunca acumular mais de um ciclo sem auditoria.
8. Mudar de tese quando os dados mandarem (aqui: do "método" para a "atribuição + solvers sem aprendizado").
9. Empacotar: artigo curto, suplemento, relatório técnico longo, repositório com pré-registro público.
10. Decidir o periódico como pergunta factual (escopo e política), não de ambição (`LEI_GERAL` II.11).

---

## 8. Mapa de arquivos

| O quê | Onde |
|---|---|
| Diário (fonte primária de cada número) | `diario_pesquisa.md` |
| Artigo curto, suplemento, versão longa | `paper/learnheuristic_pmsp.tex` (+ `paper/short/*.tex`), `paper/supplementary_material.tex` (+ `paper/s_moved.tex`), `paper/learnheuristic_pmsp_long.tex` |
| Build | `bash paper/build.sh`; material movido: `python paper/make_s_moved.py` |
| Código dos métodos | `pmsp/` (ILS, IG, Q-Learning, UCB, bandits de janela, LP-IG, MILP restrito, flowshop, limites) |
| Execuções e relatórios por entrada | `experiment/run_*_entrada*.py`, `analysis/report_entrada*.py` |
| Estatística reutilizável | `analysis/stats.py` (Holm, rank-biserial, bootstrap, família de Wilcoxon) |
| Auditorias | `analysis/audit_*.py`, `analysis/audit_short_numbers.py` |
| Hashes de pré-registro | `results/entrada*_prereg_hashes.txt` |
| Planos e críticas | `PLANO_*`, `AUDITORIA_REVISOR_CICLO2/3/4.md`, `AVALIACAO_CRITICA_05102026.md`, `ENQUADRAMENTO_EJOR.md`, `DECISAO_SUBMISSAO.md`, `SUBMISSAO_CHECKLIST.md` |
| Regras de trabalho | `LEI_GERAL_PESQUISA.md` (incl. regra mestra II.13), `PLAYBOOK_METODOLOGIA_CLAUDE_CODE.md` |
| Benchmarks de terceiros (fora do git) | `external_benchmarks/` |

---

## 9. Pendências que só o Carlos resolve

Coautoria; periódico final e título; palavra-chave oficial; financiamento; repositório (URL/DOI) e **primeiro commit** (nenhum foi feito); depósito público do pré-registro (Zenodo/OSF); leitura do texto novo (resumo, introdução, discussão); conferência das 8 referências adicionadas na E158; (opcional) autorização para baixar as instâncias clássicas de Taillard do flowshop.
