# Playbook de metodologia e de trabalho com o Claude Code

Versão 1.0 — 14/10/2026. Escrito a partir do projeto PMSP-Learnheuristic (158 entradas de diário; ver `TRAJETORIA_E_LICOES.md`). Complementa `LEI_GERAL_PESQUISA.md` (regras e jurisprudência gerais) e não a substitui: aqui estão o **protocolo experimental reutilizável**, os **modelos prontos** e as **práticas de trabalho com o agente**.

---

## 0. Como iniciar um chat novo (copie e cole)

> Você está assumindo um projeto de pesquisa em andamento. Responda **sempre em português**; o manuscrito é em inglês. Leia, nesta ordem: (1) `README.md`; (2) `LEI_GERAL_PESQUISA.md` (Partes I e IV e o item II.13); (3) `PLAYBOOK_METODOLOGIA_CLAUDE_CODE.md`; (4) `TRAJETORIA_E_LICOES.md`; (5) a memória do projeto (`MEMORY.md` e os arquivos que ele aponta); (6) as últimas 3 entradas de `diario_pesquisa.md` e o documento de estado/plano mais recente. Antes de propor algo novo, confira em `TRAJETORIA_E_LICOES.md` §3 se já foi tentado. **Restrições:** não fazer commit, download, instalação de pacote nem envio externo sem autorização explícita; fazer backup antes de editar o diário, a bibliografia e scripts; usar as ferramentas Write/Edit (e não heredoc do Bash) para LaTeX, regex e qualquer texto com barras invertidas; registrar no diário *antes* de rodar (hipótese, regra de decisão, hashes, previsões); reportar toda não-confirmação; ao fim de cada ciclo, auditar como revisor. Pergunte a mim apenas decisões estratégicas ou irreversíveis.

---

## 1. O protocolo de atribuição justo ao orçamento (reutilizável)

Use sempre que alguém afirma que um componente aprendido (política, preditor, seletor) melhora uma busca ou um solver. Oito regras:

| # | Regra | Como fazer |
|---|---|---|
| 1 | **Controle aleatório a orçamento igual** | Mesmas ações, mesmas iterações, mesmas instâncias; o seletor aleatório escolhe uniformemente. Diferença pareada por instância: `d = 100 (A − B)/B` (negativo: A melhor). |
| 2 | **CPU igual, calibrada em sequência** | Veja §1.1. Nunca use tempos de execuções paralelas. |
| 3 | **Concorrente validado offline** | Escolher a melhor ação *fixa* em instâncias de ajuste do mesmo tipo/tamanho; testar em instâncias novas. |
| 4 | **Folga e captura** | Folga = ganho do oráculo por instância (melhor ação fixa em cada instância) sobre o aleatório; captura = ganho da política ÷ ganho do concorrente validado. |
| 5 | **Limites certificados e terceiros** | GAP contra limite inferior certificado (relaxação) e contra melhores soluções conhecidas de benchmark padrão. |
| 6 | **Testes pareados com controle de multiplicidade** | Wilcoxon bicaudal pareado; Holm dentro de cada família pré-especificada; rank-biserial; IC90 por bootstrap; margem de equivalência (±0,05%; ±0,10% quando declarado). |
| 7 | **Pré-registro com hashes** | §3.1. O registro é datado e só cresce. |
| 8 | **Instâncias separadas** | Ajuste (tuning), teste e reserva; a reserva é tocada uma vez. |

### 1.1 Calibração de CPU igual
1. Escolha o par (A caro, B barato) e o orçamento de A (por exemplo `T = 1000`).
2. Em instâncias de calibração *separadas*, num único processo, **em máquina ociosa**, meça a CPU média por execução de A e de B com o mesmo `T`; `K = cpu_A / cpu_B`.
3. B recebe `T_eq = round(T · K)` iterações.
4. Reverifique a razão em máquina ociosa (esperado 0,96–1,02) e **congele K no pré-registro** antes do teste.
5. Reporte os dois cenários (iterações iguais e CPU igual); a diferença é o tamanho do "efeito de orçamento".

*Caso real:* a busca local no IG de P||Cmax multiplicava a CPU por 2,1–52,4; a 0,150% de ganho a iterações iguais cai para 0,010% a CPU igual (93% era orçamento). Os tempos medidos em execução paralela davam razões entre 0,74 e 1,17, enquanto a razão real era de 2,3 a 82,9.

### 1.2 Estatística (código pronto em `analysis/stats.py`)
`holm_bonferroni`, `rank_biserial_paired`, `bootstrap_ci_mean_diff`, `pairwise_wilcoxon_family`, `variance_equality_family`. Convenção do projeto: famílias confirmatórias fixas, "confirmada = p_holm < 0,05 e d < 0", e uma margem de equivalência declarada antes. Efeitos de ordem 0,01–0,05 ponto só aparecem em testes agregados (todas as classes juntas) ou em médias por seed.

### 1.3 O que o protocolo deve produzir
Uma tabela "prática comum × orçamento justo" (como a Tabela `fair_vs_naive` do artigo) e uma tabela síntese entre ambientes (oráculo, validado, aprendiz online, custo relativo de CPU). Se as duas colunas concordam, o resultado é robusto; se discordam, essa é a contribuição.

---

## 2. Ciclo de trabalho (regra mestra, `LEI_GERAL` II.13)

1. **Plano** de melhorias, cada uma com hipótese e regra de decisão.
2. **Pré-registro** no diário (§3.1) e hashes.
3. **Execução** (em lotes; relatório automático).
4. **Resultados** no diário (§3.2): hashes conferidos, emendas só se anteriores à leitura do resultado, previsões acertadas e erradas.
5. **Auditoria como revisor** (§3.3) do trabalho inteiro.
6. Novo plano com as fragilidades corrigíveis. Só se encerra quando a auditoria não achar fragilidade corrigível relevante.

---

## 3. Modelos prontos

### 3.1 Entrada de pré-registro (diário)
```markdown
## Entrada N (dd/mm/aaaa) — CICLO k, frente j — <título>
**Motivação (auditoria X, fragilidade Y):** <por que esta frente existe>
**Hipóteses:** H1 ...; H2 ...
**Delineamento:** instâncias (ajuste / teste / reserva; índices), métodos, orçamento (T, CPU igual? K congelado), seeds, scripts (`experiment/run_*`, `analysis/report_*`).
**Família confirmatória:** testes, correção (Holm/m), regra de decisão ("confirmada = p_holm < 0,05 e d < 0"), margem de equivalência.
**Critérios descritivos fixados antes:** <ex.: GAP ≤ 0,25% no total e ≤ 0,5% por tipo>
**Previsões (antes de rodar):** <direção e ordem de grandeza de cada teste>
**Hashes:** `results/entradaN_prereg_hashes.txt` (md5 de scripts, entradas e saída)
```

### 3.2 Entrada de resultados
```markdown
## Entrada N+1 (dd/mm/aaaa) — Resultados do CICLO k (Entrada N)
Hashes conferidos? emendas (e por quê, antes de ler resultados)? execuções, tempo, erros, integridade.
**Testes:** id — comparação — d, IC90, W/L/T, p_holm — CONFIRMADA / NÃO.
**Previsões:** acertou / errou (com os números).
**Conclusão honesta** e **o que muda no texto** (seção, tabela, alegação).
```

### 3.3 Auditoria como revisor (`AUDITORIA_REVISOR_CICLOk.md`)
Tabela: **Fragilidade** · **Evidência (arquivo/linha/número)** · **Classe** (corrigível com experimento / corrigível com texto / limitação declarada) · **Ação** · **Ciclo**. Incluir sempre: alegações sem suporte, confundimentos (o que mais mudou entre os grupos?), baselines fracos, escopo estreito, limitações sem teste, erros factuais, reprodutibilidade, citações não conferidas.

### 3.4 Lista "antes de afirmar um efeito"
- Há controle aleatório a orçamento igual e concorrente offline?
- O orçamento é igual em CPU (calibrado em sequência), não só em iterações?
- A amostra é grande o bastante (N=30 é piloto)? A conclusão sobrevive à exclusão de casos degenerados?
- Há limite certificado e referência de terceiros?
- O efeito sobrevive a seeds, hiperparâmetros e estado/reward alternativos?
- A hipótese foi escrita *antes* dos dados? Se não, está rotulada como exploratória e testada em instâncias novas?
- Cada número do texto tem rastro até um arquivo (e foi recalculado por um auditor)?

---

## 4. Trabalhando com o Claude Code

1. **O estado vive em arquivos, não na conversa.** Diário (append-only), `README`, documentos de plano/auditoria, memória persistente (`~/.claude/projects/<projeto>/memory/`), `SUBMISSAO_CHECKLIST.md`. O resumo automático no fim do contexto perde detalhes; a transcrição completa da sessão (`.jsonl`) pode ser relida quando preciso.
2. **Memória persistente enxuta:** perfil do usuário (idioma, nível), regras de feedback (regra mestra, armadilhas de ferramenta), estado do projeto, referências. Atualize o índice (`MEMORY.md`) ao fim de cada ciclo.
3. **Portões de decisão:** perguntas estratégicas e irreversíveis vão ao usuário (título, coautoria, periódico, mudança de tese, downloads, commits, instalações). Experimentos já pré-especificados podem rodar com autonomia.
4. **Peça a crítica antes da construção.** As perguntas mais valiosas do projeto foram: "quais são as nossas fraquezas?", "temos algo novo?", "já chegamos no limite?".
5. **Segundas opiniões** (outros modelos): tratar como hipótese; conferir toda citação, DOI e fonte de dados no original.
6. **Tarefas longas:** rodar em segundo plano; um `ProcessPool` por vez no Windows; MILP não concorrente com jobs pesados; calibrar CPU em máquina ociosa.
7. **Backups e arquivo:** `archive/root_backups/` para diário e bibliografia; `archive/paper_backups/` para versões do artigo; nunca apagar sem nota.
8. **Fechar a sessão com sincronização:** diário, documento de estado, memória e `README` coerentes entre si (`LEI_GERAL` Parte IV, item 7).

---

## 5. Catálogo de armadilhas técnicas (Windows + Python + LaTeX)

| Armadilha | Sintoma | Remédio |
|---|---|---|
| Heredoc do Bash com `\` | `\textsc` vira TAB+`extsc`; `\\` colapsa | Ferramenta Write/Edit; ou `chr(92)` em Python |
| Regex que remove comentário TeX | `\%` tratado como início de comentário; tokens perdidos | `(?<!\\)%.*` |
| Dois `ProcessPool` simultâneos | `WinError 6`, queda de processos | Um por vez; retomar a partir de checkpoint |
| Backup dentro de `tests/` | Pytest coleta erros | Prefixo `_backup_` ou pasta `archive/` |
| Overflow int64 em instâncias de faixa larga | Avisos de overflow, resultados corrompidos | `dtype` adaptativo/inteiros de precisão arbitrária, com teste de regressão |
| `resume` com chave colidindo entre gerações do manifesto | Linhas faltantes ou trocadas | Conferir o CSV final por junção em chave estável (`member`) |
| Console cp1252 | `UnicodeEncodeError` ao imprimir `→` | `PYTHONIOENCODING=utf-8` |
| Rótulos repetidos entre artigo e suplemento (`xr`) | "multiply defined" ou referência vazia | Cada rótulo em um só documento; prefixo `S-` no principal |
| Ordem de compilação | Referências cruzadas desatualizadas | principal → suplemento → principal (`paper/build.sh`) |
| `rm` com variável de shell | Bloqueio de segurança do ambiente | Caminho literal ou `"${VAR:?}"/*.png` |
| Medir tempo em paralelo | Razões de CPU erradas | Sequencial, processo único, máquina ociosa |

---

## 6. Candidatos a nova jurisprudência (para anexar a `LEI_GERAL_PESQUISA.md` após sua revisão)

**II.14 — Compare com a configuração validada offline, não só com o controle aleatório.** *Caso:* o Q-Learning sobre o tamanho de destruição perdia 0,152% para a escolha aleatória, mas 0,360% para o tamanho validado em instâncias separadas; contra o aleatório só, o veredito é "neutro/pior"; contra o concorrente real, "muito pior" (E146).

**II.15 — Iterações iguais não são orçamento igual.** *Caso:* 0,150% de ganho da busca local a iterações iguais viram 0,010% a CPU igual; na receita, o sinal inverte (+0,034%) (E128, E131).

**II.16 — Uma lei confirmada em duas amostras ainda é hipótese: teste-a prospectivamente em outro espaço.** *Caso:* ρ = 0,59 e 0,74 para estratégias de destruição; ρ = −0,37 para o tamanho de destruição, com critério pré-fixado ρ ≥ 0,5 (E146).

**II.17 — Registre previsões antes de rodar e relate as erradas.** *Caso:* previsões erradas sobre LP-IG vs MILP, política sobre d, flowshop e LP-RC foram todas reportadas; isso sustenta a credibilidade das acertadas (E144, E146, E153, E157).

**II.18 — Audite o auditor e a transcrição dos números.** *Caso:* "0,27%" no resumo (valor de uma variante desprezível) em vez de "0,25–0,30%"; "74,7% → 97,5%" misturando amostras de 12 e 6 classes; e um auditor que tratava `\%` como comentário e deixava passar números (E158). Remédio: auditoria de proveniência que exige que cada número do texto exista numa fonte já auditada.

**II.19 — Ao cortar para caber no limite, preserve a versão longa e gere o material movido por script.** *Caso:* 46 → 24 páginas, com o suplemento montado por `make_s_moved.py` a partir do texto original, sem retranscrever nada (E158).

**II.20 — "Achados" apoiados em poucas instâncias são suspeitos até a exclusão de casos degenerados.** *Caso:* ganho na família E que era artefato de 2 instâncias (E66→E67); sinal com N=30 que sumiu com N=100 (E17→E18).

**II.21 — Verifique o ineditismo contra a literatura mais recente, e de novo antes de submeter.** *Caso:* o trabalho mais próximo era de 2026 e só protegíamos `ils_ql`, não `ig_ql` (E8, E72, E78).

**II.22 — Uma comparação "a CPU igual" só é tão justa quanto a eficiência das implementações dos dois braços.** *Caso:* o "93% do ganho da busca local era orçamento" (manchete do resumo) vinha de uma busca local em laços Python (razão de CPU com/sem busca local K = 9,6 em média geométrica); vetorizada, com resultados idênticos (120/120 execuções, 188 testes), K cai para 1,8 e o ganho a CPU igual é 0,105% (só 30% era orçamento); na receita, o sinal passa de +0,034% (a busca local perdia) para −0,008% (neutra) (E162–E164). *Como detectar:* antes de atribuir um ganho a "orçamento", pergunte se a razão de CPU vem do algoritmo ou de uma implementação ingênua; vetorize/compile os dois braços, teste a equivalência exata e reporte K.

**II.23 — Máquina compartilhada: meça CPU do processo, não o relógio de parede, e não use um único `ProcessPool`.** *Caso:* com outros jobs do usuário ocupando a máquina, meus workers ficaram quase parados e o pool caiu várias vezes com `WinError 6`; a solução foi rodar N processos independentes (fatias `--shard/--nshards`, cada um com seu CSV, e `--merge`), retomáveis, e calibrar razões de CPU com `time.process_time` acumulado (E163–E165).

**II.24 — Uma regra de decisão que combina um teste de postos com o sinal da média pode dar um veredito que nenhuma das duas estatísticas sustenta.** *Caso:* "confirmada = p_holm < 0,05 e d < 0" classificou como confirmado um seletor MLP que era pior em 160 instâncias e melhor em 36 (rank-biserial +0,76) com média −0,011% equivalente a zero; o texto reporta "paridade" e registra que a regra não previu a discordância (E169). *Como evitar:* pré-especifique a leitura quando o teste de postos e a média discordam, use equivalência pela média, e relate a discordância em vez do veredito literal.

**II.25 — Uma unidade de orçamento "independente de implementação" (contagem de operações) não é livre de suposições.** *Caso:* ao compilar o IG e a busca local (numba, 403 testes de equivalência) e contar avaliações de movimento, a CPU de uma implementação eficiente (K = 2,0; ganho −0,115%, 28% de orçamento) concordou com a vetorizada (K = 1,8; −0,105%), mas a contagem deu K entre 1,9 e 9,1 e ganho −0,080% (50% de orçamento): ela embute pesos entre tipos de operação (a reconstrução custa 1,9 vezes menos que a busca local em CPU, correlação de postos 0,71) e o ganho vai de −0,026% a −0,124% quando o peso da reconstrução vai de 0,25 a 4 (E170–E171). *Como aplicar:* use como orçamento principal a CPU de uma implementação eficiente de todos os braços; trate a contagem como verificação, declare o modelo de custo e reporte a sensibilidade aos pesos.

**II.26 — Estenda a amostra de uma auditoria sem mudar o livro de códigos, e conte as previsões que erram nos limites.** *Caso:* a auditoria de literatura passou de 12 para 23 artigos com uma emenda congelada (hash) antes de ler, com os mesmos critérios; 5 de 7 previsões acertaram e as 2 que erraram, por 1 ponto percentual (26% contra <25%; 20% contra <20%), foram relatadas como erros, sem ajustar o limiar; o critério de inclusão foi mantido mais estreito de propósito onde uma leitura liberal inflaria a tese (E172). *Como evitar:* registre a emenda e as previsões antes de ler; liste todos os excluídos e o motivo; não redefina o limiar depois.

**II.27 — Ao ampliar o espaço de ação, o concorrente estático certo pode ser uma mistura, não o melhor braço único.** *Caso:* nos quatro operadores do ILS (efeitos e custos muito diferentes) a mistura aleatória uniforme vence cada operador unico por classe, o oráculo entre os operadores únicos é 0,04% pior que a mistura e o operador único validado (sempre a inter-troca) perde para ela por 0,062%; DQN e LinUCB também perdem (+0,041% e +0,066%) e, com o ILS eficiente, custam 3,3× e 1,6× a CPU do aleatório. A tese "a validação offline vence o online" teve de virar "uma configuração estática (validada ou uma mistura) vence o online" (E176–E177). *Como aplicar:* antes de declarar a validação como o competidor, meça se uma mistura bate cada ação única; se bater, a mistura entra como concorrente estático e o oráculo entre ações únicas deixa de medir folga.

**II.28 — Quando uma conclusão a CPU igual é corrigida, refaça também as figuras e comparações derivadas do mesmo tempo medido.** *Caso:* depois de corrigir o "93%", a Figura Pareto do texto principal e a subseção 'IG versus ILS a CPU igual' do suplemento ainda usavam o ILS em laços (razões até 24,8×); refeitas, o ILS custa 0,34–1,08× o IG e, a CPU igual, ele recebe até 2,2× mais iterações: a vantagem do ILS em tempos finos cresce (+0,21%) e a do IG em tempos grossos diminui pouco (−0,65%) (E174, E178–E179). *Como aplicar:* liste todo artefato (figura, tabela, razão) que depende da medição corrigida e regenere cada um; registre o que mudou.

**II.29 — Um aprendiz pode "ganhar a CPU igual" sem decidir melhor: separe o efeito de custo do efeito de decisão.** *Caso:* com agentes compilados em espaços de 5 a 30 ações, o Q-Learning ganha 0,05–0,06% do aleatório a CPU igual mas nada a iterações iguais; ele aprendeu a evitar as ações caras (0,38–0,50 das avaliações do aleatório), então o aleatório, calibrado para a mesma CPU, ficou com menos iterações; e a ação fixa barata validada por classe ainda o vence por 0,08–0,10% (E180–E181). *Como aplicar:* reporte sempre a CPU igual E as iterações iguais, o custo do agente por unidade de trabalho e a razão de avaliações, e confronte o aprendiz com o competidor estático barato.

**II.30 — Ao derivar um artigo curto de um manuscrito longo auditado, audite a versão curta do zero.** *Caso:* o artigo do CEC herdou números do periódico e a auditoria própria (recalculando do dado) achou nove problemas que a auditoria da versão longa não achou: uma contagem de testes que tinha crescido (188 → 357), uma faixa do resumo arredondada para o lado errado, uma ressalva de exceções perdida (10 de 12 classes), um "22 concluem" que incluía 5 casos parciais, "72 de 96" virando "todas", uma contagem de instâncias que omitia o held-out e um argumento de outro problema (E182). *Como aplicar:* escreva um script de auditoria específico do artigo curto, que recalcule da fonte cada número da prosa e confira que o literal está no texto; números herdados só por presença nas fontes já auditadas.

**II.31 — Um resultado nulo precisa de poder e de controle positivo, e a extrapolação entre famílias de aprendizes falha.** *Caso:* antes de afirmar 'nenhum aprendiz ganha', medi o poder por efeito plantado (o teste de postos com Holm só detecta 0,03 pp com ≥ 97%, bem menos que o MDE analítico de 0,009), plantei um bônus de recompensa (todos os seis aprendizes o capturam) e confirmei que a ação validada é detectada (p ~ 1e-98); e o primeiro AOS clássico novo (roleta tipo ALNS) quebrou 5 de 8 previsões que eu tinha extrapolado do UCB1: ele tem efeito de decisão (0,039 pp) (E184-E187). *Como aplicar:* para cada nulo, relate o MDE empírico por efeito plantado, um controle positivo do aprendiz e do teste, e registre previsões por família de método, não por analogia com outra família; trate o erro de previsão como informação e corrija a frase forte demais.

**II.32 — Quando um resultado contraria a previsão, trate a exploração como hipótese e confirme-a em dados que não participaram dela.** *Caso:* nas instâncias publicadas, 4 de 9 previsões erraram (eu tinha extrapolado das sintéticas, com folga ≤ 0,13%); uma análise exploratória por (família, tercil de n) sugeriu que o ganho do Q-Learning acompanha a folga do oráculo; registrei isso como cinco hipóteses confirmatórias (Holm sobre 5, limiares escritos antes) e doze previsões, congelei os JSON do ajuste e rodei 3.950 instâncias held-out: todas confirmadas, rho 0,65 (E189-E192). *Como aplicar:* (i) declare a origem exploratória da hipótese no pré-registro; (ii) congele o ajuste (hiperparâmetros, ações validadas, K) antes de tocar no conjunto de replicação; (iii) separe a família confirmatória (com limiares) das descritivas; (iv) reescreva a tese em torno do que foi confirmado, inclusive quando ela contraria a conclusão anterior.

**II.33 — Uma vitória do método novo sobre a linha de base tem de sobreviver a controles que igualem os DADOS e a REPRESENTACAO, nao so' o orcamento.** *Caso:* o seletor offline (5.720 instancias de treino) superou a acao validada (525 instancias) por 0,024-0,044 pontos, a iteracoes e a avaliacoes iguais; so' depois de treinar a estatica nas mesmas 5.720 instancias (so' 0,001-0,019 melhor que a validada) e uma tabela de consulta por classe (empata com o seletor onde 95% das classes foram vistas no treino) ficou claro que a maior parte do ganho e' dado e classe, nao adaptacao a instancia (E200-E202). *Como aplicar:* (i) antes de comemorar, liste o que o novo metodo tem a mais (dados, granularidade, custo por decisao) e construa um controle para cada item; (ii) registre os controles como estagio 2 ANTES de rodar (a ordem importa: eles foram escritos depois de ver o estagio 1 e antes de rodar); (iii) reporte a decomposicao (dados + classe + instancia).

**II.34 — Meca o teto antes de pre-registrar uma bateria e defina a populacao de avaliacao em funcao dele.** *Caso:* no flowshop de permutacao, as classes com m = 5 chegam ao limite inferior em todos os metodos (ruido 0, VNR indefinido); na familia F publicada, 5 de 26 celulas sao teto. Sem definir 'celula avaliavel' (VNR definido) antes, a acuracia do pre-teste seria inflada por empates (E197, E199). *Como aplicar:* faca uma sonda curta so' em instancias de ajuste (folga do oraculo por classe e fracao no limite inferior), declare no pre-registro o que foi visto e como isso mudou as previsoes, e conte as celulas excluidas.

**II.35 — Valide um limiar fixado numa familia nunca vista e relate a previsao que falhou, mesmo por pouco.** *Caso:* o pre-teste VNR (tau 0,877 congelado) teve AUC 0,75/0,73 na familia F, mas o Spearman do S15 ficou 0,002 abaixo do limiar registrado (0,348 contra 0,35); a leitura registrada ('generaliza') foi NAO e a 'generalizacao parcial' SIM, e o artigo diz 'preditivo, mas modesto' (E201). *Como aplicar:* escreva as leituras (cheia e parcial) junto com os limiares; nao mude o limiar depois; descreva a diferenca em unidades de efeito, nao so' o veredito.

---

## 7. Mapa de código reutilizável

| Necessidade | Onde |
|---|---|
| Seletores (aleatório, UCB1, UCB1 por estado, Q-Learning, ação fixa) | `pmsp/selectors.py`, `pmsp/qlearning.py`, `pmsp/fixed_selector.py` |
| Thompson e torneio de eliminação (janela curta) | `pmsp/window_bandits.py` |
| IG e ILS para P||Cmax | `pmsp/ig.py`, `pmsp/ils.py`, `pmsp/operators.py` |
| R||Cmax: IG, LP-IG, MILP restrito | `pmsp/unrelated.py`, `pmsp/unrelated_lp.py`, `pmsp/unrelated_srmilp.py`, `pmsp/unrelated_exact.py` |
| Flowshop de permutação (aceleração de Taillard, IG, ILS) | `pmsp/pfsp.py` |
| Limites (clássico e fluxo em arco certificado) | `pmsp/bounds.py`, `pmsp/exact.py` |
| Baselines (LPT, MULTIFIT, LDM, SA, BCO) | `pmsp/constructive.py`, `pmsp/multifit.py`, `pmsp/ldm.py`, `pmsp/sa.py`, `pmsp/bco.py` |
| Estatística | `analysis/stats.py` |
| Padrão de uma bateria | `experiment/run_*_entrada*.py` (execução) + `analysis/report_entrada*.py` (relatório) + `results/entrada*_prereg_hashes.txt` |
| Tabelas e figuras do artigo | `analysis/make_tables_cycle5.py`, `analysis/make_table_synthesis.py`, `analysis/make_fig_headroom.py` |
| Auditorias | `analysis/audit_entrada*_claims.py`, `analysis/audit_short_numbers.py` |
