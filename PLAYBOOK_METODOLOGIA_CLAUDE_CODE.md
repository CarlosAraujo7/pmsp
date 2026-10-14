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
