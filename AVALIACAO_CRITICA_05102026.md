# Avaliação crítica do trabalho completo — 05/10/2026 (Entrada 97)

**Natureza:** auditoria do manuscrito (`paper/learnheuristic_pmsp.tex`, 29 p.) e do suplemento contra
`results/results.csv`, o código e a literatura. Cada número abaixo foi reproduzido por execução
(`analysis/audit_entrada97_*.py`, `analysis/report_entrada9*.py`) ou verificado em fonte; o que é
hipótese está marcado. **Nada foi alterado no manuscrito nesta auditoria.** Decisões de
enquadramento e de correção são de Carlos (Lei Geral, I.6).

---

## 1. Avaliação geral (como um revisor do EJOR leria hoje)

**O que se sustenta (reproduzido):** F1 — `ig_ql` nunca é batido por LPT/7 ILS nas 12 classes sintéticas
(72/96 Holm-significativas, todas favoráveis); F3 — 0/48 comparações de ablação significativas por classe;
F4 — `ig_random` > `ils_random` em 10/12 classes; os números da Seção 5.5 reproduzem
(B 25/30, C 27/30, E 25/30, F 23/28; ILS 31/15/16/20); determinismo 850/850 makespans; 295 testes passam;
o efeito pequeno do Q-Learning no holdout pré-especificado; o despacho (ganho pooled de 0,013 pp, negativo em C);
o gap ao ótimo (iAF) e a folga do LB.

**Por que ainda seria difícil publicar assim (em ordem de gravidade):**
1. **O espaço de estados está degenerado** (§2, C1): o agente "context-sensitive" enxerga, na prática, só
   progresso e estagnação. Isso mina a conclusão central sobre o papel do aprendizado.
2. **Relevância prática limitada**: o iAF exato resolve 85,5% das instâncias de C/F em < 1 s; o `ig_ql` é pior
   exatamente nas instâncias difíceis para o exato (gap 1,6–2,1%); o MULTIFIT domina em n ≥ 75.
3. **Baselines fracos/subjustificados** e literatura de P||Cmax não usada (§2, C7).
4. **Dois rótulos conceituais errados** ("variância"; "UCB1 é pior") sustentados por artefatos (§2, C4, C5).
5. **Erros factuais de texto** (descrição do BCO, amostra externa, contagens) que um revisor encontra.
Com as correções P0 (§5) o trabalho pode virar um estudo metodológico defensável; o encaixe no EJOR
continua incerto e deve ter plano B (periódicos que valorizam rigor experimental).

---

## 2. O que o trabalho afirma e NÃO se cumpre (ou não está sustentado)

| # | Alegação (onde) | Evidência contrária (reproduzida) | Gravidade | Correção |
|---|---|---|---|---|
| C1 | "Representação compacta de 81 estados, sensível ao contexto" (Abstract, Intro, §4.2, Highlights) | Em execuções reais do `ig_ql` (4 classes × 8 inst.), **f1 (gap) e f2 (CV das cargas) ficam no bin 0 em 100% dos passos**; f3 é só o progresso (uniforme por construção); f4 está no bin 2 em 90–96%. Estados distintos por execução: 5,0–6,5 (máx. 9 por classe). Limiares 0,10/0,35 e 0,15/0,40 são 1–2 ordens de grandeza acima dos valores reais (gap < 1%). O próprio diário já havia medido isso (Entrada 17: mediana de f1 = 0,0076) e **nunca foi usado nem divulgado** | **Crítica** | Recalibrar (quantis ou escala relativa à instância), rerodar, e reportar diagnóstico de visitação; reescrever "context-sensitive" |
| C2 | "Codificação compacta eleva visitas por estado de 1,6 para ~12" (§4.2) | Cobertura real 5–9 estados (IG) e ~21 (ILS); média "12" assume visita uniforme | Alta | Remover/substituir pelo diagnóstico real |
| C3 | "Condicionar por estado não contribui" (ablação 2×2, §5.4, Discussão) | Confundida por C1: `ig_ucb_ctx` e `ig_ql` condicionam em (progresso, estagnação), não no estado da solução | **Crítica** | Reexecutar a ablação com estado recalibrado antes de concluir |
| C4 | "UCB1 consistentemente pior; Q-Learning vence UCB em todas as classes" (§5.9, suplemento; "propriedade estrutural de bandits") | O `UCBSelector` usa recompensa **bruta** (UCB1 exige [0,1]). Piloto exploratório (N=30; 20×6, 50×10, 100×20): UCB com recompensa normalizada melhora significativamente (p=0,0017; 0,0001; 0,001) e **fica indistinguível de `ils_random` e `ils_ql`** (p=0,57; 0,49; 0,26 vs QL). No IG não muda | Alta | Normalizar, rerodar ILS-UCB (N≥100), retratar a alegação (e a "vitória real" da Entrada 13) |
| C5 | "A **variância** dos tempos, não n/m, decide IG vs ILS" (Abstract, §5.7, §5.8, Discussão, Conclusões) | O fatorial compara U[1,100] vs U[1,10⁸]: **CV idêntico (0,559 vs 0,565)**; os métodos são invariantes a escala (makespan(1000p)=1000·makespan(p) em 6/6, exceto UCB bruto); o que muda é a duplicação: `dup_ratio` 0,49 → 0,00. Evidência do texto = "uma seed representativa" (1,24 vs 6,83%), sem N nem teste | Alta | Renomear para granularidade/duplicação (faixa relativa a n); refazer o fatorial com N e testes |
| C6 | "Os três baselines partem do mesmo LPT e rodam sob o mesmo orçamento de iterações" (§4.6) | Falso para o BCO (parte de solução vazia; orçamento por regra de movimentos; docstring de `bco.py`) e para o MULTIFIT | Média/Alta | Corrigir o texto |
| C7 | "Estado da arte publicado" para MULTIFIT (1978), SA, BCO (2012) | SA de Li & Perkins segue preprint (busca de 05/10/2026 não achou versão revisada); SA ≡ LPT em 7/12 classes; BCO depende da regra de orçamento. **Não citados/comparados**: arc-flow melhorado com VNS (Processes 2022, *a verificar*), branch-and-bound de Akram et al. (arXiv:2405.15371, 2024), Dell'Amico et al. (heurísticas/exatos para P‖Cmax) | Alta | Chamar de "baselines publicados"; busca de literatura dirigida; incluir comparador mais forte |
| C8 | "Métodos exatos viáveis só para instâncias pequenas" (§2.3) | iAF: 85,5% de C/F em < 1 s (mediana 1 ms) | Média | Corrigir (a Intro já foi) |
| C9 | "O trabalho foca em n/m ∈ [3,3; 5,0]; LPT com gap 2,0–4,4%" (§3) | O estudo cobre n/m 3,0–15,0; gap do LPT 0,16–6,5% | Média | Atualizar |
| C10 | "Amostra externa de 1.612 instâncias (512 de B…)" (§5.5); Abstract/Highlights "6.780 instâncias… a vantagem replica" | Os números da 5.5 reproduzem **com o manifest de 1.500** (B=400, C=400, E=400, F=300), não com 1.612/512. Os testes formais da família SOTA não foram rodados nas 6.780 | Média | Corrigir a descrição **ou** rodar o relatório nas 6.780 (minutos) e atualizar |
| C11 | "41 de 48 comparações (6 competidores…)"; "o punhado de células não significativas se concentra em (20,4) e (40,5)" (§5.1) | 6 competidores × 12 classes = 72; **52/72** significativas, todas favoráveis. As 24 células não significativas estão em **quatro** classes: (20,4), (40,5), (150,15), (300,20), 6 cada | Média | Corrigir |
| C12 | "ILS-HI usam até T=4.000" (§5.1) | `MAX_ITER_BY_SIZE` chega a 8.000 em quatro classes | Baixa | Corrigir |
| C13 | "`ils_ql` vs `ils_random`: p ≥ 0,075" (§5.9, suplemento) | Mínimo = 0,0745 | Baixa | Escrever p > 0,07 |
| C14 | `d = 0,3 n` justificado por "análise de sensibilidade preliminar não reportada" (§4.5) | Evidência não apresentada; o diário (Entrada 24) registra que `d` não passou por varredura sistemática | Média | Reportar a varredura (apêndice) ou remover a justificativa |
| C15 | "Pré-registrado" (Abstract, Intro, 5.6, 5.8, Discussão) | Congelamento no próprio diário do projeto com hashes; não há registro público/independente | Média | Dizer "pré-especificado (hashes registrados em…)", depositar os arquivos num repositório com DOI; evitar o termo sem registro |
| C16 | "Perturbação double-bridge (3 movimentos aleatórios)" (§4.1) | Não é double-bridge | Baixa | Renomear |
| C17 | "(20,4): o LPT já é quase ótimo" (§5.1) | LPT tem gap médio 1,96% ali; o teto é dos métodos de busca (gap 0%) | Baixa | Reescrever |
| C18 | Highlight 4 / Abstract: "Validado em 6.780 instâncias" | 6.780 = exploratório + gap ao ótimo (2.780); confirmatório = 4.160 novas; SOTA formal = 1.500 | Média | Alinhar com C10 |
| C19 | Seção 2.3: "GA, busca tabu amplamente aplicados ao PMSP" | Sem citação | Baixa | Citar ou remover |

---

## 3. Fragilidades metodológicas (não são "afirmações falsas", mas um revisor aponta)

1. **Desenho do agente nunca variado**: α, γ, cronograma de ε, recompensa (apenas melhoria estrita; ~3 eventos por execução), estados (C1). A conclusão "pouco aprendizado" é sobre *este desenho*, não sobre aprendizado.
2. **Gap vs LB fraco**: folga média do LB 0,72% (C) e 1,01% (F); em n/m=3 só 17% dos melhores atingem o LB. Gaps absolutos nas sintéticas não são sub-otimalidade. Só (20,6) tem ótimo conhecido nas sintéticas (via MIP, piloto).
3. **Uma execução por instância**; variância do algoritmo não separada da variância entre instâncias.
4. **Distribuição de instâncias estreita** (U[1,100] nas sintéticas) e sem instâncias reais; o fatorial só testa dois níveis de faixa.
5. **Orçamento por iterações**, não por tempo; cpu_time de `results.csv` é de outra máquina (só razões valem).
6. **Mesmo holdout em dois testes confirmatórios** (Q-Learning e despacho), sem ajuste entre as duas famílias.
7. **Teste de postos e magnitude**: em amostras grandes com muitos empates (despacho: rank-biserial −0,61 com 0,013 pp) o p-valor engana; a magnitude absoluta deve vir primeiro (já feito no texto novo).
8. **Reprodutibilidade**: sem git, sem LICENSE, `requirements.txt` com `>=` (sem versões fixas), raiz com ~180 backups, 52 pares duplicados no CSV externo, README com "180 instâncias materializadas" (hoje há 12 classes × até 100 geradas por seed), placeholders de repositório/financiamento.
9. **Narrativa**: a Seção 5.7 é relato de processo (tentativas e correções) com evidência de uma seed; deve virar resultado com N, testes e tabela, ou ir ao suplemento.

---

## 4. O que está em aberto

| Item | Estado |
|---|---|
| Efeito do aprendizado com estado calibrado | **Não testado** (C1/C3) — a pergunta central do título |
| UCB normalizado nas 6 classes (N≥100) e no suplemento | Piloto só (N=30, 3 classes) |
| Fatorial granularidade × n/m com N e testes | Não feito |
| Testes formais (Holm) da validação externa nas 6.780 | Não feitos |
| Comparador mais forte para P‖Cmax (+ literatura 2022–2026) | Não feito; literatura apenas sondada |
| Varredura de `d`, α, γ, ε | Não feita/não reportada |
| Hipótese da recompensa esparsa | Não testada |
| Piloto "IG semeado por MULTIFIT" | Exploratório; citado na Discussão sem formalização |
| Título, §5.8–5.9 e Discussão novos | Aguardam revisão de Carlos |
| Seção do EJOR, keyword oficial, CRediT/financiamento/repositório/IA, coautoria | Em aberto |
| Verificação das novas referências (iAF-2022, Akram 2024, Dell'Amico) | Pendente (princípio I.1) |

---

## 5. Recomendações priorizadas

**P0 — corrigem o que está errado ou ameaça a conclusão central (maior valor por custo):**
1. **Recalibrar o estado e reexecutar** (`ig_ql` e `ils_ql` com f1/f2 em escala relativa ou por quantis; 12 classes, N=100; + diagnóstico de visitação). Custo: ~15–30 min de CPU + pré-especificação. Define se o resultado do título muda. *Recomendo pré-especificar antes, como nos testes anteriores.*
2. **UCB normalizado**: rerodar `ils_ucb`/`ig_ucb` normalizados (N≥100), atualizar §5.9 e suplemento, retratar C4. Custo: ~30 min.
3. **Fatorial granularidade**: 3–4 níveis de H, n/m ∈ {3,10,30}, N≥30, testes formais, relatar `dup_ratio`; renomear "variância". Custo: ~1 h.
4. **Testes formais externos nas 6.780** com `report_entrada58` (já existe) e alinhar Abstract/Highlights/§5.5 (C10, C18). Custo: minutos.
5. **Correções de texto sem computação**: C6, C8, C9, C11, C12, C13, C16, C17, C19 e a redação de C15. ~1–2 h.

**P1 — fortalecem a contribuição:**
6. Literatura dirigida de P‖Cmax (2022–2026) + um comparador mais forte, com orçamento de tempo e ótimo do iAF como referência; reformular "SOTA" → "baselines publicados".
7. Varredura pequena de `d`, α, γ, ε (apêndice) e relato do desenho de recompensa.
8. Novas famílias de instâncias discriminantes (n/m ≈ 2–3; log-normal, bimodal) e, se possível, uma instância "real".
9. Depositar código/dados/hashes dos pré-registros em repositório com DOI; rotular "pré-especificado".

**P2 — higiene e submissão:** `git init` + LICENSE + versões fixas; limpar a raiz; remover duplicatas do CSV externo; README; recompilar e conferir ≤ 30 páginas; keyword oficial do EJOR e seção; CRediT/financiamento/IA; coautoria; plano B de periódico.

**Árvore de decisão após P0-1:** se o estado calibrado produzir efeito de aprendizado claro → o artigo vira "o aprendizado ajuda quando o estado é informativo" (mais forte e positivo); se o nulo persistir → a conclusão "aprendizado contribui pouco" passa a ser **sustentada** (hoje não é) e o estudo metodológico fica defensável.

---

## 6. Rastreabilidade
`analysis/audit_entrada97_manuscript_claims.py` (cobertura 23×12 sem células vazias, F1/F3/F4, ILS, MAX_ITER, estados),
`audit_entrada97_ucb_normalisation.py`, `audit_entrada97_scale_invariance.py`,
`audit_entrada97_scale_and_external.py` (CV/dup_ratio, reprodução da 5.5 com o manifest de 1.500),
`audit_entrada90_*` e `report_entrada90/94_*` (rodadas anteriores). Literatura: busca de 05/10/2026 (arXiv:2410.11880
permanece preprint; arXiv:2405.15371; MDPI *Processes* 10(11):2293 — este último **não verificado** (HTTP 403)).

---

## 7. Atualização (05/10/2026, Entradas 98-99): prioridade 0 executada

| Item | Resultado | Efeito sobre o trabalho |
|---|---|---|
| P0-1 estado recalibrado (pré-especificado) | Estados distintos 5–7 → 18 (IG) e 27 (ILS); IG: **nulo** (−3,1% / −3,6%, p_holm 0,28); ILS: **pior que aleatório** (+4,8% / +4,7%, p_holm 0,009) | O estado degenerado era real, mas corrigi-lo **não reverte** o resultado central; a hipótese da recompensa esparsa (≈3 eventos/corrida) passa a ser a explicação principal e continua **não testada** |
| P0-2 UCB normalizado | Normalizado ≫ bruto (−23,7%, p_holm 8×10⁻⁶⁶); `ils_ql` vs UCB norm: p=0,78 | Alegação "UCB1 pior / QL vence UCB" **retratada** (corpo e suplemento) |
| P0-3 fatorial granularidade | rho 0,88/0,67/0,57 (3/3 confirmados); CV idêntico (KW p=0,30) | "Variância" → **granularidade**; n/m desloca o cruzamento e escala a magnitude; regra de `dup_ratio` não pode ser exata |
| P0-4 testes formais externos (6.780) | SOTA 133/150, 121/150, 128/150, 67/78; ablação 10/704; ILS vence em B (157/163) | Seção 5.5 reescrita; descrição de amostra corrigida |
| P0-5 correções de texto | C1–C19 aplicadas (exceto os que dependem de P1) | Manuscrito: 30 p., 0 erros; suplemento 9 p. |

**Ainda aberto:** comparador mais forte + literatura 2022–2026 (C7), varredura de hiperparâmetros, novas famílias de
instâncias, teste confirmatório externo com estado recalibrado, hipótese da recompensa esparsa, depósito com DOI,
git/LICENSE/versões fixas, itens administrativos, revisão de Carlos (título, 5.4–5.9, Discussão).


---

## 8. Atualização (08/10/2026, Entradas 100-113): prioridades 1 e 2 executadas

Tudo abaixo foi pré-especificado no diário (hipóteses, regra de decisão, hashes md5 do código e dos dados lidos) **antes** de rodar; as emendas
(todas feitas antes de ler o resultado do teste afetado) estão no fim de cada `results/entradaNNN_prereg_hashes.txt` e no Suplemento S4.7.
`analysis/audit_entrada113_final_claims.py` recalcula 63 números do texto novo a partir dos CSV (63/63 conferem).

| Item aberto (§4) | Resultado | Efeito sobre o trabalho |
|---|---|---|
| Comparador mais forte + literatura (C7) | LDM (Karmarkar-Karp m-way) implementado e testado; **IG semeado** pela melhor entre LPT/MULTIFIT/LDM; limite inferior **certificado** (relaxação linear do arc-flow, 0 violações em 150 ótimos do iAF, justo em 98,7%); 8 referências novas verificadas (CrossRef/arXiv). 8/8 hipóteses confirmadas (Entrada 102) | O MULTIFIT e o LDM batem o `ig_ql` com partida LPT em n grande (LDM em 21% das instâncias sintéticas e 36% das publicadas); o IG semeado nunca é pior que eles, melhora a melhor construtiva em 19% (sintético) / 34% (publicado) e é ótimo provado em 69,3% (MULTIFIT 48,5%). "SOTA" -> "baselines publicados" |
| Efeito do aprendizado com partida forte | d = -0,010% (S), -0,013% (X): estatisticamente detectável, **negligenciável** (margem 0,05%) | Conclusão central resiste |
| Hipótese da recompensa esparsa / pré-treino (Entrada 104) | Recompensa densa: **pior** que aleatório (+0,084%); pré-treino por classe: equivalente (+0,002%); folga: melhor estratégia fixa = aleatório (0,000%), oráculo por instância só -0,037% | O nulo da política **não** é artefato de esparsidade nem de treino; é falta de folga entre as 3 estratégias (explicação A suportada, B refutada) |
| Varredura de hiperparâmetros (Entrada 105) | α, γ, decaimento de ε: irrelevantes (|d| <= 0,023%); **d (tamanho da destruição) importa**: com d = 0,1 n a política vale até 0,11% e o GAP do `ig_ql` cai 2-4x em 75 <= n <= 150 | Regra pré-registrada "robusto" **não** satisfeita: o texto agora diz "0,01-0,05 pp (até 0,11 com d pequeno)"; d padrão (0,3 n) não foi retunado (seria "tuning on test"); limitação + trabalho futuro |
| Novas famílias de instâncias / região difícil (Entrada 108) | 900 instâncias, 6 distribuições, n/m 2,5-10: H1 (política ~ 0) e 4/4 (despacho e granularidade) **confirmadas**; **MULTIFIT é pior que o LPT** em U[50,100] | Achados de granularidade e despacho generalizam; força do MULTIFIT depende da distribuição |
| Variância entre seeds (Entrada 109) | ICC 0,29-0,84; S1 **não** confirmada: efeito da política em médias-por-seed = -0,045% (IC90 -0,063; -0,027), não equivalente a 0 na margem 0,05%; S2 confirmada (6/6) | A execução única mascarava o efeito; texto qualificado: "pequeno, não nulo" |
| IG vs ILS em CPU igual (Entrada 110) | A vantagem do ILS em granularidade fina **persiste** (d = +0,09%/+0,14%); em dados grossos o IG ganha mais (-0,70%) | Achado de granularidade não é artefato do orçamento por iterações; razões de CPU (mesma sessão): ILS-QL/IG-QL 1,3-24,8x |
| Pareto qualidade-tempo | `paper/fig_pareto.pdf`; IG-QL+ na fronteira nos 3 painéis | Figura 1 do artigo |
| Reprodutibilidade (P2) | `git init` (sem commit), `LICENSE` (MIT/CC BY), `CITATION.cff`, `requirements.txt` fixado, README reescrito, 347 backups em `archive/` | Pronto para depositar |
| Texto administrativo | Financiamento genérico ("sem bolsa específica"), IA expandida (código + análise + texto), disponibilidade de dados com placeholder de DOI, rascunho de carta | Carlos revisa à mão |

**Ainda aberto (decisão ou ação de Carlos):** coautoria/orientador; palavra-chave oficial do EJOR; financiamento real (CAPES/FUNCAP?) se houver;
URL/DOI do repositório; primeiro commit e depósito (Zenodo); revisão do título e do texto novo (5.4, 5.5, Discussão, Conclusões, S4);
teste confirmatório externo com estado recalibrado (não feito; limitação declarada); ajuste de d por validação cruzada (trabalho futuro declarado);
versão CEC (não iniciada).

---

## 9. Atualização (10/10/2026, Entradas 114-136): Ciclos 2 a 7 da regra mestre (plano -> execução -> auditoria como revisor)

Cada ciclo foi pré-especificado no diário com hashes; todas as não-confirmações estão no texto. Quatro auditorias numéricas recalculam os números citados no manuscrito a partir dos CSV
(63/63, 59/59, 21/21 e 87/87). Os documentos `AUDITORIA_REVISOR_CICLO2.md`, `..._CICLO3.md` e `..._CICLO4.md` registram, para cada ciclo, as fragilidades apontadas e o plano seguinte.

| Ciclo | Pergunta | Resultado (honesto) | Efeito no trabalho |
|---|---|---|---|
| 2-3 (Entradas 114-123) | O tamanho de destruição d como espaço de ações; lei do orçamento; R||Cmax; IG ajustado + semeado | Com folga real (d) o Q-Learning NÃO captura nem com T = 5000 ou pré-treino; a validação offline captura; lei do orçamento NÃO confirmada; receita sem aprendizado = 74,7% de ótimo provado (MULTIFIT 48,5%) | Tese reformulada: "estrutura de busca > política"; a receita é o melhor método |
| 4 (Entradas 124-126) | Lei da folga em dados novos; estado x TD no R; híbrido IG+LS no P | Lei confirmada (rho 0,59); em R só estado + TD juntos capturam; no P o ganho vem de destruir-reconstruir (0,59 de 0,72) | R||Cmax replica a decomposição; "mecanismo" -> "estrutura de busca" |
| 5 (Entradas 127-132) | Busca local na receita; R||Cmax onde o MILP não alcança; híbrido a CPU igual | A busca local ajuda a igual nº de iterações (-0,036%/-0,072%) mas **a CPU igual perde** (+0,034%): a receita com T maior chega a 89,0% de ótimo provado; **o MILP-60s vence as heurísticas até 1000 x 50** (GAP 2,04% vs 4,15%/2,78%); no híbrido a CPU igual o ganho da busca local cai de 0,150% a 0,010% | Duas previsões minhas falharam e estão no texto; a decomposição é qualificada pelo orçamento |
| 6-7 (Entradas 133-136) | A política em regime não saturado (1000 x 50, 96 instâncias novas) a CPU igual; validação offline vs política | Q1/Q2 **não confirmadas** (-0,35%, -0,19%); post hoc: -1,73% em U[1,100] (um braço domina; captura total), PIOR que a mistura em U[100,120] e U[100,200] (+0,89% a CPU igual); validação offline captura a folga (-0,46%, 17/0) a menor CPU; vs Q-Learning inconclusivo | A lei da folga ganha um caso de captura completa; "a política vale onde a busca não satura" NÃO é regra geral |

**Ainda aberto (só Carlos):** coautoria, palavra-chave oficial, financiamento, URL/DOI e primeiro commit/depósito, pré-registro público, permissão para baixar o benchmark padrão de R||Cmax, revisão do texto novo, versão CEC.
**Fragilidades remanescentes declaradas:** agente tabular, heurísticas de R||Cmax não competitivas com o MILP e instâncias sintéticas, pré-registro não público, densidade do manuscrito (40 páginas).

**Atualização (10/10/2026, Entradas 137-138): benchmark padrão de R||Cmax.** Com a autorização de Carlos, baixei o benchmark de Fanjul-Peyro e Ruiz (1.400 instâncias; cópia MIT do repositório Fraunhofer-Chalmers-Centre/RCmax, com os melhores valores conhecidos de CPLEX, Gurobi e LR) e rodei uma validação externa pré-especificada em 700 instâncias de teste (280 de ajuste): B1 mecanismo a CPU igual **-2,14%** (confirmada); B2a/B2b política **-0,055%/-0,033%** (não confirmadas); B3 lei da folga **rho = 0,74** (confirmada); B4 a validação offline vence o Q-Learning por **0,066%** (confirmada). O IG+LS-QL fica **2,08% acima do melhor conhecido** (CPLEX/Gurobi/LR: <= 0,18%), então a ressalva "heurísticas de R não competitivas" está agora medida em dados de terceiros. A fragilidade "R||Cmax só em instâncias sintéticas" (H4 da auditoria do Ciclo 4) está resolvida.

**Atualização (11/10/2026, Entradas 139-147): Ciclo 9 -- resultados melhores, dentro do protocolo.** (i) Receita sem aprendizado em P||Cmax com mais iterações: **97,5% de ótimos provados em T = 64.000** (75,5% em T = 1.000; MULTIFIT 48,5%); (ii) R||Cmax: **IG partindo da relaxação linear (LP-IG)**: -1,18% a CPU igual, GAP ao melhor conhecido 2,23% -> 0,97% (0,31% em n = 1000; 0,85% com T = 5000), critério de competitividade pré-fixado cumprido, 3,8% de soluções estritamente melhores que CPLEX/Gurobi/LR; contra o MILP-60s em 1000 x 50: GAP 1,09% vs 2,04%, igual ou melhor em 62,5% das instâncias, mas **não confirmado** no teste pareado agregado (p_holm = 0,72); (iii) **a lei da folga FALHOU prospectivamente** para o tamanho de destruição (Q-Learning online +0,15% pior que sortear; pré-treinado e online perdem para a validação; rho = -0,37): folga é necessária, não suficiente; diagnóstico exploratório: >90% da recompensa chega nas primeiras 100 de 1.000 iterações. Auditorias 175/175.

**Atualização (11/10/2026, Entradas 148-150): matheurística para R||Cmax.** O LP-IG seguido de um MILP restrito às 6 máquinas mais baratas por tarefa (30 s, HiGHS) leva o GAP médio ao melhor conhecido do benchmark padrão de **2,23% (IG original) para 0,25%** (420 instâncias reservadas; S1-S3 confirmadas: -0,49% a CPU ~ igual a T = 8000; iguala ou bate o melhor conhecido em 50% das instâncias e o supera em 4,8%; CPLEX 2 h: 0,20%; Gurobi/LR 300 s: ~0,1%). Critério de competitividade pré-fixado cumprido no total e por n, **não** por tipo (jobs correlacionados 0,67%). Auditorias 194/194; 333 testes.

**Atualização (11/10/2026, Entradas 151-154): reposicionamento e terceiro problema.** Implementado o flowshop de permutação (`pmsp/pfsp.py`, só numpy) e testado em 120 instâncias estilo Taillard: **o ingrediente ativo é a busca local** (IG = ILS a CPU igual, P1 -0,03% n.s.; busca local a CPU igual -0,21%, p_holm = 1e-6), **o Q-Learning sobre o tamanho de destruição é pior que sortear** (+0,14%) e **a validação offline vence o online** (-0,16%). Síntese transversal (Tabela `f35_synthesis`, 7 ambientes): o Q-Learning sobre estratégias nunca fica pior que sortear e captura 0,44-0,68 do ganho validado; sobre tamanhos fica sempre pior (+0,015; +0,152; +0,142%) -- hipótese pós hoc. `PLANO_CORTE_30_PAGINAS.md`: tese em uma frase, quatro contribuições, títulos candidatos e alocação de páginas por seção (46 -> ~30). Auditorias 214/214; 338 testes.
