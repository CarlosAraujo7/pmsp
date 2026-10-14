# Task #44 — Reescrita completa do artigo em inglês: resumo final

**Data:** 08/09/2026
**Status:** completa — Fases A a F executadas, verificadas e já deployadas no seu projeto (`paper/learnheuristic_pmsp.tex` e `.pdf`, 54 páginas). Nenhuma pausa de revisão foi feita no meio do caminho, conforme sua instrução ("pode ir adicionando bloco por bloco... só tenha em mente de colocar tudo que foi feito, tudo mesmo").

Este documento é o resumo de tudo. Os detalhes completos, bloco por bloco, estão no `diario_pesquisa.md` (Entradas 78 a 84) e no `ANALISE_E_PLANO_EJOR.md`, se você quiser conferir algo específico depois.

---

## Precisa de uma decisão sua: possível referência fabricada

Antes de qualquer outra coisa, porque isso é diferente de uma escolha editorial normal: encontrei uma entrada em `refs.bib` (`DeneuxEtal2021`) que muito provavelmente **não corresponde a um artigo real**.

- A entrada diz: Thierry Denoeux, "A survey of machine learning techniques applied to combinatorial optimization", EJOR 292(2):405-422, 2021.
- Busquei na web por esse título exato e por esses dados (volume/páginas) e não encontrei nada. O campo de pesquisa real do Thierry Denoeux é teoria de evidência/funções de crença — não otimização combinatória.
- Os dados bibliográficos alegados (EJOR 292(2):405-422) são muito parecidos com os de um artigo **diferente e genuíno**, que já está citado corretamente no seu artigo sob outra chave: Bengio, Lodi e Prouvost, "Machine learning for combinatorial optimization: a methodological tour d'horizon", EJOR 290(2):405-421, 2021 (`BengioEtal2021`).

Minha hipótese é que essa entrada foi criada em algum momento anterior do projeto com os metadados confundidos ou inventados, possivelmente por associação errada com o artigo do Bengio et al.

**Eu não citei `DeneuxEtal2021` em lugar nenhum do artigo** (para não colocar uma citação fabricada no texto) **e não apaguei a entrada de `refs.bib`** por conta própria, porque isso é uma questão de integridade de dados que é sua para decidir, não minha. As opções que vejo:

1. Apagar a entrada de `refs.bib` por ser inválida.
2. Tentar descobrir se existe um artigo real que ela deveria representar, e corrigir os metadados.
3. Deixar como está por enquanto (sem citar, sem apagar) até você verificar com mais calma.

Não fiz nada além de investigar e sinalizar.

---

## O que mudou, fase por fase

O artigo inteiro foi traduzido para o inglês (era 100% português) e, no processo, expandido para refletir o estado atual e real do projeto — não só uma tradução, mas uma atualização de conteúdo em vários pontos onde o texto em português estava desatualizado.

**Fase A-B (estrutura e abertura).** Título, resumo, Introdução e Related Work traduzidos e atualizados.

**Fase C.** Seção 3 (Definição do Problema) traduzida. Seção 4 renomeada "Proposed Learnheuristics" e expandida com 2 subseções novas descrevendo o mecanismo completo do `ig_ql` (as 3 estratégias de destruição, reconstrução gulosa, etc.) e toda a escada de ablação, tudo verificado direto no código-fonte. Corrigidos 2 bugs factuais que estavam no texto antigo (cronograma de epsilon errado; um `\ref` que resolvia para o número errado).

**Fase D — a maior mudança de conteúdo.** Seção 5 (Experimentos) totalmente reescrita e expandida de 1 capítulo (o antigo capítulo `ils_ql`, com 8 métodos/6 classes) para 10 subseções cobrindo o achado real e atual do projeto: `ig_ql` como resultado principal, validado contra a família ILS (72/96 comparações significativas), validado contra os 3 baselines SOTA publicados (35/36, com o cruzamento de escala com o MULTIFIT explicado estruturalmente), decomposto via ablação em mecanismo (efeito real) vs. política de Q-Learning (efeito nulo), validado externamente em 1.612 instâncias publicadas, mais os limites estruturais de n/m do ILS e a exploração híbrida preliminar. O capítulo `ils_ql` original virou um resultado motivador, preservado e reenquadrado. Encontrei e corrigi 3 erros meus nessa fase antes do deploy, incluindo uma alegação factual errada ("36 de 36" vs. SOTA — o correto é 35/36, com um empate genuíno).

**Fase E.** Seções 6 (Discussão) e 7 (Conclusões) totalmente reescritas, reordenadas para `ig_ql` vs. SOTA vir primeiro como manchete (a decomposição mecanismo-vs-política como segunda contribuição, e os achados do capítulo `ils_ql` como contribuição secundária — nada foi descartado). A Discussão inclui um parágrafo respondendo de frente a pergunta que um revisor provavelmente vai fazer ("por que manter `ig_ql`, não `ig_random`, como manchete, se a política não ajuda estatisticamente?"), com seus 3 motivos e a ressalva técnica honesta. Lista de trabalho futuro expandida de 4 para 6 itens.

**Fase F (a mais recente).** Revisão final de consistência do documento inteiro, agora 100% em inglês:
- Fechei 10 lacunas de citação que existiam desde fases anteriores (chaves preparadas em `refs.bib` mas nunca usadas no texto): Random Forest, NumPy, pandas, SciPy, teste de Wilcoxon, correção de Holm-Bonferroni, o guia de testes não-paramétricos do Derrac et al., o effect size rank-biserial, e os testes de Levene e Fligner-Killeen. Cada uma verificada contra o código-fonte ou a literatura antes de citar.
- O achado do `DeneuxEtal2021` (acima).
- Corrigi um bug real de parênteses duplicados numa frase da Seção 5.2 (renderizava "))" visível na página 16).
- Verifiquei a numeração de seções/tabelas/figuras do documento inteiro (11 tabelas, 3 figuras, tudo sequencial e sem números "chumbados" no texto fora de referências automáticas — risco zero de desatualizar).
- Varredura final por português residual: zero ocorrências em texto visível ao leitor.

---

## Verificação e testes

Em cada fase: recompilação completa (4 passadas, xelatex + bibtex) tanto no sandbox quanto no seu computador, inspeção visual das páginas tocadas, comparação byte-a-byte (`diff`) confirmando que só as seções pretendidas mudaram, e a suíte de 295 testes automatizados rodando depois de cada deploy. Estado final: 0 erros de compilação, 0 referências não resolvidas, apenas 3 avisos benignos e pré-existentes (nada relacionado a esta reescrita), 295/295 testes passando.

O PDF final (54 páginas) está anexado a esta mensagem, e o `.tex`/`.pdf` já estão salvos em `paper/` no seu projeto.

---

## Sugestão de próximos passos

1. Decidir sobre `DeneuxEtal2021` (acima) — é o único ponto que realmente precisa de você, o resto é revisão normal.
2. Quando tiver cabeça, ler o PDF completo de uma vez (é a sugestão que você mesmo deu). Se quiser o histórico detalhado de decisões e raciocínio por trás de qualquer trecho específico, as Entradas 78-84 do `diario_pesquisa.md` têm o passo a passo completo.
3. Depois da sua revisão, útil já pensar em: declaração de contribuições CRediT, coautoria (se houver), e o enquadramento final de submissão pro EJOR.
