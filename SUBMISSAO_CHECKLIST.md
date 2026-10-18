# Checklist de submissão (EJOR) — o que falta e é seu (atualizado em 10/10/2026, após os Ciclos 4-7)

Tudo o que dependia de experimento, texto técnico ou higiene do repositório foi feito (ver `diario_pesquisa.md`, Entradas 100-136, as auditorias de revisor `AUDITORIA_REVISOR_CICLO2.md`, `..._CICLO3.md`, `..._CICLO4.md` e
`AVALIACAO_CRITICA_05102026.md` §8-9). O que resta exige decisão ou dado que só você tem. Os campos marcados `[Insert ...]` no `.tex` são exatamente estes.

## 1. Decisões / dados seus
| # | Item | Onde | Observação |
|---|---|---|---|
| 1 | **Coautoria** (orientador/colaboradores) | `\author{}` em `paper/learnheuristic_pmsp.tex` (comentário `TODO(Carlos)`); CRediT; carta | O artigo está com autor único; ajuste CRediT (Supervision, etc.) se entrar coautoria |
| 2 | **Palavra-chave oficial do EJOR** | `\begin{keyword}` | As 5 atuais são rascunho; o portal exige uma da lista oficial |
| 3 | **Financiamento real** | `\section*{Funding}` | Hoje: "sem bolsa específica" (genérico). Se há bolsa CAPES/FUNCAP/CNPq, troque |
| 4 | **URL/DOI do repositório** | `\section*{Data and Code Availability}`, `CITATION.cff`, README | Depositar (Zenodo/GitHub) e inserir; o git já foi iniciado (`git init`), **nenhum commit foi feito** |
| 5 | **Declaração de IA** | `\section*{Declaration of Generative AI ...}` | Texto expandido (código, análise, redação); leia e ajuste — é declaração factual sua |
| 6 | **Carta de apresentação** | `paper/cover_letter_draft.md` | Rascunho em inglês (já cita 75%/89% e os dois resultados negativos); preencher seção do EJOR, revisores sugeridos, confirmações de originalidade |
| 7 | **Revisão do texto novo** | 5.4 (baselines/exato), 5.5-5.8 (por que a política rende pouco, híbrido, receita, R||Cmax), Discussão, Conclusões, Suplemento S4 | Título e subseções 5.4-5.9 ainda aguardam sua leitura; o texto mudou bastante nos Ciclos 4-7 |
| 8 | **Pré-registro público** | Zenodo/OSF | O diário com hashes **não** é registro público: depositar `diario_pesquisa.md` + `results/entrada*_prereg_hashes.txt` no momento da submissão e citar o DOI |
| 9 | ~~Benchmark padrão de R||Cmax~~ | **feito** (você autorizou em 10/10) | Baixado de `github.com/Fraunhofer-Chalmers-Centre/RCmax` (MIT); validação externa nas Entradas 137-138; dados fora do git (`external_benchmarks/rcmax_fanjul_ruiz/README.md`); confira a citação (Åblad et al., 2021) |

## 2. Antes de submeter
- **ATUALIZAÇÃO 14/10/2026:** versão curta montada (corpo 24 páginas, suplemento 43, versão longa de 46 páginas arquivada como relatório técnico); periódico recomendado: Journal of Heuristics (ver `DECISAO_SUBMISSAO.md`); conferir as 8 referências novas de `paper/refs.bib` (E158) e reformatar para o gabarito do periódico. Os itens de página abaixo estão desatualizados.
- [ ] Rodar `bash paper/build.sh` (corpo: 25 páginas; suplemento: 44; sem referências indefinidas) e conferir o PDF final. **18/10/2026 (Ciclo 12): Taillard clássico feito; auditoria de literatura feita (peça a dupla codificação de 8 artigos, `results/entrada159_litaudit.txt`); depósito Zenodo: `DEPOSITO_ZENODO_PASSO_A_PASSO.md`.**
- [ ] `python -m pytest tests` (338 testes) e as quatro auditorias numéricas (`audit_entrada113_final_claims` 63/63, `audit_entrada126_final_claims` 59/59, `audit_entrada127_cycle4_claims` 21/21, `audit_entrada133_cycle5_claims` 225/225).
- [ ] Primeiro commit e depósito com DOI; atualizar placeholders.
- [ ] Suplemento (17 páginas) como arquivo adicional; conferir referências cruzadas (`xr`) compilando corpo → suplemento → corpo.
- [ ] Escolher a seção do EJOR no portal (ver o Guide for Authors vigente; não verificado aqui).
- [ ] Conferir se o EJOR aceita 40 páginas (o teto de 30 foi suspenso por você em 09/10; hoje 46 páginas; se houver limite no portal, mover seções 5.x para o suplemento).

## 3. Limitações declaradas no texto (um revisor pode insistir)
1. O tamanho de destruição padrão d = 0,3 n **não foi ajustado** nas comparações principais; a receita com d validado em instâncias separadas é reportada à parte (Seção 5.8).
2. Agente **tabular** apenas, com uma recompensa densa testada; ações pequenas.
3. As conclusões sobre componentes adicionais (busca local) dependem da convenção de orçamento: a igual número de iterações a busca local ajuda, a CPU igual não (Entradas 128 e 131).
4. R||Cmax: heurísticas não competitivas com o MILP-60s até 1000 x 50 nem com CPLEX/Gurobi/LR no benchmark padrão (2,1% vs <= 0,2% do melhor conhecido); a maior parte da seção usa instâncias sintéticas; ganho da política instável entre amostras e distribuições.
5. Execução única por instância nas baterias principais; médias por seed em 12 classes.
6. Pré-especificação = diário datado com hashes, **não** registro público.
7. Famílias externas: o efeito do Q-Learning é negativo (não confirmável) em F.

## 4. Plano B
- *Computers & Operations Research*, *INFORMS Journal on Computing*, *Journal of Heuristics* (valorizam rigor experimental/reprodutibilidade).
- **IEEE CEC** (6 páginas): ver `ENQUADRAMENTO_EJOR.md` §8; sem novos experimentos.
