# Enquadramento das contribuições para o EJOR

**Data:** 04/10/2026 (Entrada 93 do diário). **Status:** análise e recomendação do agente;
a decisão de enquadramento é de Carlos (Lei Geral, I.6). Nada aqui foi aplicado ao manuscrito.

## 1. O que cada contribuição realmente sustenta

| # | Contribuição | Evidência (verificada por execução) | Força | Risco com um revisor do EJOR |
|---|---|---|---|---|
| C1 | Decomposição **mecanismo × política** em learnheuristics, com controle de seleção aleatória a orçamento igual, aplicada a dois mecanismos (ILS, IG) | `ig_random` já vence `ils_random` em 10/12 classes; `ig_ql` não se separa de `ig_random` por classe | Alta (metodologia limpa, replicada) | "Controle aleatório é boa prática padrão, não novidade" — a novidade é a aplicação com teste confirmatório e referências exatas, não a ideia |
| C2 | Efeito do aprendizado: **pequeno, mas real**, confirmado em pré-registro | Holdout de 4.160 instâncias nunca vistas: −3,0% (B), −0,85% (C, marginal), −3,5% (E) do gap; 0,01–0,03 pp absolutos; F não testável | Média (rigorosa, mas efeito minúsculo) | "Quem se importa com 0,02 pp?" — só vale como achado metodológico |
| C3 | Teste confirmatório **pré-registrado** com holdout em OR heurístico | Hashes congelados antes da execução (Entradas 90–91) | Alta e rara no campo | Pode parecer excesso de zelo se for o título; funciona como evidência |
| C4 | `ig_ql` vs. SOTA publicado | Vence SA e BCO em 12/12 classes, também sob orçamento publicado/tempo igual; **perde para o MULTIFIT em todas as classes com n ≥ 75, e o MULTIFIT é ~50× mais rápido** | Média/baixa | "SOTA" = MULTIFIT (1978), SA (preprint), BCO (2012): fraco; SA ≡ LPT em 7/12 classes |
| C5 | Gap ao **ótimo conhecido** (iAF, 9.530 instâncias C+F) | `ig_ql`: 1,14% acima do ótimo em C (21% resolvidas), 0,17% em F | Alta como evidência; é rara em estudos de AOS | Expõe a fraqueza prática (ver §2) |
| C6 | **Variância** (não n/m) decide IG vs. ILS; ILS vence na família B | Fatorial controlado + 4 famílias externas | Média/alta; o achado mais "OR" | Despacho `apriori_select` só foi validado em 95 instâncias |
| C7 | `ils_ql` ≈ seleção aleatória; política "interpretável" é artefato de cobertura de estados | N=30–100 por classe, 8 mecanismos AOS | Média | Resultado nulo com 81 estados; revisor dirá "o desenho de estados é que é fraco" (e os dados sugerem isso: ~3 eventos de recompensa por rodada) |
| C8 | Reimplementação fiel de 3 baselines, 295 testes, dados e código | Verificado | Suporte, não contribuição | — |

## 2. Três achados desta rodada que restringem o enquadramento

1. **Não dá para vender `ig_ql` como solver competitivo de P||Cmax.** O método exato iAF resolve **85,5% das instâncias de C/F em < 1 s** (mediana 1 ms; 93% em < 10 s; tempos de máquinas diferentes, só a ordem de grandeza importa). Nas instâncias **difíceis para o exato** (iAF > 1 s, ~15% da amostra) o `ig_ql` tem gap de 1,6–2,1% e resolve ≤ 3,4% ao ótimo; nas fáceis (< 0,1 s) o gap é 0,39%. Ou seja, a heurística é pior justamente onde o exato precisa de tempo.
2. **O MULTIFIT domina para n ≥ 75** (gap ≈ 0,002–0,24% contra 0,07–0,82% do `ig_ql`; ~50× menos CPU em 300×20). Piloto exploratório (N=30, `analysis/pilot_entrada93_multifit_seeded_ig.py`): iniciar o IG da melhor entre LPT/MULTIFIT nunca perde do MULTIFIT (por construção) e **coincide exatamente com ele em n ≥ 75** — o IG não acrescenta nada sobre uma solução MULTIFIT ali; melhora o MULTIFIT só nas classes pequenas (21/30 em 20×6, 26/30 em 30×8).
3. **O que o aprendizado adiciona é ~0,02 pp.** O agente recebe em média 0,7–4,6 eventos de recompensa não-nula por rodada (`reward = max(delta,0)`, aceitação gulosa) para 81×3 entradas — hipótese plausível para a pouca contribuição, **não testada**.

## 3. Alegações que o manuscrito NÃO deve fazer

- "`ig_ql` é um método competitivo/de estado da arte para P||Cmax" (contradito por 2.1 e 2.2).
- "O Q-Learning é a fonte do ganho" ou, no extremo oposto, "o Q-Learning não contribui" (C2).
- "Generaliza para as 4 famílias" sem qualificar F e C.
- Qualquer valor absoluto de cpu_time de `results.csv` (só razões valem; máquina diferente).
- O título atual ("…Hybridizing Iterated Local Search and Iterated Greedy with Online Q-Learning"): o híbrido é preliminar e a evidência aponta no sentido oposto da ênfase em aprendizado.

## 4. Enquadramentos possíveis

**A — Estudo metodológico controlado (recomendado como espinha).** Pergunta central: *de onde vem o ganho de uma learnheurística?* P||Cmax entra como banco de testes **porque tem ótimos conhecidos** (raro em estudos de AOS). Contribuições, nesta ordem: (i) protocolo de decomposição mecanismo × política + confirmação pré-registrada em instâncias nunca vistas; (ii) resultado: o mecanismo (IG) explica quase tudo; o aprendizado ≈ 1–3% do gap, detectável só agregando e confirmado; (iii) a estrutura da instância (variância) decide qual mecanismo ganha; (iv) gap ao ótimo exato e ressalvas honestas (MULTIFIT em n ≥ 75, exato nas instâncias fáceis); (v) cautela: cobertura de estados e artefato de inicialização em AOS tabular. Título ilustrativo: *"Mechanism or policy? Attributing the gains of Q-Learning-guided Iterated Greedy for parallel machine scheduling"*. Ponto forte: honestidade rigorosa; ponto fraco: contribuição "negativa/diagnóstica".

**B — Método proposto que vence.** Não sustentável pelos dados (§2.1–2.2). Descartar.

**C — Seleção de algoritmo por feature barata / portfólio.** `apriori_select` (despacho IG × ILS por `dup_ratio > 0,1`, limiar fixado desde a Entrada 61) e o semeio MULTIFIT→IG (§2.2) formam uma contribuição **prescritiva** ("qual usar quando"), mais próxima do que o EJOR costuma valorizar. Hoje a evidência é preliminar (95 instâncias; piloto N=30). **Há um teste confirmatório barato e limpo:** rodar `ils_ql` nas 4.160 instâncias do holdout (o limiar de 0,1 é anterior a elas) e avaliar o despacho sem nenhuma retunagem. Custo estimado: ~1–2 h em 8 workers; pré-registro antes.

**Recomendação:** A como espinha dorsal; C como a contribuição positiva **se** o teste confirmatório passar (senão vai para o suplemento). Posição minha, sujeita a você.

## 5. Plano para ≤ 30 páginas (limite do guia do EJOR, verificado em 29/08/2026; hoje ~40 + 3 tabelas novas)

| Bloco | Palavras (aprox.) | Destino |
|---|---|---|
| Introdução, Related Work, Problema, Método (IG) | ~3.000 | Mantém; reduzir Intro (967 palavras) e fundir §3.1–3.4 do ILS num parágrafo |
| 5.1–5.4 (setup, headline, SOTA+sensibilidade, decomposição) | ~2.700 | Núcleo — mantém |
| 5.5 Validação externa (+ holdout + gap ao ótimo) | ~1.100 | Núcleo — mantém |
| 5.6 Estrutura (variância) | ~870 | Mantém condensado |
| 5.7 Híbridos/despacho | ~1.450 | Para o suplemento, **ou** substitui por C se confirmado |
| 5.8 Capítulo `ils_ql` | ~1.500 + 7 floats | 1 parágrafo no corpo + suplemento online |
| Síntese + Discussão + Conclusões | ~4.300 | Fundir; hoje repetem as mesmas alegações 3×; alvo ~1.800 |

Corte estimado: ~35–40% das palavras, suficiente para ~25–28 páginas.

## 6. Risco de enquadramento no periódico

O EJOR avalia contribuição metodológica e relevância para OR; um resultado "pequeno efeito do aprendizado" com solver não competitivo é um risco real de rejeição por **novidade/impacto**, mesmo com rigor. Mitigações: enquadramento A com C; honestidade sobre limites (já é o ponto forte). Contingência (avaliação minha, **não verificada** por bibliometria): periódicos que valorizam rigor experimental e reprodutibilidade, como *Computers & Operations Research*, *INFORMS Journal on Computing* ou *Journal of Heuristics*. Qual seção do EJOR (p.ex. "Discrete Optimization" ou área de inteligência computacional) deve ser confirmado no guia/sistema de submissão antes de escrever o cover letter.

## 7. Decisões que cabem a Carlos
1. Enquadramento: A, A+C, ou outro?
2. Rodar o teste confirmatório do despacho (`ils_ql` no holdout)? (pré-registro + ~1–2 h)
3. Tratar a hipótese da recompensa esparsa (item 5 da lista anterior) como trabalho futuro, ou testá-la antes da submissão?
4. Aprovar o corte para ≤ 30 páginas (mover §5.7–5.8 ao suplemento) e a troca do título?
5. Instalar LaTeX (MiKTeX/tectonic) nesta máquina para compilar e conferir o PDF?


## 8. Atualização (08/10/2026): decisões da seção 7 e novo estado do enquadramento

**Decisões da seção 7 (todas resolvidas):** (1) enquadramento A + C, adotado; (2) o teste confirmatório do despacho foi rodado (Entrada 94: confirmado) e **replicado** em
900 instâncias de 6 distribuições (Entrada 108); (3) a hipótese da recompensa esparsa foi testada e **refutada** (Entrada 104); (4) corte para 30 páginas e novo título, feitos
(o corpo tem 30 páginas incluindo referências; Convergência, o framework ILS e suas variantes e todas as verificações de robustez estão no suplemento);
(5) MiKTeX instalado, `paper/build.sh` compila corpo e suplemento.

**O que o artigo vende hoje (e o que sustenta cada frase):**
1. *Atribuição:* o ganho do `ig_ql` vem do mecanismo destruir-reconstruir, não da política aprendida. Sustentação: controle aleatório pareado em 2 mecanismos,
   política = 0,01-0,05 pp (0,11 com d pequeno), folga do oráculo por instância = 0,037%, recompensa densa pior, pré-treino nulo, estado informativo nulo,
   hiperparâmetros de aprendizado irrelevantes; confirmado em 4.160 instâncias held-out, 6 distribuições, partida forte e médias por seed.
2. *Honestidade sobre o solver:* MULTIFIT/LDM superam o `ig_ql` com partida LPT em n grande e o iAF exato resolve 85% de C/F em < 1 s; o papel prático do IG é partir da
   melhor construtiva (nunca pior, melhora 19-34%, ótimo provado em 69,3%).
3. *Padrão de evidência:* limite inferior certificado, pré-especificação com hashes, equivalência (IC90 em +-0,05%), médias por seed, magnitude antes do p-valor.
4. *Granularidade decide IG vs ILS* (replicado: 6 distribuições, CPU igual) e um despacho de 1 feature confirmado fora da amostra.

**Risco de enquadramento (revisado):** continua sendo novidade/impacto para o EJOR (resultado de "efeito pequeno" com solver não competitivo). A defesa mais forte agora é
o rigor (itens 1-3 acima) e o argumento de que o resultado nulo é *explicado* (folga), não só observado. Plano B inalterado: *Computers & Operations Research*,
*INFORMS Journal on Computing*, *Journal of Heuristics*.

**Plano B para o IEEE CEC (6 páginas, não iniciado):** núcleo = decomposição mecanismo vs política em 2 mecanismos + folga/recompensa/pré-treino (Tabela f14) + IG semeado
(Tabela f12 resumida) + figura Pareto; cortar validação externa detalhada, dispatch e ILS companion para 2 frases; reaproveitar `analysis/make_table_*.py`
com `\scriptsize`. Estimativa: 2-3 h de redação, sem novos experimentos.

**Decisões que cabem a Carlos agora:** ver `SUBMISSAO_CHECKLIST.md`.

---

## 9. Atualização (10/10/2026, Ciclos 4-7): o que o artigo vende agora

O argumento central (a estrutura de busca, não a política, produz o ganho) resistiu a três ciclos adicionais de ataque de revisor, e três expectativas minhas foram **refutadas pelos dados** e estão no texto:
(i) a busca local não melhora a receita **a CPU igual** (só a igual nº de iterações): a variante mais forte do estudo é a receita sem aprendizado **com mais iterações** (89,0% de ótimo provado contra 74,7% a T = 1000 e 48,5% do MULTIFIT);
(ii) no R||Cmax o MILP de 60 s vence as heurísticas até 1000 x 50, então a seção de R é de **atribuição**, não de competitividade;
(iii) a política não é geralmente útil em regime não saturado: captura toda a folga onde uma estratégia domina (U[1,100], -1,73%) e perde da mistura onde a diversidade vale mais (+0,89% a CPU igual em U[100,200]), e a validação offline captura a folga com menor CPU.
O ganho de originalidade para o EJOR vem do rigor (pré-especificação, CPU igual, limite certificado, relato de não-confirmações) e de achados que contrariam a narrativa habitual ("aprendizado vence"); o risco de enquadramento (efeito pequeno, solver não competitivo) permanece.
