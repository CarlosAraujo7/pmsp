# Plano de Expansão Experimental — Rumo a um Estudo EJOR-Caliber

**Data:** 30/08/2026 (revisado no mesmo dia após retorno de Carlos)
**Motivação:** Carlos indicou que o trabalho feito até aqui (Fases 0-3) é bem menor do que o pretendido. O objetivo declarado é uma "repaginação inteira" com pesquisa extensa e sólida o suficiente para o EJOR, expandindo simultaneamente: (1) mais métodos e ablations, (2) escala muito maior, (3) rigor estatístico avançado, (4) comparação com a literatura.

**Revisão desta versão:** a v1 propunha um GA genérico como método de comparação com a literatura. Carlos questionou diretamente — com razão — a falta de uma referência específica e recente para o problema exato (P||Cmax puro). Pesquisa real conduzida em resposta; o GA foi substituído por `ig_ql` (Seção 1), com citação precisa. Detalhes completos do processo em `diario_pesquisa.md`, Entradas 10-12.

**Este documento é uma PROPOSTA, não uma execução.** Nada abaixo foi rodado ainda. Ele existe para Carlos aprovar, cortar ou ajustar antes de eu comprometer horas de processamento ou escrever qualquer código novo. Os números de tempo são estimativas com base em dados reais já coletados nesta sessão, não em suposições.

---

## 1. Métodos novos

Todo método abaixo tem uma referência específica por trás — nenhum é uma escolha genérica. Achado relevante desta rodada de pesquisa: Ostojić et al. (2025), revisão sistemática sobre otimização para P||Cmax (*Symmetry*, 17(2):178, cobrindo 1959–2024), conclui que **não existe um padrão estabelecido de como comparar algoritmos para este problema exato** — ou seja, a responsabilidade de justificar cada escolha de método é nossa, não uma convenção herdada da área.

| Método | Categoria | O que faz | Por que importa / referência |
|---|---|---|---|
| `ils_pm` | AOS clássico (Probability Matching) | Mantém qualidade estimada $q_i$ por operador (média com decaimento); probabilidade de seleção $p_i = p_{min} + (1-4p_{min})\frac{q_i}{\sum q_j}$ | Fialho et al. (2010) — um dos 3 mecanismos clássicos de AOS da literatura já citada; falta no desenho atual, que só tem UCB como bandido clássico |
| `ils_ap` | AOS clássico (Adaptive Pursuit) | Variante de PM que "persegue" mais agressivamente o operador de maior $q_i$ estimado | Fialho et al. (2010) — completa o trio clássico PM/AP/UCB |
| `ils_sarsa` | RL on-policy | Mesmo espaço de estados/ações do Q-Learning atual; só a regra de atualização muda para on-policy ($Q(s,a) \mathrel{+}= \alpha[r+\gamma Q(s',a')-Q(s,a)]$, usando a ação real seguinte, não o máximo) | Contraste direto on-policy/off-policy com o `ils_ql` já existente, mesmo desenho de estado — fundamento clássico de RL (Rummery \& Niranjan, 1994; Sutton \& Barto). Pergunta científica limpa: a natureza on-policy vs. off-policy importa para AOS, ou só "ter contexto" já importa? |
| `ils_op2_only` | Ablation de isolamento | Só usa Op2 (reinserção) | Completa a bateria de isolamento — hoje só existe Op1-only |
| `ils_op3_only` | Ablation de isolamento | Só usa Op3 (troca intramáquina) | Op3 nunca melhora o makespan por design — esperado ficar preso no LPT inicial; é o "controle negativo" que falta |
| `ils_op4_only` | Ablation de isolamento | Só usa Op4 (block-move) | Completa a bateria; testa se block-move sozinho compete com Op1 |
| `ig_ql` **(novo, substitui o GA da v1)** | Framework alternativo + AOS | Iterated Greedy (destruição de um subconjunto de tarefas + reconstrução gulosa, estilo NEH) usando o **mesmo** agente de Q-Learning tabular de 81 estados, agora selecionando entre estratégias de destruição em vez dos 4 operadores de vizinhança do ILS | Adaptado de **Missaoui \& O'Sullivan (2026)** — "Q-learning enhanced Iterated Greedy" para escalonamento de máquinas paralelas (não flowshop) — o trabalho mais próximo já encontrado em toda a pesquisa, publicado em 2026. Difere do original em framework de busca (eles usam IG puro; aqui é o mesmo IG) e principalmente em objetivo (eles otimizam atraso/adiantamento ponderado com setup dependente de sequência; aqui adaptamos para makespan puro, sem setup) |

**Por que `ig_ql` no lugar do GA — e por que é *mais barato* de implementar, não mais caro:** ao contrário de um GA (que exigiria codificação por população, cruzamento, mutação e seleção — um paradigma de busca populacional inteiro, novo em relação a tudo que já existe no projeto), o Iterated Greedy é estruturalmente parecido com o ILS já implementado: destruir/perturbar → reconstruir/buscar localmente → aceitar. O `ig_ql` reaproveita o agente de Q-Learning, a codificação de 81 estados e o sinal de recompensa exatamente como estão — a única peça genuinamente nova é (a) um operador de destruição (remover um subconjunto de $d$ tarefas do cronograma, com 2-3 variantes possíveis: remoção aleatória, remoção da tarefa mais cara, remoção concentrada em uma máquina) e (b) uma reconstrução gulosa (reinserir cada tarefa removida, uma de cada vez, na posição/máquina que menos aumenta o makespan). A interface `BaseSelector`/`make_selector` permanece intacta — o Q-Learning passa a escolher entre variantes de destruição, não entre operadores de vizinhança. Estimativa de esforço: comparável aos métodos-núcleo, não "uma ordem de magnitude a mais" como o GA seria.

`ig_ql` serve dois papéis ao mesmo tempo: (a) pergunta científica direta e nova — o framework de busca local (ILS vs. Iterated Greedy) importa, mantendo o mesmo mecanismo de RL/AOS fixo? — e (b) a comparação-com-a-literatura do plano original, agora com citação específica, precisa e recente, no lugar do GA genérico.

**Pendente de confirmação de Carlos:** o desenho técnico acima (variantes de destruição, reconstrução gulosa estilo NEH) antes de eu escrever qualquer código — ver Seção 6.

Implementação: os 6 primeiros métodos reaproveitam `AlwaysOp1Selector` como modelo (~15 linhas cada, mesma interface de `BaseSelector`, mesmos testes de sanidade de `test_phase2.py`/`test_phase3.py` como referência). O `ig_ql` precisa de um módulo de destruição/reconstrução novo (bem delimitado), mas reaproveita o resto do harness — agente, estado, recompensa, loop de treinamento.

---

## 2. Classes de instância novas

Mantendo as 6 atuais (para não perder os dados já coletados) e adicionando 6 novas, variando tamanho **e** razão $n/m$ (não só tamanho, para testar generalização estrutural, não só escala):

| Classe | $n$ | $m$ | $n/m$ | Motivação |
|---|---|---|---|---|
| *(6 atuais, mantidas)* | 20–100 | 4–20 | 3,3–5,0 | já coletado |
| `rand_40j_5m` | 40 | 5 | 8,0 | razão alta, porte pequeno-médio |
| `rand_60j_20m` | 60 | 20 | 3,0 | razão baixa (muitas máquinas), porte médio |
| `rand_150j_15m` | 150 | 15 | 10,0 | escala maior, razão alta |
| `rand_150j_30m` | 150 | 30 | 5,0 | escala maior, razão igual às classes médias atuais (continuidade) |
| `rand_250j_25m` | 250 | 25 | 10,0 | teste de escalabilidade |
| `rand_300j_20m` | 300 | 20 | 15,0 | extremo — poucas máquinas relativo ao volume de tarefas |

Réplicas: **50 por classe nova** (mais poder estatístico que as 30 atuais, alinhado ao pedido de rigor); as 6 classes atuais continuam em 30 para os métodos já rodados, mas passam a 50 também para os métodos novos (ver Seção 4).

---

## 3. Protocolo estatístico avançado

| Item | O quê | Onde entra |
|---|---|---|
| Teste de Friedman | Omnibus não-paramétrico comparando todos os métodos simultaneamente, por classe e agregado | Substitui comparações par-a-par soltas como primeira linha de defesa |
| Post-hoc | Nemenyi (all-pairs) **ou** Wilcoxon pareado com correção Holm-Bonferroni nos pares cientificamente relevantes (QL vs. cada AOS clássico; isolamentos entre si; SARSA vs. QL; `ig_ql` vs. `ils_ql`) | Depois do Friedman rejeitar H0 |
| Effect size | Correlação rank-biserial para cada Wilcoxon | Quantifica *quanto* importa, não só *se* importa |
| Intervalos de confiança | Bootstrap percentil (10.000 reamostragens) do GAP% médio por método×classe | Substitui só reportar média±desvio |
| Sensibilidade a hiperparâmetros do Q-Learning | Grade $\alpha \in \{0{,}05, 0{,}1, 0{,}2\} \times \gamma \in \{0{,}8, 0{,}9, 0{,}95\} \times$ decaimento de $\epsilon \in \{$lento, médio, rápido$\}$ = 27 combinações, nas 6 classes originais, 10 réplicas cada | Mostra que o resultado não depende de uma escolha de hiperparâmetro "com sorte" |
| Sensibilidade à granularidade do espaço de estados | Comparar 81 estados ($3^4$, atual) contra 16 ($2^4$) e 256 ($4^4$) — mesmo `ils_ql`, 6 classes originais, 30 réplicas | Valida a própria escolha de design já documentada no artigo (que menciona ter testado 625 estados antes) |

**Limitação técnica a resolver:** este ambiente de execução (VM usada pelo `device_bash`) não tem `scipy`/`scikit-learn`. Os testes de Wilcoxon/Friedman/bootstrap preciso implementar em Python puro aqui, **ou** Carlos roda essa etapa na própria máquina (que tem `scipy`, confirmado pelo `requirements.txt`). Proponho: eu implemento e rodo as versões em Python puro para ter os números já em mãos, e Carlos confirma/re-roda com `scipy` como verificação cruzada — mesmo padrão de dupla-checagem já usado com o pytest.

---

## 4. Escala total consolidada

| Bloco | Cálculo | Execuções novas |
|---|---|---|
| Métodos novos (7, incl. `ig_ql`) nas 6 classes atuais | $7 \times 6 \times 30$ | 1.260 |
| Todos os métodos (16, incl. `ig_ql`) nas 6 classes novas | $16 \times 6 \times 50$ | 4.800 |
| Sensibilidade a hiperparâmetros QL | $27 \times 6 \times 10$ | 1.620 |
| Sensibilidade à granularidade de estado | $2 \times 6 \times 30$ | 360 |
| **Total geral** | | **8.040 execuções novas** |

(Já existem 1.620 execuções da grade original — não recontadas aqui. Total ligeiramente maior que a v1 — 8.040 vs. 7.920 — porque `ig_ql` agora recebe o mesmo tratamento pleno dos demais métodos, em vez do escopo reduzido que o GA teria.)

### Estimativa de tempo (com base em tempos reais já medidos nesta sessão, não suposição)

Tempo por execução medido: ~0,09–0,19s (classes 20–30j) · ~0,3–0,4s (50j) · ~0,6–0,75s (75j) · ~0,9–1,7s (100j, padrão) · até ~5,8s (100j, orçamento estendido "hi"). Para as classes novas grandes (150–300j) não medidas ainda, a extrapolação da tendência aponta para ~2–12s por execução no orçamento padrão, podendo passar de 15–40s nas variantes de orçamento estendido. `ig_ql` deve ter custo por execução da mesma ordem de grandeza do `ils_ql` (mesmo número de iterações, reconstrução gulosa é O(n·m) por remoção, comparável ao custo de avaliação já existente) — não medido ainda, será confirmado no primeiro lote real.

| Bloco | Estimativa |
|---|---|
| Classes rápidas/médias (até 100j) | ~45–60 min |
| Classes grandes novas (150–300j) — domina o custo total | **~4–5 horas** |
| Sensibilidades (hiperparâmetros + granularidade) | ~20–25 min |
| **Total aproximado** | **~5–6 horas de processamento real** (essencialmente igual à v1 — a mudança GA→`ig_ql` não muda a ordem de grandeza do tempo total) |

Isso é executado em chamadas de até ~170s cada — ou seja, algo entre 100 e 150 chamadas sequenciais só para a fatia grande. É um volume real de tempo decorrido, não só de "trabalho".

---

## 5. Opções de escopo — decisão de Carlos

| Opção | O que muda | Impacto no tempo |
|---|---|---|
| **A — Escopo completo** ✅ (escolhida por Carlos) | Tudo acima, como projetado | ~5-6h, em várias ondas ao longo da sessão |
| **B — Cortar o nível "muito grande"** | Remove 250j e 300j, mantém 150j como teto (ainda 50% maior que o atual) | Cai para ~2-2,5h |
| **C — Reduzir réplicas novas de 50→30** | Uniformiza com as classes atuais | Corta ~40% do bloco principal |
| **D — Dividir a execução** | Eu rodo aqui as classes rápidas/médias; Carlos roda as classes grandes na própria máquina (mais rápida, sem limite de 170s por chamada, mesmo `experiment/runner.py` já testado) em paralelo | Tempo real decorrido cai bastante; eu sigo com o resto (métodos, estatística, escrita) enquanto isso roda |

Carlos escolheu explicitamente a Opção A (escopo completo). B, C e D continuam disponíveis como ajustes possíveis a qualquer momento, inclusive já em andamento, caso o volume real (~100-150 chamadas sequenciais) se mostre impraticável.

---

## 6. Estado das decisões e o que falta

**Já decidido por Carlos:**
- Escopo de execução: completo (Opção A da Seção 5).
- Método de comparação com a literatura: `ig_ql` (Iterated Greedy + Q-Learning, citação Missaoui \& O'Sullivan 2026) no lugar do GA — tratado como método de pleno direito, não mais um item à parte de esforço maior. Sem baseline externo adicional além dele.

**Ainda preciso da confirmação de Carlos antes de escrever qualquer código:**
1. O desenho técnico do `ig_ql` (Seção 1: variantes de destruição, reconstrução gulosa estilo NEH, reaproveitando agente/estado/recompensa) está de acordo, ou há algo a ajustar?
2. Sinal verde explícito para eu começar a implementar os 7 métodos novos (`ils_pm`, `ils_ap`, `ils_sarsa`, `ils_op2_only`, `ils_op3_only`, `ils_op4_only`, `ig_ql`). Carlos pediu para esperar a decisão completa antes de qualquer código — entendo que ela está completa agora com a resolução do método de comparação, mas confirmo explicitamente antes de escrever a primeira linha.

Depois da confirmação, implemento e valido os 7 métodos (backups, verificação por execução real, mesmo padrão já estabelecido nesta sessão) antes de rodar qualquer experimento em escala — e só então executo em ondas, registrando cada uma no diário.
