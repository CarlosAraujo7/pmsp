# Auditoria no papel de revisor — fim do Ciclo 5 (10/10/2026)

Regra mestre: plano -> execução -> auditoria como revisor -> novo plano (`LEI_GERAL_PESQUISA.md` II.13). Esta auditoria cobre o Ciclo 4 (Entradas 124–126: lei da folga confirmada em dados novos, ablação
estado x TD no R||Cmax, híbrido IG+LS no P||Cmax) e o Ciclo 5 (Entradas 127–132), cujo plano nasceu da auditoria posterior ao Ciclo 4 (a receita não tinha busca local; no R||Cmax o MILP-60s vencia
as heurísticas até n = 500 e não se sabia onde deixa de alcançar).

## Evidência do Ciclo 5 (todas pré-especificadas, hashes em `results/entrada127_prereg_hashes.txt` e `results/entrada129_prereg_hashes.txt`)

| Frente | Pergunta | Resultado | Leitura |
|---|---|---|---|
| 1 (Entradas 127/128) | A busca local melhora a receita (d ajustado + partida forte)? | F1 −0,036% (S) e F3 −0,072% (X') **a igual número de iterações**; **F2 a CPU igual: +0,034% (a receita sem busca local vence)**; 89,0% de ótimo provado com T_eq vs 81,2% (com LS) vs 74,7% (T = 1000) | O ganho é de ORÇAMENTO, não de estrutura. A receita mais forte é a sem busca local com mais iterações |
| 2 (Entradas 127/130) | O MILP-60s deixa de alcançar em R||Cmax n = 1000, m = 50? | **Não**: MILP GAP 2,04% vs IG+LS-QL 4,15% (T = 1000) e 2,78% (T = 5000); G1 +2,24%, G2 +0,89% (nenhuma confirmada a favor do IG) | Heurísticas de R não competitivas até (1000, 50); a seção de R é de atribuição |
| 3 (Entradas 129/131/132) | O ganho de 0,15% da busca local no híbrido do P||Cmax (Entrada 124) sobrevive a CPU igual? | Q1: −0,010% (IC90 −0,019; −0,002; equivalente a ±0,05%): **93% do ganho desaparece**; CPU igualado confirmado em verificação sequencial (razões 0,96–1,02) | A decomposição "destruir-reconstruir 0,59 + busca local 0,15" vale a igual número de iterações; a CPU igual o ingrediente ativo é destruir-reconstruir |

Integrado ao manuscrito (Seções 5.5, 5.6, 5.8/R, Discussão, Conclusões, resumo, destaques, tabela de evidências, tabela de pré-registro do suplemento: 39 páginas + 17).

## Fragilidades que um revisor do EJOR apontaria agora

| # | Fragilidade | Gravidade | Classe | Ação (Ciclo 6) |
|---|---|---|---|---|
| H1 | **A política foi sempre comparada a igual T, não a igual CPU.** No R||Cmax 1000 x 50 o Q-Learning custa **1,40x** o CPU da seleção aleatória (7,7 s vs 5,5 s; o cálculo do estado pesa); a lição das frentes 1 e 3 é que "igual T" pode ser vantagem de orçamento. Conclusões sobre a política (pequena, instável) podem estar **subestimadas** (se o custo extra é pago) ou ok | Alta | corrigível | Teste pré-especificado a CPU igual no regime de interesse (ver H2) |
| H2 | Em R||Cmax (1000 x 50) o ganho da política foi **0,52%** (descritivo, 32 instâncias; U[1,100] −1,53%), **5x maior** que o de n <= 500 — o único regime do estudo em que a política parece valer algo, e é justamente onde a busca NAO satura (GAP cai de 4,15% a 2,78% com 5x orçamento). A tese "a política adiciona pouco" pode estar restrita a regimes saturados; sem teste confirmatório isso é uma fraqueza (ou uma descoberta) | Alta | corrigível | Teste confirmatório em instâncias NOVAS (96, seeds 500000+): H1 ql vs random (T = 1000); H2 ql vs random a **CPU igual** (T_eq calibrado em sequência); folga dos 4 braços fixos neste tamanho |
| H3 | A receita é apresentada como o "melhor método" mas o orçamento T (e T_eq) foi escolhido pela razão de CPU de uma comparação, não otimizado; "mais iterações melhor" é monotônico mas sem curva | Média | corrigível (barato) | Curva GAP x T da receita (T = 1.000 ... 50.000) em 4 classes: onde satura? Já sugerido pelos dados (171 melhor, 0 pior vs T = 1000); fica como descritiva |
| H4 | Heurísticas de R||Cmax não são o estado da arte e as instâncias são sintéticas; o benchmark padrão (Fanjul-Peyro e Ruiz) exige download (permissão de Carlos) | Média | só Carlos | Pedir autorização; sem ela, declarado como limitação (já declarado) |
| H5 | Aprendiz tabular com recompensa única; nenhuma alternativa mais forte (função de aproximação, bandit contextual com características de instância além de ql_rf) | Média | limitação declarada | Mantido como limitação; escopo do manuscrito é a atribuição |
| H6 | Densidade do manuscrito (39 páginas; muitos subexperimentos em cadeia) | Média | editorial | Passada final de leitura (consistência de números, ordem, redundâncias) depois do Ciclo 6 |
| H7 | Pré-registro não público (diário com hashes) | Média | só Carlos | Depositar o diário/hashes em OSF/Zenodo no momento da submissão (checklist) |

## Plano do Ciclo 6 (único experimento novo; H3 só se sobrar tempo)

**Frente 1 — A política em regime não saturado, a CPU igual** (`experiment/run_unrelated_policy_entrada133.py`, `analysis/report_entrada133_unrelated_policy.py`). R||Cmax n = 1000, m = 50, 4 distribuições x 24 reps = **96 instâncias novas**
(seed 500000+), d/n = 0,02 e limiares de estado da Entrada 118 (sem reajuste), T = 1000. Métodos: ur_igls_random, ur_igls_ql, `ur_igls_random_eq` (random com T_eq = max(1000, round(1000 K)), K = razão
mediana de CPU ql/random em 8 instâncias de AJUSTE separadas, calibração sequencial), ur_igls_fix0..3. Família (Holm/2): **Q1 ql vs random (T = 1000)**; **Q2 ql vs random_eq (CPU igual)**.
Leituras pré-fixadas: 'a política vale em regime não saturado' = Q1 e Q2 confirmadas (d < 0); 'só via CPU' = Q1 sim, Q2 não; 'não' = Q1 não. Descritivas: folga (oráculo dos 4 braços) e razão política/folga com IC90; por distribuição.
Se Q2 for confirmada com d <= −0,10%, o resumo/discussão passam a delimitar a tese (a política vale onde a busca não satura); se não, a restrição "regimes saturados" cai e a tese se fortalece.

## Resultado do Ciclo 6 e 7 (Entradas 133-136) e decisão

- **Ciclo 6 (Entrada 133/134):** H1/H2 respondidas: a CPU igual a política **não** é confirmada no regime não saturado (Q1 -0,35%, p_holm 0,19; Q2 -0,19%, p_holm 0,67; a previsão registrada erra). Análise post hoc: efeito heterogêneo (U[1,100] -1,73%, 17/0; U[100,200] +0,89% a CPU igual, 0/23).
- **Ciclo 7 (Entrada 135/136):** a validação offline captura a folga (V2 -0,46%, 17/0, p_holm 8e-4) a menor CPU; V1 vs Q-Learning inconclusivo (-0,14%, IC90 -0,34; +0,07); V3 replica Q1 (-0,31%, p_holm 0,30).
- **Auditoria seguinte (revisor):** nenhuma fragilidade corrigível barata resta que mude uma conclusão; as pendentes são H4 (benchmark padrão de R||Cmax, só Carlos), H5 (agente tabular, limitação declarada), H6 (densidade; passada editorial de Carlos) e H7 (pré-registro público, só Carlos). H3 (curva GAP x T da receita) fica opcional (os dados de T_eq já mostram a monotonicidade: 171 melhor/0 pior contra T = 1000).
- **Decisão:** encerrar os ciclos de experimentos; integrar, auditar numericamente (87/87 + 63/63 + 59/59 + 21/21), recompilar (40 + 17 páginas) e entregar a Carlos.

## Adendo (Entradas 137-138): H4 resolvida
Carlos autorizou o download do benchmark padrão de R||Cmax; a validação externa pré-especificada (700 instâncias de teste, 280 de ajuste, referências exatas de terceiros) replicou a decomposição (mecanismo -2,14% a CPU igual; política não confirmada; validação offline vence o Q-Learning por 0,066%; lei da folga rho = 0,74) e mediu a distância às referências exatas (IG+LS-QL a 2,08% do melhor conhecido, métodos exatos a <= 0,18%). Restam H5 (agente tabular), H6 (densidade; 42 páginas) e H7 (pré-registro público): só Carlos ou limitação declarada.

## Adendo (Entradas 139-147): Ciclo 9 (a pedido de Carlos: buscar resultados melhores)
Frentes: (1) curva da receita (97,5% de ótimos a T = 64.000); (2) LP-IG em R||Cmax (competitivo em n >= 500 e nos tipos de faixa larga; fraco em n <= 200 e U[1,100]/U[10,100]); LP-IG vs MILP-60s (não confirmado pooled); (3) teste prospectivo da lei da folga com o tamanho de destruição: **refutada** (Q-Learning pior que sortear; validação vence). Fraquezas remanescentes: aprendiz tabular com janela de aprendizado curta (sugestão para trabalho futuro: recompensas atrasadas/horizonte longo, política treinada offline com representação mais rica); LP-IG fraco em n pequeno; densidade (44 páginas).
