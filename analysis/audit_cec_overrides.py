"""
analysis/audit_cec_overrides.py (Ciclo 51, 20/11/2026)
=======================================================
Reescrita do artigo do CEC apos a revisao duplo-cega de 18/11/2026: muitas frases mudaram de forma, algumas foram cortadas. As quatro auditorias numericas (audit_cec_claims.py, 2, 3, 4) conferem cada numero recalculado contra um TRECHO
literal do texto; aqui ficam, por rotulo de verificacao, (a) o trecho novo que contem o mesmo numero (a verificacao continua valendo, agora apontando para a frase nova), ou (b) RETIRED: a afirmacao saiu do artigo (por falta de espaco, por
ter sido substituida pela analise de captura da Entrada 283 ou por nao ser mais verdadeira na forma antiga), e a verificacao e' dispensada. Cada auditoria consulta `resolve` dentro de chk/present/journal. As verificacoes das afirmacoes NOVAS
(Entradas 283, 284, 282, 277, 281, tabela de classes) estao em analysis/audit_cec_claims5.py. Uso: importado pelas auditorias.
"""
from __future__ import annotations

RETIRED = object()

# rotulo -> RETIRED | trecho novo | (trecho novo, valor esperado, tolerancia)
OVERRIDES = {
    # ------------------------------------------------------------------ audit_cec_claims.py
    "E1 p_holm < 1e-80": RETIRED,
    "D1 parcela (30%)": r"($30\%$)",
    "C1 parcela (28%)": r"($28\%$; a reading registered before the compiled run)",
    "C2 contagem de avaliacoes (0,080)": RETIRED,
    "C2 parcela (50%)": RETIRED,
    "K vetorizada (1,8)": r"vectorized ($K=1.8$)",
    "K compilada (2,0)": r"compiled ($K=2.0$)",
    "D2 receita vetorizada (-0,008)": RETIRED,
    "maior ganho de aprendiz a iteracoes iguais (0,016)": r"gains more than $0.016$ points in any of the $12$ comparisons",
    "Q-Learning CPU igual S5 (0,059)": r"by $0.059$ and $0.049$",
    "validada vs nove metodos: p_holm < 1e-57": RETIRED,
    "K_CPU do Q em S5/S15/S30: minimo (0,50)": r"so $K=0.50$--$0.60$",
    "DQN-ILS vs mistura (0,041)": r"lose to the uniform mixture ($0.041$ and $0.066$)",
    "operador validado vs mistura (0,062)": r"as does the operator validated per class ($0.062$)",
    "controle positivo real S5 (0,092)": r"The validated action is detected at $0.092$ and $0.080$ points",
    "plantio, bonus 0,5: minimo >= 1,5 vez": r"choose it $1.5$--$13.7$ times as often as uniform choice would",
    "ALNS S5 a iteracoes iguais (0,039)": r"($0.039$ at equal iterations with five sizes",
    "ALNS S15 a CPU igual (0,020)": RETIRED,
    "decisao do Q-Learning S5 (-0,003)": r"decision effect $-0.003$ and $-0.004$",
    "custo do Q-Learning S5 (0,062)": r"($0.062$ and $0.053$; decision effect",
    "Thompson e DMAB a iteracoes iguais: |d| maximo (0,013)": r"stay within $0.013$",
    "corrida a iteracoes iguais S5 (+0,012)": r"($-0.012$ and $-0.067$)",
    "tempos finos: maior ganho a iteracoes iguais (0,015)": r"no learner gains more than $0.015$",
    "folga do oraculo T = 1000 (0,125)": r"is $0.125$ and $0.122$ points (five and fifteen actions)",
    "T = 100000: folga do oraculo (0,015)": r"falls to $0.015\%$",
    "T = 100000: vantagem da validada, minimo (0,012)": RETIRED,
    "instancias de teste do estagio 2 (2.550)": r"We ran the nine learners on $2{,}550$ published instances",
    "publicadas: Q-Learning a iteracoes iguais S15 (0,058)": r"it gains $0.058$ points at equal iterations with fifteen actions",
    "publicadas: roleta a iteracoes iguais S5 (0,057)": RETIRED,
    "familia C 1-2: Q-Learning S5 (0,16)": r"$0.16$ and $0.24$, while in family E",
    "H1 Q-Learning S15 familia C 1-2 (0,245)": r"(family C: $+0.245$ and $+0.149$",
    "H3b Q-Learning perde em E, S15 (0,052)": r"family E: $-0.052$ and $-0.059$",
    "H4 roleta S5 (0,056)": r"ALNS roulette with five sizes: $+0.056$",
    "as cinco hipoteses confirmatorias confirmadas (5)": r"all five confirmatory tests held",
    "baterias que contaram previsoes (40)": RETIRED,
    "previsoes que acertaram (244)": RETIRED,
    "23 artigos": RETIRED,
    "controle aleatorio (10)": RETIRED,
    "limite de CPU comum (6)": RETIRED,
    "estatico ajustado offline (3)": RETIRED,
    "beneficio do componente (22)": RETIRED,
    "dos quais parcial (5)": RETIRED,
    # ------------------------------------------------------------------ audit_cec_claims2.py
    "instancias de ajuste publicadas (525)": r"$525$ tuning,",
    "instancias da familia F (780)": RETIRED,
    "treino bruto B, C, E (6.000)": RETIRED,
    "excluidas do treino (280)": RETIRED,
    "instancias dos tres primeiros conjuntos (7.403)": RETIRED,
    "familia F S5 AUC (0,75)": RETIRED,
    "seletor vs validada: dentro da margem exceto S15 held-out": r"(parity, except in the larger space on held-out instances",
    "seletor vs qt e qtp, minimo (0,06)": r"(the selector does by $0.06$--$0.10$)",
    "flowshop sintetico: Q-Learning F5 (-0,050)": r"Q-Learning loses $0.05$--$0.08$",
    "flowshop: validada vence os dez pares a iteracoes iguais": r"beats the five learners in all ten learner--space pairs of the synthetic instances",
    "flowshop: validada vs aprendizes, minimo (0,020)": r"($0.020$--$0.104$)",
    "validada a CPU igual vence os 54 pares": r"in all $54$ comparisons (nine learners, two spaces, three instance sets)",
    # ------------------------------------------------------------------ audit_cec_claims3.py
    "remocao nas publicadas, minimo (28%)": r"and $28$--$40\%$ of that of the published instances",
    "rho cf held-out S5 (0,95)": RETIRED,
    "rho cf flowshop F5 (-0,27)": RETIRED,
    "limiar h* (0,048)": RETIRED,
    "celulas fora da amostra do limiar (60)": RETIRED,
    "acuracia do limiar nas 60 celulas (78%)": RETIRED,
    "Q aquecido vs frio, S5 descoberta (-0,006)": r"much worse with fifteen ($-0.146$ and $-0.143$)",
    "AG: validada vs aleatorio em G9 (-0,057)": r"($-0.057$ with nine actions)",
    "AG: Thompson G9 (0,051)": r"Thompson sampling gains $0.051$ and $0.047$",
    "mistura vs aleatorio G9 (0,050)": r"it gains $0.050$ and $0.144$",
    "mistura vs validada unica G9 (0,105)": r"beats the validated single action by $0.11$ and $0.07$",
    "testes registrados, todas as baterias (1169)": RETIRED,
    "testes confirmados (685)": RETIRED,
    "concordancia dos tres modos nos pares de cabecalho": RETIRED,
    "Q otimista vs aleatorio: ganho minimo (0,004)": r"optimistic initialization stays level with random choice ($0.004$--$0.008$)",
    "Q otimista pior que o frio, S5 held-out (0,007)": RETIRED,
    "validada vence o Q otimista, minimo (0,067)": RETIRED,
    "testes unitarios no total (1.598)": (r"$1{,}606$ unit tests in all", 1606, 0),
    "condicao necessaria: celulas fora da descoberta (150)": RETIRED,
    "condicao necessaria: celulas abaixo de h* (71)": RETIRED,
    "condicao necessaria: nenhuma abaixo de h* ganha alem da margem (0)": RETIRED,
    "condicao necessaria: celulas acima de h* (79)": RETIRED,
    "condicao necessaria: ganho > 0,05 acima de h* (29)": RETIRED,
    "referencia absoluta: fracao no limite nas sinteticas, minimo (42%)": RETIRED,
    "K de UCB1 e do gradiente: minimo (1,01)": RETIRED,
    "R||Cmax: ganho da validada S5 (0,95)": r"the validated action gains $0.95$ and $2.08$",
    "R||Cmax: a validada vence os dez pares (confirmados)": r"the validated action beats every learner in all ten learner--space pairs ($0.77$--$2.52$)",
    "R||Cmax: acuracia da regra (42%)": RETIRED,
    "cronograma: ganho no P||Cmax (0,095)": RETIRED,
    "cronograma: perde para a validada no R||Cmax (0,166)": RETIRED,
    "seletor R||Cmax: vence o melhor aprendiz em S5 (0,72)": RETIRED,
    "seletor R||Cmax: captura em S5 (60%)": RETIRED,
    "seletor R||Cmax: diferenca para a validada em S5 (-0,056)": RETIRED,
    "seletor R||Cmax: instancias de treino adicionais (1.200)": RETIRED,
    "AOS publicados: diferenca minima para a validada (0,06)": r"($0.06$--$1.82$)",
    "irace cfg: paridade com a validada no P||Cmax (dentro de 0,012)": r"(within $0.012$)",
    "irace cfg: atras da validada no R||Cmax, S5 (0,15)": r"and $0.15$--$0.19$ behind on R$||\Cmax$",
    "irace Q: piso do epsilon mediano, P||Cmax (0,26)": RETIRED,
    "irace Q: melhor que o Q padrao no R||Cmax, S5 (0,58)": RETIRED,
    "irace Q: a validada vence nos quatro pares": RETIRED,
    "CPU igual R||Cmax: Q-Learning S15 ganha sobre o aleatorio (0,58)": RETIRED,
    "CPU igual R||Cmax: efeito de decisao do Q-Learning S15 (-0,48)": RETIRED,
    "CPU igual R||Cmax: a validada bate os cinco aprendizes nos dez pares": r"also at equal CPU time ($0.94$--$2.17$)",
    "malha fechada: a validada bate la8 e ro nas oito celulas": RETIRED,
    "horizonte R||Cmax: validada S5 em T = 1.000 (1,00)": RETIRED,
    "horizonte R||Cmax: o Q-Learning nunca ganha do aleatorio (maior ganho -0,13)": r"Q-Learning never beats random choice and the validated action beats every learner at every horizon",
    "familias publicadas: a validada bate os 30 pares aprendiz-celula": r"all $30$ learner--cell pairs ($0.42$--$2.64$)",
    "familias publicadas: Q-Learning / validada em cm-S15 (65%)": r"($53\%$ and $65\%$ of the validated gain)",
    "concorrentes nas familias publicadas: irace cfg em paridade com a validada (dentro de 0,06)": RETIRED,
    "concorrentes nas familias publicadas: irace-Q ganha alem da margem em 5 celulas": RETIRED,
    "concorrentes nas familias publicadas: maior ganho dos desenhos publicados (0,54)": RETIRED,
    "concorrentes nas familias publicadas: a validada bate q, iq e fr nas 18 comparacoes": RETIRED,
    "ordem: celulas repetidas com o Q-Learning invariante a ordem (69), resumo": r"in none of the $69$ cells",
    "ordem: 24 celulas sinteticas de P||Cmax": r"$24$ synthetic and $18$ held-out published P$||\Cmax$ cells, $21$ benchmark and $6$ published-family R$||\Cmax$ cells, $69$ in all",
    "ordem: ql_perm perde 0,25 na familia C em S15": r"loses $0.25$ and $0.12$ points in family C",
    "ordem: Spearman do Q padrao nas 18 celulas (+0,65)": RETIRED,
    "ordem: ql_perm nas familias publicadas, maior ganho (-0,05)": RETIRED,
    "ordem: razao de avaliacoes do ql_perm, minimo (0,99)": RETIRED,
    "ordem: ql_v0 recupera de 81% a 86%, minimo": r"recovers $81$--$86\%$ of its gain",
    # ------------------------------------------------------------------ audit_cec_claims4.py
    "CEC: celulas do escopo (69)": r"$69$ in all",
    "CEC: o ql_rt ganha alem da margem em 25 das 69": r"Q-Learning gains beyond the margin in $25$ of the $69$ cells",
    "CEC: o ql_rt bate a validada em 2 das 69, por menos de 0,02": r"is beaten in $2$, by less than $0.02$ points",
    "CEC: ql_rt gana nas seis celulas das familias": r"all six published R$||\Cmax$ families",
    "CEC: Spearman do ql_rt nas 28 classes (0,81)": RETIRED,
    "CEC: Spearman do ql_rt nas 18 celulas publicadas (+0,72)": RETIRED,
    "CEC: ql_rt recupera 12% (B30) e 18% (B15) do ganho da validada": RETIRED,
    "CEC: ql_perm nao ganha em nenhuma das 69": r"it gains beyond the margin in none of the $69$ cells",
}


def resolve(label, expected, tol, snippet):
    """Retorna None se a verificacao foi aposentada, ou (esperado, tolerancia, trecho) ja' com a substituicao aplicada."""
    o = OVERRIDES.get(label)
    if o is None:
        return expected, tol, snippet
    if o is RETIRED:
        return None
    if isinstance(o, tuple):
        return o[1], o[2], o[0]
    return expected, tol, o
