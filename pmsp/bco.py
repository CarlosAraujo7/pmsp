"""
pmsp/bco.py
===========
Bee Colony Optimization (BCO) para o PMSP, reimplementado a partir de
Davidovic, Selmic, Teodorovic & Ramljak (2012), "Bee colony optimization
for scheduling independent tasks to identical processors", Journal of
Heuristics 18(4):549-569, DOI 10.1007/s10732-012-9197-3 -- baseline SOTA
de comparacao empirica (ver diario_pesquisa.md, Entradas 33-40;
ANALISE_E_PLANO_EJOR.md, Atualizacoes 19, 21, 23, 24).

Fidelidade ao artigo original (texto completo extraido diretamente do
PDF fornecido por Carlos via acesso institucional CAPES/UECE -- nao
resumo de IA, texto bruto via pdftotext, Entrada 40).

Implementamos a variante "BCO Modificado" (Secao 5.3 do artigo), a que
sustenta a alegacao do resumo ("competitive with state-of-the-art
methods") e deu os melhores resultados do artigo tanto nas instancias
de scheduling quanto nas de BPP -- decisao de Carlos, pergunta feita
apos a verificacao completa (Entrada 40).

Estrutura geral (Secao 3.2, pseudocodigo, linhas 344-365 do PDF): cada
iteracao, B abelhas partem de solucao vazia e alternam entre:

    * Forward pass: cada abelha realiza ate NC movimentos construtivos
      (um movimento = escolher uma tarefa livre + um processador, ambos
      via roleta) -- Secao 4.1, Eq. 5-7.
    * Backward pass: abelhas avaliam suas solucoes parciais, decidem
      lealdade via roleta (Eq. 8-9, Secao 4.2), abelhas nao-comprometidas
      escolhem uma recrutadora via roleta (Eq. 10, Secao 4.3).

Repete forward/backward ate todas as solucoes completas (uma iteracao
completa); repete por I iteracoes (orcamento, ver abaixo).

Formulas (variaveis como no artigo):

    Selecao de tarefa (Eq. 5, linha 396):
        p_i = l_i / sum_{k em livres} l_k
        -- tarefas maiores tem mais chance (vies tipo LPT, estocastico).

    Selecao de processador (Eq. 6-7, linhas 407-413):
        p_j = V_j / sum_k V_k,   V_j = (max_y - y_j) / (max_y - min_y)
        -- processadores menos carregados tem mais chance.

    Lealdade (Eq. 8-9, linhas 443-452):
        O_b = (C_max - C_b) / (C_max - C_min)     [normalizacao entre abelhas]
        p_b^(u+1) = exp(-(O_max - O_b) / u)       [u = contador de forward pass]

    Recrutamento (Eq. 10, linhas 480-484):
        p_b = O_b / sum_{k em recrutadoras} O_k

Parametros tunados pelos proprios autores, mantidos EXATOS (linhas
563-568 do PDF, "determined experimentally as the best performing
ones"): B=5 abelhas, NC=10 movimentos por forward pass.

Modificacao "BCO Modificado" (Secao 5.3, linhas 712-752 do PDF): usa o
melhor makespan global ja encontrado (entre iteracoes completas) para:

    (a) forward pass: zera p_j se atribuir a tarefa ao processador j
        excederia o melhor makespan global (fallback explicito do
        proprio artigo, linha 746-747: processador aleatorio se todas
        as probabilidades zerarem);
    (b) backward pass: zera a probabilidade de lealdade (forca a abelha
        a ficar nao-comprometida) se a solucao parcial da abelha ja
        excede o melhor makespan global; se TODAS as abelhas ficarem
        nao-comprometidas, a iteracao e interrompida e reiniciada do
        zero (linha 751-752) -- aqui, a iteracao interrompida NAO conta
        resultado mas CONSOME um slot do orcamento de I iteracoes
        (decisao documentada, o artigo nao especifica).

Duas diferencas DELIBERADAS em relacao ao artigo original (mesmo
espirito das diferencas ja documentadas em pmsp/sa.py):

    1. Orcamento/parada: o artigo usa I=100 iteracoes fixas, escolhidas
       pelos autores para as escalas de instancia DELES (n ate 5000, m
       ate 400 -- ver diario_pesquisa.md, Entrada 40). A pedido de
       Carlos ("mesmo orcamento dos nossos metodos", Entrada 37/38),
       o numero de iteracoes aqui e DERIVADO de max_iter (MAX_ITER=1000
       deste projeto), nao um valor fixo.

       Cada iteracao externa do BCO custa B*n movimentos construtivos
       (B abelhas, cada uma completando n atribuicoes) -- MUITO mais
       caro por "iteracao" que um passo do ILS/IG/SA deste projeto (que
       fazem O(1) a O(d) movimentos por iteracao, d pequeno e fixo).
       Igualar o numero de "iteracoes" diretamente daria ao BCO um
       orcamento computacional descomunalmente maior que os outros
       metodos -- nao seria "o mesmo orcamento". Em vez disso,
       equalizamos o ORCAMENTO TOTAL DE MOVIMENTOS CONSTRUTIVOS:

           I = max(1, max_iter // (B * n))

       Isto significa que instancias maiores recebem PROPORCIONALMENTE
       MENOS iteracoes externas de BCO (ex., com B=5 e max_iter=1000:
       n=20 -> I=10; n=100 -> I=2) -- o mesmo tipo de aperto relativo de
       orcamento que ja explica o "cruzamento por escala" do ig_ql vs.
       MULTIFIT (Entrada 38), aplicado de forma consistente ao BCO. B e
       NC permanecem EXATOS (5 e 10) porque sao os valores tunados pelos
       proprios autores para a QUALIDADE de cada ciclo -- so o numero de
       VEZES que o ciclo roda muda, nao o desenho interno de cada ciclo.

    2. Solucao inicial: BCO e, por desenho (linha 348, "Assign an empty
       solution to each bee"), um metodo CONSTRUTIVO (constroi do zero
       a cada iteracao), nao um metodo de busca local perturbativo --
       por isso NAO comeca do LPT, ao contrario de ils_run/ig_run/
       sa_run. Mesmo precedente ja estabelecido pelo MULTIFIT
       (pmsp/multifit.py), tambem construtivo puro, sem ponto de
       partida LPT.

Casos-limite nao explicitados no artigo, tratados aqui de forma
defensiva e documentados por transparencia (mesmo padrao de pmsp/sa.py):

    - V_j quando max_y == min_y (empate de carga entre processadores --
      o caso MAIS COMUM de todos, ja que todos os processadores comecam
      com carga 0 no inicio de cada solucao de cada abelha): usamos
      probabilidade uniforme entre os m processadores.
    - O_b quando C_max == C_min (empate de makespan parcial entre
      abelhas): O_b = 1.0 para todas (todas empatadas no melhor).
    - "Melhor global" antes de existir qualquer solucao completa
      (1a iteracao): sem restricao -- equivalente ao BCO basico ate a
      primeira solucao completa existir.

Parametro `use_global_knowledge` permite desligar a modificacao da
Secao 5.3 (BCO basico puro) -- usado nos testes para verificar as
formulas base isoladamente; o runner experimental deste projeto sempre
usa o default (True = BCO Modificado, a decisao do Carlos).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .instance import PMSPInstance
from .solution import PMSPSolution
from .bounds import gap_percent


# ---------------------------------------------------------------------------
# Estado interno de uma abelha durante a construcao de uma solucao
# ---------------------------------------------------------------------------

class _BeeState:
    """
    Espelha a estrutura (S, A, Y) descrita no artigo (Secao 4, linhas
    373-381: S=matriz de alocacao, A=contagem por maquina, Y=cargas) --
    aqui como (assignment, loads, free_tasks). Implementado a parte de
    PMSPSolution (em vez de usa-la durante a construcao incremental)
    porque PMSPSolution.move_job() pressupoe que o job ja tem uma
    maquina de origem valida, o que nao vale para tarefas ainda nao
    atribuidas (assignment == -1) -- usar move_job() aqui seria
    incorreto (indexaria loads[-1], o ULTIMO elemento, por engano).
    """

    __slots__ = ("assignment", "loads", "free_tasks")

    def __init__(self, n: int, m: int, loads_dtype=np.int64) -> None:
        self.assignment = np.full(n, -1, dtype=np.int32)
        self.loads = np.zeros(m, dtype=loads_dtype)  # ver Entrada 53
        self.free_tasks: List[int] = list(range(n))

    def is_complete(self) -> bool:
        return len(self.free_tasks) == 0

    def clone(self) -> "_BeeState":
        """Copia profunda -- usada quando uma abelha nao-comprometida
        'junta-se' a uma recrutadora (linha 494-496: 'the current
        structure (S, A, Y) is copied from the recruiter to the
        follower')."""
        new = _BeeState.__new__(_BeeState)
        new.assignment = self.assignment.copy()
        new.loads = self.loads.copy()
        new.free_tasks = self.free_tasks.copy()
        return new

    def to_solution(self, instance: PMSPInstance) -> PMSPSolution:
        """Converte para PMSPSolution -- valido apenas quando is_complete()."""
        return PMSPSolution(instance, self.assignment)


# ---------------------------------------------------------------------------
# Selecao por roleta
# ---------------------------------------------------------------------------

def _roulette_choice(
    candidates: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Seleciona um elemento de `candidates` via roleta ponderada por
    `weights` (Secao 3.2: 'Choose one move using the roulette wheel').
    Nao precisam somar 1 -- normalizados aqui. Se todos os pesos forem
    0 (empate uniforme ja resolvido antes de chamar, OU todas as
    opcoes zeradas pela modificacao da Secao 5.3), cai para escolha
    UNIFORME entre os candidatos -- e exatamente o fallback explicito
    do proprio artigo para a selecao de processador (linha 746-747:
    'a processor is selected randomly in order to complete the current
    solution').
    """
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = w.sum()
    if total <= 0.0:
        idx = int(rng.integers(len(candidates)))
    else:
        probs = w / total
        idx = int(rng.choice(len(candidates), p=probs))
    return int(candidates[idx])


# ---------------------------------------------------------------------------
# Forward pass (Secao 4.1, Eq. 5-7)
# ---------------------------------------------------------------------------

def _forward_pass(
    bee: _BeeState,
    NC: int,
    m: int,
    p: np.ndarray,
    global_best: Optional[int],
    rng: np.random.Generator,
) -> None:
    """Executa ate NC movimentos construtivos para uma abelha (menos se
    restarem menos de NC tarefas livres). Modifica `bee` in-place."""
    moves_to_do = min(NC, len(bee.free_tasks))

    for _ in range(moves_to_do):
        free_arr = np.array(bee.free_tasks, dtype=np.int64)

        # --- Selecao de tarefa (Eq. 5) ---
        weights_task = p[free_arr].astype(np.float64)
        task = _roulette_choice(free_arr, weights_task, rng)

        # --- Selecao de processador (Eq. 6-7) ---
        y = bee.loads
        max_y, min_y = float(y.max()), float(y.min())
        if max_y == min_y:
            # Caso-limite mais comum: todos empatados (ex.: inicio da
            # construcao, todas as cargas = 0) -- probabilidade uniforme.
            V = np.ones(m, dtype=np.float64)
        else:
            V = (max_y - y.astype(np.float64)) / (max_y - min_y)

        if global_best is not None:
            # BCO Modificado (Secao 5.3, linha 743-747): zera p_j se
            # alocar `task` ao processador j excederia o melhor makespan
            # global ja encontrado.
            would_exceed = (y.astype(np.float64) + float(p[task])) > global_best
            V = np.where(would_exceed, 0.0, V)

        proc = _roulette_choice(np.arange(m, dtype=np.int64), V, rng)

        bee.assignment[task] = proc
        bee.loads[proc] += int(p[task])
        bee.free_tasks.remove(task)


# ---------------------------------------------------------------------------
# Backward pass (Secao 4.2-4.3, Eq. 8-10)
# ---------------------------------------------------------------------------

def _backward_pass(
    bees: List[_BeeState],
    global_best: Optional[int],
    u: int,
    rng: np.random.Generator,
) -> bool:
    """
    Avalia solucoes parciais, decide lealdade e recrutamento. Retorna
    False se TODAS as abelhas ficarem nao-comprometidas (BCO Modificado,
    linha 751-752: 'the iteration is interrupted and a new one starts')
    -- quem chama deve descartar a iteracao inteira nesse caso.
    """
    B = len(bees)
    Cb = np.array([float(bee.loads.max()) for bee in bees])
    Cmax, Cmin = float(Cb.max()), float(Cb.min())

    if Cmax == Cmin:
        # Caso-limite: todas as abelhas empatadas no makespan parcial
        # (ex.: muito no inicio da busca) -- todas igualmente "melhores".
        O = np.ones(B, dtype=np.float64)
    else:
        O = (Cmax - Cb) / (Cmax - Cmin)
    Omax = float(O.max())  # sempre 1.0 por construcao, mantido explicito
                            # para espelhar a notacao do artigo (Eq. 9)

    loyal = np.zeros(B, dtype=bool)
    for b in range(B):
        p_loyal = math.exp(-(Omax - O[b]) / u)
        if global_best is not None and Cb[b] > global_best:
            # BCO Modificado (Secao 5.3, linha 748-750): forca abelha
            # cuja solucao parcial ja excede o melhor global a ficar
            # nao-comprometida.
            p_loyal = 0.0
        loyal[b] = rng.random() < p_loyal

    recruiter_idxs = np.where(loyal)[0]
    if len(recruiter_idxs) == 0:
        return False  # iteracao interrompida (linha 751-752)

    uncommitted_idxs = np.where(~loyal)[0]
    O_recruiters = O[recruiter_idxs]
    for uidx in uncommitted_idxs:
        chosen = _roulette_choice(recruiter_idxs, O_recruiters, rng)
        bees[int(uidx)] = bees[chosen].clone()  # Eq. 10 + copia (S,A,Y)

    return True


# ---------------------------------------------------------------------------
# Uma iteracao completa (forward/backward ate todas as B solucoes prontas)
# ---------------------------------------------------------------------------

def _run_one_bco_iteration(
    instance: PMSPInstance,
    B: int,
    NC: int,
    global_best: Optional[int],
    rng: np.random.Generator,
) -> Optional[List[_BeeState]]:
    """Retorna a lista de B abelhas com solucoes completas, ou None se a
    iteracao foi interrompida (todas nao-comprometidas em algum backward
    pass -- so possivel com BCO Modificado, global_best is not None)."""
    n, m = instance.n, instance.m
    bees = [_BeeState(n, m, loads_dtype=instance.p.dtype) for _ in range(B)]
    u = 0

    while not all(bee.is_complete() for bee in bees):
        u += 1
        for bee in bees:
            _forward_pass(bee, NC, m, instance.p, global_best, rng)
        if not _backward_pass(bees, global_best, u, rng):
            return None

    return bees


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def bco_run(
    instance: PMSPInstance,
    max_iter: int = 1000,
    seed: int = 0,
    B: int = 5,
    NC: int = 10,
    use_global_knowledge: bool = True,
    track_history: bool = False,
) -> Dict[str, Any]:
    """
    Executa o BCO (Davidovic et al., 2012) como baseline de comparacao.
    Ver docstring do modulo para a fidelidade ao artigo original, a
    variante implementada (BCO Modificado, Secao 5.3) e as diferencas
    deliberadas (orcamento e solucao inicial).

    Espelha o dict de retorno de pmsp.ils.ils_run / pmsp.ig.ig_run /
    pmsp.sa.sa_run para plugar no mesmo harness experimental sem
    mudancas.

    Parametros
    ----------
    instance : PMSPInstance
    max_iter : int
        Orcamento TOTAL de movimentos construtivos (mesma semantica de
        MAX_ITER=1000 deste projeto para os outros metodos) -- o numero
        de iteracoes EXTERNAS do BCO e derivado disto, ver docstring do
        modulo ("diferenca deliberada 1").
    seed : int
    B : int
        Numero de abelhas (5 no artigo original, valor tunado pelos
        autores -- ver docstring do modulo).
    NC : int
        Movimentos construtivos por forward pass (10 no artigo
        original).
    use_global_knowledge : bool
        Se True (default -- a decisao de Carlos), usa a variante "BCO
        Modificado" (Secao 5.3). Se False, usa o BCO basico sem
        conhecimento global entre iteracoes (Secoes 3-4) -- usado nos
        testes para isolar as formulas base.
    track_history : bool

    Retorna
    -------
    dict com as chaves 'makespan', 'gap_pct', 'cpu_time',
    'n_improvements', 'history'.
    """
    rng = np.random.default_rng(seed)
    n = instance.n
    I = max(1, max_iter // (B * n))

    global_best: Optional[int] = None
    n_improvements = 0
    history: List[int] = []

    t_start = time.perf_counter()
    for _ in range(I):
        gb_for_iteration = global_best if use_global_knowledge else None
        bees = _run_one_bco_iteration(instance, B, NC, gb_for_iteration, rng)

        if bees is None:
            # Iteracao interrompida (linha 751-752): nao produz solucao
            # nova, mas consome um slot do orcamento de I (ver docstring
            # do modulo, "diferenca deliberada 1").
            if track_history and global_best is not None:
                history.append(int(global_best))
            continue

        for bee in bees:
            makespan = int(bee.loads.max())
            if global_best is None or makespan < global_best:
                global_best = makespan
                n_improvements += 1

        if track_history:
            history.append(int(global_best))

    cpu_time = time.perf_counter() - t_start
    lb = instance.lower_bound

    return {
        "makespan": int(global_best),
        "gap_pct": gap_percent(int(global_best), lb),
        "cpu_time": cpu_time,
        "n_improvements": n_improvements,
        "history": history,
    }
