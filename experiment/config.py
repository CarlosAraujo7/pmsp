"""
experiment/config.py
====================
Configuracao central do desenho experimental.

Todos os hiperparametros estao concentrados aqui para facilitar
reproducibilidade e rastreabilidade (cada run salva uma copia desta config).
"""
from __future__ import annotations

# --- Protocolo de repeticoes ---
# N_INSTANCES: instancias DISTINTAS por configuracao (n, m).
#   Cada instancia e gerada com seed=instance_id, garantindo reproducibilidade.
#   As 30 instancias cobrem o espaco amostral; a variancia entre instancias
#   e o que interessa para generalizacao --- nao a variancia algoritmica
#   sobre a mesma instancia.
N_INSTANCES: int = 30     # instancias diferentes por (n, m)
N_REPS: int = 1           # runs por instancia (1 = determinismo + instance variance)
BASE_SEED: int = 0        # seed base; instancia i usa seed BASE_SEED + i

# --- ILS: hiperparametros compartilhados por todos os metodos ---
MAX_ITER: int = 1000              # iteracoes do loop principal
PERTURBATION_THRESHOLD: int = 50  # iters sem melhora antes de perturbar
N_PERTURB: int = 3                # movimentos aleatorios na perturbacao
STAGNATION_WINDOW: int = 50       # janela de normalizacao de f4

# --- Q-Learning: hiperparametros do agente ---
QL_ALPHA: float = 0.1
QL_GAMMA: float = 0.9
QL_EPSILON_START: float = 1.0
QL_EPSILON_MIN: float = 0.05
QL_EPSILON_DECAY: float = 0.995

# --- UCB: constante de exploracao ---
UCB_C: float = 1.4142  # sqrt(2)

# --- RF: configuracao do seletor ---
RF_N_ESTIMATORS: int = 20
RF_REFIT_INTERVAL: int = 15
RF_WINDOW_SIZE: int = 300
RF_EPSILON: float = 0.20

# --- Instancias do experimento (todas hard: LPT nao-otimo na maioria) ---
# Escolhidas com n/m in [3, 6]: poucos jobs por maquina -> alta variancia
# -> LPT nao consegue balancear bem.
# Instancias triviais (n=20x2, n=50x4, n=100x5) foram removidas.
INSTANCE_CONFIGS: list[tuple[int, int]] = [
    (20,  4),   # n/m = 5.0   pequeno
    (20,  6),   # n/m = 3.3   pequeno-hard
    (30,  8),   # n/m = 3.75  medio
    (50, 10),   # n/m = 5.0   medio-grande
    (75, 15),   # n/m = 5.0   grande
    (100, 20),  # n/m = 5.0   muito grande
]

# --- Metodos comparados ---
METHODS: list[str] = [
    "lpt",
    "ils_random",
    "ils_ucb",
    "ils_rf",
    "ils_ql",
    "ils_rand_hi",  # ILS-Random com orcamento estendido (controle de isolamento)
    "ils_ql_ws",    # ILS-QL com warm-start da Q-table
    "ils_ql_hi",    # ILS-QL com orcamento estendido (MAX_ITER_BY_SIZE)
]

# Metodos para primeira execucao (sem variantes estendidas)
METHODS_BASE: list[str] = ["lpt", "ils_random", "ils_ucb", "ils_rf", "ils_ql"]

# Metodos da segunda fase (--extend)
METHODS_EXTEND: list[str] = ["ils_rand_hi", "ils_ql_ws", "ils_ql_hi"]

# Metodos de ablacao/controle -- rodados avulsos via --method, nao fazem
# parte de METHODS nem de nenhuma rodada em lote (adicionado 29/08/2026,
# Fase 1 do plano em ANALISE_E_PLANO_EJOR.md).
METHODS_ABLATION: list[str] = ["ils_op1_only"]

# Baselines SOTA (estado da arte) para comparacao empirica real com a
# literatura -- a pedido de Carlos ("citacao nao basta"), ver
# diario_pesquisa.md, Entradas 33-37; ANALISE_E_PLANO_EJOR.md,
# Atualizacoes 19 e 21. Rodados avulsos via run_all(methods=METHODS_SOTA,
# append=True), igual a METHODS_NEW.
#   multifit       Coffman, Garey & Johnson (1978) -- classico, revisado
#                   por pares, totalmente especificado (pmsp/multifit.py).
#   sa_li_perkins  Li & Perkins (2024), arXiv:2410.11880 -- preprint,
#                   mesma distribuicao U[1,100] mas escala de instancia
#                   diferente da nossa (pmsp/sa.py). Reimplementado com
#                   LPT como solucao inicial e MAX_ITER como orcamento
#                   (mesmo orcamento dos nossos metodos, decisao de
#                   Carlos), em vez do design original (aloc. aleatoria +
#                   regra de parada por tempo M=n ms) -- ver docstring
#                   de pmsp/sa.py para a lista completa de diferencas
#                   deliberadas e documentadas.
#   bco            Davidovic, Selmic, Teodorovic & Ramljak (2012),
#                   Journal of Heuristics 18(4):549-569 -- verificado
#                   por texto bruto do PDF completo (Entrada 40, nivel
#                   de rigor pleno). Variante "BCO Modificado" (Secao
#                   5.3 do artigo, com conhecimento global entre
#                   iteracoes), decisao de Carlos. B=5, NC=10 mantidos
#                   EXATOS (valores tunados pelos proprios autores); o
#                   numero de iteracoes externas e derivado de MAX_ITER
#                   (mesmo orcamento dos nossos metodos) em vez do I=100
#                   fixo do artigo -- ver docstring de pmsp/bco.py para
#                   a lista completa de diferencas deliberadas.
METHODS_SOTA: list[str] = ["multifit", "sa_li_perkins", "bco"]

# Metodos novos aprovados em PLANO_EXPERIMENTAL_EXTENSIVO.md (Secao 1) --
# ver diario_pesquisa.md, Entrada 23 (implementacao) e Entrada 24
# (bateria completa). Rodados avulsos via run_all(methods=METHODS_NEW,
# append=True), nao fazem parte de METHODS (a grade original de 1.620
# execucoes ja publicada no artigo permanece intocada).
METHODS_NEW: list[str] = [
    "ils_sarsa", "ils_pm", "ils_ap",
    "ils_op2_only", "ils_op3_only", "ils_op4_only",
    "ig_ql",
]

# --- Orcamento estendido por tamanho de instancia ---
# Usado por ils_ql_hi e ils_rand_hi.
# Comentario ORIGINAL (linha abaixo) NAO se confirma por remedicao direta
# -- ver diario_pesquisa.md, Entrada 70 (Task #101, 05/09/2026):
# "Dimensionado para que o agente QL visite ~50% dos 81 estados com 3+ visitas."
# Remedido antes de estender a 6 classes novas: nenhuma classe testada
# (nem as originais abaixo) chega perto de 50% em orcamento pratico --
# ex.: (100,20) no seu proprio valor configurado (4000) atinge so 30/81
# (37%) e ja esta num platou real (32000 iteracoes, 8x mais, nao muda
# nada). As 6 entradas novas foram derivadas por CURVA DE COBERTURA REAL
# (1 seed, diagnostico completo em results/diag_max_iter_by_size_entrada70.json)
# -- "ponto de joelho" (ganho de cobertura vs custo por iteracao) por
# classe, nao um alvo fixo de %. Classes com mesmo n usam o mesmo valor
# (precedente ja existente de (20,4)==(20,6) abaixo). (250,25) e (300,20)
# platuam quase de imediato em coberturas baixas (~17% e ~14%) --
# orcamento estendido tem pouco efeito de exploracao nessas 2 classes.
MAX_ITER_BY_SIZE: dict = {
    (20,  4):  1500,
    (20,  6):  1500,
    (30,  8):  2000,
    (50, 10):  2500,
    (75, 15):  3000,
    (100, 20): 4000,
    # --- adicionado 05/09/2026, Entrada 70, Task #101 ---
    (40,   5): 8000,
    (60,  20): 8000,
    (150, 15): 8000,
    (150, 30): 8000,
    (250, 25): 4000,
    (300, 20): 2000,
}

# --- Warm-start: configuracao do pre-treinamento ---
WS_SOURCE_SIZES: list = [(20, 4)]
WS_N_PRETRAIN: int = 30
QL_EPSILON_START_WS: float = 0.5

# --- Diretorios de saida ---
RESULTS_DIR: str = "results"
QTABLES_SUBDIR: str = "qtables"
