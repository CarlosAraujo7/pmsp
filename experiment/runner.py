"""
experiment/runner.py
====================
Motor experimental: executa o protocolo completo sobre N_INSTANCES instancias
distintas por configuracao (n, m) e salva os resultados em CSV + Q-tables.

Desenho experimental:
  - N_INSTANCES = 30 instancias DIFERENTES por (n, m), seed = BASE_SEED + inst_id.
  - N_REPS = 1 run por instancia (a variancia de interesse e entre instancias,
    nao entre runs do mesmo algoritmo sobre a mesma instancia).
  - 'rep' no CSV = inst_id (identificador da instancia), nao rodada algoritmica.

Metodos suportados:
  lpt         LPT deterministico
  ils_random  ILS com selecao aleatoria de operador
  ils_ucb     ILS com UCB1
  ils_rf      ILS com Random Forest online
  ils_ql      ILS com Q-Learning (cold-start)
  ils_rand_hi ILS-Random com orcamento estendido (controle de isolamento)
  ils_ql_ws   ILS-QL com warm-start da Q-table
  ils_ql_hi   ILS-QL com orcamento estendido (MAX_ITER_BY_SIZE)
  ig_ql       Iterated Greedy + Q-Learning (3 estrategias de destruicao)
  ig_random   Iterated Greedy + selecao aleatoria (controle de isolamento
              do ig_ql, ver diario_pesquisa.md, Entrada 26)
  ig_ucb      Iterated Greedy + UCB1 (segunda ablacao do ig_ql, ver
              diario_pesquisa.md, Entrada 27)
  ig_ucb_ctx  Iterated Greedy + UCB1 contextual (terceira ablacao do
              ig_ql, ver diario_pesquisa.md, Entrada 28)
  ig_ql_flat  Iterated Greedy + Q-Learning sem estado (fecha o 2x2 do
              ig_ql, ver diario_pesquisa.md, Entrada 29)

Baselines SOTA (Entrada 37, ANALISE_E_PLANO_EJOR.md Atualizacao 21):
  multifit       Coffman, Garey & Johnson (1978) -- classico revisado
                 por pares (pmsp/multifit.py)
  sa_li_perkins  Li & Perkins (2024), arXiv:2410.11880 -- preprint
                 (pmsp/sa.py)
  bco            Davidovic, Selmic, Teodorovic & Ramljak (2012),
                 Journal of Heuristics -- variante "BCO Modificado"
                 (Secao 5.3), verificado por texto bruto do PDF
                 completo (Entrada 40, pmsp/bco.py)

Uso:
    python -m experiment.runner                # roda metodos base
    python -m experiment.runner --extend       # roda ils_rand_hi, ils_ql_ws, ils_ql_hi
    python -m experiment.runner --summarize    # imprime resumo do CSV existente
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from pmsp.qlearning import (QLearningAgent, SarsaAgent,
                             ContextualUCBSelector, FlatQLearningSelector,
                             N_STATES, N_OPERATORS)
from pmsp.selectors import (
    RandomSelector, UCBSelector, RFSelector, AlwaysOp1Selector,
    AlwaysOp2Selector, AlwaysOp3Selector, AlwaysOp4Selector,
    ProbabilityMatchingSelector, AdaptivePursuitSelector,
    NormalisedUCBSelector,
)
from pmsp.ils import ils_run, lpt_run
from pmsp.ig import ig_run, N_DESTROY_STRATEGIES
from pmsp.multifit import multifit_run
from pmsp.ldm import ldm_run
from pmsp.seeding import best_constructive
from pmsp.sa import sa_run
from pmsp.bco import bco_run
from experiment import config as CFG


# ---------------------------------------------------------------------------
# Pre-treinamento da Q-table (warm-start)
# ---------------------------------------------------------------------------

def pretrain_qtable(
    source_sizes=None,
    n_pretrain: int = None,
    max_iter: int = None,
    base_seed: int = CFG.BASE_SEED,
    verbose: bool = True,
) -> np.ndarray:
    """
    Pre-treina um agente Q-Learning em instancias de porte pequeno
    e retorna a Q-table media acumulada sobre todas as rodadas.

    Os features do estado sao todos normalizados e independentes de escala,
    tornando a transferencia de Q-table entre tamanhos de instancia valida.

    Returns
    -------
    np.ndarray shape (N_STATES, N_OPERATORS)
    """
    if source_sizes is None:
        source_sizes = CFG.WS_SOURCE_SIZES
    if n_pretrain is None:
        n_pretrain = CFG.WS_N_PRETRAIN
    if max_iter is None:
        max_iter = CFG.MAX_ITER

    Q_accum = np.zeros((N_STATES, N_OPERATORS), dtype=np.float64)
    count = 0

    if verbose:
        print("  [pretrain] Iniciando pre-treinamento da Q-table...")

    for n, m in source_sizes:
        for inst_id in range(n_pretrain):
            seed  = base_seed + inst_id
            inst  = PMSPInstance.random(n=n, m=m, seed=seed)
            agent = QLearningAgent(
                alpha=CFG.QL_ALPHA,
                gamma=CFG.QL_GAMMA,
                epsilon_start=CFG.QL_EPSILON_START,
                epsilon_min=CFG.QL_EPSILON_MIN,
                epsilon_decay=CFG.QL_EPSILON_DECAY,
                seed=seed,
            )
            ils_run(
                instance=inst,
                selector=agent,
                max_iter=max_iter,
                perturbation_threshold=CFG.PERTURBATION_THRESHOLD,
                n_perturb=CFG.N_PERTURB,
                seed=seed,
                stagnation_window=CFG.STAGNATION_WINDOW,
            )
            Q_accum += agent._Q
            count += 1

    Q_avg = Q_accum / count if count > 0 else Q_accum
    if verbose:
        dominant = int(np.bincount(np.argmax(Q_avg, axis=1)).argmax())
        print(f"  [pretrain] {count} rodadas concluidas. "
              f"Operador dominante na politica media: Op{dominant+1}")
    return Q_avg


# ---------------------------------------------------------------------------
# Fabrica de seletores
# ---------------------------------------------------------------------------

def make_ils_selector(
    method: str,
    seed: int,
    q_init: Optional[np.ndarray] = None,
) -> Any:
    """
    Instancia o seletor correto para o metodo pedido.

    ils_rand_hi usa RandomSelector (o orcamento estendido e controlado
    pelo max_iter passado a ils_run, nao pelo seletor em si).
    """
    if method in ("ils_random", "ils_rand_hi"):
        return RandomSelector(seed=seed)
    if method == "ils_op1_only":
        return AlwaysOp1Selector()
    if method == "ils_ucb":
        return UCBSelector(C=CFG.UCB_C, seed=seed)
    if method == "ils_rf":
        return RFSelector(
            n_estimators=CFG.RF_N_ESTIMATORS,
            refit_interval=CFG.RF_REFIT_INTERVAL,
            window_size=CFG.RF_WINDOW_SIZE,
            epsilon=CFG.RF_EPSILON,
            seed=seed,
        )
    if method in ("ils_ql", "ils_ql_hi"):
        return QLearningAgent(
            alpha=CFG.QL_ALPHA,
            gamma=CFG.QL_GAMMA,
            epsilon_start=CFG.QL_EPSILON_START,
            epsilon_min=CFG.QL_EPSILON_MIN,
            epsilon_decay=CFG.QL_EPSILON_DECAY,
            seed=seed,
        )
    if method == "ils_ql_ws":
        return QLearningAgent(
            alpha=CFG.QL_ALPHA,
            gamma=CFG.QL_GAMMA,
            epsilon_start=CFG.QL_EPSILON_START_WS,
            epsilon_min=CFG.QL_EPSILON_MIN,
            epsilon_decay=CFG.QL_EPSILON_DECAY,
            seed=seed,
            q_init=q_init,
        )
    # -----------------------------------------------------------------
    # Metodos novos (31/08/2026) -- ver PLANO_EXPERIMENTAL_EXTENSIVO.md,
    # Secao 1, e diario_pesquisa.md, Entrada 23.
    # -----------------------------------------------------------------
    if method == "ils_sarsa":
        return SarsaAgent(
            alpha=CFG.QL_ALPHA,
            gamma=CFG.QL_GAMMA,
            epsilon_start=CFG.QL_EPSILON_START,
            epsilon_min=CFG.QL_EPSILON_MIN,
            epsilon_decay=CFG.QL_EPSILON_DECAY,
            seed=seed,
        )
    if method == "ils_pm":
        return ProbabilityMatchingSelector(seed=seed)
    if method == "ils_ap":
        return AdaptivePursuitSelector(seed=seed)
    if method == "ils_op2_only":
        return AlwaysOp2Selector()
    if method == "ils_op3_only":
        return AlwaysOp3Selector()
    if method == "ils_op4_only":
        return AlwaysOp4Selector()
    if method == "ig_ql":
        # Mesmo agente Q-Learning tabular, mas com espaco de acoes de
        # 3 estrategias de destruicao (pmsp/ig.py), nao 4 operadores
        # de vizinhanca -- ver diario_pesquisa.md, Entrada 23.
        return QLearningAgent(
            alpha=CFG.QL_ALPHA,
            gamma=CFG.QL_GAMMA,
            epsilon_start=CFG.QL_EPSILON_START,
            epsilon_min=CFG.QL_EPSILON_MIN,
            epsilon_decay=CFG.QL_EPSILON_DECAY,
            seed=seed,
            n_actions=N_DESTROY_STRATEGIES,
        )
    if method == "ig_random":
        # Controle de isolamento (Entrada 26): mesmo framework de busca
        # do ig_ql (Iterated Greedy, destruir/reconstruir), mas a escolha
        # da estrategia de destruicao e uniforme aleatoria, sem Q-Learning
        # nenhum -- serve para separar o ganho do MECANISMO de busca (IG)
        # do ganho (se houver) da POLITICA de selecao dentro dele.
        return RandomSelector(seed=seed, n_actions=N_DESTROY_STRATEGIES)
    if method == "ils_ucb_norm":
        # Entrada 98: UCB1 com recompensa normalizada em [0,1] (o
        # ils_ucb original usa recompensa bruta -- ver Entrada 97).
        return NormalisedUCBSelector(C=CFG.UCB_C, seed=seed)
    if method == "ig_ucb_norm":
        return NormalisedUCBSelector(C=CFG.UCB_C, seed=seed, n_actions=N_DESTROY_STRATEGIES)
    if method == "ig_ucb":
        # Segunda ablacao (Entrada 27): um bandit simples (UCB1, sem
        # TD-learning/Q-table) entre a aleatoriedade pura do ig_random e
        # o Q-Learning completo do ig_ql -- testa se a pequena vantagem do
        # ig_ql encontrada na Entrada 26 precisa de aprendizado sequencial
        # de verdade, ou se qualquer selecao adaptativa simples ja chega la.
        return UCBSelector(C=CFG.UCB_C, seed=seed, n_actions=N_DESTROY_STRATEGIES)
    if method == "ig_ucb_ctx":
        # Terceira ablacao (Entrada 28): UCB1 CONDICIONADO POR ESTADO
        # (mesmo state_id do QLearningAgent), mas sem TD-learning/gamma/
        # propagacao de valor entre estados -- isola a variavel
        # "condicionamento por estado" da variavel "aprendizado sequencial",
        # as duas coisas que o Q-Learning combina.
        return ContextualUCBSelector(C=CFG.UCB_C, seed=seed,
                                      n_actions=N_DESTROY_STRATEGIES)
    if method == "ig_ql_flat":
        # Quarta celula do desenho fatorial 2x2 (Entrada 29): TD-learning
        # de verdade (bootstrap com gamma, epsilon-greedy), mas SEM
        # condicionamento por estado (n_states=1, um unico estado
        # agregado) -- o oposto do ig_ucb_ctx. Fecha a decomposicao
        # estado x TD-learning aberta nas Entradas 27-28.
        return FlatQLearningSelector(
            alpha=CFG.QL_ALPHA,
            gamma=CFG.QL_GAMMA,
            epsilon_start=CFG.QL_EPSILON_START,
            epsilon_min=CFG.QL_EPSILON_MIN,
            epsilon_decay=CFG.QL_EPSILON_DECAY,
            seed=seed,
            n_actions=N_DESTROY_STRATEGIES,
        )
    raise ValueError(f"Metodo desconhecido: {method!r}")


# ---------------------------------------------------------------------------
# Executa uma unica rodada (instancia x metodo)
# ---------------------------------------------------------------------------

def run_single(
    instance: PMSPInstance,
    method: str,
    inst_id: int,
    max_iter: int = CFG.MAX_ITER,
    q_init: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Executa uma rodada e retorna o dicionario de metricas.

    Parameters
    ----------
    inst_id : int
        Identificador da instancia (= seed usada para gera-la).
        Salvo como 'rep' no CSV para manter compatibilidade.
    """
    seed = CFG.BASE_SEED + inst_id

    if method == "lpt":
        result = lpt_run(instance)
        result["selector"] = None
    elif method == "ldm":
        # Entrada 102: Largest Differencing Method (Karmarkar & Karp, 1982) -- ver pmsp/ldm.py
        result = ldm_run(instance)
        result["selector"] = None
    elif method in ("ig_ql_seed", "ig_random_seed"):
        # Entrada 102: IG partindo da melhor entre LPT/MULTIFIT/LDM (pmsp/seeding.py)
        selector = make_ils_selector(method.replace("_seed", ""), seed, q_init=q_init)
        result = ig_run(
            instance=instance,
            selector=selector,
            max_iter=max_iter,
            seed=seed,
            stagnation_window=CFG.STAGNATION_WINDOW,
            initial_solution=best_constructive(instance),
        )
        result["selector"] = selector
    elif method == "multifit":
        # Baseline SOTA classico (Coffman, Garey & Johnson, 1978) --
        # construtivo, deterministico, sem busca local -- ver
        # pmsp/multifit.py e diario_pesquisa.md, Entrada 37.
        result = multifit_run(instance)
        result["selector"] = None
    elif method == "sa_li_perkins":
        # Baseline SOTA reimplementado de Li & Perkins (2024),
        # arXiv:2410.11880 -- ver pmsp/sa.py (docstring completa com as
        # diferencas deliberadas em relacao ao artigo original) e
        # diario_pesquisa.md, Entrada 37. Nao usa selector (parametros
        # fixos T0/D/a_const, nao aprendizado).
        result = sa_run(instance, max_iter=max_iter, seed=seed)
        result["selector"] = None
    elif method == "bco":
        # Baseline SOTA reimplementado de Davidovic et al. (2012),
        # variante "BCO Modificado" (Secao 5.3) -- ver pmsp/bco.py
        # (docstring completa com fidelidade ao PDF original e as
        # diferencas deliberadas de orcamento/solucao inicial) e
        # diario_pesquisa.md, Entradas 39-40. Nao usa selector
        # (populacao de B abelhas com formulas fixas, nao aprendizado).
        result = bco_run(instance, max_iter=max_iter, seed=seed)
        result["selector"] = None
    elif method in ("ig_ql", "ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql_flat", "ig_ucb_norm"):
        # Framework de busca diferente (Iterated Greedy, nao ILS) --
        # ver pmsp/ig.py e diario_pesquisa.md, Entrada 23. Nao usa
        # perturbation_threshold/n_perturb (conceitos especificos do
        # ILS); o proprio ciclo destruir/reconstruir, a cada iteracao,
        # ja cumpre o papel de busca e diversificacao.
        selector = make_ils_selector(method, seed, q_init=q_init)
        result = ig_run(
            instance=instance,
            selector=selector,
            max_iter=max_iter,
            seed=seed,
            stagnation_window=CFG.STAGNATION_WINDOW,
        )
        result["selector"] = selector
    else:
        selector = make_ils_selector(method, seed, q_init=q_init)
        result = ils_run(
            instance=instance,
            selector=selector,
            max_iter=max_iter,
            perturbation_threshold=CFG.PERTURBATION_THRESHOLD,
            n_perturb=CFG.N_PERTURB,
            seed=seed,
            stagnation_window=CFG.STAGNATION_WINDOW,
        )
        result["selector"] = selector

    result.update({
        "instance": instance.name,
        "method":   method,
        "rep":      inst_id,   # 'rep' = inst_id por compatibilidade
        "n":        instance.n,
        "m":        instance.m,
        "lb":       instance.lower_bound,
    })
    return result


# ---------------------------------------------------------------------------
# Retrato fiel da configuracao realmente usada (corrige Achado 2 da
# auditoria de 29/08/2026 -- ver diario_pesquisa.md, Entradas 1 e 2:
# run_config.json estava desatualizado e nao era regravado por nenhum
# caminho de codigo).
# ---------------------------------------------------------------------------

def write_run_config_snapshot(
    output_dir,
    methods,
    instance_configs,
    n_instances: int,
    max_iter: int,
) -> Path:
    """
    Grava (sempre sobrescrevendo) um retrato dos parametros REALMENTE
    usados nesta chamada de run_all(), em <output_dir>/run_config.json.

    Diferente de um arquivo mantido a mao, este e regravado a cada
    execucao de run_all() -- nunca fica desatualizado silenciosamente.
    """
    import datetime
    snapshot = {
        "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "methods": list(methods),
        "instance_configs": [list(c) for c in instance_configs],
        "n_instances": n_instances,
        "max_iter": max_iter,
        "max_iter_by_size": {
            f"{k[0]}x{k[1]}": v for k, v in CFG.MAX_ITER_BY_SIZE.items()
        },
        "perturbation_threshold": CFG.PERTURBATION_THRESHOLD,
        "n_perturb": CFG.N_PERTURB,
        "stagnation_window": CFG.STAGNATION_WINDOW,
        "ql_alpha": CFG.QL_ALPHA,
        "ql_gamma": CFG.QL_GAMMA,
        "ql_epsilon_start": CFG.QL_EPSILON_START,
        "ql_epsilon_min": CFG.QL_EPSILON_MIN,
        "ql_epsilon_decay": CFG.QL_EPSILON_DECAY,
        "ucb_c": CFG.UCB_C,
        "rf_n_estimators": CFG.RF_N_ESTIMATORS,
        "rf_refit_interval": CFG.RF_REFIT_INTERVAL,
        "rf_window_size": CFG.RF_WINDOW_SIZE,
        "rf_epsilon": CFG.RF_EPSILON,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    path.write_text(json.dumps(snapshot, indent=2))

    # Historico append-only (nunca sobrescrito) -- adicionado em 29/08/2026
    # para que rodar --method X isolado nao apague o retrato da rodada
    # anterior mais abrangente; run_config.json continua sendo "a rodada
    # mais recente", e run_config_history.jsonl guarda todas.
    history_path = output_dir / "run_config_history.jsonl"
    with open(history_path, "a") as f_hist:
        f_hist.write(json.dumps(snapshot) + "\n")

    return path


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_all(
    methods=None,
    instance_configs=None,
    n_instances: int = CFG.N_INSTANCES,
    max_iter: int = CFG.MAX_ITER,
    output_dir=None,
    save_qtables: bool = True,
    verbose: bool = True,
    append: bool = False,
) -> Path:
    """
    Executa o protocolo experimental completo.

    Para cada (n, m) em instance_configs:
      - Gera n_instances instancias distintas (seed = BASE_SEED + inst_id)
      - Roda cada metodo UMA vez por instancia
      - 'rep' no CSV = inst_id (identificador da instancia)

    Parameters
    ----------
    append : bool
        Se True, abre o CSV em modo append. Util para adicionar novos
        metodos sem re-executar os ja existentes.
    """
    methods         = methods         or CFG.METHODS
    instance_configs = instance_configs or CFG.INSTANCE_CONFIGS
    output_dir      = Path(output_dir or CFG.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    qtables_dir = output_dir / CFG.QTABLES_SUBDIR
    if save_qtables:
        qtables_dir.mkdir(parents=True, exist_ok=True)

    # Pre-treina Q-table se necesario (warm-start)
    Q_pretrained = None
    if "ils_ql_ws" in methods:
        Q_pretrained = pretrain_qtable(verbose=verbose)
        np.save(output_dir / "qtable_pretrained.npy", Q_pretrained)

    fieldnames = [
        "instance", "method", "rep", "n", "m", "lb",
        "makespan", "gap_pct", "cpu_time", "n_improvements",
    ]

    csv_path  = output_dir / "results.csv"
    file_mode = "a" if append else "w"
    t_start   = time.perf_counter()

    # --- Guarda contra duplicatas em modo append ------------------------
    # Adicionado em 29/08/2026 (ver diario_pesquisa.md, Entrada 2) para
    # corrigir a causa-raiz do Achado 1 da auditoria: reexecutar um metodo
    # avulso (ex.: --method ils_ql_ws) apos um --extend anterior duplicava
    # silenciosamente as linhas desse metodo no CSV. Em modo append,
    # combinacoes (instance, method, rep) ja presentes agora sao puladas.
    existing_keys = set()
    if append and csv_path.exists():
        with open(csv_path, newline="") as f_exist:
            for row in csv.DictReader(f_exist):
                existing_keys.add((row["instance"], row["method"], row["rep"]))

    with open(csv_path, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()

        for n, m in instance_configs:
            inst_name = f"rand_{n}j_{m}m"
            gaps_by_method: Dict[str, list] = {meth: [] for meth in methods}

            for inst_id in range(n_instances):
                seed = CFG.BASE_SEED + inst_id
                inst = PMSPInstance.random(n=n, m=m, seed=seed,
                                           name=inst_name)

                for method in methods:
                    if (inst_name, method, str(inst_id)) in existing_keys:
                        if verbose:
                            print(f"  [skip] {inst_name} / {method} / "
                                  f"rep{inst_id:02d} ja presente em "
                                  f"{csv_path.name} (guarda anti-duplicata)")
                        continue

                    # Determina max_iter pelo metodo
                    if method in ("ils_ql_hi", "ils_rand_hi"):
                        mi = CFG.MAX_ITER_BY_SIZE.get((n, m), max_iter)
                    else:
                        mi = max_iter

                    qi = Q_pretrained if method == "ils_ql_ws" else None

                    r = run_single(inst, method, inst_id,
                                   max_iter=mi, q_init=qi)

                    # Salva Q-table para variantes QL
                    if method.startswith("ils_ql") and save_qtables:
                        sel = r.get("selector")
                        if sel is not None:
                            fname = f"{inst_name}_{method}_rep{inst_id:02d}"
                            sel.save(qtables_dir / fname)

                    gaps_by_method[method].append(r["gap_pct"])
                    writer.writerow({k: r[k] for k in fieldnames})

            if verbose:
                for method in methods:
                    gaps = gaps_by_method[method]
                    print(f"  {inst_name:22s} | {method:14s} | "
                          f"GAP={float(np.mean(gaps)):6.2f}% "
                          f"+/- {float(np.std(gaps)):.2f}%")

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"\nConcluido em {elapsed:.1f}s  ->  {csv_path}")

    write_run_config_snapshot(
        output_dir=output_dir,
        methods=methods,
        instance_configs=instance_configs,
        n_instances=n_instances,
        max_iter=max_iter,
    )
    return csv_path


# ---------------------------------------------------------------------------
# Sumario rapido
# ---------------------------------------------------------------------------

def summarize(csv_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(df.groupby(["method", "n", "m"])["gap_pct"].agg(["mean", "std"]).round(4).to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Motor experimental PMSP Learnheuristica")
    p.add_argument("--extend",     action="store_true",
                   help="Adiciona ils_rand_hi, ils_ql_ws, ils_ql_hi ao CSV existente")
    p.add_argument("--method",     default=None,
                   help="Roda apenas este metodo")
    p.add_argument("--n-instances", type=int, default=CFG.N_INSTANCES)
    p.add_argument("--max-iter",    type=int, default=CFG.MAX_ITER)
    p.add_argument("--output-dir",  default=CFG.RESULTS_DIR)
    p.add_argument("--no-qtables",  action="store_true")
    p.add_argument("--summarize",   action="store_true",
                   help="Imprime resumo do CSV existente e encerra")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.summarize:
        summarize(Path(args.output_dir) / "results.csv")
        sys.exit(0)

    if args.extend:
        methods = CFG.METHODS_EXTEND
        append  = True
    elif args.method:
        methods = [args.method]
        append  = args.method not in CFG.METHODS_BASE
    else:
        methods = CFG.METHODS_BASE
        append  = False

    run_all(
        methods=methods,
        n_instances=args.n_instances,
        max_iter=args.max_iter,
        output_dir=Path(args.output_dir),
        save_qtables=not args.no_qtables,
        append=append,
    )
