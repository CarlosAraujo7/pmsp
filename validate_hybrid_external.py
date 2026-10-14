import os
import sys
import time
import traceback

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import experiment.config as CFG
from pmsp.instance import PMSPInstance
from pmsp.qlearning import QLearningAgent
from pmsp.apriori_select import dup_ratio as compute_dup_ratio
from pmsp.hybrid import hybrid_run, pretrain_hybrid_qtable, N_HYBRID_ACTIONS
from external_benchmarks.pcmax_symmetry2025.loader import load_instance_array

# ---------------------------------------------------------------------------
# Validacao em escala intermediaria de hybrid_ql/hybrid_ql_ws (Task #77,
# continuacao direta da Entrada 62): MESMA amostra estratificada de 95
# instancias usada para apriori_select/racing (reconstruida aqui pela
# MESMA logica de build_sample(), nao por um arquivo separado, para
# garantir -- por construcao, nao por sorte -- que sao literalmente as
# MESMAS instancias, permitindo comparacao direta nas mesmas linhas).
#
# hybrid_ql/hybrid_ql_ws NAO fazem selecao a priori de framework (ao
# contrario de apriori_select/racing): unificam os 7 movimentos (4 do
# ILS + 3 do IG) num unico espaco de acao e deixam o Q-Learning decidir
# iteracao a iteracao (pmsp/hybrid.py, Tarefa #68/#69). Por isso esta
# rodada NAO produz uma coluna "framework" nem uma acuracia de selecao --
# a comparacao pertinente aqui e' so' gap_pct contra oracle_best e contra
# os dois especialistas isolados, no MESMO pe de igualdade das Entradas
# 61-62.
#
# Warm-start (hybrid_ql_ws): pre-treino UMA UNICA VEZ via
# pretrain_hybrid_qtable(), usando os MESMOS hiperparametros ja
# estabelecidos para o warm-start padrao do projeto (ils_ql_ws,
# experiment/config.py: WS_SOURCE_SIZES=[(20,4)], WS_N_PRETRAIN=30,
# max_iter=CFG.MAX_ITER, base_seed=CFG.BASE_SEED) -- nao parametros
# novos inventados para esta rodada. A Q-table pre-treinada e' calculada
# em instancias SINTETICAS (nunca nas 95 instancias de validacao -- sem
# vazamento de informacao) alternando baixa/alta variancia (default de
# pretrain_hybrid_qtable(), motivado pela Tarefa #70/Entrada 60) e
# salva em disco (_scratch_hybrid_qtable_pretrained.npy) para nao ser
# recalculada a cada chamada retomada -- so' calculada 1x, cacheada.
#
# Retomavel por (member, method), mesmo padrao das Entradas 57/61/62:
# cada chamada roda ate' TIME_BUDGET_S segundos e para de forma limpa;
# rodar de novo continua de onde parou.
# ---------------------------------------------------------------------------

REF_CSV = "_scratch_dup_ratio_validation_full.csv"
OUT_CSV = "_scratch_validate_hybrid_external_results.csv"
ERR_LOG = "_scratch_validate_hybrid_external_errors.log"
QTABLE_CACHE = "_scratch_hybrid_qtable_pretrained.npy"

N_PER_FAMILY = {"B": 20, "C": 20, "E": 35, "F": 20}
SEED = 1000
MAX_ITER = CFG.MAX_ITER
TIME_BUDGET_S = 38  # margem sob o teto de 45s do device_bash


def build_sample():
    """IDENTICA a validate_new_mechanisms_external.py:build_sample() --
    mesma fonte, mesmo agrupamento, mesmo np.linspace -- para garantir as
    MESMAS 95 instancias por construcao."""
    ref = pd.read_csv(REF_CSV)
    pieces = []
    for fam, group in ref.groupby("family"):
        group = group.sort_values("n").reset_index(drop=True)
        k = min(N_PER_FAMILY.get(fam, 20), len(group))
        idx = np.linspace(0, len(group) - 1, k)
        idx = sorted(set(int(round(x)) for x in idx))
        pieces.append(group.iloc[idx])
    sample = pd.concat(pieces, ignore_index=True)
    return sample[["member", "family", "subfolder", "n", "m", "ils_ql", "ig_ql", "actual_winner"]]


def get_pretrained_qtable():
    if os.path.exists(QTABLE_CACHE):
        Q = np.load(QTABLE_CACHE)
        print(f"Q-table pre-treinada carregada do cache ({QTABLE_CACHE}), shape={Q.shape}", flush=True)
        return Q
    print("Pre-treinando Q-table hibrida (1x, sera cacheada)...", flush=True)
    t0 = time.time()
    Q = pretrain_hybrid_qtable(
        source_sizes=CFG.WS_SOURCE_SIZES,
        n_pretrain=CFG.WS_N_PRETRAIN,
        max_iter=MAX_ITER,
        base_seed=CFG.BASE_SEED,
        verbose=True,
    )
    np.save(QTABLE_CACHE, Q)
    print(f"Pre-treino concluido em {time.time()-t0:.0f}s, cacheado em {QTABLE_CACHE}", flush=True)
    return Q


def load_done():
    if os.path.exists(OUT_CSV):
        done = pd.read_csv(OUT_CSV)
        return set(zip(done.member, done.method))
    return set()


def append_result(row):
    df = pd.DataFrame([row])
    header = not os.path.exists(OUT_CSV)
    df.to_csv(OUT_CSV, mode="a", header=header, index=False)


def log_error(member, method, exc):
    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(f"{member} {method}: {exc}\n{traceback.format_exc()}\n---\n")


def main():
    sample = build_sample()
    Q_pretrained = get_pretrained_qtable()
    done = load_done()
    total_pairs = len(sample) * 2
    print(f"amostra total: {len(sample)} instancias ({total_pairs} pares instancia-metodo), "
          f"ja concluidos: {len(done)}", flush=True)
    print(sample.groupby("family").size().to_string(), flush=True)

    t_start = time.time()
    n_done_this_run = 0

    for row in sample.itertuples():
        if time.time() - t_start > TIME_BUDGET_S:
            print(f"\norcamento de tempo local esgotado ({TIME_BUDGET_S}s), parando (retomavel).", flush=True)
            break

        m, n, p = load_instance_array(row.family, row.member)
        inst = PMSPInstance(p=p, m=m, name=row.member)
        dr = compute_dup_ratio(inst)

        if (row.member, "hybrid_ql") not in done:
            try:
                agent = QLearningAgent(
                    alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA,
                    epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                    epsilon_decay=CFG.QL_EPSILON_DECAY, seed=SEED,
                    n_actions=N_HYBRID_ACTIONS,
                )
                r = hybrid_run(
                    instance=inst, selector=agent, max_iter=MAX_ITER,
                    seed=SEED, stagnation_window=CFG.STAGNATION_WINDOW,
                )
                append_result(dict(
                    member=row.member, family=row.family, n=row.n, m=row.m,
                    method="hybrid_ql", gap_pct=r["gap_pct"],
                    dup_ratio=dr, cpu_time=r["cpu_time"], n_improvements=r["n_improvements"],
                ))
                n_done_this_run += 1
            except Exception as exc:
                log_error(row.member, "hybrid_ql", exc)

        if (row.member, "hybrid_ql_ws") not in done:
            try:
                agent_ws = QLearningAgent(
                    alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA,
                    epsilon_start=CFG.QL_EPSILON_START_WS, epsilon_min=CFG.QL_EPSILON_MIN,
                    epsilon_decay=CFG.QL_EPSILON_DECAY, seed=SEED,
                    n_actions=N_HYBRID_ACTIONS, q_init=Q_pretrained,
                )
                r2 = hybrid_run(
                    instance=inst, selector=agent_ws, max_iter=MAX_ITER,
                    seed=SEED, stagnation_window=CFG.STAGNATION_WINDOW,
                )
                append_result(dict(
                    member=row.member, family=row.family, n=row.n, m=row.m,
                    method="hybrid_ql_ws", gap_pct=r2["gap_pct"],
                    dup_ratio=dr, cpu_time=r2["cpu_time"], n_improvements=r2["n_improvements"],
                ))
                n_done_this_run += 1
            except Exception as exc:
                log_error(row.member, "hybrid_ql_ws", exc)

        print(f"  {row.member} (fam={row.family} n={row.n} m={row.m} dup_ratio={dr:.3f}) OK  "
              f"(elapsed {time.time() - t_start:.0f}s, feitos nesta chamada={n_done_this_run})", flush=True)

    done_now = load_done()
    print(f"\nfeitos nesta chamada: {n_done_this_run}. "
          f"total concluido: {len(done_now)}/{total_pairs}.", flush=True)
    if len(done_now) < total_pairs:
        print("AINDA INCOMPLETO -- rode de novo pra continuar (retomavel por member+method).", flush=True)
    else:
        print("AMOSTRA COMPLETA.", flush=True)


if __name__ == "__main__":
    main()
