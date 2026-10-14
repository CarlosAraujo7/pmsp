import os
import sys
import time
import traceback

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import experiment.config as CFG
from pmsp.instance import PMSPInstance
from pmsp.apriori_select import apriori_run, dup_ratio as compute_dup_ratio
from pmsp.racing import racing_run
from external_benchmarks.pcmax_symmetry2025.loader import load_instance_array

# ---------------------------------------------------------------------------
# Validacao em escala intermediaria de apriori_select/racing (Task #75,
# Entrada 61): nao as 1.612 instancias inteiras (isso exigiria rodar a
# otimizacao de verdade em todas, caro), mas uma amostra ESTRATIFICADA por
# familia e por tamanho (n), priorizando a familia E -- onde a Entrada 61
# mostrou que dup_ratio fica ABAIXO do baseline trivial (56,25% contra
# 60,25%) -- exatamente o caso onde o racing (que mede empiricamente, nao
# depende da feature-proxy) deveria, em teoria, se sair melhor.
#
# Reaproveita ils_ql/ig_ql JA CALCULADOS (mesma fonte que validou dup_ratio
# em larga escala: _scratch_dup_ratio_validation_full.csv, 1.612 instancias
# com os dois metodos, gap_pct medio sobre as repeticoes originais da
# bateria externa) como referencia -- so' roda de novo apriori_run()/
# racing_run(), que sao os metodos novos desta rodada de exploracao.
#
# Arquivos de saida gravados na PASTA DO REPOSITORIO (prefixo _scratch_,
# nao comprometidos como entrega final -- Carlos pediu pra so' limpar na
# hora de submeter), nao em /tmp: a VM efemera do device_bash pode reiniciar
# entre chamadas (ja aconteceu nesta sessao), e /tmp nao sobrevive a isso.
#
# Metodologia single-seed (nao 3 reps como no piloto de 5 instancias):
# trade-off deliberado de largura por profundidade, adequado pra um primeiro
# retrato em escala maior -- documentado aqui, nao escondido. Os valores de
# referencia ils_ql/ig_ql SAO medias sobre reps da bateria original; os
# valores novos (apriori_select/racing) sao de UM seed so' (1000) -- uma
# pequena assimetria de ruido, aceitavel pra esta checagem exploratoria.
#
# Retomavel por (member, method): cada chamada roda ate' TIME_BUDGET_S
# segundos (deixando margem sob o teto de 45s do device_bash) e para de
# forma limpa; rodar de novo continua de onde parou, sem repetir trabalho
# ja' feito -- mesmo padrao de resume por chave estavel da Entrada 57 (nao
# por indice recalculado).
# ---------------------------------------------------------------------------

REF_CSV = "_scratch_dup_ratio_validation_full.csv"
OUT_CSV = "_scratch_validate_new_mechanisms_external_results.csv"
ERR_LOG = "_scratch_validate_new_mechanisms_external_errors.log"

N_PER_FAMILY = {"B": 20, "C": 20, "E": 35, "F": 20}
SEED = 1000
MAX_ITER = CFG.MAX_ITER
TIME_BUDGET_S = 38  # margem sob o teto de 45s do device_bash


def build_sample():
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

        if (row.member, "apriori_select") not in done:
            try:
                r = apriori_run(
                    inst, max_iter=MAX_ITER, seed=SEED,
                    alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA,
                    epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                    epsilon_decay=CFG.QL_EPSILON_DECAY, stagnation_window=CFG.STAGNATION_WINDOW,
                    perturbation_threshold=CFG.PERTURBATION_THRESHOLD, n_perturb=CFG.N_PERTURB,
                )
                append_result(dict(
                    member=row.member, family=row.family, n=row.n, m=row.m,
                    method="apriori_select", gap_pct=r["gap_pct"],
                    framework=r["framework_escolhido"], dup_ratio=dr, cpu_time=r["cpu_time"],
                ))
                n_done_this_run += 1
            except Exception as exc:
                log_error(row.member, "apriori_select", exc)

        if (row.member, "racing") not in done:
            try:
                r2 = racing_run(
                    inst, max_iter=MAX_ITER, seed=SEED,
                    alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA,
                    epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                    epsilon_decay=CFG.QL_EPSILON_DECAY, stagnation_window=CFG.STAGNATION_WINDOW,
                    perturbation_threshold=CFG.PERTURBATION_THRESHOLD, n_perturb=CFG.N_PERTURB,
                )
                append_result(dict(
                    member=row.member, family=row.family, n=row.n, m=row.m,
                    method="racing", gap_pct=r2["gap_pct"],
                    framework=r2["framework_vencedor"], dup_ratio=dr, cpu_time=r2["cpu_time"],
                ))
                n_done_this_run += 1
            except Exception as exc:
                log_error(row.member, "racing", exc)

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
