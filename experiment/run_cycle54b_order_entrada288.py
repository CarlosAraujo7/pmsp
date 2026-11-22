"""
experiment/run_cycle54b_order_entrada288.py (Entrada 288, Ciclo 54, parte B)
==============================================================================
O TESTE DE ORDEM e a CONTABILIDADE DA LINHA ZERADA do Q-Learning. Resposta a E da revisao cega #2 do CEC (lema do aprisionamento, teste de invariancia a ordem). So' gera dados; a analise e as previsoes ficam em
analysis/report_entrada288_order.py (congeladas antes de rodar este executor).

Celulas: as 12 classes sinteticas de P||Cmax no espaco S5 e os 7 tipos do benchmark padrao de R||Cmax no espaco B5 (as celulas em que o aprisionamento esta documentado). Para cada celula: 20 instancias (uma em cada cinco das 100 do teste),
R = 12 ORDENS da lista de acoes (identidade, invertida e 10 permutacoes aleatorias fixas), 3 sementes por (instancia, ordem) e os DOIS desempates do Q-Learning (tie_mode 0 = primeiro maximo, o padrao; 1 = maximo escolhido ao acaso),
pelos nucleos com registro (pmsp/ig_logged.py, pmsp/ur_logged.py). A permutacao e' aplicada a tabela de acoes (estrategia, tamanho, busca local) como um todo: o indice 0 da lista nova e' a acao pi[0] da antiga.
Por execucao: makespan final, limite inferior, fracao das decisoes na acao de INDICE 0 (share_first), fracao das decisoes tomadas numa linha Q ZERADA (share_zero) e a cota inferior da probabilidade de escolher o indice 0,
media_t [eps_t / K + (1 - eps_t) z_t] (z_t = 1 se a linha do estado estava zerada; com desempate pelo primeiro maximo, numa linha zerada o passo guloso escolhe o indice 0), com eps_t = max(0,05; 0,995^t).
Saida: results/order288_<celula>.csv (uma linha por execucao). Uso: python -m experiment.run_cycle54b_order_entrada288 [--workers 10]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

R = ROOT / "results"
T = 1000
HP = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
N_INST, N_ORD, N_SEED = 20, 12, 3
SEED0 = 71_000_000
FIELDS = ["coll", "cell", "inst", "perm_id", "perm", "seed_i", "tie", "makespan", "lb", "share_first", "share_zero", "bound"]


def perms(K: int) -> list[list[int]]:
    rng = np.random.default_rng(288)
    out = [list(range(K)), list(range(K))[::-1]]
    while len(out) < N_ORD:
        p = [int(v) for v in rng.permutation(K)]
        if p not in out:
            out.append(p)
    return out


def cell_list():
    from experiment import run_agents_numba_entrada180 as E
    cells = [("sintetico", f"S5|rand_{n}j_{m}m") for (n, m) in E.CLASSES]
    cells += [("padrao", f"B5|{ty}") for ty in ["U1a100", "U10a100", "U100a120", "U100a200", "U1000a1100", "JobsCorre", "MaqCorre"]]
    return cells


def do_cell(args):
    coll, cell, ci = args
    from experiment.run_cycle54_logs_entrada287 import instances_of
    from pmsp import ig_logged as IL
    from pmsp import ur_logged as UL
    out = R / ("order288_" + cell.replace("|", "_") + ".csv")
    if out.exists():
        return cell, 0, 0.0
    t0 = time.perf_counter()
    insts = list(instances_of(coll, cell, False))[::5][:N_INST]
    eps = np.maximum(0.05, 0.995 ** np.arange(T))
    rows = []
    for ii, (key, X, a0, m, od, lb, st, d, ls, core) in enumerate(insts):
        K = len(st)
        P = perms(K)
        fn = IL.ig_agent_core_logged if coll == "sintetico" else UL.ur_agent_core_logged
        for pi, perm in enumerate(P):
            stp, dp, lsp = st[perm], d[perm], ls[perm]
            for s in range(N_SEED):
                for tie in (0, 1):
                    seed = SEED0 + 1_000_000 * ci + 10_000 * ii + 100 * pi + 10 * s + tie
                    res = fn(X, a0, m, stp, dp, lsp, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, K)), 1.0, tie)
                    la, lz = res[6], res[10]
                    bound = float(np.mean(eps / K + (1.0 - eps) * lz))
                    rows.append(dict(coll=coll, cell=cell, inst=key, perm_id=pi, perm="".join(map(str, perm)) if K <= 10 else str(pi), seed_i=s, tie=tie, makespan=int(res[0]), lb=lb,
                                     share_first=float(np.mean(la == 0)), share_zero=float(np.mean(lz == 1)), bound=bound))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return cell, len(rows), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    jobs = [(c, n, i) for i, (c, n) in enumerate(cell_list())]
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        for cell, n, dt in pool.imap_unordered(do_cell, jobs):
            print(f"[order288] {cell}: {n} execucoes, {dt:.0f}s (total {time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
