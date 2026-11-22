"""
experiment/run_cycle54c_horizon_entrada290.py (Entrada 290, Ciclo 54, parte C)
================================================================================
DOSE-RESPOSTA DE HORIZONTE: a captura do aprendiz sobe com a informacao dentro da MESMA celula? Na Entrada 287b a captura subiu com os sucessos por acao por execucao (+0,55; tercos 0,036, 0,164, 0,208), mas a informacao estava confundida
com a folga e as colecoes. Aqui a informacao e' variada na MESMA celula: o horizonte T da execucao (1.000, 4.000 e 16.000 iteracoes) multiplica por 4 e por 16 os sucessos por acao por execucao. Hospedeiro: o benchmark padrao de R||Cmax
(Fanjul-Peyro e Ruiz), espaco B5, os 7 tipos (e' onde ha' folga real: o teto nao decide).
Por tipo e T: 30 instancias de teste (uma em cada tres das 100) x 2 sementes para o aleatorio (com registro, para contar os sucessos), o Q-Learning com desempate aleatorio (ql_rt) e o UCB1; a acao VALIDADA em T e' escolhida POR CLASSE (tipo, n), como na bateria principal (Entrada 256): a de menor razao media makespan / melhor conhecido nas
10 instancias de AJUSTE da classe (m de 10 a 50), 2 sementes por acao e mesmo T, e executada nas instancias de teste da classe. (A primeira rodada validava UMA acao por tipo, juntando os tamanhos: um pool sem rotulo, que a
Entrada 286b mostrou ser pior que o aleatorio; seus resultados ficaram em results/horizon290_run1_flawed/ e nao sao interpretados.) Nucleos: pmsp/ur_logged.py (aleatorio, com registro) e pmsp/ur_ties.py (os demais).
Saida: results/horizon290_<tipo>.csv (teste) e results/horizon290_tune_<tipo>.csv (ajuste). Uso: python -m experiment.run_cycle54c_horizon_entrada290 [--workers 10]
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
HP = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
TYPES = ["U1a100", "U10a100", "U100a120", "U100a200", "U1000a1100", "JobsCorre", "MaqCorre"]
HORIZONS = [1000, 4000, 16000]
N_TEST, N_SEED, N_TSEED = 30, 2, 2
SEED0 = 81_000_000
FIELDS = ["type", "instance", "T", "method", "seed_i", "makespan", "lb", "pos"]


def do_type(args):
    ty, ci = args
    from experiment import run_cycle32_fanjul_entrada256 as F
    from pmsp import ur_logged as UL
    from pmsp import ur_ties as UT
    man = F.manifest()
    test = man[(man.role == "test") & (man["type"] == ty)].sort_values("ri").iloc[::3].head(N_TEST)
    tune = man[(man.role == "tune") & (man["type"] == ty)].sort_values("ri")
    t0 = time.perf_counter()
    trows, rows = [], []
    for T in HORIZONS:
        # ---- ajuste da acao validada em T, POR CLASSE (tipo, n): menor razao media makespan / melhor conhecido, 2 sementes por acao
        val_by_n = {}
        for n_cls, tg in tune.groupby("n"):
            ratio = np.zeros(5)
            cnt = 0
            for _, r in tg.iterrows():
                Pm, a0, od, lb = F.load(r)
                st, d, ls = F.arm_table_b("B5", int(r.n))
                for a in range(5):
                    for s_ in range(N_TSEED):
                        sd = SEED0 + 100_000 * ci + 1000 * int(r.ri % 997) + 10 * a + s_ + 100 * (T // 1000)
                        res = UT.ur_agent_core_ties(Pm, a0, int(r.m), st, d, ls, T, 0, a, 0.1, 0.1, sd, od, float(lb), 0, 0, 0.0, HP, 0, np.zeros((81, 5)), 1.0, 1)
                        ratio[a] += res[0] / float(r.best_known)
                        trows.append(dict(type=ty, instance=int(r.instance), T=T, method=f"arm{a}", seed_i=s_, makespan=int(res[0]), lb=float(lb), pos=int(n_cls)))
                cnt += N_TSEED
            val_by_n[int(n_cls)] = int(np.argmin(ratio / cnt))
        # ---- teste
        for ii, (_, r) in enumerate(test.iterrows()):
            Pm, a0, od, lb = F.load(r)
            st, d, ls = F.arm_table_b("B5", int(r.n))
            val = val_by_n[int(r.n)]
            for s in range(N_SEED):
                base = SEED0 + 50_000_000 + 100_000 * ci + 1000 * ii + 10 * s + (T // 1000)
                res = UL.ur_agent_core_logged(Pm, a0, int(r.m), st, d, ls, T, 1, 0, 0.1, 0.1, base, od, float(lb), 0, 0, 0.0, HP, 0, np.zeros((81, 5)), 1.0, 1)
                rows.append(dict(type=ty, instance=int(r.instance), T=T, method="rand", seed_i=s, makespan=int(res[0]), lb=float(lb), pos=int((res[9] > 0).sum())))
                for meth, agent, fixed in (("ql_rt", 2, 0), ("ucb", 3, 0), ("val", 0, val)):
                    res = UT.ur_agent_core_ties(Pm, a0, int(r.m), st, d, ls, T, agent, fixed, 0.1, 0.1, base + 7 * (1 + agent), od, float(lb), 0, 0, 0.0, HP, 0, np.zeros((81, 5)), 1.0, 1)
                    rows.append(dict(type=ty, instance=int(r.instance), T=T, method=meth, seed_i=s, makespan=int(res[0]), lb=float(lb), pos=-1))
    for name, data in ((f"horizon290_{ty}.csv", rows), (f"horizon290_tune_{ty}.csv", trows)):
        with open(R / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(data)
    return ty, len(rows), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        for ty, n, dt in pool.imap_unordered(do_type, [(t, i) for i, t in enumerate(TYPES)]):
            print(f"[horizon290] {ty}: {n} execucoes, {dt:.0f}s (total {time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
