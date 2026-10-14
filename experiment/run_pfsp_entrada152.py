"""
experiment/run_pfsp_entrada152.py
===================================
Entrada 152 (Ciclo 10): TERCEIRO PROBLEMA do estudo de atribuicao 'mecanismo x politica': flowshop de permutacao (Fm|prmu|Cmax), IG de Ruiz e Stutzle (2007) com selecao de acoes sobre o TAMANHO DE DESTRUICAO d
(acoes d in {2, 3, 4, 6, 8}), e ILS com a mesma busca local de insercao. Instancias 'estilo Taillard' geradas aqui (p ~ U{1..99}; sem download): n in {50, 100} x m in {10, 20}; 4 tamanhos;
AJUSTE = 10 instancias por tamanho (seed 31000 + 1000*tamanho + rep), TESTE = 30 por tamanho (seed 32000 + 1000*tamanho + rep) = 120. T = 300 iteracoes (IG), seed da execucao = seed da instancia.
Fases (cada uma congelada em hash antes da seguinte):
  --tune       : (a) d fixo (5 acoes) no ajuste -> d validado por tamanho (menor makespan medio / melhor makespan do ajuste); (b) limiares de estado (terciles de f1, f2 do `ig_rand`) ; (c) razoes de CPU sequenciais
                 (K_ils = CPU(ILS)/CPU(IG com LS) por iteracao; K_nols = CPU(IG com LS)/CPU(IG sem LS)) -> T do ILS e do IG sem LS a CPU igual
  --run        : teste: ig_fix_0..4, ig_rand, ig_ql, ig_ucb, ils_eq (T_ils), ig_nols_eq (T_nols)
Saidas: results/pfsp_tune_entrada152.csv, pfsp_choice_entrada152.json, pfsp_test_entrada152.csv. Retomavel.
Uso: python -m experiment.run_pfsp_entrada152 --tune --workers 11 ; --run --workers 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIZES = [(50, 10), (50, 20), (100, 10), (100, 20)]
D_CHOICES = [2, 3, 4, 6, 8]
T_IG = 300
N_TUNE, N_TEST = 10, 30
TUNE = ROOT / "results" / "pfsp_tune_entrada152.csv"
CHOICE = ROOT / "results" / "pfsp_choice_entrada152.json"
OUT = ROOT / "results" / "pfsp_test_entrada152.csv"
FIELDS = ["n", "m", "rep", "method", "makespan", "cpu_time", "T", "lb"]
METHODS = [f"ig_fix_{k}" for k in range(5)] + ["ig_rand", "ig_ql", "ig_ucb", "ils_eq", "ig_nols_eq"]


def inst_of(si, rep, base):
    from pmsp.pfsp import PfsInstance
    n, m = SIZES[si]
    return PfsInstance.random(n, m, seed=base + 1000 * si + rep, name=f"pfsp_{n}x{m}_r{rep}")


class PfsStateFn:
    """Estado de 81 bins: f1 = melhora relativa ao inicio, f2 = folga do corrente sobre o melhor, f3 = progresso, f4 = estagnacao; limiares f1/f2 (2 cada) calibrados no ajuste."""

    def __init__(self, thr):
        self.t1, self.t2 = thr["f1"], thr["f2"]
        self.rec = None

    def __call__(self, cur, best, init, t, T, no_imp):
        f1 = (init - cur) / max(init, 1)
        f2 = (cur - best) / max(best, 1)
        if self.rec is not None:
            self.rec.append((f1, f2))
        b1 = int(f1 > self.t1[0]) + int(f1 > self.t1[1])
        b2 = int(f2 > self.t2[0]) + int(f2 > self.t2[1])
        b3 = int(t / T > 0.33) + int(t / T > 0.67)
        b4 = int(no_imp / 50.0 > 0.25) + int(no_imp / 50.0 > 0.60)
        return 27 * b1 + 9 * b2 + 3 * b3 + b4


def _selector(kind, seed, na):
    from experiment import config as CFG
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import NormalisedUCBSelector, RandomSelector
    if kind.startswith("ig_fix_"):
        return FixedActionSelector(int(kind[-1]))
    if kind == "ig_ql":
        return QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN, epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=na)
    if kind == "ig_ucb":
        return NormalisedUCBSelector(C=CFG.UCB_C, seed=seed, n_actions=na)
    return RandomSelector(seed=seed, n_actions=na)


def solve(inst, kind, T, seed, thr=None, rec=None):
    from pmsp.pfsp import pfsp_ig_run, pfsp_ils_run
    if kind in ("ils_eq", "ils"):
        r = pfsp_ils_run(inst, max_iter=T, seed=seed)
        return r["makespan"], r["cpu_time"]
    use_ls = kind != "ig_nols_eq"
    sf = None
    if kind == "ig_ql" or rec is not None:
        sf = PfsStateFn(thr if thr is not None else {"f1": [0.0, 0.0], "f2": [0.0, 0.0]})
        sf.rec = rec
    r = pfsp_ig_run(inst, _selector(kind, seed, len(D_CHOICES)), D_CHOICES, max_iter=T, seed=seed, state_fn=sf, use_ls=use_ls)
    return r["makespan"], r["cpu_time"]


def _tune_task(t):
    inst = inst_of(t["si"], t["rep"], base=31000)
    rec = [] if t["method"] == "ig_rand" else None
    ms, cpu = solve(inst, t["method"], T_IG, 31000 + 1000 * t["si"] + t["rep"], rec=rec)
    return dict(n=inst.n, m=inst.m, rep=t["rep"], method=t["method"], makespan=int(ms), cpu_time=cpu, T=T_IG, lb=inst.lower_bound), rec


def _tkey(t):
    n, m = SIZES[t["si"]]
    return (str(n), str(m), str(t["rep"]), t["method"])


def _rkey(r):
    return (str(r["n"]), str(r["m"]), str(r["rep"]), r["method"])


def _drive(fn, tasks, out, workers, every, collect=None):
    done = set()
    if out.exists():
        done = {_rkey(r) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if _tkey(t) not in done]
    pend.sort(key=lambda t: -(SIZES[t["si"]][0] * SIZES[t["si"]][1]))
    print(f"[tasks] {len(pend)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in pend}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    row, extra = fut.result()
                    w.writerow(row)
                    if collect is not None and extra:
                        collect.extend(extra)
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % every == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def tune(workers):
    import pandas as pd
    tasks = [dict(si=si, rep=r, method=mt) for si in range(len(SIZES)) for r in range(N_TUNE) for mt in [f"ig_fix_{k}" for k in range(5)] + ["ig_rand"]]
    rec = []
    _drive(_tune_task, tasks, TUNE, workers, 10, collect=rec)
    d = pd.read_csv(TUNE)
    choice = {"d_val": {}, "thr": None, "T_ils": {}, "T_nols": {}}
    for (n, m), g in d.groupby(["n", "m"]):
        # razao ao melhor makespan (entre as 6 metodos) da propria instancia; escolhe o d fixo de menor razao media
        piv = g.pivot(index="rep", columns="method", values="makespan")
        ratio = piv[[f"ig_fix_{k}" for k in range(5)]].div(piv.min(axis=1), axis=0).mean()
        k = int(ratio.values.argmin())
        choice["d_val"][f"{int(n)}x{int(m)}"] = dict(action=k, d=D_CHOICES[k], ratios={c: float(v) for c, v in ratio.items()})
        print(f"[tune] {int(n)}x{int(m)} -> d = {D_CHOICES[k]} ({', '.join(f'{c[-1]}:{v:.4f}' for c, v in ratio.items())}) | rand {float((piv['ig_rand'] / piv.min(axis=1)).mean()):.4f}", flush=True)
    if rec:
        arr = np.array(rec)
        choice["thr"] = {"f1": [float(np.quantile(arr[:, 0], q)) for q in (1 / 3, 2 / 3)], "f2": [float(np.quantile(arr[:, 1], q)) for q in (1 / 3, 2 / 3)]}
    CHOICE.write_text(json.dumps(choice, indent=1))


def calibrate_cpu():
    """razoes de CPU sequenciais (maquina em repouso) por tamanho em 3 instancias de ajuste, T = 20."""
    ch = json.loads(CHOICE.read_text())
    for si, (n, m) in enumerate(SIZES):
        kr, kn = [], []
        for rep in range(3):
            inst = inst_of(si, rep, base=31000)
            seed = 31000 + 1000 * si + rep
            c_ig = solve(inst, "ig_rand", 20, seed)[1]
            c_ils = solve(inst, "ils_eq", 20, seed)[1]
            c_no = solve(inst, "ig_nols_eq", 200, seed)[1]
            kr.append((c_ils / 20) / (c_ig / 20))
            kn.append((c_ig / 20) / (c_no / 200))
        k_ils, k_no = float(np.median(kr)), float(np.median(kn))
        ch["T_ils"][f"{n}x{m}"] = max(10, int(round(T_IG / k_ils)))
        ch["T_nols"][f"{n}x{m}"] = max(T_IG, int(round(T_IG * k_no)))
        print(f"[calib] {n}x{m}: CPU/iter ILS/IG = {k_ils:.2f} -> T_ils = {ch['T_ils'][f'{n}x{m}']}; IG/IGnoLS = {k_no:.1f} -> T_nols = {ch['T_nols'][f'{n}x{m}']}", flush=True)
    CHOICE.write_text(json.dumps(ch, indent=1))


def _test_task(t):
    inst = inst_of(t["si"], t["rep"], base=32000)
    n, m = SIZES[t["si"]]
    ch = json.loads(CHOICE.read_text())
    mt = t["method"]
    T = T_IG
    if mt == "ils_eq":
        T = ch["T_ils"][f"{n}x{m}"]
    elif mt == "ig_nols_eq":
        T = ch["T_nols"][f"{n}x{m}"]
    ms, cpu = solve(inst, mt, T, 32000 + 1000 * t["si"] + t["rep"], thr=ch["thr"])
    return dict(n=n, m=m, rep=t["rep"], method=mt, makespan=int(ms), cpu_time=cpu, T=T, lb=inst.lower_bound), None


def run(workers):
    tasks = [dict(si=si, rep=r, method=mt) for si in range(len(SIZES)) for r in range(N_TEST) for mt in METHODS]
    _drive(_test_task, tasks, OUT, workers, 40)
    print("concluido ->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.tune:
        tune(a.workers)
    if a.calibrate:
        calibrate_cpu()
    if a.run:
        run(a.workers)


if __name__ == "__main__":
    main()
