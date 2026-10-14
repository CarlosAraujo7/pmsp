"""
experiment/run_dsel_entrada114.py
===================================
Entrada 114: a politica aprendida passa a importar quando o espaco de acoes TEM folga? Na Entrada 104 mostrou-se que as 3
estrategias de destruicao do IG sao quase intercambiaveis (folga 0,04%), e na Entrada 105 que o TAMANHO de destruicao d importa
bem mais (ate 0,45%). Aqui a ACAO e' o tamanho de destruicao: 5 fracoes d/n in {0,05; 0,10; 0,20; 0,30; 0,40} com a estrategia
de destruicao fixa em `Random` (isola o efeito de d).

Metodos (T=1000, partida LPT, mesmas instancias/seeds de results.csv: PMSPInstance.random(n, m, seed=rep), seed da execucao = rep):
  igd_fix05..igd_fix40   d fixo (5 metodos)
  igd_random             d uniforme entre as 5 fracoes a cada iteracao
  igd_ql_def             Q-Learning (5 acoes), estado padrao de 81 estados
  igd_ql_cal             Q-Learning (5 acoes), estado `cal` (limiares da Entrada 98)
  igd_ucb                UCB1 com recompensa normalizada (5 bracos)
Conjunto de AJUSTE (separado): seeds 90000+rep, rep 0-29, so' os 5 metodos de d fixo, por classe -> o melhor d fixo por classe e'
escolhido AQUI e avaliado no conjunto de TESTE (validacao fora da amostra).
Saidas: results/dsel_test_entrada114.csv, results/dsel_tuning_entrada114.csv (retomaveis).
Uso: python -m experiment.run_dsel_entrada114 --tune --workers 12 ; --run --workers 12 ; --smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20),
           (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
FRACS = [0.05, 0.10, 0.20, 0.30, 0.40]
FIX = ["igd_fix05", "igd_fix10", "igd_fix20", "igd_fix30", "igd_fix40"]
TEST_METHODS = FIX + ["igd_random", "igd_ql_def", "igd_ql_cal", "igd_ucb"]
N_TEST, N_TUNE, TUNE_SEED0 = 100, 30, 90000
OUT_TEST = ROOT / "results" / "dsel_test_entrada114.csv"
OUT_TUNE = ROOT / "results" / "dsel_tuning_entrada114.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements"]


def action_map(a: int, n: int):
    return 0, max(2, round(FRACS[a] * n))


def _run(task: dict) -> dict:
    from experiment import config as CFG
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ig import ig_run
    from pmsp.instance import PMSPInstance
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import NormalisedUCBSelector, RandomSelector

    n, m, rep, mt, tune = task["n"], task["m"], task["rep"], task["method"], task["tune"]
    iseed = (TUNE_SEED0 + rep) if tune else (CFG.BASE_SEED + rep)
    rseed = CFG.BASE_SEED + rep
    inst = PMSPInstance.random(n=n, m=m, seed=iseed, name=f"rand_{n}j_{m}m")
    kw = dict(instance=inst, max_iter=1000, seed=rseed, stagnation_window=CFG.STAGNATION_WINDOW, action_map=action_map)
    if mt in FIX:
        r = ig_run(selector=FixedActionSelector(FIX.index(mt)), **kw)
    elif mt == "igd_random":
        r = ig_run(selector=RandomSelector(seed=rseed, n_actions=5), **kw)
    elif mt == "igd_ql_def":
        ag = QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                            epsilon_decay=CFG.QL_EPSILON_DECAY, seed=rseed, n_actions=5)
        r = ig_run(selector=ag, **kw)
    elif mt == "igd_ql_cal":
        from experiment.run_signal_headroom_entrada104 import _state_fn
        ag = QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                            epsilon_decay=CFG.QL_EPSILON_DECAY, seed=rseed, n_actions=5)
        r = ig_run(selector=ag, state_fn=_state_fn(n, m), **kw)
    elif mt == "igd_ucb":
        r = ig_run(selector=NormalisedUCBSelector(C=CFG.UCB_C, seed=rseed, n_actions=5), **kw)
    else:
        raise ValueError(mt)
    return dict(instance=f"rand_{n}j_{m}m", method=mt, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"],
                gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"])


def _drive(tasks, out: Path, workers: int):
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(f)}
    tasks = [t for t in tasks if (f"rand_{t['n']}j_{t['m']}m", t["method"], str(t["rep"])) not in done]
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas) -> {out.name}")
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut], e)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 300 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cl = CLASSES[:1] if a.smoke else CLASSES
    if a.smoke:
        tasks = [dict(n=n, m=m, rep=r, method=mt, tune=False) for n, m in cl for r in range(2) for mt in TEST_METHODS]
        _drive(tasks, OUT_TEST.with_name("dsel_test_entrada114_SMOKE.csv"), a.workers)
        return
    if a.tune:
        _drive([dict(n=n, m=m, rep=r, method=mt, tune=True) for n, m in cl for r in range(N_TUNE) for mt in FIX], OUT_TUNE, a.workers)
    if a.run:
        _drive([dict(n=n, m=m, rep=r, method=mt, tune=False) for n, m in cl for r in range(N_TEST) for mt in TEST_METHODS], OUT_TEST, a.workers)


if __name__ == "__main__":
    main()
