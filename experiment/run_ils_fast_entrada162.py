"""
experiment/run_ils_fast_entrada162.py
=======================================
Entrada 162 (Ciclo 13, frente D3): o achado central 'o IG vence o ILS' (mecanismo) sobrevive a um ILS EFICIENTE a CPU igual? O ILS de P||Cmax usa quatro operadores de melhor melhora em lacos Python (op1, op2, op4 caros);
`pmsp/operators_fast.py::apply_best_fast` os vetoriza com equivalencia exata (op3 e' barato e fica como esta) e `ils_run(..., best_fn=apply_best_fast)` os usa (default None = comportamento anterior).
Comparacao com ig_random (T = 1000, d = 0,3 n, 3 estrategias aleatorias, partida LPT; results/results.csv; as mesmas instancias/seeds): ils_random_fast a ITERACOES iguais (T = 1000) e a CPU igual (T_ils = round(1000 / K_ils),
K_ils = CPU por iteracao do ILS rapido / CPU por iteracao do IG, mediana de 3 instancias, calibracao sequencial com o CPU do processo, `time.process_time`, medido em maquina possivelmente compartilhada).
Fases: --verify (ILS com best_fn rapida == ILS padrao, makespans identicos: 12 classes x 3 reps, T = 150) ; --calibrate ; --run --shard i --nshards N (12 classes x 100 reps x 2 metodos, em N processos independentes) ; --merge. Saidas: results/ils_fast_verify_entrada162.csv,
results/ils_fast_cpu_ratio_entrada162.json, results/ils_eq_fast_entrada162.csv. Retomavel.
Uso: python -m experiment.run_ils_fast_entrada162 --verify ; --calibrate ; --run --shard i --nshards N ; --merge
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

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
RATIO = ROOT / "results" / "ils_fast_cpu_ratio_entrada162.json"
VERIFY = ROOT / "results" / "ils_fast_verify_entrada162.csv"
OUT = ROOT / "results" / "ils_eq_fast_entrada162.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "T"]


def _inst(n, m, rep):
    from experiment import config as CFG
    from pmsp.instance import PMSPInstance
    return PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")


def ils_solve(n, m, rep, T, fast):
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.ils import ils_run
    from pmsp.operators_fast import apply_best_fast
    inst = _inst(n, m, rep)
    return inst, ils_run(instance=inst, selector=make_ils_selector("ils_random", CFG.BASE_SEED + rep), max_iter=T, seed=CFG.BASE_SEED + rep, stagnation_window=CFG.STAGNATION_WINDOW,
                         best_fn=apply_best_fast if fast else None)


def ig_solve(n, m, rep, T):
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.ig import ig_run
    inst = _inst(n, m, rep)
    return inst, ig_run(instance=inst, selector=make_ils_selector("ig_random", CFG.BASE_SEED + rep), max_iter=T, seed=CFG.BASE_SEED + rep, stagnation_window=CFG.STAGNATION_WINDOW)


def verify():
    rows = []
    for n, m in CLASSES:
        for rep in range(3):
            _, a = ils_solve(n, m, rep, 150, False)
            _, b = ils_solve(n, m, rep, 150, True)
            rows.append(dict(instance=f"rand_{n}j_{m}m", rep=rep, standard=int(a["makespan"]), fast=int(b["makespan"]), cpu_standard=a["cpu_time"], cpu_fast=b["cpu_time"]))
        print(f"[verify] {n}x{m}", flush=True)
    d = pd.DataFrame(rows)
    d["equal"] = d.standard == d.fast
    d.to_csv(VERIFY, index=False)
    print(f"ILS rapido == ILS padrao: {int(d.equal.sum())}/{len(d)} makespans identicos; razao de CPU media padrao/rapido {d.cpu_standard.sum() / d.cpu_fast.sum():.1f}")


def calibrate():
    out = {}
    T = 300
    for n, m in CLASSES:
        ta = tb = 0.0
        na = nb = 0
        rep = 0
        while rep < 3 or (min(ta, tb) < 3.0 and rep < 400):     # a resolucao de process_time no Windows e' ~15,6 ms: acumula ate' >= 3 s de CPU por metodo (minimo de 3 instancias)
            t0 = time.process_time()
            ils_solve(n, m, rep % 20, T, True)
            ta += time.process_time() - t0                      # CPU do PROCESSO (nao o relogio de parede): robusto a outros jobs na maquina compartilhada
            t0 = time.process_time()
            ig_solve(n, m, rep % 20, T)
            tb += time.process_time() - t0
            rep += 1
        k = ta / tb
        out[f"{n}x{m}"] = dict(K_ils=k, T_ils=max(10, int(round(1000 / k))), runs=rep)
        print(f"[calib] {n}x{m}: K_ils (CPU/iter ILS rapido / IG) = {k:.2f} -> T_ils = {out[f'{n}x{m}']['T_ils']} ({rep} execucoes de cada)", flush=True)
    RATIO.write_text(json.dumps(out, indent=1))


def _task(t):
    inst, r = ils_solve(t["n"], t["m"], t["rep"], t["T"], True)
    return dict(instance=inst.name, method=t["method"], rep=t["rep"], n=t["n"], m=t["m"], lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], T=t["T"])


def run(shard, nshards):
    """Fatias independentes (sem ProcessPoolExecutor, que caiu com WinError 6 nesta maquina compartilhada); --merge junta os CSV das fatias."""
    ratio = json.loads(RATIO.read_text())
    part = OUT.with_name(OUT.stem + f"_s{shard}.csv")
    done = set()
    for src in (OUT, part):
        if src.exists() and src.stat().st_size > 0:
            done |= {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(open(src, newline=""))}
    tasks = []
    for n, m in CLASSES:
        for rep in range(100):
            for mt, T in (("ils_fast_eqT", 1000), ("ils_fast_eqCPU", ratio[f"{n}x{m}"]["T_ils"])):
                if (f"rand_{n}j_{m}m", mt, str(rep)) not in done:
                    tasks.append(dict(n=n, m=m, rep=rep, T=T, method=mt))
    tasks.sort(key=lambda t: -t["n"] * t["T"])
    tasks = [t for k, t in enumerate(tasks) if k % nshards == shard]
    print(f"[fatia {shard}/{nshards}] {len(tasks)} pendentes -> {part.name}", flush=True)
    new = not part.exists()
    t0 = time.perf_counter()
    with open(part, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, t in enumerate(tasks, 1):
            w.writerow(_task(t))
            f.flush()
            os.fsync(f.fileno())
            if i % 25 == 0 or i == len(tasks):
                print(f"  [fatia {shard} {i}/{len(tasks)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido fatia", shard)


def merge():
    parts = sorted(OUT.parent.glob(OUT.stem + "_s*.csv"))
    frames = [pd.read_csv(OUT)] if OUT.exists() and OUT.stat().st_size > 0 else []
    frames += [pd.read_csv(p) for p in parts]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True).drop_duplicates(["instance", "method", "rep"])
    d.to_csv(OUT, index=False)
    print(f"merge -> {OUT.name}: {len(d)} linhas de {len(parts)} fatias")


def main():
    ap = argparse.ArgumentParser()
    for k in ("verify", "calibrate", "run", "merge"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.verify:
        verify()
    if a.calibrate:
        calibrate()
    if a.run:
        run(a.shard, a.nshards)
    if a.merge:
        merge()


if __name__ == "__main__":
    main()
