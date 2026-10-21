"""
experiment/run_igls_fast_entrada162.py
========================================
Entrada 162 (Ciclo 13, frente D): SENSIBILIDADE DA CONCLUSAO 'A CPU IGUAL' A IMPLEMENTACAO. A busca local do IG hibrido de P||Cmax (inter-troca + reinsercao, ate 2 passos) estava em lacos Python triplos; a razao de
CPU com/sem busca local (K de 2,1 a 82,9) media em parte a ineficiencia do codigo e o '93% do ganho e orcamento' depende de K. `pmsp/operators_fast.py` traz os mesmos operadores vetorizados (equivalencia exata testada em
tests/test_operators_fast.py); este script recalibra K e refaz os dois testes a CPU igual com a busca local rapida:
  cenario H (Entrada 129): ig_random (d = 0,3 n, 3 estrategias aleatorias, partida LPT): igls_random (T = 1000, resultados da Entrada 124, identicos com a busca local rapida) vs ig_random_eq_fast (T_eq = max(1000, round(1000 K_fast)));
  cenario R (Entrada 127): receita (d/n ajustado, partida pela melhor construtiva): igcvls_seed (T = 1000, resultados da Entrada 127) vs igcv_seed_eq_fast (T_eq = max(1000, round(1000 K_fast))).
Fases:
  --verify    : igls com a busca local RAPIDA reproduz EXATAMENTE os makespans armazenados (cenario H: 12 classes x 5 reps; cenario R: 12 classes x 5 reps) -> results/igls_fast_verify_entrada162.csv
  --calibrate : K_fast por classe e cenario, sequencial, maquina em repouso, 3 instancias (reps 0-2) -> results/igls_fast_cpu_ratio_entrada162.json (K = CPU(com LS rapida)/CPU(sem LS), T = 1000)
  --run       : 12 classes x 100 reps x 2 cenarios (sem busca local, T_eq) -> results/igls_eq_fast_entrada162.csv (H), results/igcv_eq_fast_entrada162.csv (R). Retomavel.
Uso: python -m experiment.run_igls_fast_entrada162 --verify ; --calibrate ; --run --shard i --nshards N (N processos independentes) ; --merge
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

from experiment.run_igcvls_entrada127 import d_choice  # noqa: E402
from experiment.run_dsel_entrada114 import FRACS  # noqa: E402

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
RATIO = ROOT / "results" / "igls_fast_cpu_ratio_entrada162.json"
VERIFY = ROOT / "results" / "igls_fast_verify_entrada162.csv"
OUT_H = ROOT / "results" / "igls_eq_fast_entrada162.csv"
OUT_R = ROOT / "results" / "igcv_eq_fast_entrada162.csv"
FIELDS = ["scenario", "instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "T"]


def _inst(n, m, rep):
    from experiment import config as CFG
    from pmsp.instance import PMSPInstance
    return PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")


def solve(scenario, n, m, rep, T, ls, frac=None):
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ig import ig_run
    from pmsp.operators_fast import make_ls_fast
    from pmsp.seeding import best_constructive
    inst = _inst(n, m, rep)
    ls_fn = make_ls_fast() if ls else None
    if scenario == "H":
        r = ig_run(instance=inst, selector=make_ils_selector("ig_random", CFG.BASE_SEED + rep), max_iter=T, seed=CFG.BASE_SEED + rep, stagnation_window=CFG.STAGNATION_WINDOW, ls_fn=ls_fn)
    else:
        k = FRACS.index(frac)
        r = ig_run(instance=inst, selector=FixedActionSelector(k), max_iter=T, seed=CFG.BASE_SEED + rep, stagnation_window=CFG.STAGNATION_WINDOW,
                   action_map=lambda a, nn: (0, max(2, round(FRACS[a] * nn))), initial_solution=best_constructive(inst), ls_fn=ls_fn)
    return inst, r


def verify():
    hy = pd.read_csv(ROOT / "results" / "igls_p_entrada124.csv")
    hy = hy[hy.method == "igls_random"].set_index(["instance", "rep"]).makespan
    rc = pd.read_csv(ROOT / "results" / "igcvls_seed_entrada127.csv")
    rc = rc[rc.method == "igcvls_seed"].set_index(["instance", "rep"]).makespan
    ch = d_choice()
    rows = []
    for n, m in CLASSES:
        for rep in range(5):
            name = f"rand_{n}j_{m}m"
            inst, r = solve("H", n, m, rep, 1000, True)
            rows.append(dict(scenario="H", instance=name, rep=rep, stored=int(hy.loc[(name, rep)]), fast=int(r["makespan"])))
            inst, r = solve("R", n, m, rep, 1000, True, frac=ch[name])
            rows.append(dict(scenario="R", instance=name, rep=rep, stored=int(rc.loc[(name, rep)]), fast=int(r["makespan"])))
        print(f"[verify] {n}x{m}", flush=True)
    d = pd.DataFrame(rows)
    d["equal"] = d.stored == d.fast
    d.to_csv(VERIFY, index=False)
    print(f"equivalencia a nivel de execucao: {int(d.equal.sum())}/{len(d)} makespans identicos")


def calibrate():
    ch = d_choice()
    out = {"H": {}, "R": {}}
    for sc in ("H", "R"):
        for n, m in CLASSES:
            rs = []
            for rep in range(3):
                a = solve(sc, n, m, rep, 1000, False, frac=ch[f"rand_{n}j_{m}m"])[1]["cpu_time"]
                b = solve(sc, n, m, rep, 1000, True, frac=ch[f"rand_{n}j_{m}m"])[1]["cpu_time"]
                rs.append(b / a)
            k = sorted(rs)[1]
            out[sc][f"{n}x{m}"] = dict(K=k, T_eq=max(1000, int(round(1000 * k))))
            print(f"[calib {sc}] {n}x{m}: K = {k:.2f} -> T_eq = {out[sc][f'{n}x{m}']['T_eq']}", flush=True)
    RATIO.write_text(json.dumps(out, indent=1))


def _task(t):
    inst, r = solve(t["scenario"], t["n"], t["m"], t["rep"], t["T"], False, frac=t.get("frac"))
    return dict(scenario=t["scenario"], instance=inst.name, method="ig_random_eq_fast" if t["scenario"] == "H" else "igcv_seed_eq_fast", rep=t["rep"], n=t["n"], m=t["m"], lb=inst.lower_bound,
                makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"], T=t["T"])


def run(shard, nshards):
    """Execucao em FATIAS independentes (sem ProcessPoolExecutor: o pool caiu com WinError 6 varias vezes nesta maquina compartilhada). Cada fatia grava seu proprio CSV; --merge junta."""
    ratio = json.loads(RATIO.read_text())
    ch = d_choice()
    for sc, out in (("H", OUT_H), ("R", OUT_R)):
        part = out.with_name(out.stem + f"_s{shard}.csv")
        done = set()
        for src in (out, part):
            if src.exists():
                done |= {(r["instance"], r["rep"]) for r in csv.DictReader(open(src, newline=""))}
        tasks = [dict(scenario=sc, n=n, m=m, rep=r, T=ratio[sc][f"{n}x{m}"]["T_eq"], frac=ch[f"rand_{n}j_{m}m"]) for n, m in CLASSES for r in range(100) if (f"rand_{n}j_{m}m", str(r)) not in done]
        tasks.sort(key=lambda t: -t["n"] * t["T"])
        tasks = [t for k, t in enumerate(tasks) if k % nshards == shard]
        print(f"[fatia {shard}/{nshards} {sc}] {len(tasks)} pendentes -> {part.name}", flush=True)
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
                    print(f"  [fatia {shard} {sc} {i}/{len(tasks)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido fatia", shard)


def merge():
    for out in (OUT_H, OUT_R):
        parts = sorted(out.parent.glob(out.stem + "_s*.csv"))
        frames = [pd.read_csv(out)] if out.exists() and out.stat().st_size > 0 else []
        frames += [pd.read_csv(p) for p in parts]
        d = pd.concat([f for f in frames if len(f)], ignore_index=True).drop_duplicates(["instance", "rep"])
        d.to_csv(out, index=False)
        print(f"merge -> {out.name}: {len(d)} linhas de {len(parts)} fatias")


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
