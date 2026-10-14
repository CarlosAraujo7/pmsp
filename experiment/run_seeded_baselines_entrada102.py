"""
experiment/run_seeded_baselines_entrada102.py
==============================================
Baselines mais fortes (WP2 do plano de revisao; diario, Entrada 102). Roda, nas MESMAS
instancias e seeds da bateria sintetica (instancia = PMSPInstance.random(n, m,
seed=BASE_SEED+rep); run_single(inst, metodo, rep, max_iter=1000), identico a results.csv):

  ldm            Largest Differencing Method (Karmarkar & Karp, 1982; pmsp/ldm.py) -- construtiva
  ig_ql_seed     IG-QL partindo de min(LPT, MULTIFIT, LDM) em vez de so' LPT
  ig_random_seed IG com selecao uniforme (controle) partindo da mesma solucao

Saida: results/seeded_baselines_entrada102.csv (NAO toca results.csv). Retomavel:
(instance, method, rep) presentes sao pulados. Hipoteses/regra de decisao pre-especificadas
em diario_pesquisa.md (Entrada 102) e analisadas por analysis/report_entrada102_seeded_baselines.py.

Para o benchmark externo (6.780 instancias) usa-se o executor existente:
  cd external_benchmarks/pcmax_symmetry2025
  python run_external_battery.py --methods ldm,ig_ql_seed,ig_random_seed \
         --output results_external_seeded_entrada102.csv --workers 12

Uso (da raiz do repositorio):
    python -m experiment.run_seeded_baselines_entrada102 --workers 12
    python -m experiment.run_seeded_baselines_entrada102 --smoke
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
METHODS = ["ldm", "ig_ql_seed", "ig_random_seed"]
N_REPS = 100
OUT = ROOT / "results" / "seeded_baselines_entrada102.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements"]


def _run(task: dict) -> dict:
    from pmsp.instance import PMSPInstance
    from experiment.runner import run_single
    from experiment import config as CFG

    n, m, rep, method = task["n"], task["m"], task["rep"], task["method"]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    r = run_single(inst, method, rep, max_iter=1000)
    return dict(instance=f"rand_{n}j_{m}m", method=method, rep=rep, n=n, m=m, lb=inst.lower_bound,
                makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    classes = [(20, 6)] if args.smoke else CLASSES
    out = OUT.with_name("seeded_baselines_entrada102_SMOKE.csv") if args.smoke else OUT
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(f)}
    tasks = [dict(n=n, m=m, rep=rep, method=method)
             for n, m in classes for rep in range(3 if args.smoke else N_REPS) for method in METHODS
             if (f"rand_{n}j_{m}m", method, str(rep)) not in done]
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} ja feitas)")
    if args.dry_run or not tasks:
        return
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut], e)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 100 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
