"""
experiment/run_sota_sensitivity_entrada90.py
==============================================
Analise de SENSIBILIDADE dos baselines SOTA (diario_pesquisa.md,
Entrada 90, item 3 da lista de melhorias). Motivo: a auditoria de
04/10/2026 mostrou que (a) o BCO roda com I=max(1,1000//(5n)) = 10..1
iteracoes (publicado: I=100) e (b) sa_li_perkins tem gap medio IDENTICO
ao LPT em 7 das 12 classes (regra de parada em 1% do LB). Este script
roda, nas MESMAS instancias/seeds da bateria sintetica (instancia =
PMSPInstance.random(n, m, seed=BASE_SEED+rep)):

  bco_I100   BCO com o numero de iteracoes PUBLICADO (I=100)
  bco_tm     BCO em TEMPO IGUAL ao ig_ql: I calibrado por classe (3
             instancias, mesma maquina/sessao): I = round(cpu_ig_ql /
             cpu_por_iteracao_do_bco), minimo 1
  sa_noearly SA de Li & Perkins SEM a regra de parada (p_target=0), 1000
             iteracoes completas
  ig_ql_cpu  RE-EXECUCAO do ig_ql (mesmo seed): (i) da' um cpu_time na
             MESMA sessao/maquina dos baselines acima (os cpu_time de
             results.csv vem de outra maquina/condicao -- as razoes entre
             metodos se preservam, os valores absolutos nao); (ii) serve
             de checagem de determinismo (makespan deve ser IDENTICO ao
             de results.csv)

Saida: results/sota_sensitivity_entrada90.csv (NAO toca results.csv).
Retomavel: (instance, method, rep) ja presentes sao pulados.

Replicas: N=100 para n<=100 (N=30 em rand_20j_4m, como no resto do
projeto) e N=30 para n>=150 (BCO a I=100 custa dezenas de segundos por
execucao nessas classes). Ver analysis/report_entrada90_sota_sensitivity.py.

Uso (da raiz do repositorio):
    python -m experiment.run_sota_sensitivity_entrada90 --workers 8
    python -m experiment.run_sota_sensitivity_entrada90 --smoke     # 1 classe, 2 reps
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20),
           (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
METHODS = ["bco_I100", "bco_tm", "sa_noearly", "ig_ql_cpu"]
OUT = ROOT / "results" / "sota_sensitivity_entrada90.csv"
CALIB = ROOT / "results" / "sota_sensitivity_entrada90_calibration.json"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time",
          "n_improvements", "I_used"]


def n_reps(n: int, m: int) -> int:
    if (n, m) == (20, 4):
        return 30
    return 100 if n <= 100 else 30


def _run(task: dict) -> dict:
    from pmsp.instance import PMSPInstance
    from pmsp.bco import bco_run
    from pmsp.sa import sa_run
    from experiment.runner import run_single
    from experiment import config as CFG

    n, m, rep, method = task["n"], task["m"], task["rep"], task["method"]
    seed = CFG.BASE_SEED + rep
    inst = PMSPInstance.random(n=n, m=m, seed=seed, name=f"rand_{n}j_{m}m")
    I_used = ""
    if method == "bco_I100":
        I_used = 100
        r = bco_run(inst, max_iter=100 * 5 * n, seed=seed)
    elif method == "bco_tm":
        I_used = task["I_tm"]
        r = bco_run(inst, max_iter=I_used * 5 * n, seed=seed)
    elif method == "sa_noearly":
        r = sa_run(inst, max_iter=1000, seed=seed, p_target=0.0)
    elif method == "ig_ql_cpu":
        r = run_single(inst, "ig_ql", rep, max_iter=1000)
    else:
        raise ValueError(method)
    return dict(instance=f"rand_{n}j_{m}m", method=method, rep=rep, n=n, m=m, lb=inst.lower_bound,
                makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], I_used=I_used)


def calibrate(classes) -> dict:
    """I do BCO em tempo igual ao ig_ql, por classe (mesma maquina/sessao, execucao
    sequencial -- rode com a maquina em repouso)."""
    from pmsp.instance import PMSPInstance
    from pmsp.bco import bco_run
    from experiment.runner import run_single
    from experiment import config as CFG
    cal = {}
    for n, m in classes:
        c_ig, c_it = [], []
        for rep in range(3):
            inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
            c_ig.append(run_single(inst, "ig_ql", rep, max_iter=1000)["cpu_time"])
            c_it.append(bco_run(inst, max_iter=2 * 5 * n, seed=rep)["cpu_time"] / 2)
        ig, it = sum(c_ig) / 3, sum(c_it) / 3
        cal[f"{n}x{m}"] = dict(cpu_ig_ql=ig, cpu_bco_per_iter=it, I_tm=max(1, round(ig / it)))
        print(f"[calib] {n}x{m}: ig_ql {ig:.3f}s, bco {it:.3f}s/iter -> I_tm={cal[f'{n}x{m}']['I_tm']}", flush=True)
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    classes = [(20, 6)] if args.smoke else CLASSES
    out = OUT.with_name("sota_sensitivity_entrada90_SMOKE.csv") if args.smoke else OUT
    if CALIB.exists() and not args.smoke:
        cal = json.loads(CALIB.read_text())
        print(f"[calib] reaproveitando {CALIB.name}")
    else:
        cal = calibrate(classes)
        if not args.smoke:
            CALIB.write_text(json.dumps(cal, indent=1))

    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(f)}
    tasks = []
    for n, m in classes:
        for rep in range(2 if args.smoke else n_reps(n, m)):
            for method in METHODS:
                if (f"rand_{n}j_{m}m", method, str(rep)) in done:
                    continue
                tasks.append(dict(n=n, m=m, rep=rep, method=method, I_tm=cal[f"{n}x{m}"]["I_tm"]))
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
                if i % 50 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
