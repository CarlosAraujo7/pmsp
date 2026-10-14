"""
experiment/run_external_confirm2_entrada115.py
================================================
Entrada 115: SEGUNDO teste confirmatorio externo, agora tambem com o ESTADO INFORMATIVO. A confirmacao da Entrada 90/91 (4.160 instancias)
usou o estado original (f1, f2 degenerados). Aqui:
  * instancias NUNCA executadas (nem manifest, nem holdout da Entrada 90, nem sobras): CALIBRACAO (K=4 por classe) e CONFIRMACAO (K=20
    por classe), classes (subpasta, m, n) das familias B, C, E ja' presentes em manifest.csv; escolha deterministica por indices
    igualmente espacados; disjuntas entre si. inst_id = 5000 + posicao (seed de execucao).
  * calibracao: ig_random com selecao uniforme nas instancias de CALIBRACAO, registrando f1 e f2 RELATIVOS ao valor inicial da execucao;
    os limiares sao os tercis 1/3 e 2/3 da razao, POOLED sobre todas as instancias de calibracao (nao depende da familia).
  * confirmacao: ig_ql_rel (estado relativo, limiares pooled), ig_ql (estado original) e ig_random, T=1000, partida LPT.
Saidas: external_benchmarks/pcmax_symmetry2025/manifest_{calib,confirm2}_entrada115.csv, results/state_calibration_external_entrada115.json,
results_external_confirm2_entrada115.csv.
Uso (da raiz): python -m experiment.run_external_confirm2_entrada115 --build | --calibrate --workers 12 | --run --workers 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))

FAMILIES = ["B", "C", "E"]
K_CAL = {"B": 4, "C": 3, "E": 4}
K_CONF = {"B": 20, "C": 14, "E": 20}
INST_ID0 = 5000
M_CAL = EXT / "manifest_calib_entrada115.csv"
M_CONF = EXT / "manifest_confirm2_entrada115.csv"
THR = ROOT / "results" / "state_calibration_external_entrada115.json"
OUT = EXT / "results_external_confirm2_entrada115.csv"
METHODS = ["ig_ql_rel", "ig_ql", "ig_random"]
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "family", "subfolder", "member"]


def build() -> None:
    from build_sample import evenly_spaced_indices
    from loader import list_instances
    used = set()
    for fn in ["manifest.csv", "manifest_holdout_entrada90.csv"]:
        used |= {r["member"] for r in csv.DictReader(open(EXT / fn, newline=""))}
    for fn in ["results_external_symmetry2025.csv", "results_holdout_entrada90.csv"]:
        used |= {r["member"] for r in csv.DictReader(open(EXT / fn, newline=""))}
    man = list(csv.DictReader(open(EXT / "manifest.csv", newline="")))
    classes = {(r["family"], r["subfolder"], int(r["m"]), int(r["n"])) for r in man if r["family"] in FAMILIES}
    cal_rows, conf_rows = [], []
    for fam in FAMILIES:
        groups = defaultdict(list)
        for r in list_instances(fam):
            groups[(fam, r.subfolder, r.m, r.n)].append(r)
        kc, kf = K_CAL[fam], K_CONF[fam]
        # B e E: classes do manifest com instancias suficientes nao vistas; C: as classes do manifest estao esgotadas (holdout da Entrada 90),
        # entao usam-se as 75 classes NAO amostradas antes (mesma familia/gerador, classes novas)
        keys = sorted(k for k in groups if (k in classes if fam != "C" else True))
        for key in keys:
            elig = sorted((r for r in groups[key] if r.member not in used), key=lambda r: r.member)
            if len(elig) < kc + kf:
                continue
            ci = evenly_spaced_indices(len(elig), kc)
            cal = [elig[i] for i in ci]
            rest = [r for i, r in enumerate(elig) if i not in set(ci)]
            for pos, r in enumerate(cal):
                cal_rows.append(dict(family=fam, member=r.member, subfolder=r.subfolder, m=r.m, n=r.n, inst_id=INST_ID0 + pos))
            for pos, gi in enumerate(evenly_spaced_indices(len(rest), kf)):
                r = rest[gi]
                conf_rows.append(dict(family=fam, member=r.member, subfolder=r.subfolder, m=r.m, n=r.n, inst_id=INST_ID0 + 100 + pos))
    a, b = {r["member"] for r in cal_rows}, {r["member"] for r in conf_rows}
    assert not (a & b) and not ((a | b) & used)
    for path, rows in [(M_CAL, cal_rows), (M_CONF, conf_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["family", "member", "subfolder", "m", "n", "inst_id"])
            w.writeheader()
            w.writerows(rows)
        by = defaultdict(int)
        for r in rows:
            by[r["family"]] += 1
        print(path.name, len(rows), dict(by))


def _load(manifest: Path):
    from loader import load_many
    rows = list(csv.DictReader(open(manifest, newline="")))
    out = []
    byfam = defaultdict(list)
    for r in rows:
        byfam[r["family"]].append(r["member"])
    data = {}
    for fam, mem in byfam.items():
        data.update(load_many(fam, mem))
    for r in rows:
        m, n, p = data[r["member"]]
        out.append(dict(family=r["family"], subfolder=r["subfolder"], member=r["member"], m=m, p=p, inst_id=int(r["inst_id"])))
    return out


def _calib_one(t):
    from experiment import config as CFG
    from experiment.runner import make_ils_selector
    from pmsp.instance import PMSPInstance
    import pmsp.ig as ig_mod
    inst = PMSPInstance(p=t["p"], m=t["m"], name="cal")
    orig = ig_mod.compute_features
    log = []

    def logger(solution, tt, max_iter, iters_no_improve, stagnation_window=50):
        v = orig(solution, tt, max_iter, iters_no_improve, stagnation_window)
        log.append((float(v[0]), float(v[1])))
        return v

    ig_mod.compute_features = logger
    try:
        seed = t["inst_id"]
        ig_mod.ig_run(instance=inst, selector=make_ils_selector("ig_random", seed), max_iter=1000, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW)
    finally:
        ig_mod.compute_features = orig
    a = np.array(log)
    f10, f20 = a[0, 0], a[0, 1]
    return np.column_stack([a[:, 0] / f10 if f10 > 0 else np.zeros(len(a)), a[:, 1] / f20 if f20 > 0 else np.zeros(len(a))])


def calibrate(workers: int) -> None:
    items = _load(M_CAL)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        arrs = list(ex.map(_calib_one, items, chunksize=4))
    rel = np.concatenate(arrs)
    q = [1 / 3, 2 / 3]
    thr = {"rel": {"f1": [float(v) for v in np.quantile(rel[:, 0], q)], "f2": [float(v) for v in np.quantile(rel[:, 1], q)]},
           "n_instances": len(items), "n_steps": int(len(rel)), "quantiles": q}
    THR.write_text(json.dumps(thr, indent=1))
    print("limiares pooled (rel):", thr["rel"])


def _run(t):
    from experiment import config as CFG
    from experiment.run_state_recalibration_entrada98 import BINS34, make_state_fn
    from experiment.runner import make_ils_selector
    from pmsp.bounds import DEFAULT_STATE_CONFIG
    from pmsp.ig import N_DESTROY_STRATEGIES, ig_run
    from pmsp.instance import PMSPInstance
    from pmsp.qlearning import QLearningAgent
    inst = PMSPInstance(p=t["p"], m=t["m"], name=f"ext2_{t['member']}")
    seed = t["inst_id"]
    mt = t["method"]
    kw = dict(instance=inst, max_iter=1000, seed=seed, stagnation_window=CFG.STAGNATION_WINDOW)
    if mt == "ig_ql_rel":
        thr = json.loads(THR.read_text())["rel"]
        fn = make_state_fn("rel", thr, {"f1": [0, 0, 0], "f2": [0, 0, 0], "states": set()})
        ag = QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                            epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=N_DESTROY_STRATEGIES)
        r = ig_run(selector=ag, state_fn=fn, **kw)
    else:
        r = ig_run(selector=make_ils_selector(mt, seed), **kw)
    return dict(instance=inst.name, method=mt, rep=t["inst_id"], n=inst.n, m=inst.m, lb=inst.lower_bound, makespan=r["makespan"],
                gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"], family=t["family"],
                subfolder=t["subfolder"], member=t["member"])


def run(workers: int, limit=None) -> None:
    items = _load(M_CONF)
    done = set()
    if OUT.exists():
        done = {(r["member"], r["method"]) for r in csv.DictReader(open(OUT, newline=""))}
    tasks = [dict(it, method=mt) for it in items for mt in METHODS if (it["member"], mt) not in done]
    if limit:
        tasks = tasks[:limit]
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)", flush=True)
    new = not OUT.exists()
    t0 = time.perf_counter()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 200 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.build:
        build()
    if a.calibrate:
        calibrate(a.workers)
    if a.run:
        run(a.workers, a.limit)
