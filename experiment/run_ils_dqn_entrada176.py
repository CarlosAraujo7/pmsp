"""
experiment/run_ils_dqn_entrada176.py
======================================
Entrada 176 (Ciclo 14, frente I2): ACOES QUALITATIVAMENTE DIFERENTES. O limite restante do artigo e' que os espacos de acao testados (estrategias e tamanhos de destruicao) sao parecidos entre si. O ILS de P||Cmax escolhe entre QUATRO operadores de vizinhanca
(inter-troca, reinsercao, troca intra-maquina/permuta e movimento de bloco; pmsp/operators.py) com custos e efeitos muito diferentes (CPU por execucao de 0,15 s a 5,3 s com os operadores em laços). Aqui o DQN maior (pmsp/dqn.py, 2 x 64, replay, rede-alvo) e o LinUCB
escolhem o operador, nas 12 classes x 100 instancias (as mesmas do resto do estudo), com os operadores vetorizados (`best_fn=apply_best_fast`, resultados identicos aos dos operadores originais; tests/test_operators_fast.py).
Comparador aleatorio: results/ils_eq_fast_entrada162.csv, metodo ils_fast_eqT (ILS aleatorio vetorizado, T = 1000, as mesmas instancias e seeds). Braco validado: melhor operador UNICO por classe em 30 instancias de AJUSTE (seeds 90000+); o DQN tem o lr ajustado nas mesmas instancias.
Fases: --tune (lr do DQN) ; --arms-tune (4 operadores unicos) ; --choose ; --run (dqn_ils, linucb_ils, val_ils no teste) ; --calibrate (razao de CPU sequencial, 3 instancias por classe) ; --merge NOME. Fatias independentes: --shard i --nshards N. Retomavel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import config as CFG  # noqa: E402

R = ROOT / "results"
CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
LRS = [0.001, 0.003, 0.01]
N_TUNE, N_TEST, TUNE_SEED0 = 30, 100, 90000
CHOICE = R / "ilsdqn_choice_entrada176.json"
FP = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "lr", "arm"]


def out_path(name):
    return R / f"ilsdqn_{name}_entrada176.csv"


def inst_of(n, m, rep, tune):
    from pmsp.instance import PMSPInstance
    return PMSPInstance.random(n=n, m=m, seed=(TUNE_SEED0 + rep) if tune else (CFG.BASE_SEED + rep), name=f"rand_{n}j_{m}m")


def solve(method, n, m, rep, tune, lr=None, arm=None, T=1000):
    from pmsp.dqn import DQNSelector
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.ils import ils_run
    from pmsp.linucb import LinUCBSelector
    from pmsp.operators_fast import apply_best_fast
    from experiment.runner import make_ils_selector
    inst = inst_of(n, m, rep, tune)
    rseed = CFG.BASE_SEED + rep
    if method.startswith("dqn"):
        sel = DQNSelector(4, lr=lr, seed=rseed)
    elif method == "linucb_ils":
        sel = LinUCBSelector(4, seed=rseed)
    elif method in ("val_ils", "arm"):
        sel = FixedActionSelector(int(arm))
    elif method == "rand_ils":
        sel = make_ils_selector("ils_random", rseed)
    else:
        raise ValueError(method)
    r = ils_run(instance=inst, selector=sel, max_iter=T, seed=rseed, stagnation_window=CFG.STAGNATION_WINDOW, best_fn=apply_best_fast)
    return dict(instance=inst.name, method=method, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"], lr=lr, arm=arm)


def tasks(kind):
    ch = json.loads(CHOICE.read_text()) if CHOICE.exists() else {}
    out = []
    for n, m in CLASSES:
        c = f"{n}x{m}"
        if kind == "tune":
            out += [dict(method=f"dqn@{lr}", n=n, m=m, rep=r, tune=True, lr=lr) for r in range(N_TUNE) for lr in LRS]
        elif kind == "arms-tune":
            out += [dict(method=f"arm{a}", n=n, m=m, rep=r, tune=True, arm=a, _m="arm") for r in range(N_TUNE) for a in range(4)]
        else:
            for r in range(N_TEST):
                out += [dict(method="dqn_ils", n=n, m=m, rep=r, tune=False, lr=ch["lr"]), dict(method="linucb_ils", n=n, m=m, rep=r, tune=False), dict(method="val_ils", n=n, m=m, rep=r, tune=False, arm=ch["val"][c])]
    return out


def run(kind, shard, nshards):
    out = out_path(kind.replace("-", "_"))
    part = out.with_name(out.stem + f"_s{shard}.csv")
    done = set()
    for src in (out, part):
        if src.exists() and src.stat().st_size > 0:
            done |= {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(open(src, newline=""))}
    pend = [t for t in tasks(kind) if (f"rand_{t['n']}j_{t['m']}m", t["method"], str(t["rep"])) not in done]
    pend.sort(key=lambda t: -t["n"] * t["m"])
    pend = [t for k, t in enumerate(pend) if k % nshards == shard]
    print(f"[fatia {shard}/{nshards} {kind}] {len(pend)} pendentes -> {part.name}", flush=True)
    new = not part.exists()
    t0 = time.perf_counter()
    with open(part, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FP)
        if new:
            w.writeheader()
        for i, t in enumerate(pend, 1):
            row = solve(t.get("_m", t["method"]), t["n"], t["m"], t["rep"], t["tune"], t.get("lr"), t.get("arm"))
            row["method"] = t["method"]
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            if i % 100 == 0 or i == len(pend):
                print(f"  [fatia {shard} {kind} {i}/{len(pend)}] {time.perf_counter() - t0:.0f}s", flush=True)
    print("concluido fatia", shard, kind)


def choose():
    import pandas as pd
    ch = json.loads(CHOICE.read_text()) if CHOICE.exists() else {}
    d = pd.read_csv(out_path("tune"))
    ch["lr"] = float(d.groupby("lr").gap_pct.mean().idxmin())
    ch["lr_mean_gap"] = {str(k): float(v) for k, v in d.groupby("lr").gap_pct.mean().items()}
    a = pd.read_csv(out_path("arms_tune"))
    a["cls"] = a.n.astype(str) + "x" + a.m.astype(str)
    a["op"] = a.arm.astype(int)
    ch["val"] = {c: int(g.groupby("op").makespan.mean().idxmin()) for c, g in a.groupby("cls")}
    CHOICE.write_text(json.dumps(ch, indent=1))
    print(json.dumps(ch, indent=1))


def calibrate():
    """Razao de CPU (CPU do processo, sequencial) DQN/aleatorio e LinUCB/aleatorio por classe, 3 instancias; mediana das razoes."""
    import numpy as np
    ch = json.loads(CHOICE.read_text())
    res = {}
    solve("rand_ils", 20, 4, 0, False)
    for n, m in CLASSES:
        rs = {"dqn_ils": [], "linucb_ils": []}
        for rep in range(3):
            def cpu(method, **kw):
                t = time.process_time()
                solve(method, n, m, rep, False, **kw)
                return time.process_time() - t
            base = cpu("rand_ils")
            rs["dqn_ils"].append(cpu("dqn_ils", lr=ch["lr"]) / base)
            rs["linucb_ils"].append(cpu("linucb_ils") / base)
        res[f"{n}x{m}"] = {k: float(np.median(v)) for k, v in rs.items()}
        print(f"[calib] {n}x{m}: {res[f'{n}x{m}']}", flush=True)
    (R / "ilsdqn_cpu_ratio_entrada176.json").write_text(json.dumps(res, indent=1))


def merge(name):
    import pandas as pd
    out = out_path(name)
    parts = sorted(out.parent.glob(out.stem + "_s*.csv"))
    frames = [pd.read_csv(out, dtype={"instance": str})] if out.exists() and out.stat().st_size > 0 else []
    frames += [pd.read_csv(q, dtype={"instance": str}) for q in parts]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True).drop_duplicates(["instance", "method", "rep"])
    d.to_csv(out, index=False)
    print(f"merge {name} -> {out.name}: {len(d)} linhas de {len(parts)} fatias")


def main():
    ap = argparse.ArgumentParser()
    for k in ("tune", "arms-tune", "choose", "run", "calibrate"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--merge", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        run("tune", a.shard, a.nshards)
    if getattr(a, "arms_tune"):
        run("arms-tune", a.shard, a.nshards)
    if a.choose:
        choose()
    if a.run:
        run("run", a.shard, a.nshards)
    if a.calibrate:
        calibrate()
    if a.merge:
        merge(a.merge)


if __name__ == "__main__":
    main()
