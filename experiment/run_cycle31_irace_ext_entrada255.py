"""
experiment/run_cycle31_irace_ext_entrada255.py
===============================================
Entrada 255 (Ciclo 31, frente A): o IRACE como configurador (Entrada 244) nas FAMILIAS PUBLICADAS de R||Cmax (Entrada 254; pmsp/instance_ur_ext.py): para cada uma das 75 classes (familia x tamanho) e cada um dos quatro alvos (cfg5, cfg15: a configuracao estatica da IG, com tamanho de destruicao continuo e, em cfg15, a estrategia; q5, q15: os quatro hiperparametros do Q-Learning), o irace recebe como orcamento o custo de ajuste da acao validada da Entrada 254 (20 instancias de ajuste x 2 sementes x 5 ou 15 acoes = 200 ou 600 experimentos; piso de 180 para q), com as configuracoes padrao do irace; a melhor configuracao e' avaliada em 20 instancias de teste por classe x 5 sementes. So' o orcamento 1x (o comparador primario da Entrada 244).
Arquitetura e script R: os da Entrada 244 (experiment/irace_target_entrada244.R, sem mudanca): o irace em R fala por soquete local com este servidor Python. Variaveis de ambiente: IRACE_RSCRIPT, R_LIBS_USER, IRACE_NPROC.
Fases (retomaveis): --tune | --run. Saidas: results/irace_tune_ext_entrada255.jsonl, results/irace_test_ext_entrada255.csv. Pre-registro: analysis/report_entrada255_ext_competitors.py (docstring; hashes em results/entrada255_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle30_ur_external_entrada254 as X  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from pmsp import ur_numba_agents as UA  # noqa: E402

R = ROOT / "results"
RSCRIPT = os.environ.get("IRACE_RSCRIPT", r"C:\Users\vinic\R\R-4.6.1\bin\Rscript.exe")
RLIB = os.environ.get("R_LIBS_USER", r"C:\Users\vinic\R\library")
RFILE = ROOT / "experiment" / "irace_target_entrada244.R"
N_TUNE, N_TEST, N_SEED, T = X.N_TUNE, X.N_TEST, 5, 1000
KINDS = ["cfg5", "cfg15", "q5", "q15"]
RKIND = {"cfg5": "cfg5", "cfg15": "cfg15", "q5": "q", "q15": "q"}
SPACE = {"cfg5": "S5", "cfg15": "S15", "q5": "S5", "q15": "S15"}
MIN_BUDGET = {"cfg5": 0, "cfg15": 0, "q5": 180, "q15": 180}
BASE = {"S5": 200, "S15": 600}                      # custo de ajuste da acao validada: 20 x 2 x (5 ou 15)
CLASSES = X.CLASSES
FIELDS = ["host", "space", "cls", "rep", "method", "seed_i", "makespan"]


@functools.lru_cache(maxsize=None)
def instance(ci, rep, tune):
    return X.make(ci, rep, tune)


def run_cfg(kind, ci, rep, tune, params, seed):
    """Uma execucao do IG com a configuracao `params` (cfg: (estrategia, d/n); q: (alfa, gama, decaimento, piso)); devolve o makespan."""
    f, n, m = CLASSES[ci]
    P, a0, od, lb = instance(ci, rep, tune)
    seed = int(seed) % 2_147_483_647
    if kind.startswith("cfg"):
        st = np.array([int(round(params[0]))], dtype=np.int64)
        d = np.array([max(2, int(round(params[1] * n)))], dtype=np.int64)
        ls = np.zeros(1, dtype=np.int64)
        agent, hp = 0, UA.HP_DEFAULT
    else:
        st, d, ls = arm_table(SPACE[kind], n)
        agent, hp = 2, np.array([params[0], params[1], params[2], params[3], 0.0])
    out = UA.ur_agent_core(P, a0, m, st, d, ls, T, agent, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 0)
    return int(out[0])


def cost(kind, ci, rep, seed, params):
    lb = instance(ci, rep, True)[3]
    return 100.0 * (run_cfg(kind, ci, rep, True, params, seed) - lb) / lb


def warm():
    for kind, prm in (("cfg15", (1, 0.2, 0, 0)), ("q5", (0.1, 0.9, 0.995, 0.05))):
        run_cfg(kind, 0, 0, True, prm, 1)


def tune_job(job):
    try:
        return _tune_job(job)
    except Exception:
        import traceback
        return dict(error=traceback.format_exc(), job=list(job))


def _tune_job(job):
    kind, ci = job
    budget = max(BASE[SPACE[kind]], MIN_BUDGET[kind])
    seed = 1 + 1000 * ci + 10 * KINDS.index(kind) + 7
    tag = f"ext_{kind}_c{ci}"
    work = R / "_irace_work_ext"
    work.mkdir(exist_ok=True)
    out = work / f"{tag}.csv"
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    env = dict(os.environ, R_LIBS_USER=RLIB)
    t0 = time.perf_counter()
    with open(work / f"{tag}.log", "w") as lg:
        proc = subprocess.Popen([RSCRIPT, str(RFILE), str(port), RKIND[kind], str(budget), str(seed), str(out), str(N_TUNE)], stdout=lg, stderr=subprocess.STDOUT, env=env)
        srv.settimeout(1)
        conn = None
        for _ in range(120):
            try:
                conn, _a = srv.accept()
                break
            except socket.timeout:
                if proc.poll() is not None:
                    break
        if conn is None:
            proc.kill()
            raise RuntimeError(f"o R nao se conectou em {tag} (veja {work / (tag + '.log')})")
        conn.settimeout(None)
        f = conn.makefile("rw", newline="\n")
        n_exp = 0
        while True:
            line = f.readline()
            if not line:
                break
            rep, sd, *p = line.rstrip("\r\n").split("\t")
            c = cost(kind, ci, int(rep), int(sd), [float(x) for x in p])
            f.write(f"{c:.10g}\n")
            f.flush()
            n_exp += 1
        proc.wait()
    conn.close()
    srv.close()
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"irace falhou em {tag} (veja {work / (tag + '.log')})")
    import pandas as pd
    el = pd.read_csv(out)
    best = el.iloc[0]
    names = ["strat", "dfrac"] if kind == "cfg15" else ["dfrac"] if kind == "cfg5" else ["alpha", "gamma", "decay", "floor"]
    return dict(kind=kind, ci=ci, budget=budget, n_exp=n_exp, secs=round(time.perf_counter() - t0, 1), n_elites=len(el), best={k: float(best[k]) for k in names})


def _init():
    warm()


def tune():
    path = R / "irace_tune_ext_entrada255.jsonl"
    done = set()
    if path.exists():
        for ln in path.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["kind"], r["ci"]))
    jobs = [(k, ci) for k in KINDS for ci in range(len(CLASSES)) if (k, ci) not in done]
    print(f"[irace ext] {len(jobs)} jobs", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(int(os.environ.get("IRACE_NPROC", "10")), initializer=_init) as pool, open(path, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(tune_job, jobs), 1):
            if "error" in r:
                print(f"[irace ext] FALHOU {r['job']}:\n{r['error']}", flush=True)
                continue
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(f"[irace ext] {i}/{len(jobs)} ({time.perf_counter() - t0:.0f}s) {r['kind']} c{r['ci']}: {r['best']} exp={r['n_exp']} {r['secs']}s", flush=True)


def seed_of(kind, ci, rep, s):
    return 1_400_000_000 + 10_000_000 * KINDS.index(kind) + 20_000 * ci + 5 * rep + s


def test_job(arg):
    ci, choice = arg
    f, n, m = CLASSES[ci]
    cls = X.cls_name(f, n, m)
    rows = []
    for kind in KINDS:
        cfg = choice[(kind, ci)]
        if kind == "cfg5":
            prm = (0, cfg["dfrac"])
        elif kind == "cfg15":
            prm = (cfg["strat"], cfg["dfrac"])
        else:
            prm = (cfg["alpha"], cfg["gamma"], cfg["decay"], cfg["floor"])
        for rep in range(N_TEST):
            for s in range(N_SEED):
                rows.append(dict(host="ext", space=SPACE[kind], cls=cls, rep=rep, method=f"{kind}_b1", seed_i=s, makespan=run_cfg(kind, ci, rep, False, prm, seed_of(kind, ci, rep, s))))
    return rows


def run_test():
    ch = {}
    for ln in (R / "irace_tune_ext_entrada255.jsonl").read_text().splitlines():
        r = json.loads(ln)
        ch[(r["kind"], r["ci"])] = r["best"]
    missing = [(k, c) for k in KINDS for c in range(len(CLASSES)) if (k, c) not in ch]
    assert not missing, f"faltam ajustes: {missing[:5]}"
    t0 = time.perf_counter()
    rows = []
    with mp.Pool(int(os.environ.get("IRACE_NPROC", "10")), initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(test_job, [(ci, ch) for ci in range(len(CLASSES))]), 1):
            rows.extend(r)
            if i % 15 == 0:
                print(f"[irace ext test] {i}/{len(CLASSES)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    out = R / "irace_test_ext_entrada255.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.pilot:
        # depuracao: dois trabalhos, sem gravar
        warm()
        for kind, ci in (("cfg15", 0), ("q5", 40)):
            r = tune_job((kind, ci))
            print("pilot:", r if "error" in r else {k: r[k] for k in ("kind", "ci", "budget", "n_exp", "secs", "best")}, flush=True)
    if a.tune:
        tune()
    if a.run:
        run_test()
