"""
experiment/run_cycle24_irace_entrada244.py
============================================
Entrada 244 (Ciclo 24): O IRACE COMO CONFIGURADOR. A critica dos revisores cegos (#4-#8: 'o competidor estatico e' caseiro; faltam um configurador estabelecido (irace) e o Q-Learning ajustado por ele') vira dois experimentos, no mesmo protocolo:
 (a) 'cfg': o irace (Lopez-Ibanez et al. 2016; R, pacote irace 4.5, configuracoes padrao) ajusta a CONFIGURACAO ESTATICA do IG em cada uma das 12 classes, nas 30 instancias de ajuste da classe: cfg5 = tamanho de destruicao d/n continuo em [0,05; 0,40] (estrategia aleatoria, o espaco S5);
     cfg15 = estrategia {0, 1, 2} x d/n continuo em [0,05; 0,40] (o espaco S15). O espaco do irace contem a grade das acoes validadas (d/n em {0,05; ...; 0,40}) e todos os pontos entre elas;
 (b) 'q': o irace ajusta, por classe e por espaco (q5, q15), os quatro hiperparametros do Q-Learning de estados (alfa em [0,01; 0,9] em escala log, gama em [0; 0,99], decaimento do epsilon por passo em [0,98; 0,9999], piso do epsilon em [0; 0,5]).
Orcamento (execucoes do IG de T = 1000 iteracoes; custo = lacuna ao limite inferior em %): bm vezes o custo de ajuste da acao validada na mesma classe e espaco (pc: 1 semente por (instancia, acao): 150 (S5) e 450 (S15); ur: 2 sementes: 300 e 900), com bm em {1, 3, 10}; o principal e' bm = 1 ('a custo de ajuste igual').
Hospedeiros: 'pc' (P||Cmax sintetico, Entradas 180/209) e 'ur' (R||Cmax, Entrada 234). O alvo roda em Python (numba; os nucleos congelados) e o irace (em R) fala com ele por um soquete local (experiment/irace_target_entrada244.R).
Teste: a melhor configuracao do irace por classe, nas 100 instancias de teste x 5 sementes, T = 1000, contra o aleatorio, a acao validada e o Q-Learning padrao dos arquivos existentes. Metodos: cfg5_b1, cfg5_b3, cfg5_b10, cfg15_*, q5_*, q15_*.
Fases (retomaveis): --pilot (so' instancias de AJUSTE; depura sem tocar nas de teste) | --tune HOSPEDEIRO | --run HOSPEDEIRO. Saidas: results/irace_tune_HOSPEDEIRO_entrada244.jsonl, results/irace_test_HOSPEDEIRO_entrada244.csv.
Pre-registro: analysis/report_entrada244_irace.py (docstring; hashes em results/entrada244_prereg_hashes.txt).
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

from experiment import run_cycle22_ur_entrada234 as UR  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from experiment.run_agents_numba_entrada180 import make as make_pc  # noqa: E402
from pmsp import ig_numba_agents4 as G4  # noqa: E402
from pmsp import ur_numba_agents as UA  # noqa: E402

R = ROOT / "results"
RSCRIPT = os.environ.get("IRACE_RSCRIPT", r"C:\Users\vinic\R\R-4.6.1\bin\Rscript.exe")
RLIB = os.environ.get("R_LIBS_USER", r"C:\Users\vinic\R\library")
RFILE = ROOT / "experiment" / "irace_target_entrada244.R"
N_TUNE, N_TEST, N_SEED, T = 30, 100, 5, 1000
KINDS = ["cfg5", "cfg15", "q5", "q15"]
BMS = [1, 3, 10]
HOSTS = {"pc": 0, "ur": 1}
RKIND = {"cfg5": "cfg5", "cfg15": "cfg15", "q5": "q", "q15": "q"}
SPACE = {"cfg5": "S5", "cfg15": "S15", "q5": "S5", "q15": "S15"}
MIN_BUDGET = {"cfg5": 0, "cfg15": 0, "q5": 180, "q15": 180}   # o irace exige pelo menos 180 experimentos com quatro parametros
BASE = {("pc", "S5"): 150, ("pc", "S15"): 450, ("ur", "S5"): 300, ("ur", "S15"): 900}
FIELDS = ["host", "space", "cls", "rep", "method", "seed_i", "makespan"]


@functools.lru_cache(maxsize=None)
def instance(host, ci, rep, tune):
    n, m = CLASSES[ci]
    if host == "pc":
        inst, p, a0, od = make_pc(n, m, rep, tune)
        return p, a0, od, inst.lower_bound
    return UR.make(n, m, rep, ci, tune)


def run_cfg(host, kind, ci, rep, tune, params, seed):
    """Uma execucao do IG com a configuracao `params` (cfg: (estrategia, d/n); q: (alfa, gama, decaimento, piso)); devolve o makespan."""
    n, m = CLASSES[ci]
    X, a0, od, lb = instance(host, ci, rep, tune)
    seed = int(seed) % 2_147_483_647
    if kind.startswith("cfg"):
        st = np.array([int(round(params[0]))], dtype=np.int64)
        d = np.array([max(2, int(round(params[1] * n)))], dtype=np.int64)
        ls = np.zeros(1, dtype=np.int64)
        agent, hp = 0, G4.HP_DEFAULT
    else:
        st, d, ls = arm_table(SPACE[kind], n)
        agent, hp = 2, np.array([params[0], params[1], params[2], params[3], 0.0])
    if host == "pc":
        out = G4.ig_agent_core4(X, a0, m, st, d, ls, T, agent, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 0)
    else:
        out = UA.ur_agent_core(X, a0, m, st, d, ls, T, agent, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 0)
    return int(out[0])


def cost(host, kind, ci, rep, seed, params):
    lb = instance(host, ci, rep, True)[3]
    return 100.0 * (run_cfg(host, kind, ci, rep, True, params, seed) - lb) / lb


def warm():
    for host in HOSTS:
        for kind, prm in (("cfg15", (1, 0.2, 0, 0)), ("q5", (0.1, 0.9, 0.995, 0.05))):
            run_cfg(host, kind, 0, 0, True, prm, 1)


def tune_job(job):
    try:
        return _tune_job(job)
    except Exception:
        import traceback
        return dict(error=traceback.format_exc(), job=list(job))


def _tune_job(job):
    host, kind, ci, bm, pilot = job
    base = BASE[(host, SPACE[kind])]
    budget = max(base * bm, MIN_BUDGET[kind])
    seed = 1 + 1000 * ci + 100 * bm + 10 * KINDS.index(kind) + 5 * HOSTS[host]
    tag = f"{host}_{kind}_c{ci}_b{bm}" + ("_pilot" if pilot else "")
    work = R / "_irace_work"
    work.mkdir(exist_ok=True)
    out = work / f"{tag}.csv"
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(120)
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
        n = 0
        while True:
            line = f.readline()
            if not line:
                break
            rep, sd, *p = line.rstrip("\r\n").split("\t")
            c = cost(host, kind, ci, int(rep), int(sd), [float(x) for x in p])
            f.write(f"{c:.10g}\n")
            f.flush()
            n += 1
        proc.wait()
    conn.close()
    srv.close()
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"irace falhou em {tag} (veja {work / (tag + '.log')})")
    import pandas as pd
    el = pd.read_csv(out)
    best = el.iloc[0]
    names = ["strat", "dfrac"] if kind == "cfg15" else ["dfrac"] if kind == "cfg5" else ["alpha", "gamma", "decay", "floor"]
    return dict(host=host, kind=kind, ci=ci, bm=bm, budget=budget, n_exp=n, secs=round(time.perf_counter() - t0, 1), n_elites=len(el), best={k: float(best[k]) for k in names},
                elites=[{k: float(r[k]) for k in names} for _, r in el.iterrows()])


def _init():
    warm()


def tune(host, pilot=False):
    path = R / f"irace_tune_{host}_entrada244{'_pilot' if pilot else ''}.jsonl"
    done = set()
    if path.exists():
        for ln in path.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["kind"], r["ci"], r["bm"]))
    if pilot:
        jobs = [(host, k, ci, bm, True) for k in KINDS for ci in (0, 7) for bm in (1,)]
    else:
        jobs = [(host, k, ci, bm, False) for k in KINDS for ci in range(len(CLASSES)) for bm in BMS]
    jobs = [j for j in jobs if (j[1], j[2], j[3]) not in done]
    print(f"[irace {host}] {len(jobs)} jobs", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(int(os.environ.get("IRACE_NPROC", "10")), initializer=_init) as pool, open(path, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(tune_job, jobs), 1):
            if "error" in r:
                print(f"[irace {host}] FALHOU {r['job']}:\n{r['error']}", flush=True)
                continue
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 10 == 0 or pilot:
                print(f"[irace {host}] {i}/{len(jobs)} ({time.perf_counter() - t0:.0f}s) {r['kind']} c{r['ci']} b{r['bm']}: {r['best']} exp={r['n_exp']} {r['secs']}s", flush=True)


def seed_of(host, kind, bm, ci, rep, s):
    return 700_000_000 + 100_000_000 * HOSTS[host] + 10_000_000 * KINDS.index(kind) + 1_000_000 * BMS.index(bm) + 20_000 * ci + 5 * rep + s


def test_job(arg):
    host, ci, choice = arg
    n, m = CLASSES[ci]
    cls = f"rand_{n}j_{m}m"
    rows = []
    for kind in KINDS:
        for bm in BMS:
            cfg = choice[(kind, ci, bm)]
            if kind == "cfg5":
                prm = (0, cfg["dfrac"])
            elif kind == "cfg15":
                prm = (cfg["strat"], cfg["dfrac"])
            else:
                prm = (cfg["alpha"], cfg["gamma"], cfg["decay"], cfg["floor"])
            for rep in range(N_TEST):
                for s in range(N_SEED):
                    rows.append(dict(host=host, space=SPACE[kind], cls=cls, rep=rep, method=f"{kind}_b{bm}", seed_i=s, makespan=run_cfg(host, kind, ci, rep, False, prm, seed_of(host, kind, bm, ci, rep, s))))
    return rows


def run_test(host):
    ch = {}
    for ln in (R / f"irace_tune_{host}_entrada244.jsonl").read_text().splitlines():
        r = json.loads(ln)
        ch[(r["kind"], r["ci"], r["bm"])] = r["best"]
    missing = [(k, c, b) for k in KINDS for c in range(len(CLASSES)) for b in BMS if (k, c, b) not in ch]
    assert not missing, f"faltam ajustes: {missing[:5]}"
    t0 = time.perf_counter()
    rows = []
    with mp.Pool(int(os.environ.get("IRACE_NPROC", "10")), initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(test_job, [(host, ci, ch) for ci in range(len(CLASSES))]), 1):
            rows.extend(r)
            print(f"[irace test {host}] {i}/{len(CLASSES)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    out = R / f"irace_test_{host}_entrada244.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=None, metavar="HOST")
    ap.add_argument("--tune", default=None, metavar="HOST")
    ap.add_argument("--run", default=None, metavar="HOST")
    a = ap.parse_args()
    if a.pilot:
        tune(a.pilot, pilot=True)
    if a.tune:
        tune(a.tune)
    if a.run:
        run_test(a.run)
