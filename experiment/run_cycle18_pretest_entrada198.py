"""
experiment/run_cycle18_pretest_entrada198.py
==============================================
Entrada 198 (Ciclo 18, frente C): SONDAS do PRE-TESTE de folga/ruido (pmsp.budgetfair.validated_to_noise) no P||Cmax, SO' em instancias de AJUSTE: por instancia, 3 sementes do controle aleatorio e 3 sementes de cada acao fixa (T = 1000), nos espacos S5 e S15.
  syn  as 12 classes sinteticas, 30 instancias de ajuste por classe (as mesmas das Entradas 180 e 185: sementes 90000 + rep).
  pub  as 525 instancias publicadas de ajuste (manifest_calib_entrada115.csv, sem as 28 de soma >= 2^62), com o grupo (familia, tercil de n) de results/pub_groups_entrada189.json.
Nenhum dado de teste e' lido. Fases: --syn ESPACO | --pub ESPACO. Saidas: results/pretest_syn_ESPACO_entrada198.csv e pretest_pub_ESPACO_entrada198.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from experiment.run_cycle16_entrada185 import make, write  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402

R = ROOT / "results"
N_TUNE_SYN, N_SEED = 30, 3


def run_syn_one(space, p, a0, m, od, lb, n, agent, fixed, seed):
    st, d, ls = arm_table(space, n)
    best, _, ev, used, _ = H.ig_agent_core3(p, a0, m, st, d, ls, 1000, agent, fixed, 0.1, 0.1, seed, od, lb, 0, 0, 0.0)
    return int(best), float(ev)


def syn(space):
    P.warm()
    A = len(arm_table(space, 100)[0])
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(N_TUNE_SYN):
            inst, p, a0, od = make(n, m, rep, True)
            lb = inst.lower_bound
            for s in range(N_SEED):
                seed0 = 7_000_000 + 1000 * ci + rep + 50000 * s + (3_000_000 if space == "S15" else 0)
                mk, ev = run_syn_one(space, p, a0, m, od, lb, n, H.AGENT_RANDOM, 0, seed0)
                rows.append(dict(space=space, cls=cls, rep=rep, method="rand", seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
                for a in range(A):
                    mk, ev = run_syn_one(space, p, a0, m, od, lb, n, H.AGENT_FIXED, a, seed0 + 100_000 * (a + 1))
                    rows.append(dict(space=space, cls=cls, rep=rep, method=f"arm{a}", seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
        print(f"[pretest syn {space}] {cls} ok", flush=True)
    write(R / f"pretest_syn_{space}_entrada198.csv", rows, True)


def pub(space):
    P.warm()
    info = json.loads(P.groups_path().read_text())
    cal = P.load_set(P.M_CAL)
    A = len(arm_table(space, 100)[0])
    rows = []
    t0 = time.perf_counter()
    for i, rec in enumerate(cal):
        fam, g = info["calib"][rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        lb, a0, od = P.prepare_inst(rec)
        for s in range(N_SEED):
            mk, ev, _ = P.run(space, rec, lb, a0, od, H.AGENT_RANDOM, 0, 0.1, 0.1, P.seed_of(space, "calib", fam, rec["inst_id"], "rand", s))
            rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method="rand", seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
            for a in range(A):
                mk, ev, _ = P.run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, P.seed_of(space, "calib", fam, rec["inst_id"], f"arm{a}", s))
                rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=f"arm{a}", seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
        if i % 100 == 0:
            print(f"[pretest pub {space}] {i}/{len(cal)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    write(R / f"pretest_pub_{space}_entrada198.csv", rows, True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", default=None)
    ap.add_argument("--pub", default=None)
    a = ap.parse_args()
    if a.syn:
        syn(a.syn)
    if a.pub:
        pub(a.pub)
