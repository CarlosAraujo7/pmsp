"""
experiment/run_cycle21_cffine_entrada228.py
=============================================
Entrada 228 (Ciclo 21, frente A): o ORACULO COM VALIDACAO CRUZADA DE SEMENTES NOS TEMPOS FINOS p ~ U[1,10^4] (a revisao cega #4 notou que a folga de uma semente dessa bateria, 0,142 e 0,175, nao tem a versao cf e que ali nenhum aprendiz ganha: um contraexemplo nao colocado).
Mesmo desenho da Entrada 209: cada acao fixa recebe 3 sementes NOVAS (1, 2, 3) nas mesmas 1.200 instancias de teste da bateria de tempos finos da Entrada 185 (12 classes x 100; make(n, m, rep, False, high=10000)); com a semente 0 ja' existente em results/fine_ESPACO_entrada185.csv, a acao do oraculo e' escolhida nas sementes {0, 1} e avaliada nas {2, 3} (analysis/report_entrada228_cf_fine.py). T = 1000; S5 (5 acoes) e S15 (15 acoes).
Fase: --space ESPACO. Saida: results/cfo_fine_ESPACO_entrada228.csv (espaco, cls, rep, arm, seed_i, makespan). Pre-registro: analysis/report_entrada228_cf_fine.py (docstring; hashes em results/entrada228_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from experiment.run_cycle16_entrada185 import make  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402

R = ROOT / "results"
SEEDS = (1, 2, 3)
FIELDS = ["space", "cls", "rep", "arm", "seed_i", "makespan"]


def seed_cf(space, key, arm, s):
    return (zlib.crc32(key.encode()) % 1000003 * 137 + s * 7919 + arm * 104729 + ["S5", "S15"].index(space) * 32452843 + 11 * 49979687) % (2 ** 31 - 1)


def fine(space):
    P.warm()
    A = len(arm_table(space, 100)[0])
    out = R / f"cfo_fine_{space}_entrada228.csv"
    rows = []
    t0 = time.perf_counter()
    for n, m in CLASSES:
        cls = f"rand_{n}j_{m}m"
        st, d, ls = arm_table(space, n)
        for rep in range(100):
            inst, p, a0, od = make(n, m, rep, False, 10000)
            lb = inst.lower_bound
            for a in range(A):
                for s in SEEDS:
                    best, _, ev, used, _ = H.ig_agent_core3(p, a0, m, st, d, ls, 1000, H.AGENT_FIXED, a, 0.1, 0.1, seed_cf(space, f"{cls}#{rep}", a, s), od, lb, 0, 0, 0.0)
                    rows.append(dict(space=space, cls=cls, rep=rep, arm=a, seed_i=s, makespan=int(best)))
        print(f"[cfo fine {space}] {cls} ({time.perf_counter() - t0:.0f}s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    print(f"[cfo fine {space}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, choices=["S5", "S15"])
    fine(ap.parse_args().space)
