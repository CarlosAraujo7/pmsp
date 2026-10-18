"""
experiment/run_pfsp_taillard_entrada159.py
============================================
Entrada 159 (Ciclo 12, frente A): REPLICACAO do estudo do flowshop (Entrada 152) nas instancias CLASSICAS de Taillard (1993): ta041-ta050 (50 x 10), ta051-ta060 (50 x 20), ta071-ta080 (100 x 10) e ta081-ta090 (100 x 20),
40 instancias, regeneradas do gerador e das sementes publicados (external_benchmarks/pfsp_taillard/generate_taillard.py; verificadas contra a matriz da ta001 impressa na fonte). TUDO congelado da Entrada 152:
os 10 metodos, T = 300, o d validado por tamanho, os limiares de estado e as razoes de CPU (T_ils, T_nols), todos obtidos em instancias GERADAS de ajuste (results/pfsp_choice_entrada152.json); nada e' reajustado.
seed da execucao = 5000 + numero da instancia de Taillard. Saida: results/pfsp_taillard_entrada159.csv (rep = indice dentro do bloco de 10; numero Taillard = primeiro + rep). Retomavel.
Uso: python -m experiment.run_pfsp_taillard_entrada159 --run --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_benchmarks" / "pfsp_taillard"))

import experiment.run_pfsp_entrada152 as P  # noqa: E402

FIRST = {(50, 10): 41, (50, 20): 51, (100, 10): 71, (100, 20): 81}
OUT = ROOT / "results" / "pfsp_taillard_entrada159.csv"
INST_DIR = ROOT / "external_benchmarks" / "pfsp_taillard" / "instances"


def taillard_instance(si, rep):
    from generate_taillard import read_instance
    from pmsp.pfsp import PfsInstance
    n, m = P.SIZES[si]
    num = FIRST[(n, m)] + rep
    p = np.array(read_instance(INST_DIR / f"ta{num:03d}.txt"), dtype=np.int64).T      # (n, m)
    assert p.shape == (n, m)
    return PfsInstance(p, name=f"ta{num:03d}"), num


def _task(t):
    inst, num = taillard_instance(t["si"], t["rep"])
    n, m = P.SIZES[t["si"]]
    ch = json.loads(P.CHOICE.read_text())
    mt = t["method"]
    T = P.T_IG
    if mt == "ils_eq":
        T = ch["T_ils"][f"{n}x{m}"]
    elif mt == "ig_nols_eq":
        T = ch["T_nols"][f"{n}x{m}"]
    ms, cpu = P.solve(inst, mt, T, 5000 + num, thr=ch["thr"])
    return dict(n=n, m=m, rep=t["rep"], method=mt, makespan=int(ms), cpu_time=cpu, T=T, lb=inst.lower_bound), None


def run(workers):
    tasks = [dict(si=si, rep=r, method=mt) for si in range(len(P.SIZES)) for r in range(10) for mt in P.METHODS]
    P._drive(_task, tasks, OUT, workers, 40)
    print("concluido ->", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.run:
        run(a.workers)


if __name__ == "__main__":
    main()
