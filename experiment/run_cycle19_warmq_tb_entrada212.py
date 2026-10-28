"""
experiment/run_cycle19_warmq_tb_entrada212.py
===============================================
Entrada 212 (Ciclo 19, frente B, ESTAGIO 2; registrado depois de ver o estagio 1 e antes de rodar): o Q aquecido do estagio 1 (Entrada 211) foi PIOR que o frio em S15 (-0,146 pontos). Diagnostico (sem dados de teste): a tabela pre-treinada e' quase toda zero (a recompensa e' esparsa) e a media de Q por acao e' igual entre acoes (0,09 +- 0,01); o argmax cai
na acao 0 em 65 dos 81 estados (S15) por EMPATE (o primeiro indice), e a acao 0 (tamanho de destruicao 0,05 n, a mais barata) e' ruim. Estagio 2: a MESMA tabela pre-treinada, mas com DESEMPATE ALEATORIO (ruido 1e-6 x uniforme somado a tabela inicial, sorteado por execucao), metodo wqn (adaptacao alfa 0,05, epsilon 0,05), e wqn0 (congelada).
Mesmas instancias (descoberta e held-out), S5 e S15, 5 sementes, T = 1000. Fases (retomaveis): --run CONJUNTO ESPACO --shard i --nshards n. Saidas: results/warmqtb_CONJUNTO_ESPACO_s*of*_entrada212.csv. Pre-registro: analysis/report_entrada212_warmq_tb.py (docstring; hashes em results/entrada212_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402
from experiment import run_cycle19_warmq_entrada211 as W  # noqa: E402

R = ROOT / "results"


def _append(out, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=W.FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def run_space(conj, space, shard, nshards):
    Q0 = np.load(W.q_path(space))
    recs, info = S.test_records(conj)
    recs = recs[shard::nshards]
    out = R / f"warmqtb_{conj}_{space}_s{shard}of{nshards}_entrada212.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        fam, g = info[rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        rows = []
        for meth, hp in (("wqn", W.HP_WQ), ("wqn0", W.HP_WQ0)):
            for s in range(S.N_SEED_TEST):
                sd = P.seed_of(space, "test", fam, rec["inst_id"], meth, s)
                Qn = Q0 + 1e-6 * np.random.default_rng(sd).random(Q0.shape)
                mk, ev, _ = W.run5(space, rec, lb, a0, od, sd, hp, Qn, 0.05)
                rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=meth, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
        _append(out, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[warmqtb {conj} {space} s{shard}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[warmqtb {conj} {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, default=None, metavar=("CONJUNTO", "ESPACO"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.run:
        run_space(a.run[0], a.run[1], a.shard, a.nshards)
