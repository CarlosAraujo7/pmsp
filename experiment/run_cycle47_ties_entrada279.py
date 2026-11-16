"""
experiment/run_cycle47_ties_entrada279.py (Entrada 279, Ciclo 47)
Aprendizes SEM travamento na primeira acao e a ordem das acoes, no benchmark de Fanjul-Peyro e Ruiz (700 instancias de TESTE, B15 e B30, 3 sementes, T = 1.000; as mesmas instancias e os mesmos hiperparametros congelados da Entrada 256).
Motivo: as revisoes cegas de 15/11/2026 observaram que o Q-Learning 'invariante a ordem' do manuscrito (ql_perm) permuta as acoes UMA vez por execucao, o que fixa uma acao aleatoria, e nao desempata ao acaso a cada decisao; e que UCB1, o gradiente
de politica, a roleta, o Q-Learning episodico e o FRRMAB nao foram repetidos com a ordem permutada. Aqui:
  ql_rt     Q-Learning padrao (tabela inicial zero) com DESEMPATE ALEATORIO A CADA DECISAO (pmsp/ur_ties.py, tie_mode = 1)
  qo_rt     Q-Learning otimista (o 'qo' da Entrada 256) com desempate aleatorio a cada decisao
  qo_perm   o Q-Learning otimista com as acoes permutadas ao acaso em cada execucao (como o ql_perm)
  ucb_perm, pg_perm, alns_perm, iq_perm, fr_perm   UCB1, gradiente de politica, roleta ALNS, Q-Learning episodico e FRRMAB com as acoes permutadas em cada execucao.
Uso: python -m experiment.run_cycle47_ties_entrada279 --space B30 --shard i --nshards n [--pilot]     Saida: results/ties_fanjul_{espaco}_s{i}of{n}_entrada279.csv (type, instance, m, n, idx, method, seed_i, makespan, bks, evals)
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle32_fanjul_entrada256 as F  # noqa: E402
from pmsp import ur_ties as UT  # noqa: E402

R = F.R
T = F.T
VARIANTS = ["ql_rt", "qo_rt", "qo_perm", "ucb_perm", "pg_perm", "alns_perm", "iq_perm", "fr_perm"]
FIELDS = ["type", "instance", "m", "n", "idx", "method", "seed_i", "makespan", "bks", "evals"]
OFFSET = 41_000_000


def seed_of(space, ri, variant, s):
    return F.seed_of(space, ri, "ql", s, False) + OFFSET + 1_000_000 * VARIANTS.index(variant)


def run_variant(variant, space, P, a0, m, od, lb, n, seed, qo_cfg, pub_cfg):
    st, d, ls = F.arm_table_b(space, n)
    A = len(st)
    UA, U5, U6 = F.UA, F.UA5, F.U6
    if variant == "ql_rt":
        out = UT.ur_agent_core_ties(P, a0, m, st, d, ls, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0, np.zeros((81, A)), 1.0, 1)
        return int(out[0]), float(out[2])
    if variant == "qo_rt":
        c, a = qo_cfg
        hp = np.array([a, 0.9, 1.0, 0.05, 0.0])
        out = UT.ur_agent_core_ties(P, a0, m, st, d, ls, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 2, np.full((81, A), c), 0.05, 1)
        return int(out[0]), float(out[2])
    perm = np.random.default_rng(seed).permutation(A)
    st, d, ls = st[perm].copy(), d[perm].copy(), ls[perm].copy()
    if variant == "qo_perm":
        c, a = qo_cfg
        hp = np.array([a, 0.9, 1.0, 0.05, 0.0])
        out = U5.ur_agent_core5(P, a0, m, st, d, ls, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 2, np.full((81, A), c), 0.05)
        return int(out[0]), float(out[2])
    if variant in ("iq_perm", "fr_perm"):
        name = variant[:2]
        cfg = pub_cfg[name]
        if name == "iq":
            ag, lr, lr2, hp = 11, 0.1, float(cfg[0]), np.array([0.1, 0.9, cfg[1], 0.05, 0.5])
        else:
            ag, lr, lr2, hp = 12, cfg[0], cfg[1], U6.HP_DEFAULT
        best, _, ev, used, _ = U6.ur_agent_core6(P, a0, m, st, d, ls, T, ag, 0, lr, lr2, seed, od, lb, 0, 0, 0.0, hp, 0)
        return int(best), float(ev)
    name = variant[:-5]
    ag = {"ucb": UA.AGENT_UCB, "pg": UA.AGENT_PG, "alns": UA.AGENT_ALNS}[name]
    lr, lr2 = (F.PG_LR[space], 0.1) if name == "pg" else (F.ALNS_HP if name == "alns" else (0.1, 0.1))
    best, _, ev, used, _ = UA.ur_agent_core(P, a0, m, st, d, ls, T, ag, 0, lr, lr2, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
    return int(best), float(ev)


def warm(space):
    P = np.random.default_rng(1).integers(1, 50, size=(30, 4)).astype(np.int64)
    a0, od = F.constructive_ur(P)
    lb = float(F.lower_bound_ur(P))
    qo, pub = F.frozen(space)
    for v in VARIANTS:
        run_variant(v, space, P, a0, 4, od, lb, 30, 1, qo, pub)


def run(space, shard, nshards, pilot):
    man = F.manifest()
    man = man[man.role == ("tune" if pilot else "test")]      # a fumaca usa instancias de AJUSTE: nenhum resultado do teste e visto antes do registro
    qo, pub = F.frozen(space)
    out = R / (f"ties_pilot_{space}_entrada279.csv" if pilot else f"ties_fanjul_{space}_s{shard}of{nshards}_entrada279.csv")
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"]) for r in csv.DictReader(open(out, newline=""))}
    warm(space)
    new = not out.exists()
    t0, cnt = time.perf_counter(), 0
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard or (r["type"], str(int(r.instance))) in done:
            continue
        P, a0, od, lb = F.load(r)
        rows = []
        for v in VARIANTS:
            for s in range(F.N_SEED):
                mk, ev = run_variant(v, space, P, a0, int(r.m), od, lb, int(r.n), seed_of(space, int(r.ri), v, s), qo, pub)
                rows.append(dict(type=r["type"], instance=int(r.instance), m=int(r.m), n=int(r.n), idx=int(r.idx), method=v, seed_i=s, makespan=mk, bks=int(r.best_known), evals=ev))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if pilot and cnt >= 2:
            break
        if cnt % 10 == 0:
            print(f"[ties {space} s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[ties {space} s{shard}] concluido, {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    run(a.space, a.shard, a.nshards, a.pilot)
