"""
experiment/run_cycle34_mechanism_entrada258.py (Entrada 258, Ciclo 34, frente 3)
Diagnostico do MECANISMO do resultado inesperado da Entrada 256: em B30 o Q-Learning padrao perde para o aleatorio (ate -0,94 pontos) e avalia 0,72x as solucoes do aleatorio.
Hipoteses (registradas em analysis/report_entrada258_mechanism.py ANTES de rodar): o Q-Learning padrao inicia a tabela em zero, a recompensa e' a melhoria bruta (>= 0) e o desempate do argmax escolhe a menor
indice; apos o decaimento de epsilon, em estados nao visitados (81 estados x 30 acoes em mil passos) ele repete a acao 0 (a menor destruicao, sem busca local). Mede-se o USO por acao.
Instancias: as de TESTE com idx = 2 (uma por celula tipo x m x n: 140), espacos B5, B15, B30, metodos rand, ql, ucb, pg, alns, 2 sementes; saida results/mech_{espaco}_entrada258.csv (colunas used_0..used_29).
Uso: python -m experiment.run_cycle34_mechanism_entrada258 --space B30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle32_fanjul_entrada256 as F  # noqa: E402

R = F.R
UA = F.UA
METH = {"rand": UA.AGENT_RANDOM, "ql": UA.AGENT_Q, "ql_rev": UA.AGENT_Q, "ucb": UA.AGENT_UCB, "pg": UA.AGENT_PG, "alns": UA.AGENT_ALNS}   # ql_rev: o mesmo Q-Learning com a ORDEM das acoes invertida (teste causal do desempate)


def main2(space):
    """Etapa 2 (registrada depois do resultado da etapa 1): o Q-Learning padrao com a acao VALIDADA da classe movida para o indice 0 (ql_v0), o aleatorio e a validada, mesmas instancias e sementes."""
    man = F.manifest()
    man = man[(man.role == "test") & (man.idx == 2)]
    val = json.loads((R / f"fr_choice_{space}_entrada256.json").read_text())["val"]
    F.warm()
    rows = []
    t0 = time.perf_counter()
    for _, r in man.iterrows():
        P, a0, od, lb = F.load(r)
        m_, n_ = int(r.m), int(r.n)
        st, d, ls = F.arm_table_b(space, n_)
        A = len(st)
        v = val[f"{r['type']}|{n_}"]
        perm = np.array([v] + [k for k in range(A) if k != v])
        for me in ("rand", "ql", "ql_v0", "val"):
            for s in range(2):
                seed = F.seed_of(space, int(r.ri), "ql" if me == "ql_v0" else me, s, False) + 777
                if me == "ql_v0":
                    best, _, ev, used, _ = UA.ur_agent_core(P, a0, m_, st[perm].copy(), d[perm].copy(), ls[perm].copy(), F.T, UA.AGENT_Q, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
                    uo = np.zeros(A, dtype=np.int64)
                    uo[perm] = used
                    used = uo
                elif me == "val":
                    best, _, ev, used, _ = UA.ur_agent_core(P, a0, m_, st, d, ls, F.T, UA.AGENT_FIXED, v, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
                else:
                    best, _, ev, used, _ = UA.ur_agent_core(P, a0, m_, st, d, ls, F.T, METH[me], 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
                row = dict(space=space, type=r["type"], m=m_, n=n_, method=me, seed_i=s, makespan=int(best), bks=int(r.best_known), evals=float(ev), val_arm=v)
                row.update({f"used_{k}": int(used[k]) for k in range(A)})
                rows.append(row)
    pd.DataFrame(rows).to_csv(R / f"mech2_{space}_entrada258.csv", index=False)
    print(f"[mech2 {space}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


def main(space):
    man = F.manifest()
    man = man[(man.role == "test") & (man.idx == 2)]
    F.warm()
    rows = []
    t0 = time.perf_counter()
    for _, r in man.iterrows():
        P, a0, od, lb = F.load(r)
        m_, n_ = int(r.m), int(r.n)
        st, d, ls = F.arm_table_b(space, n_)
        for me, ag in METH.items():
            lr, lr2 = (F.PG_LR[space], 0.1) if me == "pg" else (F.ALNS_HP if me == "alns" else (0.1, 0.1))
            for s in range(2):
                seed = F.seed_of(space, int(r.ri), "ql" if me == "ql_rev" else me, s, False) + 777      # ql_rev usa as MESMAS sementes do ql (pareamento)
                if me == "ql_rev":
                    stx, dx, lsx = st[::-1].copy(), d[::-1].copy(), ls[::-1].copy()
                else:
                    stx, dx, lsx = st, d, ls
                best, _, ev, used, _ = UA.ur_agent_core(P, a0, m_, stx, dx, lsx, F.T, ag, 0, lr, lr2, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
                if me == "ql_rev":
                    used = used[::-1]       # volta ao indice original das acoes
                row =dict(space=space, type=r["type"], m=m_, n=n_, method=me, seed_i=s, makespan=int(best), bks=int(r.best_known), evals=float(ev))
                row.update({f"used_{k}": int(used[k]) for k in range(len(used))})
                rows.append(row)
    pd.DataFrame(rows).to_csv(R / f"mech_{space}_entrada258.csv", index=False)
    print(f"[mech {space}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--stage", type=int, default=1)
    a = ap.parse_args()
    (main if a.stage == 1 else main2)(a.space)
