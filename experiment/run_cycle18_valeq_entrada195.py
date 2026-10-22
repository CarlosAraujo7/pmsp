"""
experiment/run_cycle18_valeq_entrada195.py
============================================
Entrada 195 (Ciclo 18, frente A): a ACAO VALIDADA contra cada aprendiz a CPU IGUAL. A revisao cega apontou que 'validada vs aprendiz' (Entradas 180, 185, 188, 189, 190) foi comparada a ITERACOES iguais (T = 1000 para os dois), enquanto o artigo prefere a CPU igual como orcamento e a acao validada tem
seu proprio custo K_val (CPU da acao fixa validada sobre a do aleatorio, T = 1000). Aqui o K_val e' calibrado em sequencia (maquina em repouso) e a acao validada roda T_v = round(1000 K_aprendiz / K_val) iteracoes, de modo que consome a mesma CPU que o aprendiz em T = 1000 (K_aprendiz: as razoes ja' calibradas
nas Entradas 180, 185, 188 e 189). 5 sementes, 'val_eq_<aprendiz>' por instancia; nada e' reajustado (as acoes validadas e as razoes dos aprendizes sao as congeladas).
Baterias: publicadas, conjuntos descoberta (2.550) e held-out (3.950) nos espacos S5 e S15 (as acoes validadas por (familia, tercil de n) de results/pub_choice_*_entrada189.json); sinteticas, S5 e S15, 12 classes x 100 instancias (acoes validadas por classe de results/agents_choice_*_entrada180.json) e os nove aprendizes.
Fases (retomaveis): --calibrate | --run-pub CONJUNTO ESPACO --shard i --nshards n | --run-syn ESPACO. Pre-registro: analysis/report_entrada195_valeq.py (docstring, congelado antes de rodar; hashes em results/entrada195_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle17_replication_entrada190 as RP  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from experiment.run_cycle16_entrada185 import FIELDS, N_TEST, make, write  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402

R = ROOT / "results"
LEARNERS = P.LEARNERS


def kval_path():
    return R / "valeq_kval_entrada195.json"


def run_syn_one(space, p, a0, m, od, lb, n, agent, fixed, seed, T):
    st, d, ls = arm_table(space, n)
    best, _, ev, used, _ = H.ig_agent_core3(p, a0, m, st, d, ls, T, agent, fixed, 0.1, 0.1, seed, od, lb, 0, 0, 0.0)
    return int(best), float(ev)


def synthetic_learner_k(space, cls_key):
    """K do aprendiz sobre o aleatorio por classe: Entradas 180 (ql, ucb, pg), 185 (pm, ap, alns) e 188 (ts, dmab, race)."""
    k180 = json.loads((R / "agents_cpu_ratio_entrada180.json").read_text())[space][cls_key]
    k185 = json.loads((R / "aos_cpu_ratio_entrada185.json").read_text())[space][cls_key]
    k188 = json.loads((R / "new_cpu_ratio_entrada188.json").read_text())[space][cls_key]
    return {**k180, **k185, **k188}


def calibrate():
    info = json.loads(P.groups_path().read_text())
    cal = P.load_set(P.M_CAL)
    chs = {sp: json.loads(P.choice_path(sp).read_text()) for sp in ("S5", "S15")}
    P.warm()
    res = {"pub": {}, "syn": {}}
    for sp in ("S5", "S15"):
        res["pub"][sp] = {}
        for fam in P.FAMS:
            for g in range(3):
                mem = [r for r in cal if r["family"] == fam and info["calib"][r["member"]][1] == g]
                med = float(np.median([r["n"] for r in mem]))
                pick = sorted(mem, key=lambda r: (abs(r["n"] - med), r["inst_id"]))[:2]
                ratios = []
                for rec in pick:
                    lb, a0, od = P.prepare_inst(rec)
                    arm = chs[sp][fam]["val"][str(g)]

                    def cpu(agent, fixed):
                        reps, t0 = 0, time.process_time()
                        while time.process_time() - t0 < 1.0:
                            P.run(sp, rec, lb, a0, od, agent, fixed, 0.1, 0.1, reps + 7)
                            reps += 1
                        return (time.process_time() - t0) / reps
                    ratios.append(cpu(H.AGENT_FIXED, arm) / cpu(H.AGENT_RANDOM, 0))
                res["pub"][sp][f"{fam}|{g}"] = float(np.median(ratios))
                print(f"[valeq-calib pub {sp}] {fam} grupo {g}: K_val {res['pub'][sp][f'{fam}|{g}']:.2f} (acao {chs[sp][fam]['val'][str(g)]})", flush=True)
        ch = json.loads((R / f"agents_choice_{sp}_entrada180.json").read_text())
        res["syn"][sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            cls = f"rand_{n}j_{m}m"
            ratios = []
            for rep in range(2):
                inst, p, a0, od = make(n, m, rep, False)

                def cpu(agent, fixed):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        run_syn_one(sp, p, a0, m, od, inst.lower_bound, n, agent, fixed, reps + 7, 1000)
                        reps += 1
                    return (time.process_time() - t0) / reps
                ratios.append(cpu(H.AGENT_FIXED, ch["val"][cls]) / cpu(H.AGENT_RANDOM, 0))
            res["syn"][sp][f"{n}x{m}"] = float(np.median(ratios))
            print(f"[valeq-calib syn {sp}] {n}x{m}: K_val {res['syn'][sp][f'{n}x{m}']:.2f}", flush=True)
    kval_path().write_text(json.dumps(res, indent=1))


def run_pub(setname, space, shard, nshards):
    kv = json.loads(kval_path().read_text())["pub"][space]
    K = json.loads(P.k_path().read_text())[space]
    ch = json.loads(P.choice_path(space).read_text())
    if setname == "discovery":
        info = json.loads(P.groups_path().read_text())
        test = P.load_set(P.M_TEST)[shard::nshards]
        out = R / f"valeq_pub_discovery_{space}_s{shard}of{nshards}_entrada195.csv"
    else:
        info = json.loads(RP.groups_path().read_text())
        test = P.load_set(RP.M_HOLD)[shard::nshards]
        out = R / f"valeq_pub_heldout_{space}_s{shard}of{nshards}_entrada195.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    t00 = time.perf_counter()
    new = not out.exists()
    for i, rec in enumerate(test):
        fam, g = info["test"][rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        arm = ch[fam]["val"][str(g)]
        rows = []
        for s in range(P.N_SEED):
            for k in LEARNERS:
                Tv = max(50, int(round(1000 * K[f"{fam}|{g}"][k] / kv[f"{fam}|{g}"])))
                mk, ev, _ = P.run(space, rec, lb, a0, od, H.AGENT_FIXED, arm, 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], "valeq" + k, s), T=Tv)
                rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=f"val_eq_{k}", seed_i=s, makespan=mk, lb=lb, T=Tv, evals=ev))
        write(out, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[valeq pub {setname} {space} shard {shard}] {i}/{len(test)} ({time.perf_counter() - t00:.0f}s)", flush=True)
    print(f"[valeq pub {setname} {space} shard {shard}] concluido ({time.perf_counter() - t00:.0f}s)", flush=True)


def run_syn(space):
    kv = json.loads(kval_path().read_text())["syn"][space]
    ch = json.loads((R / f"agents_choice_{space}_entrada180.json").read_text())
    out = R / f"valeq_syn_{space}_entrada195.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    t00 = time.perf_counter()
    new = not out.exists()
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        K = synthetic_learner_k(space, f"{n}x{m}")
        for rep in range(N_TEST):
            if (cls, str(rep)) in done:
                continue
            inst, p, a0, od = make(n, m, rep, False)
            rows = []
            for s in range(5):
                for k in LEARNERS:
                    Tv = max(50, int(round(1000 * K[k] / kv[f"{n}x{m}"])))
                    mk, ev = run_syn_one(space, p, a0, m, od, inst.lower_bound, n, H.AGENT_FIXED, ch["val"][cls], 1_000_000 * (LEARNERS.index(k) + 31) + 50000 * s + 1000 * ci + rep + 20_000_000 * (60 + ["S5", "S15"].index(space)), Tv)
                    rows.append(dict(space=space, cls=cls, rep=rep, method=f"val_eq_{k}", seed_i=s, makespan=mk, lb=inst.lower_bound, T=Tv, evals=ev))
            write(out, rows, new)
            new = False
        print(f"[valeq syn {space}] {n}x{m} ok ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run-pub", nargs=2, default=None, metavar=("CONJUNTO", "ESPACO"))
    ap.add_argument("--run-syn", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.run_pub:
        run_pub(a.run_pub[0], a.run_pub[1], a.shard, a.nshards)
    if a.run_syn:
        run_syn(a.run_syn)
