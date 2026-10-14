"""
experiment/run_large_core_battery_entrada48.py
================================================
Complementa run_large_ig_battery_entrada48.py: roda os 3 baselines
SOTA (multifit, sa_li_perkins, bco) e a familia ils_* original
(ils_random, ils_ucb, ils_ql, ils_rf) nas 4 classes grandes, fechando
a "bateria completa" pedida por Carlos na Entrada 46 -- ver
diario_pesquisa.md, Entrada 48, para a decisao de escopo (13 metodos
"nucleo": lpt+multifit+sa+bco+4 ils_* baseline+5 ig_*, excluindo a
familia de ablacao ils_pm/ap/sarsa/op*_only -- ja respondida nas
classes validadas, Entradas 26-30 -- e as variantes de orcamento
estendido/warm-start ils_ql_hi/rand_hi/ql_ws, que dependem de
MAX_ITER_BY_SIZE sem entrada para estas classes novas).

ils_* usa perturbation_threshold=15 (nao o default 50) -- o ajuste
validado na Entrada 22 para rand_150j_30m, e testado nesta rodada
(Entrada 48, piloto N=10) nas outras 3 classes sem prejuizo (nunca
pior, ganho pequeno/nao robusto onde o teto estrutural domina).
Monkeypatch local em CFG.PERTURBATION_THRESHOLD -- so afeta este
processo, nao o config.py em disco nem qualquer outro script.

lpt ja foi coletado por run_large_ig_battery_entrada48.py (mesma
seed/protocolo) -- nao repetido aqui.

Mesmo padrao idempotente/crash-safe (flush+fsync, existing_keys) dos
scripts anteriores. Saida no MESMO staging CSV para manter os 13
metodos "nucleo" juntos num unico arquivo antes da revisao/merge.

Uso: python3 experiment/run_large_core_battery_entrada48.py REP_START REP_END
"""
import csv, os, sys, time
sys.path.insert(0, ".")
from pmsp.instance import PMSPInstance
from experiment.runner import run_single
from experiment import config as CFG

CFG.PERTURBATION_THRESHOLD = 15  # so nesta rodada/processo -- ver docstring

STAGING_CSV = "results/large_ig_battery_staging.csv"
FIELDNAMES = ["instance", "method", "rep", "n", "m", "lb",
              "makespan", "gap_pct", "cpu_time", "n_improvements"]

TARGETS = [
    (150, 15, "rand_150j_15m"),
    (150, 30, "rand_150j_30m"),
    (250, 25, "rand_250j_25m"),
    (300, 20, "rand_300j_20m"),
]
METHODS = ["multifit", "sa_li_perkins", "bco", "ils_random", "ils_ucb", "ils_ql", "ils_rf"]


def main():
    rep_start, rep_end = int(sys.argv[1]), int(sys.argv[2])

    existing_keys = set()
    with open(STAGING_CSV, newline="") as f_exist:
        for row in csv.DictReader(f_exist):
            existing_keys.add((row["instance"], row["method"], row["rep"]))

    n_new = 0
    t0 = time.perf_counter()
    with open(STAGING_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for n, m, inst_name in TARGETS:
            for method in METHODS:
                for inst_id in range(rep_start, rep_end):
                    key = (inst_name, method, str(inst_id))
                    if key in existing_keys:
                        continue
                    seed = CFG.BASE_SEED + inst_id
                    inst = PMSPInstance.random(n=n, m=m, seed=seed, name=inst_name)
                    r = run_single(inst, method, inst_id, max_iter=CFG.MAX_ITER, q_init=None)
                    writer.writerow({k: r[k] for k in FIELDNAMES})
                    f.flush()
                    os.fsync(f.fileno())
                    n_new += 1
    print(f"Novo: {n_new} linhas, {time.perf_counter()-t0:.1f}s "
          f"(rep {rep_start}-{rep_end}, {len(TARGETS)} classes x {len(METHODS)} metodos, threshold={CFG.PERTURBATION_THRESHOLD})")


if __name__ == "__main__":
    main()
