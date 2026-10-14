"""
experiment/run_large_ig_battery_entrada48.py
=============================================
Bateria da familia IG (Iterated Greedy) + LPT nas 4 classes grandes
(150-300 tarefas) do plano aprovado -- Entrada 48 do diario.

Contexto: Entradas 19-22 diagnosticaram um "teto estrutural" (LPT ja
quase otimo em razao n/m alta) usando APENAS a familia ils_* (RandomSelector
dentro de ils_run). Mas ig_* usa um mecanismo de busca estruturalmente
diferente (destruir/reconstruir a cada iteracao, sem perturbation_threshold
-- ver experiment/runner.py, comentario em run_single) que NAO foi testado
nas classes grandes ate agora. Um piloto rapido (N=20) mostrou ig_random
melhorando sobre LPT de forma consistente e nunca pior nas 4 classes --
o oposto do padrao "quase-empate total" visto na familia ils_*. Este
script formaliza essa checagem com N=100 (mesmo rigor estatistico ja
usado no resto do projeto) para toda a familia IG (isolamento de
mecanismo + Q-Learning completo + as 2 ablacoes intermediarias).

Protocolo identico ao run_single oficial (experiment/runner.py),
CFG.MAX_ITER (1000, o mesmo orcamento das 6 classes ja validadas --
Entrada 19/20 ja descartaram orcamento como fator limitante nesta
escala). NAO usa perturbation_threshold=15 (o ajuste da Entrada 22):
esse parametro so afeta ils_*, nao tem efeito em ig_* (ver comentario
em run_single).

Saida: results/large_ig_battery_staging.csv -- schema IDENTICO a
results/results.csv (mesmos fieldnames), para permitir um append
trivial (cat) apos revisao, mas mantido separado ate entao para nao
tocar no arquivo oficial sem decisao explicita.

Idempotente/resumavel (existing_keys), flush+fsync por linha -- mesmo
padrao de todos os scripts de extensao anteriores do projeto.

Uso: python3 experiment/run_large_ig_battery_entrada48.py REP_START REP_END
"""
import csv, os, sys, time
sys.path.insert(0, ".")
from pmsp.instance import PMSPInstance
from experiment.runner import run_single
from experiment import config as CFG

STAGING_CSV = "results/large_ig_battery_staging.csv"
FIELDNAMES = ["instance", "method", "rep", "n", "m", "lb",
              "makespan", "gap_pct", "cpu_time", "n_improvements"]

TARGETS = [
    (150, 15, "rand_150j_15m"),
    (150, 30, "rand_150j_30m"),
    (250, 25, "rand_250j_25m"),
    (300, 20, "rand_300j_20m"),
]
METHODS = ["lpt", "ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql", "ig_ql_flat"]


def main():
    rep_start, rep_end = int(sys.argv[1]), int(sys.argv[2])

    existing_keys = set()
    write_header = not os.path.exists(STAGING_CSV)
    if not write_header:
        with open(STAGING_CSV, newline="") as f_exist:
            for row in csv.DictReader(f_exist):
                existing_keys.add((row["instance"], row["method"], row["rep"]))

    n_new = 0
    t0 = time.perf_counter()
    with open(STAGING_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())
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
          f"(rep {rep_start}-{rep_end}, {len(TARGETS)} classes x {len(METHODS)} metodos)")


if __name__ == "__main__":
    main()
