"""
run_external_battery.py
========================
Roda a bateria "nucleo" de 13 metodos (mesma usada nas 4 classes
grandes sinteticas, Entrada 48 do diario) contra a amostra do
benchmark publico Symmetry 2025 (manifest.csv, gerado por
build_sample.py) -- execucao PARALELA (ProcessPoolExecutor).

Uso (rodar de dentro desta pasta):
    python3 build_sample.py                          # gera manifest.csv (uma vez)
    python3 run_external_battery.py                  # roda tudo, 10 workers
    python3 run_external_battery.py --limit 20        # smoke test rapido
    python3 run_external_battery.py --workers 4        # menos paralelismo
    python3 run_external_battery.py --methods lpt,multifit,ig_ql

Resultado: results_external_symmetry2025.csv, no mesmo diretorio.
Schema = results.csv oficial do projeto (instance, method, rep, n, m,
lb, makespan, gap_pct, cpu_time, n_improvements) + colunas extras de
proveniencia (family, subfolder, member, source). gap_pct usa sempre
o lower bound analitico padrao do projeto (instance.lower_bound) --
identico ao resto do results.csv, para as 4 familias igualmente. Ver
loader.py para o porque de NAO haver gap-vs-CPLEX automatico (C/F tem
cplex.csv/Results_iAF.xlsx no repositorio, mas o join exato por
instancia nao foi verificado com confianca -- ficam disponiveis pra
consulta manual em C.zip/F.zip/Results_iAF.xlsx).

Idempotente/resumavel: (instance, method, rep) ja presentes no CSV de
saida sao pulados -- pode interromper (Ctrl+C) e rodar de novo que
continua de onde parou. Escreve+flush+fsync a cada linha (mesmo
padrao de todos os scripts de extensao do projeto).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]   # .../PMSP_Learnheuristic
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

from loader import load_many  # noqa: E402

# Bateria "nucleo" -- identica a usada nas 4 classes grandes sinteticas
# (Entrada 48: run_large_ig_battery_entrada48.py + run_large_core_battery_entrada48.py).
# Ajuste aqui ou via --methods se quiser outro conjunto.
DEFAULT_METHODS = [
    "lpt", "multifit", "sa_li_perkins", "bco",
    "ils_random", "ils_ucb", "ils_ql", "ils_rf",
    "ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql", "ig_ql_flat",
]

MANIFEST_PATH = HERE / "manifest.csv"
DEFAULT_OUTPUT = HERE / "results_external_symmetry2025.csv"

FIELDNAMES = [
    "instance", "method", "rep", "n", "m", "lb",
    "makespan", "gap_pct", "cpu_time", "n_improvements",
    "family", "subfolder", "member", "source",
]


def _run_one_task(task: dict) -> dict:
    """
    Executado em processo worker -- por isso importa pmsp/experiment
    AQUI DENTRO (cada processo novo precisa do proprio import; em
    start method 'spawn' o modulo inteiro e reimportado do zero, o
    que ja cobre isso, mas o import local deixa explicito e evita
    depender de fork/copy-on-write).
    """
    from pmsp.instance import PMSPInstance
    from experiment.runner import run_single
    from experiment import config as CFG

    inst = PMSPInstance(p=task["p"], m=task["m"], name=task["instance_name"])
    r = run_single(inst, task["method"], task["inst_id"], max_iter=task["max_iter"])
    r.pop("selector", None)
    r.update({
        "family": task["family"],
        "subfolder": task["subfolder"],
        "member": task["member"],
        "source": "symmetry2025_gitlab",
    })
    return r


def build_tasks(manifest_rows, methods, max_iter):
    # Agrupa por familia pra abrir cada .zip so uma vez.
    by_family = defaultdict(list)
    for row in manifest_rows:
        by_family[row["family"]].append(row["member"])

    contents = {}
    for family, members in by_family.items():
        print(f"[load] {family}: lendo {len(members)} instancias do .zip...")
        contents[family] = load_many(family, members)

    tasks = []
    for row in manifest_rows:
        family = row["family"]
        member = row["member"]
        m_file, n_file, p = contents[family][member]
        instance_name = f"ext_{family}_{row['subfolder']}_{m_file}x{n_file}_{row['inst_id']}"
        for method in methods:
            tasks.append({
                "family": family,
                "subfolder": row["subfolder"],
                "member": member,
                "m": m_file,
                "p": p,
                "inst_id": int(row["inst_id"]),
                "instance_name": instance_name,
                "method": method,
                "max_iter": max_iter,
            })
    return tasks


def load_existing_keys(output_path: Path) -> set:
    keys = set()
    if output_path.exists():
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                keys.add((row["member"], row["method"]))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=10,
                     help="numero de processos paralelos (padrao: 10)")
    ap.add_argument("--methods", type=str, default=None,
                     help="lista separada por virgula (padrao: bateria nucleo de 13 metodos)")
    ap.add_argument("--max-iter", type=int, default=1000,
                     help="orcamento de iteracoes (padrao: 1000, igual ao resto do projeto)")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--manifest", type=str, default=str(MANIFEST_PATH),
                     help="manifest a executar (padrao: manifest.csv; Entrada 90 adicionou "
                          "esta opcao para rodar manifest_holdout_entrada90.csv)")
    ap.add_argument("--limit", type=int, default=None,
                     help="roda so as N primeiras tarefas (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                     help="so mostra quantas tarefas rodariam, nao executa nada")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERRO: {manifest_path} nao existe. Rode 'python3 build_sample.py' primeiro.")
        sys.exit(1)

    methods = args.methods.split(",") if args.methods else DEFAULT_METHODS
    output_path = Path(args.output)

    with open(manifest_path, newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    print(f"[manifest] {len(manifest_rows)} instancias em {manifest_path.name}")

    tasks = build_tasks(manifest_rows, methods, args.max_iter)
    print(f"[tasks] {len(tasks)} tarefas totais ({len(manifest_rows)} instancias x "
          f"{len(methods)} metodos)")

    existing = load_existing_keys(output_path)
    if existing:
        print(f"[resume] {len(existing)} tarefas ja concluidas em {output_path.name}, serao puladas")

    pending = [t for t in tasks
               if (t["member"], t["method"]) not in existing]
    print(f"[pending] {len(pending)} tarefas a rodar")

    if args.limit is not None:
        pending = pending[:args.limit]
        print(f"[limit] restrito as primeiras {len(pending)} tarefas (--limit {args.limit})")

    if args.dry_run:
        print("[dry-run] nada executado. Metodos:", methods)
        return

    if not pending:
        print("Nada a fazer -- todas as tarefas ja estao no CSV de saida.")
        return

    write_header = not output_path.exists()
    n_done = 0
    n_error = 0
    t0 = time.perf_counter()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one_task, t): t for t in pending}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    n_error += 1
                    print(f"  [ERRO] {t['instance_name']} / {t['method']}: {e}")
                    continue
                writer.writerow({k: r.get(k, "") for k in FIELDNAMES})
                f.flush()
                os.fsync(f.fileno())
                n_done += 1
                if n_done % 25 == 0 or n_done == len(pending):
                    elapsed = time.perf_counter() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - n_done) / rate if rate > 0 else float("inf")
                    print(f"  [{n_done}/{len(pending)}] {elapsed:.0f}s decorridos, "
                          f"{rate:.2f} tarefas/s, ETA {eta/60:.1f} min"
                          + (f"  ({n_error} erros)" if n_error else ""))

    elapsed = time.perf_counter() - t0
    print(f"\nConcluido: {n_done} tarefas em {elapsed:.1f}s ({n_error} erros) -> {output_path}")


if __name__ == "__main__":
    main()
