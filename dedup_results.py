"""
dedup_results.py
=================
Remove duplicatas exatas de (instance, method, rep) em results/results.csv.

Contexto: auditoria de 29/08/2026 (ver diario_pesquisa.md, Entrada 1) encontrou
360 linhas duplicadas para ils_ql_ws e ils_ql_hi, causadas por reexecucoes
avulsas (--method X) apos o --extend original, sem guarda contra duplicata
(guarda adicionada em experiment/runner.py na Entrada 2).

Uso:
    python dedup_results.py --dry-run   # so reporta, nao escreve nada
    python dedup_results.py             # aplica, faz backup e sobrescreve
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "results.csv"

# Campos que devem ser IDENTICOS entre duplicatas para serem consideradas
# seguras de deduplicar automaticamente (cpu_time fica de fora: e uma
# medida de parede, pode variar levemente entre reexecucoes do mesmo seed).
STRICT_FIELDS = ["n", "m", "lb", "makespan", "gap_pct", "n_improvements"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Apenas reporta, nao modifica results.csv")
    parser.add_argument("--csv-path", default=str(CSV_PATH))
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERRO: {csv_path} nao encontrado.")
        return 1

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    groups: dict = defaultdict(list)
    for row in rows:
        key = (row["instance"], row["method"], row["rep"])
        groups[key].append(row)

    dup_keys = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"Total de linhas lidas: {len(rows)}")
    print(f"Combinacoes (instance, method, rep) unicas: {len(groups)}")
    print(f"Combinacoes com duplicata: {len(dup_keys)}")

    mismatches = []
    for key, dup_rows in dup_keys.items():
        first = dup_rows[0]
        for other in dup_rows[1:]:
            for field in STRICT_FIELDS:
                if first[field] != other[field]:
                    mismatches.append((key, field, first[field], other[field]))

    if mismatches:
        print(f"\nATENCAO: {len(mismatches)} divergencias encontradas entre "
              f"duplicatas. NAO prosseguindo automaticamente:")
        for key, field, v1, v2 in mismatches[:20]:
            print(f"  {key}: campo '{field}' difere ({v1!r} vs {v2!r})")
        return 2

    print("\nTodas as duplicatas encontradas sao identicas nos campos "
          f"{STRICT_FIELDS} (cpu_time e medida de parede, fica fora dessa "
          "checagem de proposito).")

    seen = set()
    deduped_rows = []
    removed = 0
    for row in rows:
        key = (row["instance"], row["method"], row["rep"])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped_rows.append(row)

    print(f"\nLinhas removidas: {removed}")
    print(f"Linhas restantes: {len(deduped_rows)}")

    if args.dry_run:
        print("\n--dry-run: nada foi escrito.")
        return 0

    backup_path = csv_path.with_name(csv_path.stem + "_raw_backup_20260829.csv")
    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        print(f"Backup do CSV original (com duplicatas) salvo em: {backup_path}")
    else:
        print(f"Backup ja existia em {backup_path}, nao sobrescrito.")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_rows)

    print(f"\n{csv_path} regravado sem duplicatas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
