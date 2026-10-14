"""
build_holdout_sample.py
========================
Gera manifest_holdout_entrada90.csv: instancias do benchmark Symmetry
2025 que NUNCA foram executadas por este projeto, para o teste
CONFIRMATORIO pre-registrado em
analysis/report_entrada90_family_level_ql_effect.py (diario, Entrada 90).

Desenho (congelado antes de rodar qualquer instancia do holdout):
  * familias B, C, E apenas (F foi 100% amostrada -- nao ha F nao vista);
  * MESMAS classes (subfolder, m, n) ja presentes em manifest.csv -- o
    holdout testa a mesma distribuicao em instancias novas, nao uma
    distribuicao diferente;
  * por classe, ate K_PER_CLASS instancias que (i) nao estao em
    manifest.csv e (ii) nao tem NENHUMA linha em
    results_external_symmetry2025.csv (exclui tambem as sobras da
    amostra antiga de 640/1.500 instancias);
  * escolha determinista: indices igualmente espacados entre as
    instancias elegiveis ordenadas por nome de arquivo;
  * inst_id = 1000 + posicao (seed distinta das usadas na amostra
    original, inst_id 0-39).

Uso: python build_holdout_sample.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loader import list_instances  # noqa: E402
from build_sample import evenly_spaced_indices  # noqa: E402

FAMILIES = ["B", "C", "E"]
K_PER_CLASS = 30
INST_ID_OFFSET = 1000
OUT = HERE / "manifest_holdout_entrada90.csv"


def main() -> None:
    with open(HERE / "manifest.csv", newline="") as f:
        manifest = list(csv.DictReader(f))
    used = {r["member"] for r in manifest}
    with open(HERE / "results_external_symmetry2025.csv", newline="") as f:
        for r in csv.DictReader(f):
            used.add(r["member"])
    print(f"[seen] {len(used)} membros ja vistos (manifest + resultados)")

    classes = {(r["family"], r["subfolder"], int(r["m"]), int(r["n"])) for r in manifest
               if r["family"] in FAMILIES}
    rows = []
    for fam in FAMILIES:
        refs = list_instances(fam)
        groups = defaultdict(list)
        for r in refs:
            groups[(fam, r.subfolder, r.m, r.n)].append(r)
        n_cls = n_short = 0
        for key in sorted(k for k in classes if k[0] == fam):
            elig = sorted((r for r in groups[key] if r.member not in used), key=lambda r: r.member)
            n_cls += 1
            if len(elig) < K_PER_CLASS:
                n_short += 1
            for pos, gi in enumerate(evenly_spaced_indices(len(elig), K_PER_CLASS)):
                r = elig[gi]
                rows.append(dict(family=fam, member=r.member, subfolder=r.subfolder,
                                 m=r.m, n=r.n, inst_id=INST_ID_OFFSET + pos))
        print(f"[{fam}] {n_cls} classes, {n_short} com < {K_PER_CLASS} instancias elegiveis")

    members = [r["member"] for r in rows]
    assert len(members) == len(set(members)), "membros duplicados no holdout"
    assert not (set(members) & used), "holdout NAO e disjunto do que ja foi visto"
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "member", "subfolder", "m", "n", "inst_id"])
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(int)
    for r in rows:
        by[r["family"]] += 1
    print(f"Total: {len(rows)} instancias -> {OUT.name}  por familia: {dict(by)}")


if __name__ == "__main__":
    main()
