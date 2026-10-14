"""
data/materialize_instances.py
==============================
Materializa em disco (data/instances/) as instancias que ja sao usadas
implicitamente pelo experimento (geradas on-the-fly em experiment/runner.py
via PMSPInstance.random(n, m, seed=BASE_SEED+inst_id)).

Motivacao: LEI_GERAL_PESQUISA.md pede reprodutibilidade auditavel; gerar
sob demanda e reprodutivel (seed fixa) mas nao e *publicavel* como dado
bruto. Este script fecha essa lacuna sem inventar nenhum parametro novo —
importa INSTANCE_CONFIGS, N_INSTANCES e BASE_SEED diretamente de
experiment/config.py, entao nunca pode divergir da config vigente (evita
a classe de bug "config drift" ja identificada neste projeto em 29/08/2026).

Uso:
    cd PMSP_Learnheuristic/
    python -m data.materialize_instances

Nao sobrescreve nem interage com data/generate_instances.py (script
pre-existente, nao usado, que tenta baixar instancias de Taillard —
mantido intacto para decisao futura separada).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from experiment import config as CFG

OUT_DIR = ROOT / "data" / "instances"


def class_name(n: int, m: int) -> str:
    return f"rand_{n}j_{m}m"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "gerado_por": "data/materialize_instances.py",
        "fonte_config": "experiment/config.py",
        "base_seed": CFG.BASE_SEED,
        "n_instances_por_classe": CFG.N_INSTANCES,
        "distribuicao": "p_j ~ U[1,100] inteiro (PMSPInstance.random, low=1, high=100 default)",
        "formula_seed": "seed = BASE_SEED + inst_id, inst_id em range(N_INSTANCES)",
        "classes": [],
        "nota": (
            "Estas instancias sao EXATAMENTE as mesmas usadas para gerar "
            "results/results.csv (mesma formula de seed, mesma chamada "
            "PMSPInstance.random). Materializadas em disco para permitir "
            "Data Availability / reproducibilidade auditavel no EJOR."
        ),
    }

    total = 0
    for (n, m) in CFG.INSTANCE_CONFIGS:
        cname = class_name(n, m)
        class_dir = OUT_DIR / cname
        class_dir.mkdir(parents=True, exist_ok=True)
        for inst_id in range(CFG.N_INSTANCES):
            seed = CFG.BASE_SEED + inst_id
            inst = PMSPInstance.random(n=n, m=m, seed=seed)
            out_path = class_dir / f"rep{inst_id:02d}.json"
            inst.save(out_path)
            total += 1
        manifest["classes"].append({
            "classe": cname, "n": n, "m": m, "n_arquivos": CFG.N_INSTANCES,
        })
        print(f"  {cname:16s} -> {CFG.N_INSTANCES} arquivos em {class_dir}")

    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nTotal: {total} instancias materializadas em {OUT_DIR}")
    print(f"Manifest: {OUT_DIR / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
