"""
data/generate_instances.py
==========================
Script responsável por:

    1. Tentar baixar as instâncias de Taillard (1993) do repositório
       oficial (http://mistic.heig-vd.ch/taillard/).
    2. Parsear o formato de texto desse repositório para PMSPInstance.
    3. Gerar instâncias aleatórias U[1,100] com os tamanhos definidos
       no desenho experimental.
    4. Salvar tudo em data/instances/ no formato JSON interno.

Uso rápido
----------
    cd PMSP_Learnheuristic/
    python -m data.generate_instances          # gera tudo
    python -m data.generate_instances --only-random  # só aleatórias

Nota sobre Taillard
-------------------
O PMSP de Taillard (1993) usa instâncias de *flow shop*, mas o benchmark
de máquinas paralelas deriva-se tomando apenas os tempos da primeira
operação de cada job (coluna 1 da matriz de tempos). Aqui implementamos
essa conversão E também oferecemos geração de instâncias puras PMSP
aleatórias para benchmarking.

Se o download falhar (sem internet), o script avisa e gera apenas as
instâncias aleatórias.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Ajuste de sys.path para importar pmsp/ a partir da raiz do projeto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance


# ===========================================================================
# Configurações do Desenho Experimental
# ===========================================================================

# Instâncias aleatórias: (n_jobs, n_machines, n_seeds)
RANDOM_CONFIGS = [
    # Pequeno porte
    (20,  2, 30),
    (20,  4, 30),
    # Médio porte
    (50,  4, 30),
    (50,  8, 30),
    # Grande porte
    (100, 5, 30),
    (100, 10, 30),
]

# URLs das instâncias de Taillard (máquinas paralelas uniformes)
# Taillard disponibiliza em texto simples, uma instância por bloco.
# O script abaixo usa os arquivos de uniform parallel machine (upm).
TAILLARD_URLS = {
    # Formato: (n_jobs, n_machines) → URL
    # As instâncias reais estão hospedadas como arquivos .txt
    "tai_20j_2m":  "http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/uniform_parallel_machine/tai20_2.txt",
    "tai_20j_4m":  "http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/uniform_parallel_machine/tai20_4.txt",
    "tai_50j_4m":  "http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/uniform_parallel_machine/tai50_4.txt",
    "tai_50j_10m": "http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/uniform_parallel_machine/tai50_10.txt",
    "tai_100j_5m": "http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/uniform_parallel_machine/tai100_5.txt",
    "tai_100j_10m":"http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/uniform_parallel_machine/tai100_10.txt",
}

OUT_DIR = ROOT / "data" / "instances"


# ===========================================================================
# Parser do formato Taillard
# ===========================================================================

def parse_taillard_block(text: str, name_prefix: str) -> list[PMSPInstance]:
    """
    Parseia um arquivo de instâncias Taillard (uniform parallel machines).

    Formato esperado (cada instância):
        número de jobs   número de máquinas
        p_1  p_2  ...  p_n    ← uma linha com n tempos de processamento
        (repetido para cada instância no arquivo)

    Retorna uma lista de PMSPInstance.
    """
    instances: list[PMSPInstance] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    inst_idx = 1
    while i < len(lines):
        try:
            header = lines[i].split()
            n, m = int(header[0]), int(header[1])
            i += 1
            # Coleta as linhas de tempos de processamento
            p_vals: list[int] = []
            while len(p_vals) < n and i < len(lines):
                p_vals.extend(int(x) for x in lines[i].split())
                i += 1
            if len(p_vals) >= n:
                inst = PMSPInstance(
                    p=p_vals[:n],
                    m=m,
                    name=f"{name_prefix}_{inst_idx:02d}",
                )
                instances.append(inst)
                inst_idx += 1
        except (ValueError, IndexError):
            i += 1   # linha mal-formada: pula
    return instances


# ===========================================================================
# Download das instâncias Taillard
# ===========================================================================

def download_taillard(out_dir: Path) -> list[PMSPInstance]:
    """
    Tenta baixar cada URL de Taillard. Se uma falhar, registra aviso e
    continua. Retorna lista de instâncias baixadas e parseadas.
    """
    all_instances: list[PMSPInstance] = []
    print("\n=== Baixando instâncias de Taillard (1993) ===")
    for name, url in TAILLARD_URLS.items():
        try:
            print(f"  → {name} ... ", end="", flush=True)
            with urllib.request.urlopen(url, timeout=15) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            insts = parse_taillard_block(text, name_prefix=name)
            all_instances.extend(insts)
            print(f"OK ({len(insts)} instâncias)")
        except Exception as exc:
            print(f"FALHOU ({exc})")
    return all_instances


# ===========================================================================
# Geração de instâncias aleatórias
# ===========================================================================

def generate_random_instances() -> list[PMSPInstance]:
    """
    Gera instâncias aleatórias U[1,100] conforme o desenho experimental.
    Cada configuração (n, m) é repetida com seeds 0..n_seeds-1 para
    garantir reprodutibilidade das 30 repetições experimentais.
    """
    instances: list[PMSPInstance] = []
    print("\n=== Gerando instâncias aleatórias U[1,100] ===")
    for n, m, n_seeds in RANDOM_CONFIGS:
        for seed in range(n_seeds):
            inst = PMSPInstance.random(n=n, m=m, seed=seed)
            instances.append(inst)
        print(f"  → {n} jobs / {m} máquinas: {n_seeds} instâncias geradas")
    return instances


# ===========================================================================
# Persistência
# ===========================================================================

def save_instances(instances: list[PMSPInstance], out_dir: Path) -> None:
    """
    Salva cada instância como arquivo JSON em out_dir/<name>.json.
    Também cria um índice instances/index.json com metadados.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []

    for inst in instances:
        path = out_dir / f"{inst.name}.json"
        inst.save(path)
        index.append({
            "name": inst.name,
            "n": inst.n,
            "m": inst.m,
            "lower_bound": inst.lower_bound,
            "file": path.name,
        })

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"\n✓ {len(instances)} instâncias salvas em '{out_dir}'")
    print(f"  Índice: {index_path}")


def load_all_instances(out_dir: Path | None = None) -> list[PMSPInstance]:
    """
    Carrega todas as instâncias listadas em data/instances/index.json.
    Função utilitária para uso nos experimentos (Fase 4).
    """
    if out_dir is None:
        out_dir = OUT_DIR
    index_path = out_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Índice não encontrado: {index_path}. "
            "Execute generate_instances.py primeiro."
        )
    index = json.loads(index_path.read_text())
    instances = []
    for meta in index:
        path = out_dir / meta["file"]
        instances.append(PMSPInstance.load(path))
    return instances


# ===========================================================================
# Ponto de entrada
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera e/ou baixa instâncias PMSP para o experimento."
    )
    parser.add_argument(
        "--only-random",
        action="store_true",
        help="Gera apenas instâncias aleatórias (sem tentar baixar Taillard).",
    )
    parser.add_argument(
        "--only-taillard",
        action="store_true",
        help="Apenas tenta baixar instâncias de Taillard.",
    )
    args = parser.parse_args()

    all_instances: list[PMSPInstance] = []

    if not args.only_random:
        tai = download_taillard(OUT_DIR)
        all_instances.extend(tai)
        if not tai:
            print(
                "\n⚠ Nenhuma instância de Taillard baixada. "
                "Verifique sua conexão ou os URLs no script."
            )

    if not args.only_taillard:
        rand = generate_random_instances()
        all_instances.extend(rand)

    if not all_instances:
        print("\n✗ Nenhuma instância gerada. Abortando.")
        sys.exit(1)

    save_instances(all_instances, OUT_DIR)

    # Resumo
    print(f"\n{'='*55}")
    print(f"  Total de instâncias: {len(all_instances)}")
    ns = sorted({inst.n for inst in all_instances})
    ms = sorted({inst.m for inst in all_instances})
    print(f"  Tamanhos de n: {ns}")
    print(f"  Tamanhos de m: {ms}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
