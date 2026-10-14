"""
external_benchmarks/pcmax_symmetry2025/loader.py
==================================================
Utilitarios de carga para o benchmark publico do Symmetry 2025
(Ostojic, Ramljak, Urosevic et al., DOI 10.3390/sym17020178),
baixado de https://gitlab.com/pcmax-problem/pcmax-instances em
01/09/2026 -- ver diario_pesquisa.md, Entradas 49-51.

4 familias padronizadas (B, C, E, F), cada uma um .zip contendo
arquivos .dat de texto simples (uma instancia por arquivo).

Formato dos .dat (confirmado por leitura direta do codigo-fonte de
E_generator.py e por inspecao do conteudo real dos 4 .zip -- Entrada
51 do diario): os 2 primeiros tokens (separados por QUALQUER espaco
em branco -- B/C/E usam uma linha "m n"; F usa m e n em linhas
separadas, terminador \r\n) sao m e n; os n tokens seguintes sao os
tempos de processamento p_1..p_n. parse_dat_content() abaixo tokeniza
o arquivo inteiro (content.split(), que ja trata \n/\r\n/espacos
multiplos de forma uniforme) -- funciona para as 4 familias sem
logica condicional.

C e F trazem solucao de referencia (arquivo cplex.csv embutido no
proprio .zip, colunas: Class,NumOfJobs,NumOfMach,NumberOfExample,
ObjValue,time,CPLEXStatus -- e tambem Results_iAF.xlsx, solver
independente). A correspondencia exata entre um arquivo .dat
especifico e uma linha do cplex.csv (via a coluna NumberOfExample,
que nao segue um padrao obviamente decodificavel com confianca) NAO
foi verificada -- por isso load_reference_cplex_raw() abaixo devolve
os dados BRUTOS, agrupados por (subfolder, n, m), sem tentar casar
por instancia individual. gap_pct no CSV de saida usa sempre o lower
bound analitico padrao do projeto (instance.lower_bound, mesma
formula usada no resto do results.csv) -- consistente e ja
implementado, sem risco de join errado.

E e B NAO trazem solucao de referencia neste repositorio -- so os
geradores (B_generator.py, E_generator.py, nao usados aqui; usamos
os .dat ja prontos).

Nao extrai os .zip para arquivos soltos (extracao integral e lenta
demais atraves da ponte para o computador -- ver diario_pesquisa.md,
Entrada 51: uma tentativa anterior estourou o tempo limite no meio de
27.000 arquivos). Le tudo direto de dentro do .zip via o modulo
zipfile padrao do Python -- testado e funcionando.
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES = ["B", "C", "E", "F"]


@dataclass(frozen=True)
class ExternalInstanceRef:
    """Referencia leve a uma instancia (ainda nao carregada em memoria)."""
    family: str        # "B" | "C" | "E" | "F"
    member: str         # caminho dentro do .zip, ex "E/E3/05_017_U_100_200_00.dat"
    m: int
    n: int
    subfolder: str       # ex "E3", "C9", "FFNU", "10" (classe B), "FFU"


def zip_path(family: str) -> Path:
    p = HERE / f"{family}.zip"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} nao encontrado. Baixe o repositorio primeiro "
            f"(https://gitlab.com/pcmax-problem/pcmax-instances -- "
            f"ver diario_pesquisa.md, Entrada 51)."
        )
    return p


def parse_dat_content(content: str) -> tuple[int, int, list[int]]:
    """Parser unico e robusto para as 4 familias -- ver docstring do modulo."""
    tokens = content.split()
    m = int(tokens[0])
    n = int(tokens[1])
    p = [int(x) for x in tokens[2:2 + n]]
    if len(p) != n:
        raise ValueError(
            f"Esperava {n} tempos de processamento, encontrei {len(p)} "
            f"(m={m}, primeiros tokens={tokens[:4]!r})."
        )
    return m, n, p


def list_instances(family: str) -> list[ExternalInstanceRef]:
    """Lista todas as instancias .dat de uma familia sem carregar o vetor p inteiro."""
    refs = []
    with zipfile.ZipFile(zip_path(family)) as zf:
        for name in zf.namelist():
            if not name.endswith(".dat"):
                continue
            with zf.open(name) as fh:
                head = fh.read(64).decode("utf-8", errors="replace")
            tokens = head.split()
            if len(tokens) < 2:
                content = zf.read(name).decode("utf-8", errors="replace")
                tokens = content.split()
            m, n = int(tokens[0]), int(tokens[1])
            parts = name.split("/")
            subfolder = parts[1] if len(parts) > 2 else parts[0]
            refs.append(ExternalInstanceRef(family=family, member=name,
                                             m=m, n=n, subfolder=subfolder))
    return refs


def load_instance_array(family: str, member: str) -> tuple[int, int, list[int]]:
    """Le e parseia uma unica instancia direto de dentro do .zip."""
    with zipfile.ZipFile(zip_path(family)) as zf:
        content = zf.read(member).decode("utf-8", errors="replace")
    return parse_dat_content(content)


def load_many(family: str, members: list[str]) -> dict[str, tuple[int, int, list[int]]]:
    """
    Carrega varias instancias da MESMA familia numa unica abertura do
    .zip (mais eficiente que chamar load_instance_array em loop).
    Retorna {member: (m, n, p)}.
    """
    out = {}
    with zipfile.ZipFile(zip_path(family)) as zf:
        for member in members:
            content = zf.read(member).decode("utf-8", errors="replace")
            out[member] = parse_dat_content(content)
    return out


# ---------------------------------------------------------------------
# Solucoes de referencia BRUTAS (C e F apenas) -- ver aviso na docstring
# do modulo: NAO casadas por instancia individual, so agrupadas.
# ---------------------------------------------------------------------

def load_reference_cplex_raw(family: str) -> dict[str, list[dict]]:
    """
    Le todos os cplex.csv embutidos no .zip de C ou F.

    Retorna {subfolder: [linha_do_csv_como_dict, ...]}. Dict vazio
    para B/E (nao tem cplex.csv). USO SUGERIDO: inspecao manual /
    comparacao agregada por classe -- nao ha join automatico contra
    arquivos .dat individuais nesta biblioteca (ver aviso acima).
    """
    ref: dict[str, list[dict]] = defaultdict(list)
    if family not in ("C", "F"):
        return dict(ref)
    with zipfile.ZipFile(zip_path(family)) as zf:
        csv_members = [n for n in zf.namelist() if n.endswith("cplex.csv")]
        for member in csv_members:
            subfolder = member.split("/")[1]
            content = zf.read(member).decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(content))
            ref[subfolder].extend(reader)
    return dict(ref)


if __name__ == "__main__":
    # Smoke test manual: python3 loader.py
    for fam in FAMILIES:
        refs = list_instances(fam)
        ns = sorted(set(r.n for r in refs))
        print(f"{fam}: {len(refs)} instancias, n de {ns[0]} a {ns[-1]}, "
              f"{len(set((r.subfolder, r.m, r.n) for r in refs))} grupos (subfolder,m,n) distintos")
