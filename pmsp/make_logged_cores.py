"""
pmsp/make_logged_cores.py (Entrada 287, Ciclo 54)
===================================================
Gera pmsp/ig_logged.py (P||Cmax) e pmsp/ur_logged.py (R||Cmax) por TRANSFORMACAO EXATA DE TEXTO de pmsp/ig_ties.py e pmsp/ur_ties.py (que mantem os seus hashes congelados), como a Entrada 279 fez com os nucleos anteriores.
A unica mudanca: o nucleo registra, a cada decisao t, a acao escolhida (log_a), o estado (log_s, 0 a 80), o makespan corrente antes da candidata (log_c), a recompensa BRUTA max(melhora, 0) (log_r, antes de qualquer
reescala de rmode) e se a linha da tabela Q do estado estava toda em zero na hora da decisao (log_z; 1 = linha zerada), e os devolve ao final. Nenhuma chamada ao gerador aleatorio e' adicionada ou removida: com a mesma semente o
nucleo registrado devolve o mesmo (best, n_imp, ev, used, hit2, Q) que o original, bit a bit (tests/test_logged_cores_entrada287.py).
Cada trecho substituido tem de aparecer EXATAMENTE uma vez no original; senao o gerador aborta.
Uso: python -m pmsp.make_logged_cores
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

HEADER = '''"""
{name} (Entrada 287, Ciclo 54): GERADO por pmsp/make_logged_cores.py a partir de {src} por transformacao exata de texto. NAO EDITAR a mao.
Mesmo nucleo, com o registro por decisao de (acao, estado, makespan corrente, recompensa bruta, linha Q zerada); devolve os seis valores do original mais (log_a, log_s, log_c, log_r, log_z).
"""
'''

DECL_OLD = "    used = np.zeros(A, dtype=np.int64)\n"
DECL_NEW = DECL_OLD + (
    "    log_a = np.zeros(T, dtype=np.int64)\n"
    "    log_s = np.zeros(T, dtype=np.int64)\n"
    "    log_c = np.zeros(T, dtype=np.float64)\n"
    "    log_r = np.zeros(T, dtype=np.float64)\n"
    "    log_z = np.zeros(T, dtype=np.int64)\n"
)
USED_OLD = "        used[a] += 1\n"
USED_NEW = USED_OLD + (
    "        log_a[t] = a\n"
    "        log_s[t] = s\n"
    "        zr = 1\n"
    "        for kk in range(A):\n"
    "            if Q[s, kk] != 0.0:\n"
    "                zr = 0\n"
    "        log_z[t] = zr\n"
)
R_OLD = "        r = float(cur - cm) if cur > cm else 0.0\n"
R_NEW = R_OLD + "        log_c[t] = cur\n        log_r[t] = r\n"
RET_OLD = "    return best, n_imp, ev, used, hit2, Q\n"
RET_NEW = "    return best, n_imp, ev, used, hit2, Q, log_a, log_s, log_c, log_r, log_z\n"


def transform(src_name: str, out_name: str, fn_old: str, fn_new: str) -> None:
    src = (HERE / src_name).read_text(encoding="utf-8")
    body = src[src.index('"""', 3) + 3:].lstrip("\n")           # descarta o docstring original
    for old, new in ((DECL_OLD, DECL_NEW), (USED_OLD, USED_NEW), (R_OLD, R_NEW), (RET_OLD, RET_NEW), (fn_old, fn_new)):
        assert body.count(old) == 1, (src_name, old.strip()[:60], body.count(old))
        body = body.replace(old, new)
    out = HEADER.format(name=f"pmsp/{out_name}", src=f"pmsp/{src_name}") + body
    (HERE / out_name).write_text(out, encoding="utf-8")
    print("gerado", out_name, len(out.splitlines()), "linhas")


if __name__ == "__main__":
    transform("ig_ties.py", "ig_logged.py", "def ig_agent_core_ties(", "def ig_agent_core_logged(")
    transform("ur_ties.py", "ur_logged.py", "def ur_agent_core_ties(", "def ur_agent_core_logged(")
