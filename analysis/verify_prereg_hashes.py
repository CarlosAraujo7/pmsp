"""
analysis/verify_prereg_hashes.py
==================================
Verifica os hashes md5 congelados antes de cada bateria (results/entradaNNN_prereg_hashes.txt) IGNORANDO a diferenca de fim de linha: o hash vale o conteudo do arquivo, e o Windows (CRLF) e o git com .gitattributes (LF) so' diferem na quebra de linha.
Cada linha e' aceita se o md5 do arquivo bate com o registrado em uma destas tres leituras: bytes como estao, quebras convertidas para LF ou quebras convertidas para CRLF. Linhas que comecam com '#' sao comentarios.
Uso: python -m analysis.verify_prereg_hashes [padrao, ex.: entrada180]      (sem argumento: todos os arquivos de hashes)
Retorna 1 se algum arquivo listado mudou de conteudo ou esta' ausente.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv) -> int:
    pat = argv[1] if len(argv) > 1 else "entrada"
    bad = 0
    total = 0
    for h in sorted((ROOT / "results").glob("*prereg_hashes.txt")):
        if pat not in h.name:
            continue
        for ln in h.read_bytes().decode("utf-8").replace("\r\n", "\n").split("\n"):
            if not ln.strip() or ln.startswith("#") or " *" not in ln:
                continue
            rec, name = ln.split(" *", 1)
            f = ROOT / name
            total += 1
            if not f.exists():
                print(f"AUSENTE  {h.name}: {name}")
                bad += 1
                continue
            raw = f.read_bytes()
            lf = raw.replace(b"\r\n", b"\n")
            crlf = lf.replace(b"\n", b"\r\n")
            if rec not in {hashlib.md5(x).hexdigest() for x in (raw, lf, crlf)}:
                print(f"MUDOU    {h.name}: {name}")
                bad += 1
    print(f"{total - bad}/{total} hashes conferem (ignorando fim de linha)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
