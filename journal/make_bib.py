"""Gera journal/refs.bib: a UNIAO das entradas verificadas de paper/refs.bib (manuscrito anterior) e de cec/refs_cec.bib (artigo do CEC, que acrescenta as referencias das Entradas 18x-25x), sem duplicar chaves (vale a primeira de paper/refs.bib). Uso: python journal/make_bib.py"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
seen, out = set(), []
for src in (ROOT / "paper" / "refs.bib", ROOT / "cec" / "refs_cec.bib", ROOT / "journal" / "refs_extra.bib"):
    if not src.exists():
        continue
    text = src.read_text(encoding="utf-8")
    for m in re.finditer(r"^@\w+\{([^,\s]+),.*?^\}\s*$", text, flags=re.S | re.M):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(0))
(ROOT / "journal" / "refs.bib").write_text("\n\n".join(out) + "\n", encoding="utf-8")
print(len(out), "entradas")
