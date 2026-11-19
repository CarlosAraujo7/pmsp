"""
analysis/audit_journal_short.py
===============================
Auditoria da versao CONDENSADA do manuscrito (journal/main_short.tex e os arquivos journal/s_*.tex). A versao completa (journal/main.tex) e' auditada numero a numero contra os arquivos de resultados por analysis/audit_journal_claims.py;
esta auditoria garante que a versao condensada NAO introduz numero algum que a completa nao tenha: todo numero decimal e todo inteiro >= 100 do texto condensado (fora de comentarios, de \\label, \\ref, \\cite e \\input) tem de aparecer nos
arquivos da versao completa (secoes, tabelas, resumo, apendices). Inteiros pequenos (< 100) sem decimais sao contados mas nao conferidos (aparecem por acaso em qualquer texto). Falha (codigo de saida 1) se houver um numero estranho,
se um rotulo \\ref nao existir no arquivo condensado, ou se restar um marcador %%...%% de substituicao pendente.
Uso: python -m analysis.audit_journal_short
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
J = ROOT / "journal"
SHORT = ["abstract.tex", "s_intro.tex", "s_related.tex", "s_method.tex", "s_results.tex", "s_external.tex", "s_discussion.tex", "main_short.tex"]
SHORT_TABLES = ["tab_order_b.tex", "tab_granularity.tex", "tab_tsp.tex", "tab_tsplib.tex", "tab_tsp_warm.tex", "tab_cost.tex", "tab_ties.tex", "tab_scarce.tex", "tab_cvrp.tex"]
FULL_EXTRA = [ROOT / "cec" / "tab_further.tex", ROOT / "cec" / "tab_selector.tex"]
NUM = re.compile(r"(?<![\w\\])\d+(?:,\d{3})*(?:\.\d+)?")


def clean(text: str) -> str:
    text = re.sub(r"(?m)(?<!\\)%.*$", "", text)  # comentarios
    text = text.replace("{,}", ",")
    text = re.sub(r"\\(?:label|ref|cite|citep|citet|input|includegraphics|bibliography|bibliographystyle|hyphenation|newcommand|documentclass|usepackage|graphicspath)\*?(?:\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}|\\tabcolsep|width=[0-9.]+\\textwidth|\{@\{\}[^}]*\}", " ", text)
    text = re.sub(r"\{(?:[0-9.]+(?:cm|pt|em))\}", " ", text)
    return text


def numbers(text: str):
    return [m.group(0).replace(",", "") for m in NUM.finditer(clean(text))]


def main() -> int:
    full_files = [p for p in J.glob("*.tex") if not p.name.startswith("s_") and p.name not in ("main_short.tex",)] + FULL_EXTRA
    full = set()
    for p in full_files:
        full.update(numbers(p.read_text(encoding="utf-8")))
    bad, small, total = [], 0, 0
    for name in SHORT + SHORT_TABLES:
        p = J / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        for ln, line in enumerate(txt.splitlines(), 1):
            for tok in numbers(line):
                total += 1
                strict = "." in tok or int(tok) >= 100
                if not strict:
                    small += 1
                    continue
                if tok not in full:
                    bad.append((name, ln, tok, line.strip()[:110]))
    ok = True
    labels = set()
    refs = []
    for name in SHORT + SHORT_TABLES:
        p = J / name
        if p.exists():
            t = re.sub(r"(?m)(?<!\\)%.*$", "", p.read_text(encoding="utf-8"))
            labels.update(re.findall(r"\\label\{([^}]*)\}", t))
            refs += [(name, r) for m in re.findall(r"\\ref\{([^}]*)\}", t) for r in m.split(",")]
    miss = sorted({(n, r) for n, r in refs if r not in labels and not r.startswith("app:")})
    pend = [(n, i) for n in SHORT if (J / n).exists() for i, l in enumerate((J / n).read_text(encoding="utf-8").splitlines(), 1) if "%%" in l and re.search(r"%%[A-Z]+%%", l)]
    print(f"numeros conferidos: {total - small} (decimais e inteiros >= 100); inteiros pequenos nao conferidos: {small}")
    for b in bad:
        ok = False
        print("  NUMERO AUSENTE DA VERSAO COMPLETA:", b)
    for m in miss:
        ok = False
        print("  \\ref SEM \\label NA VERSAO CONDENSADA:", m)
    for n, i in pend:
        ok = False
        print("  MARCADOR PENDENTE:", n, i)
    print("AUDITORIA DA VERSAO CONDENSADA:", "OK" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
