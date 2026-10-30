"""Gera entrega/OSF_REGISTRO_PREVISOES.md: um documento unico com as hipoteses/previsoes pre-registradas de cada bateria (o bloco 'PRE-REGISTRO' da docstring de cada relatorio), os hashes md5 do codigo congelado antes de cada bateria e a data do primeiro commit do arquivo de hashes. Serve para o REGISTRO PUBLICO (OSF) que o artigo diz nao ter: o registro e' hoje um diario com hashes sob controle de versao; este documento, ao ser depositado no OSF, passa a ter carimbo de terceiros daqui em diante (nao retroage). Uso: python analysis/make_osf_registry.py"""
import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "entrega" / "OSF_REGISTRO_PREVISOES.md"


def first_commit(path):
    try:
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%aI %h", "--", str(path.relative_to(ROOT)).replace("\\", "/")], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
        return out[-1] if out else "not yet committed"
    except Exception:
        return "unknown"


rows = []
for f in sorted((ROOT / "analysis").glob("report_entrada*.py")):
    m = re.match(r"report_entrada(\d+)_", f.name)
    if not m:
        continue
    try:
        doc = ast.get_docstring(ast.parse(f.read_text(encoding="utf-8")))
    except SyntaxError:
        continue
    if not doc or "PRE-REGISTRO" not in doc:
        continue
    i = doc.index("PRE-REGISTRO")
    head = " ".join(doc[:i].split())[:600]
    block = doc[i:]
    j = block.find("\nUso:")
    block = (block[:j] if j > 0 else block).strip()
    rows.append((int(m.group(1)), f, head, block))

L = ["# Registry of pre-specified hypotheses and predictions", "",
     "Project: *Budget-fair attribution of the gains of learning-guided Iterated Greedy* (P||Cmax, R||Cmax, permutation flowshop, genetic algorithm).", "",
     "Each section below reproduces, verbatim from the repository, the pre-registration block of one analysis script (hypotheses, decision rules and probabilistic predictions written before the corresponding battery was run), followed by the md5 hashes of the code and frozen inputs recorded before the run and the date and commit in which the hash file first entered the repository. "
     "**Limitation, stated plainly:** until this document is deposited in a public registry the ordering 'hashes before data' rests on the research diary and on file times, not on a third-party timestamp; from the deposit date on, this document is timestamped by the registry and does not apply retroactively. "
     "Predictions are scored as held / not held in the paper (Table I and the 'registered predictions' paragraph of the Discussion); the analysis scripts print the outcome of each prediction.", "",
     f"Sections: {len(rows)}.", ""]
for n, f, head, block in rows:
    h = ROOT / "results" / f"entrada{n}_prereg_hashes.txt"
    L += [f"## Entry {n} - `{f.name}`", "", f"> {head}", "", "```text", block, "```", ""]
    if h.exists():
        L += [f"Hash file `results/{h.name}` (first commit: {first_commit(h)}):", "", "```text", h.read_text(encoding="utf-8").strip(), "```", ""]
    else:
        L += ["Hash file: none recorded for this entry (the script itself is under version control).", ""]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"{OUT} - {len(rows)} sections, {OUT.stat().st_size // 1024} KB")
