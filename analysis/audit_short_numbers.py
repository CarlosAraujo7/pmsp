"""analysis/audit_short_numbers.py (Entrada 158)

Number-provenance audit of the 30-page version.  Every numeric token of the text of the short version
(paper/short/*.tex) must also occur in an audited source: the 46-page version (whose claims are audited
by audit_entrada1xx_*), the generated tables (paper/f*.tex, s_*.tex), the supplement, or the results of
the figure script (results/fig_headroom_rho.txt).  Tokens that occur only in the short version are
listed for manual review (they are either new arithmetic, e.g. a ratio, or a transcription error).

Usage: python analysis/audit_short_numbers.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "paper"

NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def strip_latex(t: str) -> str:
    t = re.sub(r"(?<!\\)%.*", "", t)  # comments only: an escaped \% is a percent sign
    t = re.sub(r"\\(?:ref|label|cite[a-z]*|input|includegraphics|externaldocument|bibliography[a-z]*|eqref)(?:\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\citep?\[[^\]]*\]\{[^}]*\}", " ", t)
    t = t.replace("{,}", "").replace("\\,", "")
    t = re.sub(r"\\texttt\{[^}]*\}", " ", t)
    t = re.sub(r"\\text(?:sc|bf|it)\{[^}]*\}", " ", t)
    t = re.sub(r"\\_", "_", t)
    return t


def tokens(t: str) -> set[str]:
    out = set()
    for m in NUM.findall(strip_latex(t)):
        out.add(m)
        if "." in m:
            out.add(m.rstrip("0").rstrip("."))
    return out


def read_all(paths):
    return "\n".join(Path(p).read_text(encoding="utf-8", errors="ignore") for p in paths)


short_files = sorted((P / "short").glob("*.tex"))
short_files = [f for f in short_files if f.name != "backmatter.tex"]
source_files = [P / "learnheuristic_pmsp_long.tex", P / "supplementary_material.tex"] + sorted(P.glob("f*.tex")) + sorted(P.glob("s_*.tex"))
source_files += [ROOT / "results" / "fig_headroom_rho.txt"] + sorted((ROOT / "results").glob("entrada159_*.txt"))
src = tokens(read_all(source_files))
# the short version may reformat 0.15 as 0.150 and so on; both are in `src` through rstrip

bad = {}
for f in short_files:
    toks = tokens(f.read_text(encoding="utf-8"))
    miss = sorted(t for t in toks if t not in src)
    if miss:
        bad[f.name] = miss

if not bad:
    print("OK: every numeric token of the short version occurs in an audited source")
else:
    total = 0
    for k, v in bad.items():
        print(f"{k}: {v}")
        total += len(v)
    print(f"{total} token(s) to review manually")
sys.exit(0)
