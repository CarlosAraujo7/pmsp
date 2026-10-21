"""Extrai de paper/refs.bib as entradas citadas no artigo do CEC e acrescenta as tres que so' ele cita (LinUCB, DQN, REINFORCE). Uso: python cec/make_bib.py"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "paper" / "refs.bib").read_text(encoding="utf-8")
entries = {}
for m in re.finditer(r"^@\w+\{([^,\s]+),.*?^\}\s*$", src, flags=re.S | re.M):
    entries[m.group(1)] = m.group(0)
tex = (HERE / "cec2027_paper.tex").read_text(encoding="utf-8")
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cited |= {k.strip() for k in m.group(1).split(",")}
EXTRA = r"""
@inproceedings{LiEtAl2010,
  author    = {Lihong Li and Wei Chu and John Langford and Robert E. Schapire},
  title     = {A contextual-bandit approach to personalized news article recommendation},
  booktitle = {Proceedings of the 19th International Conference on World Wide Web},
  pages     = {661--670},
  year      = {2010},
  doi       = {10.1145/1772690.1772758}
}

@article{MnihEtAl2015,
  author  = {Volodymyr Mnih and Koray Kavukcuoglu and David Silver and others},
  title   = {Human-level control through deep reinforcement learning},
  journal = {Nature},
  volume  = {518},
  number  = {7540},
  pages   = {529--533},
  year    = {2015},
  doi     = {10.1038/nature14236}
}

@article{Williams1992,
  author  = {Ronald J. Williams},
  title   = {Simple statistical gradient-following algorithms for connectionist reinforcement learning},
  journal = {Machine Learning},
  volume  = {8},
  number  = {3--4},
  pages   = {229--256},
  year    = {1992},
  doi     = {10.1007/BF00992696}
}
"""
extra_keys = {"LiEtAl2010", "MnihEtAl2015", "Williams1992"}
missing = sorted(k for k in cited if k not in entries and k not in extra_keys)
assert not missing, f"chaves citadas sem entrada: {missing}"
out = "\n\n".join(entries[k] for k in sorted(cited) if k in entries) + "\n" + EXTRA
out = out.replace("{P}||{C}max", "{P}$||${C}max")
out = re.sub(r",\s*note\s*=\s*\{Report date.*?1982 report\}", "", out, flags=re.S)
(HERE / "refs_cec.bib").write_text(out, encoding="utf-8")
print(len(cited), "chaves;", len([k for k in cited if k in entries]), "de refs.bib + 3 novas")
