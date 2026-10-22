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
EXTRA += r"""
@article{DrakeEtal2020,
  author  = {John H. Drake and Ahmed Kheiri and Ender {\"O}zcan and Edmund K. Burke},
  title   = {Recent advances in selection hyper-heuristics},
  journal = {European Journal of Operational Research},
  volume  = {285},
  number  = {2},
  pages   = {405--428},
  year    = {2020},
  doi     = {10.1016/j.ejor.2019.07.073}
}

@article{Talbi2021,
  author  = {El-Ghazali Talbi},
  title   = {Machine learning into metaheuristics: A survey and taxonomy},
  journal = {ACM Computing Surveys},
  volume  = {54},
  number  = {6},
  pages   = {129:1--129:32},
  year    = {2021},
  doi     = {10.1145/3459664}
}

@article{LaTorreEtal2021,
  author  = {Antonio LaTorre and Daniel Molina and Eneko Osaba and Javier Poyatos and Javier {Del Ser} and Francisco Herrera},
  title   = {A prescription of methodological guidelines for comparing bio-inspired optimization algorithms},
  journal = {Swarm and Evolutionary Computation},
  volume  = {67},
  pages   = {100973},
  year    = {2021},
  doi     = {10.1016/j.swevo.2021.100973}
}

@article{EibenSmit2011,
  author  = {Agoston E. Eiben and Selmar K. Smit},
  title   = {Parameter tuning for configuring and analyzing evolutionary algorithms},
  journal = {Swarm and Evolutionary Computation},
  volume  = {1},
  number  = {1},
  pages   = {19--31},
  year    = {2011}
}

@article{Lakens2017,
  author  = {Dani{\"e}l Lakens},
  title   = {Equivalence tests: A practical primer for t tests, correlations, and meta-analyses},
  journal = {Social Psychological and Personality Science},
  volume  = {8},
  number  = {4},
  pages   = {355--362},
  year    = {2017},
  doi     = {10.1177/1948550617697177}
}

@article{Goldberg1990,
  author  = {David E. Goldberg},
  title   = {Probability matching, the magnitude of reinforcement, and classifier system bidding},
  journal = {Machine Learning},
  volume  = {5},
  number  = {4},
  pages   = {407--425},
  year    = {1990},
  doi     = {10.1007/BF00116878}
}

@misc{SonneratEtal2021,
  author        = {Nicolas Sonnerat and Pengming Wang and Ira Ktena and Sergey Bartunov and Vinod Nair},
  title         = {Learning a large neighborhood search algorithm for mixed integer programs},
  year          = {2021},
  eprint        = {2107.10201},
  archivePrefix = {arXiv}
}

@misc{BartzBeielsteinEtal2020,
  author        = {Thomas Bartz-Beielstein and Carola Doerr and Daan van den Berg and others},
  title         = {Benchmarking in optimization: Best practice and open issues},
  year          = {2020},
  eprint        = {2007.03488},
  archivePrefix = {arXiv}
}
"""
EXTRA += r"""
@article{Russo2018,
  author  = {Daniel J. Russo and Benjamin {Van Roy} and Abbas Kazerouni and Ian Osband and Zheng Wen},
  title   = {A tutorial on {T}hompson sampling},
  journal = {Foundations and Trends in Machine Learning},
  volume  = {11},
  number  = {1},
  pages   = {1--96},
  year    = {2018},
  doi     = {10.1561/2200000070}
}

@inproceedings{DaCostaEtal2008,
  author    = {Lu{\'\i}s {Da Costa} and {\'A}lvaro Fialho and Marc Schoenauer and Mich{\`e}le Sebag},
  title     = {Adaptive operator selection with dynamic multi-armed bandits},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  pages     = {913--920},
  publisher = {ACM},
  year      = {2008}
}
"""
extra_keys = {"LiEtAl2010", "MnihEtAl2015", "Williams1992", "DrakeEtal2020", "Talbi2021", "LaTorreEtal2021", "EibenSmit2011", "Lakens2017", "Goldberg1990", "SonneratEtal2021", "BartzBeielsteinEtal2020", "Russo2018", "DaCostaEtal2008"}
missing = sorted(k for k in cited if k not in entries and k not in extra_keys)
assert not missing, f"chaves citadas sem entrada: {missing}"
out = "\n\n".join(entries[k] for k in sorted(cited) if k in entries) + "\n" + EXTRA
out = out.replace("{P}||{C}max", "{P}$||${C}max")
out = re.sub(r",\s*note\s*=\s*\{Report date.*?1982 report\}", "", out, flags=re.S)
(HERE / "refs_cec.bib").write_text(out, encoding="utf-8")
print(len(cited), "chaves;", len([k for k in cited if k in entries]), "de refs.bib + 3 novas")
