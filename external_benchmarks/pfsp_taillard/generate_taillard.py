"""Regenerate Taillard's 120 permutation-flowshop instances (ta001-ta120) from his published generator and seeds.

Source of the procedure and of the seeds: E. Taillard (1993), "Benchmarks for basic scheduling problems",
EJOR 64, 278-285, as reproduced in OR-Library (flowshop2.txt, contributed by D. C. Mattfeld and R. J. M. Vaessens;
downloaded to raw/flowshop2.txt on 2026-10-18 with Carlos's authorisation). The generator is a Lehmer LCG
(Schrage's method); processing times are U{1,...,99}; machine i, job j is drawn in machine-major order.

Usage:  python external_benchmarks/pfsp_taillard/generate_taillard.py     (writes instances/taXXX.txt and checks ta001)
Format of each file: first line "n m", then m lines (one per machine) with n processing times, so that
p[i][j] is the time of job j on machine i (the same layout as the matrix printed in the verification block of flowshop2.txt).
"""
from __future__ import annotations

from pathlib import Path

M, A, B, C = 2147483647, 16807, 127773, 2836

# (first_index, n_jobs, n_machines, seeds)  -- seeds copied from the OR-Library file (ta001 ... ta120)
BLOCKS = [
    (1, 20, 5, [873654221, 379008056, 1866992158, 216771124, 495070989, 402959317, 1369363414, 2021925980, 573109518, 88325120]),
    (11, 20, 10, [587595453, 1401007982, 873136276, 268827376, 1634173168, 691823909, 73807235, 1273398721, 2065119309, 1672900551]),
    (21, 20, 20, [479340445, 268827376, 1958948863, 918272953, 555010963, 2010851491, 1519833303, 1748670931, 1923497586, 1829909967]),
    (31, 50, 5, [1328042058, 200382020, 496319842, 1203030903, 1730708564, 450926852, 1303135678, 1273398721, 587288402, 248421594]),
    (41, 50, 10, [1958948863, 575633267, 655816003, 1977864101, 93805469, 1803345551, 49612559, 1899802599, 2013025619, 578962478]),
    (51, 50, 20, [1539989115, 691823909, 655816003, 1315102446, 1949668355, 1923497586, 1805594913, 1861070898, 715643788, 464843328]),
    (61, 100, 5, [896678084, 1179439976, 1122278347, 416756875, 267829958, 1835213917, 1328833962, 1418570761, 161033112, 304212574]),
    (71, 100, 10, [1539989115, 655816003, 960914243, 1915696806, 2013025619, 1168140026, 1923497586, 167698528, 1528387973, 993794175]),
    (81, 100, 20, [450926852, 1462772409, 1021685265, 83696007, 508154254, 1861070898, 26482542, 444956424, 2115448041, 118254244]),
    (91, 200, 10, [471503978, 1215892992, 135346136, 1602504050, 160037322, 551454346, 519485142, 383947510, 1968171878, 540872513]),
    (101, 200, 20, [2013025619, 475051709, 914834335, 810642687, 1019331795, 2056065863, 1342855162, 1325809384, 1988803007, 765656702]),
    (111, 500, 20, [1368624604, 450181436, 1927888393, 1759567256, 606425239, 19268348, 1298201670, 2041736264, 379756761, 28837162]),
]

# verification block printed in flowshop2.txt for ta001 (20 jobs x 5 machines; rows = machines)
TA001 = [
    [54, 83, 15, 71, 77, 36, 53, 38, 27, 87, 76, 91, 14, 29, 12, 77, 32, 87, 68, 94],
    [79, 3, 11, 99, 56, 70, 99, 60, 5, 56, 3, 61, 73, 75, 47, 14, 21, 86, 5, 77],
    [16, 89, 49, 15, 89, 45, 60, 23, 57, 64, 7, 1, 63, 41, 63, 47, 26, 75, 77, 40],
    [66, 58, 31, 68, 78, 91, 13, 59, 49, 85, 85, 9, 39, 41, 56, 40, 54, 77, 51, 31],
    [58, 56, 20, 85, 53, 35, 53, 41, 69, 13, 86, 72, 8, 49, 47, 87, 58, 18, 68, 28],
]


def unif(seed: int, low: int, high: int) -> tuple[int, int]:
    k = seed // B
    seed = A * (seed % B) - k * C
    if seed < 0:
        seed += M
    return seed, low + int((seed / M) * (high - low + 1))


def generate(n: int, m: int, seed: int) -> list[list[int]]:
    p = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            seed, p[i][j] = unif(seed, 1, 99)
    return p


def all_instances() -> dict[str, tuple[int, int, list[list[int]]]]:
    out = {}
    for first, n, m, seeds in BLOCKS:
        for k, s in enumerate(seeds):
            out[f"ta{first + k:03d}"] = (n, m, generate(n, m, s))
    return out


def read_instance(path: Path) -> list[list[int]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    n, m = map(int, lines[0].split())
    p = [list(map(int, lines[1 + i].split())) for i in range(m)]
    assert all(len(r) == n for r in p)
    return p


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "instances"
    out_dir.mkdir(exist_ok=True)
    inst = all_instances()
    assert len(inst) == 120
    assert inst["ta001"][2] == TA001, "generator does not reproduce the verification matrix of ta001"
    for name, (n, m, p) in inst.items():
        (out_dir / f"{name}.txt").write_text(f"{n} {m}\n" + "\n".join(" ".join(str(x) for x in row) for row in p) + "\n", encoding="utf-8")
    print("ok: 120 instances written; ta001 matches the verification block of the published generator")
