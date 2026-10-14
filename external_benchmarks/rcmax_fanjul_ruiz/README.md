# R||Cmax benchmark of Fanjul-Peyro and Ruiz (EJOR 207, 2010)

**Not redistributed with this repository** (listed in `.gitignore`). To reproduce: download the three files below into `raw/` and extract `instances.7z` into `extracted/`
(the archive contains only CSV files; `bsdtar` / `tar -xf instances.7z -C extracted` works on Windows 10+).

- Source (public copy, MIT licence): https://github.com/Fraunhofer-Chalmers-Centre/RCmax, folder `Fanjul_Ruiz_instances` (companion of Aablad, Stromberg and Spensieri, *Exact makespan minimization of unrelated parallel machines*, Open Journal of Mathematical Optimization 2, 2021, doi 10.5802/ojmo.4).
  The original host (http://soa.iti.es/files/RCmax.7z, group of R. Ruiz, UPV) was unreachable on 2026-10-10; the same instances are described on the page archived by the Wayback Machine (snapshot 2024-10-03).
- Files downloaded on 2026-10-10 with the authorisation of the author of this project:

| file | bytes | md5 |
|---|---|---|
| `raw/instances.7z` | 16,638,265 | f700bfda53d39853e2d0c32f54c7a4c3 |
| `raw/res_all_instances.csv` | 45,362 | a1b0a80b96ee9e7fbb90e64a2a318710 |
| `raw/Summary_of_results.md` | 5,831 | 1f94ddd4e0f22e2332b5c4ba61274870 |

- Content: 1,400 instances = 7 types (U[1,100], U[10,100], U[100,120], U[100,200], U[1000,1100], job-correlated, machine-correlated) x m in {10,20,30,40,50} x n in {100,200,500,1000} x 10 instances;
  one CSV per instance, one row per machine, one column per job (`p[i][j]`). `res_all_instances.csv` has, per instance, the value and optimality flag of CPLEX 11.1 (2 h, original record), Gurobi 9.1.1 (300 s) and
  the LR exact method (300 s), all produced by third parties. The study uses the minimum of the three as the best known solution.
- Split used by the study (deterministic, `results/rcmax_manifest_entrada137.csv`): per cell (type, m, n) the 10 instances sorted by id have idx 0-9; idx 0-1 tuning, idx 2-6 test, idx 7-9 unused.
- Pre-registration: diary Entrada 137 (hashes in `results/entrada137_prereg_hashes.txt`); results: Entrada 138.
