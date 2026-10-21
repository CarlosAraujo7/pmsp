# The 46-page version (`learnheuristic_pmsp_long.tex`, `learnheuristic_pmsp_long_v46pp.pdf`) is a historical record, superseded in places

The long version is kept because `paper/make_s_moved.py` selects line ranges of `learnheuristic_pmsp_long.tex` to build the moved sections of the supplement (`paper/s_moved.tex`); do not edit it, and do not cite it for the points below. The text of record is `paper/learnheuristic_pmsp.tex` (main) and `paper/supplementary_material.tex`.

Superseded points (the corrected statements are in the main text and in the diary, `diario_pesquisa.md`, Entradas 164, 171):

1. **"93% of the gain of the local search was budget"** came from a first, loop-based local search. Vectorized, 30% was budget; compiled, 28%; with the loop version the CPU ratio was 9.6, with the vectorized and compiled ones 1.8 and 2.0. The recipe with local search is neutral at equal CPU (not harmful).
2. **CPU times and cost ratios of the ILS family** (quality--time figure and the ratios in the sections on the ILS and on the published baselines) were measured with loop-based operators in parallel runs, and the ILS is 6.2 times slower than its vectorized version (same results). The IG still beats the ILS with the vectorized ILS (0.46% at equal CPU, 0.50% at equal iterations).
3. **Learners**: the long version reports the tabular Q-Learning agent, UCB1, a pre-trained agent, Thompson sampling and racing; the main text adds a contextual bandit (LinUCB), an instance-conditional kNN and MLP trained offline, a small neural Q-learner and a larger DQN (nine and more learners in all).
4. **Literature audit, implementation check and the extra environments** (Taillard replication, the standard unrelated-machine benchmark with 700 test instances, the larger training set) exist only in the main text and the supplement.
