# Cover letter (DRAFT — to be edited by the author before submission)

**To:** The Editors, *European Journal of Operational Research*
**Subject:** Submission of "Mechanism or policy? Attributing the gains of Q-Learning-guided Iterated Greedy for parallel machine scheduling"
**Suggested area/section:** [to be chosen in the submission system — e.g. a computational intelligence / heuristics section; confirm in the Guide for Authors]

Dear Editors,

Please find enclosed our manuscript for consideration as a research article in the *European Journal of Operational Research*.

**What the paper does.** Learning-guided metaheuristics are routinely reported as winning on the strength of the learned policy, yet the search mechanism they sit in is rarely isolated. For the identical parallel machine problem (P||Cmax) we embed the same tabular Q-Learning agent in two unrelated mechanisms (Iterated Greedy and Iterated Local Search) and decompose the gain with random-selection controls at matched budget. The mechanism accounts for most of the advantage; the learned policy adds about 0.01–0.05 percentage points. We then test why: the three destruction strategies leave almost no headroom (an omniscient per-instance choice would gain 0.04%), and an informative state, a dense reward, pre-training and the learning hyperparameters do not change this.

**Why it should interest *EJOR* readers.**
1. *A methodological protocol, not a new solver claim.* We do not present the algorithm as competitive with exact methods and say so. We show how a certified lower bound (the linear relaxation of an arc-flow model, validated against 150 proven optima), pre-specified hypotheses with hash-stamped code, equivalence margins, seed averaging and held-out instances change the conclusions drawn from heuristic comparisons (for example, one effect that looked large in exploration shrank by more than half out of sample).
2. *Stronger baselines than usual.* Beyond LPT, MULTIFIT, simulated annealing and bee colony optimisation, we add the Largest Differencing Method and a learning-free Iterated Greedy (destruction size chosen on separate instances, started from the best constructive solution), which is provably optimal on 75% of 1,130 synthetic instances (MULTIFIT: 49%) and on 89% when given the CPU time of a local-search hybrid. Two of our own pre-specified expectations failed and are reported: a local search added to that recipe pays only at an equal number of iterations, not at equal CPU time, and on unrelated machines the heuristics do not compete with exact methods (a general-purpose MILP beats them up to 1000 jobs on 50 machines; on the standard benchmark of Fanjul-Peyro and Ruiz they end 2.1% above the best known solutions, where the mechanism-versus-policy decomposition nevertheless replicates on 700 test instances; an Iterated Greedy started from the linear relaxation of the assignment model, introduced here, halves that distance to 1.0%, and followed by a restricted MILP it reaches 0.25% of the best known solutions, comparable to CPLEX with two hours). We also report that a prospective test of our own "headroom law" failed: for the destruction size, where choosing offline captures a real headroom, Q-Learning trained online or offline is worse than random choice.
3. *Generality.* Results are checked on 6,780 published instances (plus 4,160 held out), on six processing-time distributions, at equal CPU time and across algorithm seeds, and a simple instance feature (the granularity of the processing times) is shown to decide which mechanism is stronger.

**Originality and related work.** The closest work applies Q-Learning to destruction-size selection in Iterated Greedy for parallel machines with setups and weighted earliness/tardiness; we address the pure makespan problem, select among destruction strategies, and focus on attribution rather than on a new algorithm.

**Declarations.** [The manuscript is original, is not under consideration elsewhere, and has been approved by all authors — *confirm before sending*.] The author declares no competing interests. A generative AI assistant was used for software implementation support, analysis scripts and drafting; the declaration is included in the manuscript. Code, data, hashes of the frozen analysis files and the research diary are available at [repository/DOI — *to be inserted*].

**Suggested reviewers (optional — *to be filled by the author*):** [name, affiliation, e-mail] ×3; **reviewers to exclude (optional):** [—].

Sincerely,
Carlos Vinicio Dantas Araújo
Graduate Program in Computer Science, State University of Ceará (UECE), Fortaleza, Brazil
carlos.vinicio@aluno.uece.br
