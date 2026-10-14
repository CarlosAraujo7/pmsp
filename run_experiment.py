#!/usr/bin/env python3
"""
run_experiment.py
=================
Script de execucao completa do experimento PMSP Learnheuristic.

Uso
---
Primeira execucao (metodos base: LPT, ILS-Rand, ILS-UCB, ILS-RF, ILS-QL):
    python run_experiment.py

Extensao (adicionar ILS-QL-WS e ILS-QL-HI aos resultados existentes):
    python run_experiment.py --extend

Analise completa (figuras + tabelas LaTeX):
    python run_experiment.py --analyze

Tudo de uma vez:
    python run_experiment.py --extend --analyze
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment import runner, config as cfg
from analysis.run_analysis import main as run_analysis_main


def main():
    parser = argparse.ArgumentParser(description="PMSP Learnheuristic experiment runner.")
    parser.add_argument("--extend",  action="store_true",
                        help="Add ILS-QL-WS and ILS-QL-HI to existing results.csv.")
    parser.add_argument("--analyze", action="store_true",
                        help="Run full analysis after experiment.")
    args = parser.parse_args()

    results_dir = ROOT / "results"

    if args.extend:
        print("=== Running extended experiment (ILS-Rand-HI, ILS-QL-WS, ILS-QL-HI) ===")
        runner.run_all(
            methods=cfg.METHODS_EXTEND,
            output_dir=results_dir,
            append=True,
            verbose=True,
        )
    else:
        print("=== Running base experiment (LPT, ILS-Rand, ILS-UCB, ILS-RF, ILS-QL) ===")
        runner.run_all(
            methods=cfg.METHODS_BASE,
            output_dir=results_dir,
            append=False,
            verbose=True,
        )

    if args.analyze:
        print("\n=== Running analysis ===")
        sys.argv = ["run_analysis"]  # reset argv for argparse inside run_analysis
        run_analysis_main()


if __name__ == "__main__":
    main()
