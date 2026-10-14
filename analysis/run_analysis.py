"""
analysis/run_analysis.py
========================
Master script for Phase 5: XAI analysis and visualization.

Usage
-----
From project root:
    python -m analysis.run_analysis                 # all figures + tables
    python -m analysis.run_analysis --xai-only      # only Q-table heatmaps + policy reports
    python -m analysis.run_analysis --plots-only    # only convergence/boxplots
    python -m analysis.run_analysis --tables-only   # only LaTeX tables
    python -m analysis.run_analysis --summary       # print summary to stdout

Outputs are written to results/figures/ and results/tables/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.xai    import generate_xai_figures, generate_policy_reports
from analysis.plots  import load_results, generate_all_plots
from analysis.tables import print_summary, generate_all_tables


# ---------------------------------------------------------------------------
# Instance groups for XAI (one Q-table analysis per size)
# ---------------------------------------------------------------------------

INSTANCE_GROUPS = [
    ("rand_20j_4m",   "n=20, m=4"),
    ("rand_20j_6m",   "n=20, m=6"),
    ("rand_30j_8m",   "n=30, m=8"),
    ("rand_50j_10m",  "n=50, m=10"),
    ("rand_75j_15m",  "n=75, m=15"),
    ("rand_100j_20m", "n=100, m=20"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: XAI analysis and visualization for PMSP Learnheuristic."
    )
    parser.add_argument("--xai-only",    action="store_true",
                        help="Generate only Q-table heatmaps and policy reports.")
    parser.add_argument("--plots-only",  action="store_true",
                        help="Generate only GAP boxplots and bar charts.")
    parser.add_argument("--tables-only", action="store_true",
                        help="Generate only LaTeX tables.")
    parser.add_argument("--summary",     action="store_true",
                        help="Print text summary to stdout and exit.")
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"),
                        help="Path to results/ directory (default: <root>/results).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path    = results_dir / "results.csv"
    qtables_dir = results_dir / "qtables"
    figures_dir = results_dir / "figures"
    tables_dir  = results_dir / "tables"
    xai_dir     = figures_dir / "xai"

    if not csv_path.exists():
        print(f"ERROR: results.csv not found at {csv_path}")
        print("Run the experiment first: python -m experiment.runner")
        sys.exit(1)

    df = load_results(csv_path)

    # --- Summary only ---
    if args.summary:
        print_summary(df)
        return

    run_all    = not (args.xai_only or args.plots_only or args.tables_only)
    run_xai    = run_all or args.xai_only
    run_plots  = run_all or args.plots_only
    run_tables = run_all or args.tables_only

    # --- XAI heatmaps + policy reports ---
    if run_xai:
        print("\n=== XAI: Q-table heatmaps ===")
        if not qtables_dir.exists():
            print(f"WARNING: qtables dir not found: {qtables_dir}")
        else:
            for inst_filter, inst_label in INSTANCE_GROUPS:
                try:
                    paths = generate_xai_figures(
                        qtables_dir=qtables_dir,
                        instance_filter=inst_filter,
                        instance_label=inst_label,
                        out_dir=xai_dir,
                    )
                    for p in paths:
                        print(f"  Saved: {p.relative_to(ROOT)}")
                except FileNotFoundError as e:
                    print(f"  SKIP {inst_label}: {e}")

        print("\n=== XAI: Policy textual reports ===")
        if qtables_dir.exists():
            rep_paths = generate_policy_reports(
                qtables_dir=qtables_dir,
                instance_groups=INSTANCE_GROUPS,
                out_dir=xai_dir / "reports",
            )
            for p in rep_paths:
                print(f"  Saved: {p.relative_to(ROOT)}")

    # --- GAP boxplots and bar charts ---
    if run_plots:
        print("\n=== Plots: GAP boxplots and bar charts ===")
        paths = generate_all_plots(csv_path, figures_dir / "plots")
        for p in paths:
            if p:
                print(f"  Saved: {p.relative_to(ROOT)}")

    # --- LaTeX tables ---
    if run_tables:
        print("\n=== Tables: LaTeX results, Wilcoxon and improvement ===")
        paths = generate_all_tables(df, tables_dir)
        for p in paths:
            print(f"  Saved: {p.relative_to(ROOT)}")

    print("\nDone. All outputs in results/figures/ and results/tables/")


if __name__ == "__main__":
    main()
