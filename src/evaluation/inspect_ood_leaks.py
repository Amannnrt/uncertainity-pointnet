"""
Inspects exactly which OOD (never-seen-class) samples got routed to "Grasp" by the decision
policy — i.e. the model was confident enough to act, despite the object being from a class
it was never trained on. Checks whether there's a pattern (e.g. consistently confused with a
specific, geometrically-similar known class) rather than random noise.

Reuses risk_coverage_curve.csv, already saved by decision_policy.py — no re-inference needed.

Usage:
    python3 src/evaluation/inspect_ood_leaks.py
    python3 src/evaluation/inspect_ood_leaks.py --curve path/to/risk_coverage_curve.csv
"""

import os
import sys
import glob
import argparse

import pandas as pd

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


def find_latest_curve(run_prefix="dropout_ablation_p0.1"):
    candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, f"{run_prefix}_*")))
    if not candidates:
        raise FileNotFoundError(f"No experiment folders matching '{run_prefix}_*' found.")
    latest_run = candidates[-1]
    path = os.path.join(latest_run, "risk_coverage_curve.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No risk_coverage_curve.csv in {latest_run} — run decision_policy.py first.")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=str, default=None,
                         help="Path to risk_coverage_curve.csv. Defaults to the latest "
                              "dropout_ablation_p0.1 run.")
    args = parser.parse_args()

    curve_path = args.curve or find_latest_curve()
    print(f"Loading: {curve_path}\n")
    df = pd.read_csv(curve_path)

    leaked = df[(df["is_ood"] == True) & (df["action"] == "Grasp")].copy()
    total_ood = (df["is_ood"] == True).sum()

    print(f"=== OOD samples that leaked through to 'Grasp' ===")
    print(f"{len(leaked)} out of {total_ood} total OOD samples ({len(leaked)/total_ood*100:.1f}%)\n")

    if len(leaked) == 0:
        print("None leaked through — nothing further to inspect.")
        return

    cols_to_show = ["true_class_name", "pred_class_name", "confidence", "total_entropy", "epistemic"]
    cols_to_show = [c for c in cols_to_show if c in leaked.columns]
    print(leaked[cols_to_show].sort_values("confidence", ascending=False).to_string(index=False))

    print(f"\n--- Which true OOD class leaks through most? ---")
    print(leaked["true_class_name"].value_counts().to_string())

    print(f"\n--- What known class does the model mistake them for? ---")
    print(leaked["pred_class_name"].value_counts().to_string())

    print(f"\n--- True class -> predicted class pairs (the actual confusion pattern) ---")
    pair_counts = leaked.groupby(["true_class_name", "pred_class_name"]).size().sort_values(ascending=False)
    print(pair_counts.to_string())

    # Save a small summary for the report/paper
    out_dir = os.path.dirname(curve_path)
    out_path = os.path.join(out_dir, "ood_leak_analysis.csv")
    leaked[cols_to_show].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
