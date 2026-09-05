"""
The core "does any of this actually help" comparison: a naive baseline that always grasps
regardless of confidence, versus the uncertainty-aware Grasp/Re-scan/Ask-help policy.

Metric: unsafe grasp rate as a fraction of ALL encounters (not just among grasped ones) —
since Re-scan and Ask-help never result in an autonomous grasp attempt at all, any error
occurring in those tiers is not an "unsafe grasp," it's a deferred decision. Only the Grasp
tier can produce an unsafe grasp.

    baseline unsafe rate  = (# wrong predictions + # OOD samples) / N   [everything grasped]
    policy unsafe rate    = grasp_tier_coverage * grasp_tier_risk        [only Grasp tier acts]

Reuses the already-computed per-sample MC Dropout results for the chosen (dropout=0.1) model —
no new inference needed.

Usage:
    python3 src/evaluation/baseline_vs_policy_comparison.py
"""

import os
import sys
import glob
import json
import argparse

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.decision_policy import classify_action

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
GRASP_THRESH = 0.877
RESCAN_THRESH = 0.777


def find_latest_results(run_prefix="dropout_ablation_p0.1"):
    candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, f"{run_prefix}_*")))
    latest_run = candidates[-1]
    result_files = sorted(glob.glob(os.path.join(latest_run, "mc_dropout_results_T*.csv")))
    return result_files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()

    results_path = args.results or find_latest_results()
    print(f"Loading: {results_path}\n")
    df = pd.read_csv(results_path)
    df["is_ood"] = df["is_ood"].astype(bool)
    df["error"] = df.apply(lambda r: True if r["is_ood"] else (not bool(r["correct"])), axis=1)

    n_total = len(df)
    n_errors = df["error"].sum()

    # --- Baseline: always grasp, regardless of confidence ---
    baseline_unsafe_rate = n_errors / n_total
    print(f"=== Baseline: always Grasp (ignore confidence entirely) ===")
    print(f"Total encounters: {n_total}")
    print(f"Wrong/unsafe outcomes: {n_errors}")
    print(f"Unsafe grasp rate: {baseline_unsafe_rate*100:.2f}% of ALL encounters\n")

    # --- Policy: Grasp / Re-scan / Ask-help ---
    df["action"] = df["confidence"].apply(lambda c: classify_action(c, GRASP_THRESH, RESCAN_THRESH))
    grasp_df = df[df["action"] == "Grasp"]

    grasp_coverage = len(grasp_df) / n_total
    grasp_risk = grasp_df["error"].mean() if len(grasp_df) > 0 else 0.0
    policy_unsafe_rate = grasp_coverage * grasp_risk  # as a fraction of ALL encounters, not just grasped ones

    rescan_df = df[df["action"] == "Re-scan"]
    askhelp_df = df[df["action"] == "Ask for help"]

    print(f"=== Uncertainty-aware policy: Grasp / Re-scan / Ask-help ===")
    print(f"Grasp tier:     {len(grasp_df):4d} samples ({grasp_coverage*100:.1f}% of all encounters), "
          f"risk within tier = {grasp_risk*100:.2f}%")
    print(f"Re-scan tier:   {len(rescan_df):4d} samples ({len(rescan_df)/n_total*100:.1f}% of all encounters) "
          f"-> deferred, no autonomous grasp attempted")
    print(f"Ask-help tier:  {len(askhelp_df):4d} samples ({len(askhelp_df)/n_total*100:.1f}% of all encounters) "
          f"-> deferred to human, no autonomous grasp attempted")
    print(f"\nUnsafe grasp rate (policy): {policy_unsafe_rate*100:.2f}% of ALL encounters "
          f"(= {grasp_coverage*100:.1f}% coverage x {grasp_risk*100:.2f}% risk within that tier)")

    # --- The headline comparison ---
    reduction_factor = baseline_unsafe_rate / policy_unsafe_rate if policy_unsafe_rate > 0 else float("inf")
    absolute_reduction = (baseline_unsafe_rate - policy_unsafe_rate) * 100

    print(f"\n=== Headline result ===")
    print(f"Always-grasp baseline unsafe rate: {baseline_unsafe_rate*100:.2f}%")
    print(f"Uncertainty-aware policy unsafe rate: {policy_unsafe_rate*100:.2f}%")
    print(f"Absolute reduction: {absolute_reduction:.2f} percentage points")
    print(f"Relative reduction: {reduction_factor:.1f}x fewer unsafe autonomous grasps")
    print(f"\n(Trade-off: the policy only autonomously acts on {grasp_coverage*100:.1f}% of encounters; "
          f"the rest are deferred to re-scanning or human help rather than resolved autonomously.)")

    out_dir = os.path.dirname(results_path)
    summary = {
        "n_total": n_total,
        "baseline_unsafe_rate": baseline_unsafe_rate,
        "policy_unsafe_rate": policy_unsafe_rate,
        "grasp_coverage": grasp_coverage,
        "grasp_risk": grasp_risk,
        "rescan_coverage": len(rescan_df) / n_total,
        "askhelp_coverage": len(askhelp_df) / n_total,
        "absolute_reduction_pct_points": absolute_reduction,
        "relative_reduction_factor": reduction_factor,
    }
    with open(os.path.join(out_dir, "baseline_vs_policy_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {os.path.join(out_dir, 'baseline_vs_policy_summary.json')}")


if __name__ == "__main__":
    main()
