"""
Confidence-gated decision policy: derives real Grasp / Re-scan / Ask-help thresholds from
a risk-coverage curve, instead of the placeholder 0.35/0.65 values from the original design.

Core idea (selective classification): if the robot only acts on the K% of inputs it's most
confident about, and abstains (re-scans / asks for help) on the rest, how low can the error
rate get among the inputs it DOES act on? Sweeping K from 0% to 100% traces out the
risk-coverage curve. We pick confidence thresholds where risk crosses acceptable levels.

Critically, this uses BOTH the in-distribution test set AND the OOD set together (not just
clean test data) — a real robot encounters both known and unknown objects, and confidently
acting on an OOD object is always an error (the object isn't even in the model's vocabulary),
regardless of which class it happened to guess.

Usage:
    python3 src/evaluation/decision_policy.py --results path/to/mc_dropout_results_T30.csv
    python3 src/evaluation/decision_policy.py --grasp_risk 0.02 --rescan_risk 0.15
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


def find_latest_results(run_prefix="dropout_ablation_p0.1"):
    candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, f"{run_prefix}_*")))
    if not candidates:
        raise FileNotFoundError(f"No experiment folders matching '{run_prefix}_*' found. "
                                 f"Pass --results explicitly if using a different checkpoint.")
    latest_run = candidates[-1]
    result_files = sorted(glob.glob(os.path.join(latest_run, "mc_dropout_results_T*.csv")))
    if not result_files:
        raise FileNotFoundError(f"No mc_dropout_results_T*.csv found in {latest_run}")
    return result_files[-1]


def build_risk_coverage_curve(df: pd.DataFrame):
    """
    df must have 'confidence', 'is_ood', 'correct' columns.
    'error' is defined as: wrong classification (test samples) OR any OOD sample
    (since confidently acting on an unknown object is always unsafe, regardless of
    which known class it was mistakenly assigned to).
    Returns a DataFrame sorted by confidence descending, with cumulative coverage/risk.
    """
    df = df.copy()
    df["error"] = df.apply(
        lambda row: True if row["is_ood"] else (not bool(row["correct"])), axis=1
    )
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    n = len(df)
    df["coverage"] = (df.index + 1) / n  # fraction of samples "kept" if we cut here
    df["cumulative_risk"] = df["error"].cumsum() / (df.index + 1)  # error rate among kept samples

    return df


def find_threshold_for_risk(df: pd.DataFrame, target_risk: float):
    """
    Finds the confidence threshold such that, among all samples with confidence >= threshold,
    the error rate is <= target_risk. Returns (threshold, actual_coverage, actual_risk).
    """
    eligible = df[df["cumulative_risk"] <= target_risk]
    if len(eligible) == 0:
        return None, 0.0, None  # even the single most confident sample exceeds this risk level
    row = eligible.iloc[-1]  # last row satisfying the risk constraint = highest coverage at this risk
    return row["confidence"], row["coverage"], row["cumulative_risk"]


def plot_risk_coverage(df: pd.DataFrame, grasp_thresh, rescan_thresh, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df["coverage"], df["cumulative_risk"], color="tab:blue", label="Risk-coverage curve")

    if grasp_thresh is not None:
        cov = df[df["confidence"] >= grasp_thresh]["coverage"].max()
        ax.axvline(cov, color="tab:green", linestyle="--", alpha=0.7,
                    label=f"Grasp threshold (conf >= {grasp_thresh:.3f})")
    if rescan_thresh is not None:
        cov = df[df["confidence"] >= rescan_thresh]["coverage"].max()
        ax.axvline(cov, color="tab:orange", linestyle="--", alpha=0.7,
                    label=f"Re-scan threshold (conf >= {rescan_thresh:.3f})")

    ax.set_xlabel("Coverage (fraction of samples the robot acts on)")
    ax.set_ylabel("Risk (error rate among samples acted on)")
    ax.set_title("Risk-Coverage Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(df["cumulative_risk"].max(), 0.1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def classify_action(confidence, grasp_thresh, rescan_thresh):
    if grasp_thresh is not None and confidence >= grasp_thresh:
        return "Grasp"
    elif rescan_thresh is not None and confidence >= rescan_thresh:
        return "Re-scan"
    else:
        return "Ask for help"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                         help="Path to mc_dropout_results_T*.csv. Defaults to the latest "
                              "dropout_ablation_p0.1 run (our chosen primary model).")
    parser.add_argument("--grasp_risk", type=float, default=0.02,
                         help="Target max error rate for the Grasp tier (default 2%%).")
    parser.add_argument("--rescan_risk", type=float, default=0.15,
                         help="Target max error rate for the Re-scan tier (default 15%%).")
    args = parser.parse_args()

    results_path = args.results or find_latest_results()
    print(f"Loading: {results_path}")
    df = pd.read_csv(results_path)
    df["is_ood"] = df["is_ood"].astype(bool)

    curve = build_risk_coverage_curve(df)

    grasp_thresh, grasp_cov, grasp_risk = find_threshold_for_risk(curve, args.grasp_risk)
    rescan_thresh, rescan_cov, rescan_risk = find_threshold_for_risk(curve, args.rescan_risk)

    print(f"\n--- Decision Policy Thresholds ---")
    if grasp_thresh is not None:
        print(f"GRASP    tier: confidence >= {grasp_thresh:.4f}  "
              f"(covers {grasp_cov*100:.1f}% of all encounters, actual risk = {grasp_risk*100:.2f}%)")
    else:
        print(f"GRASP    tier: no confidence level achieves <= {args.grasp_risk*100:.1f}% risk — "
              f"even the single most confident prediction exceeds this target.")

    if rescan_thresh is not None:
        print(f"RE-SCAN  tier: confidence >= {rescan_thresh:.4f}  "
              f"(covers {rescan_cov*100:.1f}% of all encounters, actual risk = {rescan_risk*100:.2f}%)")
    else:
        print(f"RE-SCAN  tier: no confidence level achieves <= {args.rescan_risk*100:.1f}% risk.")

    print(f"ASK-HELP tier: everything below the re-scan threshold")

    # Apply the policy to every sample and report the resulting action distribution
    curve["action"] = curve["confidence"].apply(lambda c: classify_action(c, grasp_thresh, rescan_thresh))
    action_counts = curve["action"].value_counts()
    print(f"\n--- Resulting action distribution (on this dataset) ---")
    for action in ["Grasp", "Re-scan", "Ask for help"]:
        count = action_counts.get(action, 0)
        print(f"  {action:15s}: {count:5d} samples ({count/len(curve)*100:.1f}%)")

    # Break down by test vs OOD — the key safety check: how often does OOD get routed to Grasp?
    ood_actions = curve[curve["is_ood"]]["action"].value_counts()
    ood_grasp_pct = ood_actions.get("Grasp", 0) / curve["is_ood"].sum() * 100 if curve["is_ood"].sum() > 0 else 0
    print(f"\nSafety check: {ood_grasp_pct:.1f}% of OOD (never-seen) objects would still be "
          f"routed to Grasp under this policy — lower is better, this is the key failure mode "
          f"the whole project is trying to minimize.")

    out_dir = os.path.dirname(results_path)
    plot_risk_coverage(curve, grasp_thresh, rescan_thresh,
                        os.path.join(out_dir, "risk_coverage_curve.png"))

    summary = {
        "results_source": results_path,
        "grasp_threshold": grasp_thresh,
        "grasp_coverage": grasp_cov,
        "grasp_actual_risk": grasp_risk,
        "rescan_threshold": rescan_thresh,
        "rescan_coverage": rescan_cov,
        "rescan_actual_risk": rescan_risk,
        "action_distribution": action_counts.to_dict(),
        "ood_grasp_percentage": ood_grasp_pct,
        "target_grasp_risk": args.grasp_risk,
        "target_rescan_risk": args.rescan_risk,
    }
    summary_path = os.path.join(out_dir, "decision_policy_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")

    curve.to_csv(os.path.join(out_dir, "risk_coverage_curve.csv"), index=False)


if __name__ == "__main__":
    main()
