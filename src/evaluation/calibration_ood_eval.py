"""
Computes the two core evaluation numbers for the uncertainty method:

1. Expected Calibration Error (ECE) on the in-distribution test set — does confidence
   match actual accuracy? Also saves a reliability diagram (confidence vs accuracy per bin).

2. OOD detection AUROC — if we rank all samples (test + OOD) by an uncertainty score,
   how well does that ranking separate OOD from in-distribution? Computed separately for
   total_entropy, epistemic, and (1 - confidence), since we don't yet know which uncertainty
   signal is most useful (recall: epistemic separation was weak in the raw numbers, this
   quantifies that properly instead of eyeballing it).

Usage:
    python3 src/evaluation/calibration_ood_eval.py --results path/to/mc_dropout_results_T30.csv
    (defaults to the latest baseline_pointnet run's T30 results if --results is omitted)
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


def find_latest_results(run_prefix="baseline_pointnet", pattern="mc_dropout_results_T*.csv"):
    candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, f"{run_prefix}_*")))
    if not candidates:
        raise FileNotFoundError(f"No experiment folders matching '{run_prefix}_*' found.")
    latest_run = candidates[-1]
    result_files = sorted(glob.glob(os.path.join(latest_run, pattern)))
    if not result_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {latest_run}")
    return result_files[-1]


def compute_ece(confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10):
    """
    Standard ECE: partition [0,1] confidence range into num_bins equal-width bins,
    for each bin compare average confidence vs actual accuracy, weight by bin size.
    Returns (ece, per_bin_dataframe) — the latter is what you plot as a reliability diagram.
    """
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1])  # which bin each sample falls in

    rows = []
    ece = 0.0
    n = len(confidences)
    for b in range(num_bins):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            rows.append({"bin": b, "bin_range": f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}",
                         "count": 0, "avg_confidence": np.nan, "accuracy": np.nan})
            continue
        avg_conf = confidences[mask].mean()
        acc = correct[mask].mean()
        ece += (count / n) * abs(acc - avg_conf)
        rows.append({"bin": b, "bin_range": f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}",
                     "count": int(count), "avg_confidence": avg_conf, "accuracy": acc})

    return ece, pd.DataFrame(rows)


def plot_reliability_diagram(bin_df: pd.DataFrame, ece: float, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    valid = bin_df.dropna(subset=["accuracy"])

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(valid.avg_confidence, valid.accuracy, width=0.08, alpha=0.7,
           edgecolor="black", label="Model")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram: {save_path}")


def compute_ood_auroc(df: pd.DataFrame, score_col: str):
    """
    is_ood=True is the positive class (what we want the score to rank HIGHER).
    total_entropy / epistemic: higher = more uncertain = should rank OOD higher, use as-is.
    confidence: higher = more certain = should rank OOD LOWER, so we use (1 - confidence).
    """
    y_true = df["is_ood"].astype(int).values
    if score_col == "confidence":
        score = 1 - df["confidence"].values
    else:
        score = df[score_col].values
    return roc_auc_score(y_true, score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                         help="Path to mc_dropout_results_T*.csv. Defaults to latest baseline_pointnet run.")
    parser.add_argument("--num_bins", type=int, default=10)
    args = parser.parse_args()

    results_path = args.results or find_latest_results()
    print(f"Loading: {results_path}")
    df = pd.read_csv(results_path)

    test_df = df[~df.is_ood].copy()
    test_df["correct"] = test_df["correct"].astype(bool)

    # --- Calibration (test set only — OOD has no ground-truth class in this label space) ---
    ece, bin_df = compute_ece(test_df["confidence"].values, test_df["correct"].values, args.num_bins)
    print(f"\n--- Calibration (in-distribution test set) ---")
    print(f"ECE: {ece:.4f}  (lower is better; well-calibrated models are typically < 0.05-0.10)")
    print(bin_df.to_string(index=False))

    out_dir = os.path.dirname(results_path)
    reliability_plot_path = os.path.join(out_dir, "reliability_diagram.png")
    plot_reliability_diagram(bin_df, ece, reliability_plot_path)
    bin_df.to_csv(os.path.join(out_dir, "reliability_bins.csv"), index=False)

    # --- OOD detection AUROC, compared across uncertainty signals ---
    print(f"\n--- OOD Detection AUROC (test vs OOD, {len(test_df)} vs {df.is_ood.sum()} samples) ---")
    auroc_results = {}
    for score_col in ["total_entropy", "epistemic", "aleatoric", "confidence"]:
        auc = compute_ood_auroc(df, score_col)
        auroc_results[score_col] = auc
        print(f"  {score_col:15s} AUROC: {auc:.4f}  (0.5 = random, 1.0 = perfect separation)")

    best_signal = max(auroc_results, key=auroc_results.get)
    print(f"\nBest-performing uncertainty signal for OOD detection: '{best_signal}' "
          f"(AUROC={auroc_results[best_signal]:.4f})")

    # --- Save everything ---
    summary = {
        "results_source": results_path,
        "ece": ece,
        "ood_auroc": auroc_results,
        "best_ood_signal": best_signal,
        "test_accuracy": float(test_df["correct"].mean()),
        "n_test": int(len(test_df)),
        "n_ood": int(df.is_ood.sum()),
    }
    summary_path = os.path.join(out_dir, "calibration_ood_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
