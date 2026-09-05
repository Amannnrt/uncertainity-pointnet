"""
Corrected decision policy: fixes the data-leakage issue in the original decision_policy.py,
where thresholds were picked AND evaluated on the exact same test+OOD data.

Since the OOD classes were excluded from train/val entirely (that's the point of the OOD
design), there's no separate val-OOD pool to pick thresholds on. Instead, this script splits
the existing test+OOD results into two disjoint halves:

    - threshold_selection_set (50%): used ONLY to pick the grasp/re-scan confidence thresholds
    - held_out_eval_set (50%):        used ONLY to report the final risk/coverage numbers

The split is stratified by is_ood, so both halves keep a similar OOD proportion. Crucially,
the reported risk numbers come from applying FIXED thresholds (chosen on the other half) to
data that had zero influence on choosing them — this is what actually closes the leakage gap,
not just a relabeling of the same numbers.

Usage:
    python3 src/evaluation/decision_policy_corrected.py
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.decision_policy import (
    build_risk_coverage_curve, find_threshold_for_risk, classify_action, find_latest_results
)

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


def stratified_split(df: pd.DataFrame, frac=0.5, seed=42):
    """Splits df into two halves, preserving the is_ood ratio in both."""
    rng = np.random.RandomState(seed)
    id_df = df[~df["is_ood"]].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ood_df = df[df["is_ood"]].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    id_split = int(len(id_df) * frac)
    ood_split = int(len(ood_df) * frac)

    part_a = pd.concat([id_df.iloc[:id_split], ood_df.iloc[:ood_split]]).reset_index(drop=True)
    part_b = pd.concat([id_df.iloc[id_split:], ood_df.iloc[ood_split:]]).reset_index(drop=True)
    return part_a, part_b


def apply_fixed_thresholds(df: pd.DataFrame, grasp_thresh, rescan_thresh):
    """Applies already-chosen thresholds to new data and reports the resulting risk/coverage.
    This is the honest generalization check: thresholds are FIXED here, not re-derived."""
    df = df.copy()
    df["error"] = df.apply(lambda row: True if row["is_ood"] else (not bool(row["correct"])), axis=1)
    df["action"] = df["confidence"].apply(lambda c: classify_action(c, grasp_thresh, rescan_thresh))

    results = {}
    for tier in ["Grasp", "Re-scan", "Ask for help"]:
        tier_df = df[df["action"] == tier]
        coverage = len(tier_df) / len(df)
        risk = tier_df["error"].mean() if len(tier_df) > 0 else None
        results[tier] = {"coverage": coverage, "risk": risk, "n": len(tier_df)}

    ood_df = df[df["is_ood"]]
    ood_grasp_pct = (ood_df["action"] == "Grasp").mean() * 100 if len(ood_df) > 0 else 0.0

    return df, results, ood_grasp_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--grasp_risk", type=float, default=0.02)
    parser.add_argument("--rescan_risk", type=float, default=0.15)
    parser.add_argument("--split_frac", type=float, default=0.5)
    parser.add_argument("--n_repeats", type=int, default=1,
                         help="Repeat the random split N times with different seeds and report "
                              "mean +/- std of the honest held-out risk — use this to check "
                              "whether a single split's result is stable or just noisy given "
                              "the small OOD sample size.")
    args = parser.parse_args()

    results_path = args.results or find_latest_results()
    print(f"Loading: {results_path}\n")
    df = pd.read_csv(results_path)
    df["is_ood"] = df["is_ood"].astype(bool)

    if args.n_repeats > 1:
        grasp_risks, rescan_risks, ood_leaks = [], [], []
        for seed in range(args.n_repeats):
            threshold_set, eval_set = stratified_split(df, frac=args.split_frac, seed=seed)
            curve = build_risk_coverage_curve(threshold_set)
            g_thresh, _, _ = find_threshold_for_risk(curve, args.grasp_risk)
            r_thresh, _, _ = find_threshold_for_risk(curve, args.rescan_risk)
            _, honest, ood_pct = apply_fixed_thresholds(eval_set, g_thresh, r_thresh)
            if honest["Grasp"]["risk"] is not None:
                grasp_risks.append(honest["Grasp"]["risk"])
            if honest["Re-scan"]["risk"] is not None:
                rescan_risks.append(honest["Re-scan"]["risk"])
            ood_leaks.append(ood_pct)

        print(f"--- Repeated held-out evaluation across {args.n_repeats} random splits ---")
        print(f"Grasp risk:   mean={np.mean(grasp_risks)*100:.2f}%  std={np.std(grasp_risks)*100:.2f}%  "
              f"(range {min(grasp_risks)*100:.2f}%-{max(grasp_risks)*100:.2f}%)")
        print(f"Re-scan risk: mean={np.mean(rescan_risks)*100:.2f}%  std={np.std(rescan_risks)*100:.2f}%  "
              f"(range {min(rescan_risks)*100:.2f}%-{max(rescan_risks)*100:.2f}%)")
        print(f"OOD->Grasp leak: mean={np.mean(ood_leaks):.2f}%  std={np.std(ood_leaks):.2f}%  "
              f"(range {min(ood_leaks):.2f}%-{max(ood_leaks):.2f}%)")

        out_dir = os.path.dirname(results_path)
        summary = {
            "n_repeats": args.n_repeats,
            "grasp_risk_mean": float(np.mean(grasp_risks)), "grasp_risk_std": float(np.std(grasp_risks)),
            "rescan_risk_mean": float(np.mean(rescan_risks)), "rescan_risk_std": float(np.std(rescan_risks)),
            "ood_leak_mean_pct": float(np.mean(ood_leaks)), "ood_leak_std_pct": float(np.std(ood_leaks)),
        }
        with open(os.path.join(out_dir, "decision_policy_repeated_split_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {os.path.join(out_dir, 'decision_policy_repeated_split_summary.json')}")
        return

    threshold_set, eval_set = stratified_split(df, frac=args.split_frac)
    print(f"Threshold-selection set: {len(threshold_set)} samples "
          f"({threshold_set['is_ood'].sum()} OOD)")
    print(f"Held-out evaluation set: {len(eval_set)} samples "
          f"({eval_set['is_ood'].sum()} OOD)\n")

    # --- Step 1: pick thresholds using ONLY the threshold-selection half ---
    curve = build_risk_coverage_curve(threshold_set)
    grasp_thresh, grasp_cov_biased, grasp_risk_biased = find_threshold_for_risk(curve, args.grasp_risk)
    rescan_thresh, rescan_cov_biased, rescan_risk_biased = find_threshold_for_risk(curve, args.rescan_risk)

    print("--- Thresholds selected (on threshold-selection half only) ---")
    print(f"Grasp threshold:   confidence >= {grasp_thresh:.4f}")
    print(f"Re-scan threshold: confidence >= {rescan_thresh:.4f}")
    print(f"(On the selection half itself: grasp risk={grasp_risk_biased*100:.2f}%, "
          f"re-scan risk={rescan_risk_biased*100:.2f}% — this is the OPTIMISTIC number, "
          f"same-data bias included, shown for comparison only.)\n")

    # --- Step 2: apply those FIXED thresholds to the untouched held-out half ---
    eval_df, honest_results, ood_grasp_pct = apply_fixed_thresholds(eval_set, grasp_thresh, rescan_thresh)

    print("--- Honest held-out evaluation (thresholds fixed, never seen this data) ---")
    for tier in ["Grasp", "Re-scan", "Ask for help"]:
        r = honest_results[tier]
        risk_str = f"{r['risk']*100:.2f}%" if r["risk"] is not None else "n/a (0 samples)"
        print(f"  {tier:15s}: {r['n']:4d} samples ({r['coverage']*100:.1f}% coverage), risk = {risk_str}")

    print(f"\nSafety check (held-out half only): {ood_grasp_pct:.1f}% of OOD objects in the "
          f"held-out set were routed to Grasp.")

    print(f"\n--- Comparison: optimistic (same-data) vs honest (held-out) ---")
    print(f"Grasp tier risk   -> optimistic: {grasp_risk_biased*100:.2f}%   "
          f"honest: {honest_results['Grasp']['risk']*100:.2f}%" if honest_results['Grasp']['risk'] is not None else "n/a")
    print(f"Re-scan tier risk -> optimistic: {rescan_risk_biased*100:.2f}%   "
          f"honest: {honest_results['Re-scan']['risk']*100:.2f}%" if honest_results['Re-scan']['risk'] is not None else "n/a")

    out_dir = os.path.dirname(results_path)
    summary = {
        "results_source": results_path,
        "methodology_note": "Thresholds selected on a 50% split, reported risk computed on the "
                             "disjoint held-out 50% — closes the same-data leakage issue.",
        "grasp_threshold": grasp_thresh,
        "rescan_threshold": rescan_thresh,
        "optimistic_grasp_risk_same_data": grasp_risk_biased,
        "optimistic_rescan_risk_same_data": rescan_risk_biased,
        "honest_grasp_risk_heldout": honest_results["Grasp"]["risk"],
        "honest_rescan_risk_heldout": honest_results["Re-scan"]["risk"],
        "honest_grasp_coverage_heldout": honest_results["Grasp"]["coverage"],
        "honest_rescan_coverage_heldout": honest_results["Re-scan"]["coverage"],
        "honest_ood_grasp_percentage_heldout": ood_grasp_pct,
        "n_threshold_selection": len(threshold_set),
        "n_held_out_eval": len(eval_set),
    }
    summary_path = os.path.join(out_dir, "decision_policy_corrected_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")

    eval_df.to_csv(os.path.join(out_dir, "decision_policy_heldout_eval.csv"), index=False)


if __name__ == "__main__":
    main()
