"""
Properly separated policy-development vs. frozen-holdout evaluation, per the design:

    all data -> split ONCE into [policy-development pool] and [frozen held-out test pool]
    policy-development pool -> repeated resampling to CV-estimate risk of each candidate threshold
    -> select threshold (highest coverage among candidates with CV risk <= target)
    -> FREEZE threshold
    -> apply EXACTLY ONCE to the frozen held-out pool (never touched until this point)
    -> that single number is the final, honest risk estimate

This fixes the remaining leakage in decision_policy_cv_sweep.py: there, every candidate's CV
estimate was individually honest, but picking the "best" candidate out of many still
reintroduces selection bias at a smaller scale. Here, no candidate selection ever touches the
final evaluation pool at all.

This also directly tests two competing explanations for the original 15%->36% gap:
    Hypothesis A (selection overfitting): frozen held-out risk comes back close to the
        CV-estimated risk from the dev pool (e.g. both ~15-20%) -> the original 36% was
        mostly an artifact of searching too many thresholds on one small noisy sample.
    Hypothesis B (intrinsic instability): frozen held-out risk is still much higher than the
        CV estimate (e.g. CV~15%, held-out~35%) -> confidence genuinely cannot reliably
        control risk in this middle band, regardless of how carefully the threshold is chosen.

Usage:
    python3 src/evaluation/decision_policy_frozen_holdout.py
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.decision_policy import find_latest_results, classify_action
from src.evaluation.decision_policy_corrected import stratified_split, apply_fixed_thresholds
from src.evaluation.decision_policy import build_risk_coverage_curve, find_threshold_for_risk


def bootstrap_risk_in_pool(pool_df, grasp_thresh, rescan_thresh, n_repeats, subsample_frac=0.7, seed_offset=0):
    """CV-estimates risk for a FIXED threshold pair using repeated random subsamples drawn
    ONLY from within pool_df. No separate selection/eval split needed here since the threshold
    is a fixed input, not fit to any particular subsample — that's what makes each subsample's
    result an unbiased read on that threshold, as long as pool_df itself stays isolated from
    the final frozen test pool (enforced by the caller)."""
    risks, coverages = [], []
    id_df = pool_df[~pool_df["is_ood"]]
    ood_df = pool_df[pool_df["is_ood"]]
    for i in range(n_repeats):
        seed = seed_offset + i
        sub_id = id_df.sample(frac=subsample_frac, random_state=seed)
        sub_ood = ood_df.sample(frac=subsample_frac, random_state=seed)
        sub = pd.concat([sub_id, sub_ood])
        _, honest, _ = apply_fixed_thresholds(sub, grasp_thresh, rescan_thresh)
        if honest["Re-scan"]["risk"] is not None:
            risks.append(honest["Re-scan"]["risk"])
            coverages.append(honest["Re-scan"]["coverage"])
    return np.mean(risks), np.std(risks), np.mean(coverages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--dev_frac", type=float, default=0.65,
                         help="Fraction of all data used for policy development (the rest is "
                              "the frozen final test pool, touched exactly once at the end).")
    parser.add_argument("--grasp_risk_target", type=float, default=0.02)
    parser.add_argument("--rescan_risk_target", type=float, default=0.15)
    parser.add_argument("--n_candidates", type=int, default=20)
    parser.add_argument("--min_coverage", type=float, default=0.10,
                         help="Minimum acceptable coverage for the re-scan tier — prevents the "
                              "selection from degenerating to a near-empty, noise-dominated tier "
                              "that only LOOKS low-risk because almost nothing falls in it.")
    parser.add_argument("--n_cv_repeats", type=int, default=25,
                         help="Bootstrap resamples per candidate, all within the dev pool only.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_path = args.results or find_latest_results()
    print(f"Loading: {results_path}\n")
    df = pd.read_csv(results_path)
    df["is_ood"] = df["is_ood"].astype(bool)

    # --- Step 0: split ONCE into dev pool and frozen held-out pool. The frozen pool is not
    # touched again until the very last step. ---
    dev_pool, frozen_pool = stratified_split(df, frac=args.dev_frac, seed=args.seed)
    print(f"Policy-development pool: {len(dev_pool)} samples ({dev_pool['is_ood'].sum()} OOD)")
    print(f"Frozen held-out pool:    {len(frozen_pool)} samples ({frozen_pool['is_ood'].sum()} OOD)")
    print("(the frozen pool will not be touched again until the final step)\n")

    # --- Step 1: fix the grasp threshold using CV entirely within the dev pool ---
    grasp_candidates = np.linspace(0.7, 0.98, 20)
    grasp_rows = []
    for cand in grasp_candidates:
        mean_risk, std_risk, mean_cov = [], [], []
        risks, covs = [], []
        id_df = dev_pool[~dev_pool["is_ood"]]
        ood_df = dev_pool[dev_pool["is_ood"]]
        for i in range(args.n_cv_repeats):
            sub = pd.concat([id_df.sample(frac=0.7, random_state=i), ood_df.sample(frac=0.7, random_state=i)])
            sub = sub.copy()
            sub["error"] = sub.apply(lambda r: True if r["is_ood"] else (not bool(r["correct"])), axis=1)
            grasp_mask = sub["confidence"] >= cand
            if grasp_mask.sum() > 0:
                risks.append(sub[grasp_mask]["error"].mean())
                covs.append(grasp_mask.mean())
        if risks:
            grasp_rows.append({"threshold": cand, "mean_risk": np.mean(risks), "mean_coverage": np.mean(covs)})
    grasp_df = pd.DataFrame(grasp_rows)
    grasp_under = grasp_df[grasp_df["mean_risk"] <= args.grasp_risk_target]
    grasp_thresh = (grasp_under.loc[grasp_under["mean_coverage"].idxmax(), "threshold"]
                    if len(grasp_under) > 0 else grasp_df.loc[grasp_df["mean_risk"].idxmin(), "threshold"])
    print(f"Grasp threshold selected (dev pool CV): {grasp_thresh:.4f}\n")

    # --- Step 2: sweep re-scan candidates, CV-estimated entirely within the dev pool ---
    candidates = np.linspace(0.3, grasp_thresh - 0.01, args.n_candidates)
    print(f"Sweeping {args.n_candidates} re-scan candidates within the dev pool "
          f"({args.n_cv_repeats} bootstrap resamples each)...\n")
    rows = []
    for cand in candidates:
        mean_risk, std_risk, mean_cov = bootstrap_risk_in_pool(
            dev_pool, grasp_thresh, cand, args.n_cv_repeats)
        rows.append({"threshold": cand, "cv_mean_risk": mean_risk, "cv_std_risk": std_risk,
                     "cv_mean_coverage": mean_cov})
        print(f"  threshold={cand:.4f}  CV risk={mean_risk*100:.2f}% (+/-{std_risk*100:.2f}%)  "
              f"CV coverage={mean_cov*100:.1f}%")

    sweep_df = pd.DataFrame(rows)
    viable = sweep_df[sweep_df["cv_mean_coverage"] >= args.min_coverage]
    if len(viable) == 0:
        print(f"\nWARNING: no candidate reaches the minimum coverage of {args.min_coverage*100:.0f}%. "
              f"Falling back to the full candidate set (results may be noise-dominated).")
        viable = sweep_df

    under_target = viable[viable["cv_mean_risk"] <= args.rescan_risk_target]
    if len(under_target) > 0:
        best = under_target.loc[under_target["cv_mean_coverage"].idxmax()]
        target_achievable = True
    else:
        best = viable.loc[viable["cv_mean_risk"].idxmin()]
        target_achievable = False

    rescan_thresh = float(best.threshold)
    print(f"\nSelected re-scan threshold (frozen now): {rescan_thresh:.4f}")
    print(f"  Dev-pool CV estimate: risk={best.cv_mean_risk*100:.2f}% (+/-{best.cv_std_risk*100:.2f}%), "
          f"coverage={best.cv_mean_coverage*100:.1f}%")
    if not target_achievable:
        print(f"  NOTE: no candidate met the {args.rescan_risk_target*100:.0f}% target within the dev pool either.")

    # --- Step 3: the ONLY touch of the frozen pool — apply the frozen thresholds exactly once ---
    print(f"\n--- Applying frozen thresholds to the held-out pool (first and only touch) ---")
    _, final_honest, final_ood_leak = apply_fixed_thresholds(frozen_pool, grasp_thresh, rescan_thresh)
    for tier in ["Grasp", "Re-scan", "Ask for help"]:
        r = final_honest[tier]
        risk_str = f"{r['risk']*100:.2f}%" if r["risk"] is not None else "n/a"
        print(f"  {tier:15s}: {r['n']:4d} samples ({r['coverage']*100:.1f}% coverage), risk = {risk_str}")
    print(f"  OOD->Grasp leak on frozen pool: {final_ood_leak:.1f}%")

    print(f"\n--- Hypothesis check ---")
    print(f"Dev-pool CV estimate for re-scan risk:  {best.cv_mean_risk*100:.2f}%")
    print(f"Frozen held-out re-scan risk (final):    {final_honest['Re-scan']['risk']*100:.2f}%")
    gap = abs(final_honest['Re-scan']['risk'] - best.cv_mean_risk) * 100
    print(f"Gap: {gap:.2f} percentage points")
    if gap < 5:
        print("-> Small gap: consistent with Hypothesis A (the original 36% was substantially "
              "a threshold-selection overfitting artifact).")
    else:
        print("-> Large gap persists even with proper separation: consistent with Hypothesis B "
              "(confidence genuinely struggles to control risk in this middle region, "
              "independent of how carefully the threshold was chosen).")

    out_dir = os.path.dirname(results_path)
    summary = {
        "note": "Original single-split search found rescan_threshold~0.522 with in-sample risk "
                "15.00% and honest held-out risk 36.07% (n_repeats=20 mean) -- KEPT, not discarded, "
                "as the finding that motivated this redesign.",
        "grasp_threshold": float(grasp_thresh),
        "rescan_threshold_frozen": rescan_thresh,
        "dev_pool_cv_risk": float(best.cv_mean_risk),
        "dev_pool_cv_risk_std": float(best.cv_std_risk),
        "frozen_holdout_risk": float(final_honest["Re-scan"]["risk"]) if final_honest["Re-scan"]["risk"] is not None else None,
        "frozen_holdout_coverage": float(final_honest["Re-scan"]["coverage"]),
        "frozen_holdout_ood_leak_pct": float(final_ood_leak),
        "gap_percentage_points": float(gap),
        "target_achievable_in_dev": target_achievable,
        "n_dev": len(dev_pool), "n_frozen": len(frozen_pool),
    }
    with open(os.path.join(out_dir, "decision_policy_frozen_holdout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {os.path.join(out_dir, 'decision_policy_frozen_holdout_summary.json')}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sweep_df["cv_mean_coverage"], sweep_df["cv_mean_risk"], yerr=sweep_df["cv_std_risk"],
                fmt="o-", capsize=3, color="tab:blue", label="Dev-pool CV estimate")
    ax.scatter([final_honest["Re-scan"]["coverage"]], [final_honest["Re-scan"]["risk"]],
               color="red", s=100, zorder=5, label="Frozen held-out (final, single point)")
    ax.axhline(args.rescan_risk_target, color="gray", linestyle="--", label="Target")
    ax.set_xlabel("Re-scan coverage")
    ax.set_ylabel("Risk")
    ax.set_title("Dev-pool CV estimate vs. frozen held-out reality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frozen_holdout_comparison_plot.png"), dpi=150)
    plt.close(fig)
    print(f"Saved plot: {os.path.join(out_dir, 'frozen_holdout_comparison_plot.png')}")


if __name__ == "__main__":
    main()
