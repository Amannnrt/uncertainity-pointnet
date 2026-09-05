"""
Two things this script adds beyond multiview_probability_aggregation.py:

1. ORACLE DIAGNOSTIC: for each Re-scan-tier case, checks whether any INDIVIDUAL additional
   scan (not averaged with anything) alone reaches Grasp-level confidence and is correct.
   This distinguishes two very different conclusions:
       - If individual scans often DO contain a correct high-confidence answer -> the
         information exists, naive averaging was just destroying it.
       - If individual scans are themselves rarely confident/correct -> the re-scan premise
         itself doesn't hold under this occlusion setup, independent of aggregation method.

2. AGREEMENT-BASED AGGREGATION: instead of always averaging probability distributions
   (which dilutes confidence when scans disagree), only commits to a combined prediction when
   a majority of scans agree on the same class — using the mean confidence of just the
   agreeing scans. When scans disagree, uncertainty increases (routed to Ask-for-help) rather
   than being papered over by an averaged distribution.

Run on the FULL test+OOD set (not a subset) for a statistically defensible Re-scan sample
count, using T=10 MC passes (justified by the T-ablation in Section 8.1, which found T=10
performs comparably to T=30 at a fraction of the cost).

Usage:
    python3 src/evaluation/multiview_agreement_and_oracle.py
"""

import os
import sys
import glob
import argparse
from collections import Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.modelnet40_dataset import ModelNet40Dataset
from src.data.corruption import visible_mask, to_fixed_size
from src.models.pointnet import PointNetClassifier, enable_mc_dropout
from src.utils.config import OOD_CLASSES, NUM_POINTS
from src.inference.mc_dropout_inference import mc_dropout_predict, compute_uncertainty
from src.evaluation.decision_policy import classify_action

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
GRASP_THRESH = 0.877
RESCAN_THRESH = 0.777


@torch.no_grad()
def predict_distribution(model, points_1024: np.ndarray, device, T):
    x = torch.from_numpy(points_1024).float().unsqueeze(0).to(device)
    probs_T = mc_dropout_predict(model, x, T)
    stats = compute_uncertainty(probs_T)
    return stats["mean_probs"][0].cpu().numpy()


def generate_independent_scan(full_points, n_points, visible_fraction=0.5):
    mask, _ = visible_mask(full_points, visible_fraction=visible_fraction)
    return to_fixed_size(full_points[mask], n_points)


def agreement_aggregate(preds, confs):
    """preds, confs: lists of per-scan (pred_class, confidence). Returns (combined_pred,
    combined_conf, agreed: bool). If a majority class exists, combined_conf is the mean
    confidence of ONLY the agreeing scans. If no majority (all scans disagree), returns
    (None, 0.0, False) — signaling the caller to treat this as maximally uncertain."""
    counts = Counter(preds)
    top_class, top_count = counts.most_common(1)[0]
    if top_count < 2:  # no two scans agree at all
        return None, 0.0, False
    agreeing_confs = [c for p, c in zip(preds, confs) if p == top_class]
    return top_class, float(np.mean(agreeing_confs)), True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--T", type=int, default=10,
                         help="MC passes per scan. Default 10, justified by the T-ablation "
                              "(Section 8.1) showing negligible difference vs T=30.")
    parser.add_argument("--num_points", type=int, default=NUM_POINTS)
    parser.add_argument("--visible_fraction", type=float, default=0.5)
    parser.add_argument("--n_scans", type=int, default=3)
    parser.add_argument("--subset_size", type=int, default=0, help="0 = full test+OOD set.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "dropout_ablation_p0.1_*")))
        ckpt_path = os.path.join(candidates[-1], "checkpoints", "best.pth")
    print(f"Using device: {device}\nLoading checkpoint: {ckpt_path}\nT={args.T} passes/scan\n")

    test_ds = ModelNet40Dataset(split="test", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    ood_ds = ModelNet40Dataset(split="ood", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    num_classes = test_ds.num_classes

    model = PointNetClassifier(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    enable_mc_dropout(model)

    all_items = [(test_ds, i, False) for i in range(len(test_ds))] + \
                [(ood_ds, i, True) for i in range(len(ood_ds))]
    if args.subset_size and 0 < args.subset_size < len(all_items):
        rng = np.random.RandomState(args.seed)
        chosen_idx = rng.choice(len(all_items), args.subset_size, replace=False)
        all_items = [all_items[i] for i in chosen_idx]
    print(f"Running on {len(all_items)} samples.\n")

    rows = []
    for ds, idx, is_ood in tqdm(all_items, desc="agreement + oracle"):
        full_points = ds.get_full_points(idx)
        label = int(ds.labels[idx])

        scan_preds, scan_confs = [], []
        for _ in range(args.n_scans):
            scan = generate_independent_scan(full_points, args.num_points, args.visible_fraction)
            dist = predict_distribution(model, scan, device, args.T)
            pred = int(np.argmax(dist))
            conf = float(dist[pred])
            scan_preds.append(pred)
            scan_confs.append(conf)

        k1_action = classify_action(scan_confs[0], GRASP_THRESH, RESCAN_THRESH)

        row = {
            "is_ood": is_ood, "true_label": label,
            "true_class_name": ds.original_class_name(label) if is_ood else ds.classes[label],
            "k1_pred": scan_preds[0], "k1_confidence": scan_confs[0], "k1_action": k1_action,
            "k1_correct": None if is_ood else (scan_preds[0] == label),
        }
        for i in range(1, args.n_scans):
            row[f"scan{i+1}_pred"] = scan_preds[i]
            row[f"scan{i+1}_confidence"] = scan_confs[i]
            row[f"scan{i+1}_correct"] = None if is_ood else (scan_preds[i] == label)
            row[f"scan{i+1}_high_conf_correct"] = (
                None if is_ood else (scan_confs[i] >= GRASP_THRESH and scan_preds[i] == label)
            )

        # Agreement-based aggregation using all n_scans
        combined_pred, combined_conf, agreed = agreement_aggregate(scan_preds, scan_confs)
        if not agreed:
            agreement_action = "Ask for help"  # disagreement -> maximal caution, not averaged confidence
            agreement_correct = None if is_ood else False
        else:
            agreement_action = classify_action(combined_conf, GRASP_THRESH, RESCAN_THRESH)
            agreement_correct = None if is_ood else (combined_pred == label)

        row["agreement_pred"] = combined_pred
        row["agreement_confidence"] = combined_conf
        row["agreement_action"] = agreement_action
        row["agreement_correct"] = agreement_correct
        row["scans_agreed"] = agreed

        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.dirname(ckpt_path))
    out_path = os.path.join(out_dir, "multiview_agreement_oracle_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # === Oracle diagnostic ===
    rescan_initial = df[df["k1_action"] == "Re-scan"].copy()
    n_rescan = len(rescan_initial)
    print(f"\n=== Re-scan-tier cases (full dataset): {n_rescan} ===")

    if n_rescan == 0:
        print("No Re-scan cases found.")
        return

    id_rescan = rescan_initial[~rescan_initial["is_ood"]]
    oracle_hits = 0
    for i in range(2, args.n_scans + 1):
        col = f"scan{i}_high_conf_correct"
        if col in id_rescan.columns:
            oracle_hits += id_rescan[col].fillna(False).astype(bool).sum()
    # count samples with AT LEAST ONE additional scan being a high-confidence correct hit
    any_hit_mask = np.zeros(len(id_rescan), dtype=bool)
    for i in range(2, args.n_scans + 1):
        col = f"scan{i}_high_conf_correct"
        if col in id_rescan.columns:
            any_hit_mask |= id_rescan[col].fillna(False).astype(bool).values
    n_any_hit = any_hit_mask.sum()

    print(f"\n--- Oracle diagnostic (ID Re-scan cases only, n={len(id_rescan)}) ---")
    print(f"Cases where AT LEAST ONE additional individual scan alone was high-confidence AND correct: "
          f"{n_any_hit} ({n_any_hit/len(id_rescan)*100:.1f}% of ID Re-scan cases)" if len(id_rescan) > 0 else "n/a")
    print("-> If this percentage is substantial, the information needed IS present in additional "
          "views; if it's near zero, the Re-scan premise itself is weak under this occlusion setup, "
          "independent of aggregation method.")

    # === Agreement-based aggregation results ===
    print(f"\n--- Agreement-based aggregation outcomes for all {n_rescan} Re-scan-tier cases ---")
    action_counts = rescan_initial["agreement_action"].value_counts()
    for action in ["Grasp", "Re-scan", "Ask for help"]:
        n = action_counts.get(action, 0)
        print(f"  {action:15s}: {n:4d} ({n/n_rescan*100:.1f}%)")

    agree_rate = rescan_initial["scans_agreed"].mean() * 100
    print(f"\nScans agreed (majority class) for {agree_rate:.1f}% of Re-scan cases.")

    promoted = rescan_initial[rescan_initial["agreement_action"] == "Grasp"]
    promoted_id = promoted[~promoted["is_ood"]]
    promoted_ood = promoted[promoted["is_ood"]]
    n_promoted = len(promoted)
    n_correct = promoted_id["agreement_correct"].fillna(False).astype(bool).sum() if len(promoted_id) > 0 else 0
    n_incorrect = len(promoted_id) - n_correct
    n_ood_promoted = len(promoted_ood)

    if n_promoted > 0:
        print(f"\nOf {n_promoted} promoted to Grasp via agreement rule: "
              f"{n_correct} correct (ID), {n_incorrect} incorrect (ID), {n_ood_promoted} OOD (unsafe)")
    safe_rate = n_correct / n_rescan * 100 if n_rescan > 0 else 0
    print(f"Headline (agreement-based): {safe_rate:.1f}% of all Re-scan-tier encounters "
          f"correctly promoted to Grasp.")


if __name__ == "__main__":
    main()
