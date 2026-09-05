"""
Alternative to spatial point-cloud fusion: instead of combining point clouds (which risks
producing geometry unlike anything the network was trained on — as observed in
multiview_rescan_experiment.py), each "scan" stays a completely normal, independent 1024-point
partial view, run through MC Dropout separately. The scans' MEAN PROBABILITY DISTRIBUTIONS
(not just their top-class confidence numbers) are then averaged together — a standard
ensemble/multi-view-voting approach — to obtain a final combined distribution, confidence,
and action.

Uses the SAME random subset/seed as multiview_rescan_experiment.py so results are directly
comparable between the two fusion strategies.

Usage:
    python3 src/evaluation/multiview_probability_aggregation.py
"""

import os
import sys
import glob
import argparse

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
def predict_distribution(model, points_1024: np.ndarray, device, T=30):
    """Returns the full mean probability vector (C,) from T MC Dropout passes — not just the
    top-class confidence — since averaging needs the whole distribution, not single numbers."""
    x = torch.from_numpy(points_1024).float().unsqueeze(0).to(device)
    probs_T = mc_dropout_predict(model, x, T)
    stats = compute_uncertainty(probs_T)
    return stats["mean_probs"][0].cpu().numpy()  # (C,)


def generate_independent_scan(full_points: np.ndarray, n_points: int, visible_fraction: float = 0.5):
    """A single independent partial view, fixed to n_points — matches the exact format/density
    the network was trained on, unlike the fused clouds in the spatial-fusion experiment."""
    mask, _ = visible_mask(full_points, visible_fraction=visible_fraction)
    return to_fixed_size(full_points[mask], n_points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--num_points", type=int, default=NUM_POINTS)
    parser.add_argument("--visible_fraction", type=float, default=0.5)
    parser.add_argument("--n_scans", type=int, default=3)
    parser.add_argument("--subset_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "dropout_ablation_p0.1_*")))
        ckpt_path = os.path.join(candidates[-1], "checkpoints", "best.pth")
    print(f"Using device: {device}\nLoading checkpoint: {ckpt_path}")

    test_ds = ModelNet40Dataset(split="test", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    ood_ds = ModelNet40Dataset(split="ood", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    num_classes = test_ds.num_classes

    model = PointNetClassifier(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    enable_mc_dropout(model)

    # SAME subset-selection logic/seed as multiview_rescan_experiment.py, for direct comparability
    all_items = [(test_ds, i, False) for i in range(len(test_ds))] + \
                [(ood_ds, i, True) for i in range(len(ood_ds))]
    rng = np.random.RandomState(args.seed)
    if args.subset_size and args.subset_size > 0 and args.subset_size < len(all_items):
        chosen_idx = rng.choice(len(all_items), args.subset_size, replace=False)
        all_items = [all_items[i] for i in chosen_idx]
        print(f"Using the same random subset selection as the fusion experiment: {len(all_items)} samples.")

    rows = []
    for ds, idx, is_ood in tqdm(all_items, desc="probability aggregation"):
        full_points = ds.get_full_points(idx)
        label = int(ds.labels[idx])

        scan_dists = []
        for _ in range(args.n_scans):
            scan = generate_independent_scan(full_points, args.num_points, args.visible_fraction)
            dist = predict_distribution(model, scan, device, args.T)
            scan_dists.append(dist)

        row = {"is_ood": is_ood, "true_label": label}
        cumulative = np.zeros_like(scan_dists[0])
        for k in range(1, args.n_scans + 1):
            cumulative = np.mean(scan_dists[:k], axis=0)  # average of first k scans' distributions
            pred = int(np.argmax(cumulative))
            conf = float(cumulative[pred])
            action = classify_action(conf, GRASP_THRESH, RESCAN_THRESH)

            if is_ood:
                correct = None
            else:
                correct = (pred == label)

            row[f"k{k}_pred"] = pred
            row[f"k{k}_confidence"] = conf
            row[f"k{k}_action"] = action
            row[f"k{k}_correct"] = correct

        if is_ood:
            row["true_class_name"] = ds.original_class_name(label)
        else:
            row["true_class_name"] = ds.classes[label]

        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.dirname(ckpt_path))
    out_path = os.path.join(out_dir, "multiview_probability_aggregation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # --- Key analysis: same framing as the fusion experiment, for direct comparison ---
    rescan_initial = df[df["k1_action"] == "Re-scan"].copy()
    n_rescan = len(rescan_initial)
    print(f"\n=== Samples whose single-scan (k=1) confidence fell in the Re-scan tier: {n_rescan} ===")

    if n_rescan == 0:
        print("No samples in the Re-scan tier in this subset.")
        return

    for k in range(2, args.n_scans + 1):
        print(f"\n--- After combining k={k} scans' probability distributions ---")
        transition_counts = rescan_initial[f"k{k}_action"].value_counts()
        for action in ["Grasp", "Re-scan", "Ask for help"]:
            n = transition_counts.get(action, 0)
            print(f"  {action:15s}: {n:4d} ({n/n_rescan*100:.1f}%)")

        promoted = rescan_initial[rescan_initial[f"k{k}_action"] == "Grasp"]
        promoted_id = promoted[~promoted["is_ood"]]
        promoted_ood = promoted[promoted["is_ood"]]
        n_promoted = len(promoted)
        n_correct = promoted_id[f"k{k}_correct"].sum() if len(promoted_id) > 0 else 0
        n_incorrect_id = len(promoted_id) - n_correct
        n_ood_promoted = len(promoted_ood)

        if n_promoted > 0:
            print(f"  Of {n_promoted} promoted to Grasp: {n_correct} correct (ID), "
                  f"{n_incorrect_id} incorrect (ID), {n_ood_promoted} OOD (unsafe)")
        safe_promotion_rate = n_correct / n_rescan * 100 if n_rescan > 0 else 0
        print(f"  Headline (k={k}): {safe_promotion_rate:.1f}% of Re-scan-tier encounters "
              f"correctly promoted to Grasp.")

        mean_conf_before = rescan_initial["k1_confidence"].mean()
        mean_conf_after = rescan_initial[f"k{k}_confidence"].mean()
        print(f"  Mean confidence: {mean_conf_before:.4f} (k=1) -> {mean_conf_after:.4f} (k={k})")


if __name__ == "__main__":
    main()
