"""
Tests the actual premise behind the Re-scan tier: does adding a second (different-viewpoint)
observation and fusing it with the first genuinely improve confidence and accuracy for
medium-confidence cases — or is Re-scan just a theoretical middle tier with no demonstrated
benefit?

Procedure per sample:
    1. Take the full-resolution (2048-pt) point cloud.
    2. Generate View 1: a random-direction partial view (~50% of points visible), fixed to 1024 pts.
    3. Generate View 2: an INDEPENDENT random-direction partial view.
    4. Fused: the UNION of View 1's and View 2's visible points (real new geometry, not
       duplicated padding), fixed to 1024 pts.
    5. Run MC Dropout inference (T=30, frozen dropout=0.1 checkpoint) on View 1 alone, and
       again on the Fused cloud.
    6. Classify each stage's action (Grasp / Re-scan / Ask-help) using the already-derived
       frozen thresholds (0.877 / 0.777).

The key analysis: among samples whose View-1-only action was "Re-scan", what fraction moved
to "Grasp" after fusion — and of those, what fraction are actually correct (for in-distribution
samples) or are OOD objects being dangerously promoted to Grasp?

Usage:
    python3 src/evaluation/multiview_rescan_experiment.py
    python3 src/evaluation/multiview_rescan_experiment.py --subset_size 0   # full test+ood set
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
from src.inference.mc_dropout_inference import mc_dropout_predict, compute_uncertainty, find_latest_checkpoint
from src.evaluation.decision_policy import classify_action

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")

# Frozen thresholds derived in the decision-policy analysis (Section 9.3 of the report)
GRASP_THRESH = 0.877
RESCAN_THRESH = 0.777


@torch.no_grad()
def predict_confidence(model, points_1024: np.ndarray, device, T=30):
    """points_1024: (1024, 3) numpy array, already fixed-size. Returns (confidence, pred_label)."""
    x = torch.from_numpy(points_1024).float().unsqueeze(0).to(device)  # (1, 1024, 3)
    probs_T = mc_dropout_predict(model, x, T)  # (T, 1, C)
    stats = compute_uncertainty(probs_T)
    pred = int(stats["pred_class"][0].item())
    conf = float(stats["mean_probs"][0, pred].item())
    return conf, pred


def generate_views(full_points: np.ndarray, n_points: int, visible_fraction: float = 0.5):
    """Returns (view1_fixed, fused) — a single partial view fixed to n_points (matching the
    density the model was trained on), and the fused union of two independent views KEPT AT
    ITS NATURAL SIZE (only padded up if smaller than n_points, never compressed down) — since
    a real second scan adds new points rather than compressing back to the original budget."""
    mask1, _ = visible_mask(full_points, visible_fraction=visible_fraction)
    mask2, _ = visible_mask(full_points, visible_fraction=visible_fraction)  # independent direction

    view1 = to_fixed_size(full_points[mask1], n_points)

    fused_mask = mask1 | mask2
    fused_points = full_points[fused_mask]
    if len(fused_points) < n_points:
        fused = to_fixed_size(fused_points, n_points)  # pad up only if genuinely too small
    else:
        fused = fused_points  # keep the natural, larger point count — no artificial compression

    return view1, fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Defaults to the latest dropout_ablation_p0.1 checkpoint (our chosen primary model).")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--num_points", type=int, default=NUM_POINTS)
    parser.add_argument("--visible_fraction", type=float, default=0.5,
                         help="Fraction of points visible in each single partial view.")
    parser.add_argument("--subset_size", type=int, default=400,
                         help="Random subset of test+OOD samples to use (0 = full set, ~2468 samples, slower).")
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

    # Build a combined index list: (dataset_ref, index, is_ood)
    all_items = [(test_ds, i, False) for i in range(len(test_ds))] + \
                [(ood_ds, i, True) for i in range(len(ood_ds))]

    rng = np.random.RandomState(args.seed)
    if args.subset_size and args.subset_size > 0 and args.subset_size < len(all_items):
        chosen_idx = rng.choice(len(all_items), args.subset_size, replace=False)
        all_items = [all_items[i] for i in chosen_idx]
        print(f"Using a random subset of {len(all_items)} samples "
              f"(pass --subset_size 0 for the full {len(test_ds)+len(ood_ds)}-sample set).")

    rows = []
    for ds, idx, is_ood in tqdm(all_items, desc="multi-view fusion"):
        full_points = ds.get_full_points(idx)
        label = int(ds.labels[idx])

        view1, fused = generate_views(full_points, args.num_points, args.visible_fraction)

        conf1, pred1 = predict_confidence(model, view1, device, args.T)
        conf2, pred2 = predict_confidence(model, fused, device, args.T)

        action1 = classify_action(conf1, GRASP_THRESH, RESCAN_THRESH)
        action2 = classify_action(conf2, GRASP_THRESH, RESCAN_THRESH)

        if is_ood:
            true_name = ds.original_class_name(label)
            correct1 = correct2 = None  # not a meaningful concept for OOD
        else:
            true_name = ds.classes[label]
            correct1 = (pred1 == label)
            correct2 = (pred2 == label)

        rows.append({
            "is_ood": is_ood, "true_label": label, "true_class_name": true_name,
            "view1_pred": pred1, "view1_confidence": conf1, "view1_action": action1, "view1_correct": correct1,
            "fused_pred": pred2, "fused_confidence": conf2, "fused_action": action2, "fused_correct": correct2,
        })

    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(os.path.dirname(ckpt_path))
    out_path = os.path.join(out_dir, "multiview_rescan_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # --- The key analysis: what happens to samples that started in the Re-scan tier? ---
    rescan_initial = df[df["view1_action"] == "Re-scan"].copy()
    n_rescan = len(rescan_initial)
    print(f"\n=== Samples whose single-view confidence fell in the Re-scan tier: {n_rescan} ===")

    if n_rescan == 0:
        print("No samples in the Re-scan tier in this subset — try a larger --subset_size.")
        return

    transition_counts = rescan_initial["fused_action"].value_counts()
    print(f"\nAfter fusion, these moved to:")
    for action in ["Grasp", "Re-scan", "Ask for help"]:
        n = transition_counts.get(action, 0)
        print(f"  {action:15s}: {n:4d} ({n/n_rescan*100:.1f}%)")

    promoted = rescan_initial[rescan_initial["fused_action"] == "Grasp"]
    promoted_id = promoted[~promoted["is_ood"]]
    promoted_ood = promoted[promoted["is_ood"]]

    n_promoted = len(promoted)
    n_correct = promoted_id["fused_correct"].sum() if len(promoted_id) > 0 else 0
    n_incorrect_id = len(promoted_id) - n_correct
    n_ood_promoted = len(promoted_ood)

    print(f"\n=== Of the {n_promoted} promoted to Grasp after fusion ===")
    print(f"  Correct (ID):        {n_correct} ({n_correct/n_promoted*100:.1f}% of promotions)" if n_promoted else "")
    print(f"  Incorrect (ID):      {n_incorrect_id} ({n_incorrect_id/n_promoted*100:.1f}% of promotions)" if n_promoted else "")
    print(f"  OOD (unsafe):        {n_ood_promoted} ({n_ood_promoted/n_promoted*100:.1f}% of promotions)" if n_promoted else "")

    safe_promotion_rate = n_correct / n_rescan * 100 if n_rescan > 0 else 0
    print(f"\nHeadline number: {safe_promotion_rate:.1f}% of all Re-scan-tier encounters were "
          f"successfully and correctly promoted to Grasp after a second, independent view was fused in.")

    # Confidence shift summary (does fusion move confidence up on average for this group?)
    mean_conf_before = rescan_initial["view1_confidence"].mean()
    mean_conf_after = rescan_initial["fused_confidence"].mean()
    print(f"\nMean confidence for this group: {mean_conf_before:.4f} (single view) "
          f"-> {mean_conf_after:.4f} (after fusion)")


if __name__ == "__main__":
    main()
