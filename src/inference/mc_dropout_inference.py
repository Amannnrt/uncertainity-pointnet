"""
MC Dropout inference on an already-trained PointNet checkpoint. No training happens here —
this loads best.pth and runs T stochastic forward passes per sample (dropout kept active via
enable_mc_dropout, BatchNorm still uses running stats).

Computes, per sample:
    - mean prediction (averaged softmax across T passes) and predicted class
    - total predictive entropy      H[y|x]              = -sum_k p_bar_k log p_bar_k
    - aleatoric uncertainty         E_t[H[y|x,W_t]]      = mean over T of each pass's own entropy
    - epistemic uncertainty (BALD)  H[y|x] - E_t[H[...]] = the part MC Dropout is meant to surface
    - variance across passes (mean over classes, as a simple scalar summary)

Runs this on both the in-distribution test set and the OOD set (held-out classes), saving
both to CSV for the calibration/OOD-AUROC analysis in the next step.

Usage:
    python3 src/inference/mc_dropout_inference.py                     # auto-finds latest baseline_pointnet checkpoint
    python3 src/inference/mc_dropout_inference.py --checkpoint path/to/best.pth --T 30
"""

import os
import sys
import glob
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.modelnet40_dataset import ModelNet40Dataset
from src.models.pointnet import PointNetClassifier, enable_mc_dropout
from src.utils.config import OOD_CLASSES, NUM_POINTS

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
EPS = 1e-10


def find_latest_checkpoint(run_prefix="baseline_pointnet"):
    candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, f"{run_prefix}_*")))
    if not candidates:
        raise FileNotFoundError(f"No experiment folders matching '{run_prefix}_*' found under {EXPERIMENTS_DIR}")
    latest_run = candidates[-1]
    ckpt_path = os.path.join(latest_run, "checkpoints", "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best.pth found in {latest_run}/checkpoints/")
    return ckpt_path


@torch.no_grad()
def mc_dropout_predict(model, points, T):
    """
    points: (B, N, 3)
    Returns probs_T: (T, B, C) — softmax output for each of the T stochastic passes.
    """
    probs_T = []
    for _ in range(T):
        logits, _, _ = model(points)
        probs_T.append(F.softmax(logits, dim=1))
    return torch.stack(probs_T, dim=0)  # (T, B, C)


def compute_uncertainty(probs_T):
    """
    probs_T: (T, B, C)
    Returns dict of (B,) tensors: mean_probs (B,C), pred_class, total_entropy,
    aleatoric, epistemic, mean_variance
    """
    T = probs_T.shape[0]
    mean_probs = probs_T.mean(dim=0)  # (B, C)
    pred_class = mean_probs.argmax(dim=1)  # (B,)

    total_entropy = -(mean_probs * torch.log(mean_probs + EPS)).sum(dim=1)  # (B,)

    per_pass_entropy = -(probs_T * torch.log(probs_T + EPS)).sum(dim=2)  # (T, B)
    aleatoric = per_pass_entropy.mean(dim=0)  # (B,)

    epistemic = total_entropy - aleatoric  # (B,)  BALD mutual information

    variance = probs_T.var(dim=0).mean(dim=1)  # (B,)  mean across classes of per-class variance

    return {
        "mean_probs": mean_probs,
        "pred_class": pred_class,
        "total_entropy": total_entropy,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "variance": variance,
    }


def run_inference(model, dataset, device, T, is_ood: bool, classes_lookup):
    """Returns a list of dict rows, one per sample, ready for a DataFrame."""
    rows = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    sample_idx = 0
    for points, labels in tqdm(loader, desc="OOD" if is_ood else "test"):
        points = points.to(device)
        probs_T = mc_dropout_predict(model, points, T)
        stats = compute_uncertainty(probs_T)

        B = points.shape[0]
        for i in range(B):
            true_label_raw = int(labels[i].item())
            pred_label = int(stats["pred_class"][i].item())

            if is_ood:
                # labels are ORIGINAL ModelNet40 ids (see modelnet40_dataset.py) — never
                # equal to a valid prediction id in this classifier's output space, so
                # "correct" is not meaningful here.
                true_class_name = classes_lookup.original_class_name(true_label_raw)
                correct = None
            else:
                true_class_name = classes_lookup.classes[true_label_raw]
                correct = (pred_label == true_label_raw)

            rows.append({
                "sample_idx": sample_idx,
                "is_ood": is_ood,
                "true_label": true_label_raw,
                "true_class_name": true_class_name,
                "pred_label": pred_label,
                "pred_class_name": classes_lookup.classes[pred_label],
                "correct": correct,
                "confidence": stats["mean_probs"][i, pred_label].item(),
                "total_entropy": stats["total_entropy"][i].item(),
                "aleatoric": stats["aleatoric"][i].item(),
                "epistemic": stats["epistemic"][i].item(),
                "variance": stats["variance"][i].item(),
            })
            sample_idx += 1

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a best.pth checkpoint. Defaults to the latest baseline_pointnet run.")
    parser.add_argument("--T", type=int, default=30, help="Number of MC Dropout forward passes.")
    parser.add_argument("--num_points", type=int, default=NUM_POINTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = args.checkpoint or find_latest_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")

    test_ds = ModelNet40Dataset(split="test", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    ood_ds = ModelNet40Dataset(split="ood", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    num_classes = test_ds.num_classes
    print(f"test: {len(test_ds)}  ood: {len(ood_ds)}  num_classes: {num_classes}")

    model = PointNetClassifier(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    enable_mc_dropout(model)  # eval() for BatchNorm, but keeps Dropout stochastic

    print(f"Running MC Dropout inference with T={args.T} passes...")
    test_rows = run_inference(model, test_ds, device, args.T, is_ood=False, classes_lookup=test_ds)
    ood_rows = run_inference(model, ood_ds, device, args.T, is_ood=True, classes_lookup=test_ds)

    df = pd.DataFrame(test_rows + ood_rows)

    out_dir = os.path.dirname(os.path.dirname(ckpt_path))  # the run's top-level folder
    out_path = os.path.join(out_dir, f"mc_dropout_results_T{args.T}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved per-sample results to: {out_path}")

    # quick summary — the core sanity check for whether MC Dropout is doing anything useful
    test_df = df[~df.is_ood]
    ood_df = df[df.is_ood]
    print("\n--- Summary ---")
    print(f"Test accuracy (mean-prediction, T={args.T}): {test_df['correct'].mean():.4f}")
    print(f"Test set  -> mean total_entropy: {test_df.total_entropy.mean():.4f}  "
          f"mean epistemic: {test_df.epistemic.mean():.4f}  mean aleatoric: {test_df.aleatoric.mean():.4f}")
    print(f"OOD set   -> mean total_entropy: {ood_df.total_entropy.mean():.4f}  "
          f"mean epistemic: {ood_df.epistemic.mean():.4f}  mean aleatoric: {ood_df.aleatoric.mean():.4f}")
    print("\nIf MC Dropout is working as intended: OOD epistemic uncertainty should be "
          "noticeably higher than test epistemic uncertainty.")


if __name__ == "__main__":
    main()
