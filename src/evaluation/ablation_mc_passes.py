"""
Ablation: number of MC Dropout passes (T). No retraining — reuses the existing trained
checkpoint, varying only the inference-time procedure. For each T, computes:
    - accuracy (mean prediction)
    - ECE (calibration)
    - OOD detection AUROC (using total_entropy and confidence, the two strongest signals
      from the main calibration/OOD analysis)
    - wall-clock inference time (the practical cost side of the trade-off)

Usage:
    python3 src/evaluation/ablation_mc_passes.py
    python3 src/evaluation/ablation_mc_passes.py --T_values 5 10 20 30 50
"""

import os
import sys
import time
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.modelnet40_dataset import ModelNet40Dataset
from src.models.pointnet import PointNetClassifier, enable_mc_dropout
from src.utils.config import OOD_CLASSES, NUM_POINTS
from src.inference.mc_dropout_inference import (
    mc_dropout_predict, compute_uncertainty, find_latest_checkpoint, run_inference
)
from src.evaluation.calibration_ood_eval import compute_ece, compute_ood_auroc

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


def evaluate_T(model, test_ds, ood_ds, device, T):
    start = time.time()
    test_rows = run_inference(model, test_ds, device, T, is_ood=False, classes_lookup=test_ds)
    ood_rows = run_inference(model, ood_ds, device, T, is_ood=True, classes_lookup=test_ds)
    elapsed = time.time() - start

    df = pd.DataFrame(test_rows + ood_rows)
    test_df = df[~df.is_ood].copy()
    test_df["correct"] = test_df["correct"].astype(bool)

    ece, _ = compute_ece(test_df["confidence"].values, test_df["correct"].values)
    auroc_entropy = compute_ood_auroc(df, "total_entropy")
    auroc_confidence = compute_ood_auroc(df, "confidence")
    auroc_epistemic = compute_ood_auroc(df, "epistemic")

    return {
        "T": T,
        "accuracy": float(test_df["correct"].mean()),
        "ece": ece,
        "auroc_entropy": auroc_entropy,
        "auroc_confidence": auroc_confidence,
        "auroc_epistemic": auroc_epistemic,
        "inference_time_sec": elapsed,
    }


def plot_ablation(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    T_vals = df["T"]  # NOTE: df.T is pandas' transpose property, NOT the "T" column — must use df["T"]

    axes[0].plot(T_vals, df.accuracy, "o-", label="Accuracy")
    axes[0].plot(T_vals, df.ece, "s--", label="ECE")
    axes[0].set_xlabel("Number of MC passes (T)")
    axes[0].set_title("Accuracy & Calibration vs T")
    axes[0].legend()

    axes[1].plot(T_vals, df.auroc_entropy, "o-", label="Entropy")
    axes[1].plot(T_vals, df.auroc_confidence, "s-", label="Confidence")
    axes[1].plot(T_vals, df.auroc_epistemic, "^-", label="Epistemic")
    axes[1].axhline(0.5, color="gray", linestyle=":", label="Random")
    axes[1].set_xlabel("Number of MC passes (T)")
    axes[1].set_title("OOD Detection AUROC vs T")
    axes[1].legend()

    axes[2].plot(T_vals, df.inference_time_sec, "o-", color="tab:red")
    axes[2].set_xlabel("Number of MC passes (T)")
    axes[2].set_title("Inference time (sec) vs T")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--T_values", type=int, nargs="+", default=[5, 10, 20, 30, 50])
    parser.add_argument("--num_points", type=int, default=NUM_POINTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    print(f"Using device: {device}\nLoading checkpoint: {ckpt_path}")

    test_ds = ModelNet40Dataset(split="test", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    ood_ds = ModelNet40Dataset(split="ood", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    num_classes = test_ds.num_classes

    model = PointNetClassifier(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    enable_mc_dropout(model)

    print(f"Test set: {len(test_ds)}  OOD set: {len(ood_ds)}  Sweeping T = {args.T_values}")
    print("(Total runtime roughly scales with sum(T_values) relative to a single T=30 run "
          "you've already timed — use that to sanity-check how long this will take.)\n")

    rows = []
    for T in args.T_values:
        print(f"Running T={T} ...")
        result = evaluate_T(model, test_ds, ood_ds, device, T)
        rows.append(result)
        print(f"  acc={result['accuracy']:.4f}  ece={result['ece']:.4f}  "
              f"auroc_entropy={result['auroc_entropy']:.4f}  "
              f"auroc_confidence={result['auroc_confidence']:.4f}  "
              f"auroc_epistemic={result['auroc_epistemic']:.4f}  "
              f"time={result['inference_time_sec']:.1f}s")

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.dirname(ckpt_path))
    csv_path = os.path.join(out_dir, "ablation_mc_passes.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_ablation(df, os.path.join(out_dir, "ablation_mc_passes_plot.png"))
    print("\nLook for: the T beyond which accuracy/ECE/AUROC stop improving meaningfully — "
          "that's your justification for T=30 (or a different value) as a compute/quality trade-off.")


if __name__ == "__main__":
    main()
