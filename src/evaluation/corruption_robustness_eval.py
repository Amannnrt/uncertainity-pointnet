"""
Corruption robustness sweep: for increasing severity of noise and occlusion (applied
independently), run MC Dropout inference on the corrupted in-distribution test set and
record accuracy, mean confidence, and mean uncertainty at each severity level.

This is the core "does confidence honestly degrade with accuracy, or does the model stay
falsely confident" experiment — the central evidence for the project's motivation.

Usage:
    python3 src/evaluation/corruption_robustness_eval.py
    python3 src/evaluation/corruption_robustness_eval.py --T 30 --checkpoint path/to/best.pth
"""

import os
import sys
import glob
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.modelnet40_dataset import ModelNet40Dataset
from src.data.corruption import apply_corruption
from src.models.pointnet import PointNetClassifier, enable_mc_dropout
from src.utils.config import OOD_CLASSES, NUM_POINTS
from src.inference.mc_dropout_inference import mc_dropout_predict, compute_uncertainty, find_latest_checkpoint

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")

NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1]          # sigma, in the same units as ModelNet40's normalized coords
OCCLUSION_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]          # fraction of points removed


class CorruptedDataset(Dataset):
    """Wraps a base ModelNet40Dataset, applying corruption on the fly per __getitem__ call."""

    def __init__(self, base_dataset, noise_sigma=0.0, occlusion_fraction=0.0):
        self.base = base_dataset
        self.noise_sigma = noise_sigma
        self.occlusion_fraction = occlusion_fraction

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        points, label = self.base[i]
        points = apply_corruption(points, self.noise_sigma, self.occlusion_fraction)
        return points.astype(np.float32), label


@torch.no_grad()
def evaluate_severity(model, dataset, device, T):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    all_correct, all_conf, all_entropy, all_epistemic = [], [], [], []

    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        probs_T = mc_dropout_predict(model, points, T)
        stats = compute_uncertainty(probs_T)

        pred = stats["pred_class"]
        correct = (pred == labels).cpu().numpy()
        conf = stats["mean_probs"].gather(1, pred.unsqueeze(1)).squeeze(1).cpu().numpy()

        all_correct.extend(correct.tolist())
        all_conf.extend(conf.tolist())
        all_entropy.extend(stats["total_entropy"].cpu().numpy().tolist())
        all_epistemic.extend(stats["epistemic"].cpu().numpy().tolist())

    return {
        "accuracy": float(np.mean(all_correct)),
        "mean_confidence": float(np.mean(all_conf)),
        "mean_entropy": float(np.mean(all_entropy)),
        "mean_epistemic": float(np.mean(all_epistemic)),
        "n": len(all_correct),
    }


def plot_sweep(df, x_col, x_label, title, save_path):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(df[x_col], df["accuracy"], "o-", color="tab:blue", label="Accuracy")
    ax1.plot(df[x_col], df["mean_confidence"], "s--", color="tab:cyan", label="Mean confidence")
    ax2.plot(df[x_col], df["mean_entropy"], "^-", color="tab:red", label="Mean entropy")

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Accuracy / Confidence", color="tab:blue")
    ax2.set_ylabel("Predictive entropy", color="tab:red")
    ax1.set_ylim(0, 1)
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--num_points", type=int, default=NUM_POINTS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    print(f"Using device: {device}\nLoading checkpoint: {ckpt_path}")

    test_ds = ModelNet40Dataset(split="test", num_points=args.num_points, excluded_classes=OOD_CLASSES)
    num_classes = test_ds.num_classes

    model = PointNetClassifier(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    enable_mc_dropout(model)

    out_dir = os.path.dirname(os.path.dirname(ckpt_path))

    # --- Noise sweep ---
    print(f"\n=== Noise sweep (T={args.T}) ===")
    noise_rows = []
    for sigma in tqdm(NOISE_LEVELS, desc="noise levels"):
        corrupted = CorruptedDataset(test_ds, noise_sigma=sigma, occlusion_fraction=0.0)
        result = evaluate_severity(model, corrupted, device, args.T)
        result["noise_sigma"] = sigma
        noise_rows.append(result)
        print(f"  sigma={sigma:.3f}  acc={result['accuracy']:.4f}  "
              f"conf={result['mean_confidence']:.4f}  entropy={result['mean_entropy']:.4f}")

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(os.path.join(out_dir, "corruption_noise_sweep.csv"), index=False)
    plot_sweep(noise_df, "noise_sigma", "Gaussian noise sigma",
               "Robustness to sensor noise", os.path.join(out_dir, "noise_sweep_plot.png"))

    # --- Occlusion sweep ---
    print(f"\n=== Occlusion sweep (T={args.T}) ===")
    occ_rows = []
    for frac in tqdm(OCCLUSION_LEVELS, desc="occlusion levels"):
        corrupted = CorruptedDataset(test_ds, noise_sigma=0.0, occlusion_fraction=frac)
        result = evaluate_severity(model, corrupted, device, args.T)
        result["occlusion_fraction"] = frac
        occ_rows.append(result)
        print(f"  fraction={frac:.2f}  acc={result['accuracy']:.4f}  "
              f"conf={result['mean_confidence']:.4f}  entropy={result['mean_entropy']:.4f}")

    occ_df = pd.DataFrame(occ_rows)
    occ_df.to_csv(os.path.join(out_dir, "corruption_occlusion_sweep.csv"), index=False)
    plot_sweep(occ_df, "occlusion_fraction", "Fraction of points occluded",
               "Robustness to occlusion", os.path.join(out_dir, "occlusion_sweep_plot.png"))

    print(f"\nAll results and plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
