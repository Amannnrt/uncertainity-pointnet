"""
Baseline / ablation PointNet training (MC Dropout is an inference-time technique applied to
this same trained model later, not a different training procedure).

Trains on ModelNet40 with the 5 OOD classes (bottle, bowl, cup, keyboard, laptop) excluded
entirely, validates on the held-out val split, and reports final test accuracy + per-class
precision/recall/F1 at the end.

Usage:
    python3 src/training/train.py                          # default run, dropout=0.3, 100 epochs
    python3 src/training/train.py --quick                   # fast smoke test, ignore accuracy numbers
    python3 src/training/train.py --dropout_p 0.1 --epochs 70   # dropout-rate ablation run
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.modelnet40_dataset import ModelNet40Dataset
from src.models.pointnet import PointNetClassifier, feature_transform_regularizer
from src.utils.logger import ExperimentLogger
from src.utils.config import OOD_CLASSES, SEED, NUM_POINTS, VAL_FRACTION

CONFIG = {
    "run_name": "baseline_pointnet",
    "num_points": NUM_POINTS,
    "val_fraction": VAL_FRACTION,
    "seed": SEED,
    "ood_classes": OOD_CLASSES,
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "lr_step_size": 20,      # decay LR every N epochs
    "lr_gamma": 0.5,          # multiply LR by this at each step
    "feature_transform_reg_weight": 0.001,
    "dropout_p": 0.3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer, criterion, device, reg_weight, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for points, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            points, labels = points.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits, input_trans, feature_trans = model(points)
            loss = criterion(logits, labels)
            if feature_trans is not None:
                loss = loss + reg_weight * feature_transform_regularizer(feature_trans)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * points.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Fast smoke test: 2 epochs, tiny data subset, 0 dataloader workers. "
                              "Use this first to catch bugs before committing to a full run.")
    parser.add_argument("--dropout_p", type=float, default=None,
                         help="Override the dropout rate (default 0.3). Use this for the "
                              "dropout-rate ablation, e.g. --dropout_p 0.1 / 0.5 / 0.7.")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override number of training epochs (default 100). The baseline "
                              "run plateaued by ~epoch 65, so ablation runs can likely use "
                              "--epochs 70 to save time without losing accuracy.")
    args = parser.parse_args()

    if args.dropout_p is not None:
        CONFIG["dropout_p"] = args.dropout_p
        CONFIG["run_name"] = f"dropout_ablation_p{args.dropout_p}"
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs

    if args.quick:
        CONFIG["run_name"] = "quick_smoke_test"
        CONFIG["epochs"] = 2
        CONFIG["batch_size"] = 8
        print("[QUICK MODE] 2 epochs, tiny data subset, 0 workers. "
              "This only checks the pipeline runs end-to-end — ignore the accuracy numbers.")

    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    train_ds = ModelNet40Dataset(split="train", num_points=CONFIG["num_points"],
                                  val_fraction=CONFIG["val_fraction"], seed=CONFIG["seed"],
                                  excluded_classes=CONFIG["ood_classes"], augment=True)
    val_ds = ModelNet40Dataset(split="val", num_points=CONFIG["num_points"],
                                val_fraction=CONFIG["val_fraction"], seed=CONFIG["seed"],
                                excluded_classes=CONFIG["ood_classes"])
    test_ds = ModelNet40Dataset(split="test", num_points=CONFIG["num_points"],
                                 excluded_classes=CONFIG["ood_classes"])

    num_classes = train_ds.num_classes
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}  num_classes: {num_classes}")

    if args.quick:
        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = Subset(val_ds, range(min(32, len(val_ds))))
        test_ds = Subset(test_ds, range(min(32, len(test_ds))))
        print(f"[QUICK MODE] subset sizes -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    num_workers = 0 if args.quick else 2
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=num_workers)

    model = PointNetClassifier(num_classes=num_classes, dropout_p=CONFIG["dropout_p"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG["lr_step_size"], gamma=CONFIG["lr_gamma"])
    criterion = nn.CrossEntropyLoss()

    logger = ExperimentLogger(run_name=CONFIG["run_name"], config=CONFIG)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, optimizer, criterion, device,
                                                  CONFIG["feature_transform_reg_weight"], train=True)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, optimizer, criterion, device,
                                              CONFIG["feature_transform_reg_weight"], train=False)
        scheduler.step()

        logger.log_epoch(epoch, {
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        logger.save_checkpoint(model, name="last", extra={"epoch": epoch, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.save_checkpoint(model, name="best", extra={"epoch": epoch, "val_acc": val_acc})

    elapsed = time.time() - start_time
    print(f"Training done in {elapsed / 60:.1f} min. Best val acc: {best_val_acc:.4f}")

    # Final test evaluation using the best checkpoint
    best_ckpt = torch.load(os.path.join(logger.ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc, test_preds, test_labels = run_epoch(
        model, test_loader, optimizer, criterion, device,
        CONFIG["feature_transform_reg_weight"], train=False)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="macro", zero_division=0)

    results = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision_macro": precision,
        "test_recall_macro": recall,
        "test_f1_macro": f1,
        "training_time_minutes": elapsed / 60,
        "num_classes": num_classes,
        "ood_classes_excluded": CONFIG["ood_classes"],
        "dropout_p": CONFIG["dropout_p"],
    }
    logger.save_json("final_results", results)
    print("Final test results:", results)

    logger.close()


if __name__ == "__main__":
    main()
