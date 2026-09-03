"""
Local experiment logging: every run gets its own timestamped folder under experiments/,
containing:
  - config.yaml         (exact settings used, for reproducibility)
  - metrics.csv          (one row per logged step/epoch, easy to reload into pandas later)
  - tensorboard/         (live loss/accuracy curves during training: tensorboard --logdir experiments)
  - checkpoints/         (model weights, best + last)

Usage:
    from src.utils.logger import ExperimentLogger

    logger = ExperimentLogger(run_name="baseline_pointnet", config=config_dict)

    for epoch in range(num_epochs):
        ... train ...
        logger.log_epoch(epoch, {"train_loss": 0.42, "train_acc": 0.81,
                                  "val_loss": 0.51, "val_acc": 0.77})

    logger.save_checkpoint(model, name="best", extra={"epoch": epoch, "val_acc": best_acc})
    logger.close()
"""

import os
import csv
import json
import yaml
import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

import torch

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


class ExperimentLogger:
    def __init__(self, run_name: str, config: dict | None = None, experiments_dir: str = EXPERIMENTS_DIR):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(experiments_dir, f"{run_name}_{timestamp}")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.run_name = run_name
        self._csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields = None  # locked in on first log_epoch call

        if config is not None:
            self.save_config(config)

        self.tb_writer = SummaryWriter(os.path.join(self.run_dir, "tensorboard")) if _HAS_TB else None
        if not _HAS_TB:
            print("[ExperimentLogger] tensorboard not installed, skipping TB logging "
                  "(CSV logging still works). pip install tensorboard to enable.")

        print(f"[ExperimentLogger] Logging this run to: {self.run_dir}")

    def save_config(self, config: dict):
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

    def log_epoch(self, epoch: int, metrics: dict):
        """metrics: dict of name -> float, e.g. {'train_loss': 0.4, 'val_acc': 0.81}"""
        row = {"epoch": epoch, **metrics}

        if self._csv_writer is None:
            self._csv_fields = list(row.keys())
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()

        # if new metric keys show up later that weren't in the first call, ignore silently
        # rather than crash a long training run
        safe_row = {k: row.get(k, "") for k in self._csv_fields}
        self._csv_writer.writerow(safe_row)
        self._csv_file.flush()

        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, epoch)

        metric_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        print(f"[epoch {epoch}] {metric_str}")

    def log_scalar(self, name: str, value: float, step: int):
        """For finer-grained logging than per-epoch (e.g. per-batch loss)."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, step)

    def save_checkpoint(self, model, name: str = "last", extra: dict | None = None):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        payload = {"model_state_dict": model.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[ExperimentLogger] Saved checkpoint: {path}")

    def save_json(self, name: str, data: dict):
        """For one-off results (e.g. final test metrics, calibration numbers, OOD AUROC)."""
        path = os.path.join(self.run_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[ExperimentLogger] Saved results: {path}")

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()


if __name__ == "__main__":
    # smoke test
    logger = ExperimentLogger(run_name="smoke_test", config={"lr": 1e-3, "batch_size": 32})
    for epoch in range(3):
        logger.log_epoch(epoch, {"train_loss": 1.0 / (epoch + 1), "val_acc": 0.5 + 0.1 * epoch})
    logger.save_json("final_results", {"test_acc": 0.83, "ece": 0.04})
    logger.close()
    print(f"Check {logger.run_dir} for config.yaml, metrics.csv, tensorboard/, final_results.json")
