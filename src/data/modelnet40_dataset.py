"""
ModelNet40 point cloud dataset loader.

- Loads the pre-sampled h5 files (2048 points/object) downloaded via download_modelnet40.py
- Subsamples down to a fixed number of points per object (default 1024, standard for PointNet)
- Carves a stratified validation split out of the official train split
  (ModelNet40 ships with train/test only)
"""

import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modelnet40_ply_hdf5_2048")

MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]


def _load_h5_files(pattern):
    all_data, all_labels = [], []
    for fname in sorted(glob.glob(os.path.join(DATA_DIR, pattern))):
        with h5py.File(fname, "r") as f:
            all_data.append(f["data"][:])
            all_labels.append(f["label"][:])
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64).squeeze()
    return data, labels


class ModelNet40Dataset(Dataset):
    def __init__(self, split="train", num_points=1024, val_fraction=0.1, seed=42,
                 excluded_classes=None, augment=False):
        assert split in ("train", "val", "test", "ood")
        self.num_points = num_points
        self.augment = augment and split == "train"
        excluded_classes = excluded_classes or []
        excluded_ids = {MODELNET40_CLASSES.index(c) for c in excluded_classes}

        if split in ("train", "val"):
            data, labels = _load_h5_files("ply_data_train*.h5")
            rng = np.random.RandomState(seed)
            idx = np.arange(len(labels))
            rng.shuffle(idx)
            n_val = int(len(idx) * val_fraction)
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            chosen = train_idx if split == "train" else val_idx
            data, labels = data[chosen], labels[chosen]

            if excluded_ids:
                keep = np.array([lbl not in excluded_ids for lbl in labels])
                data, labels = data[keep], labels[keep]

        elif split == "test":
            data, labels = _load_h5_files("ply_data_test*.h5")
            if excluded_ids:
                keep = np.array([lbl not in excluded_ids for lbl in labels])
                data, labels = data[keep], labels[keep]

        elif split == "ood":
            data, labels = _load_h5_files("ply_data_test*.h5")
            assert excluded_ids, "Must pass excluded_classes to build an OOD split."
            keep = np.array([lbl in excluded_ids for lbl in labels])
            data, labels = data[keep], labels[keep]

        self.data = data
        self.labels = labels

        active_class_ids = [i for i in range(len(MODELNET40_CLASSES)) if i not in excluded_ids]
        self.classes = [MODELNET40_CLASSES[i] for i in active_class_ids]
        self.num_classes = len(self.classes)
        self._label_map = {orig: new for new, orig in enumerate(active_class_ids)}

        if split in ("train", "val", "test"):
            self.labels = np.array([self._label_map[lbl] for lbl in self.labels], dtype=np.int64)
        # split == "ood": labels intentionally left as ORIGINAL ModelNet40 ids.

    def original_class_name(self, label: int) -> str:
        """For OOD split labels only (kept as original ModelNet40 ids, not remapped)."""
        return MODELNET40_CLASSES[label]

    def __len__(self):
        return len(self.labels)

    def get_full_points(self, i) -> np.ndarray:
        """Returns the full stored-resolution point cloud (2048 points, as shipped by the
        dataset) BEFORE the num_points subsampling applied by __getitem__. Used for multi-view
        generation, where we need more points than the final 1024 to carve out meaningfully
        different partial views before fusing them back down to 1024."""
        return self.data[i].astype(np.float32)

    def _subsample(self, points):
        idx = np.random.choice(points.shape[0], self.num_points, replace=False)
        return points[idx]

    def _augment(self, points):
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        points = points @ rot.T
        points += np.random.normal(0, 0.01, size=points.shape)
        return points

    def __getitem__(self, i):
        points = self._subsample(self.data[i]).astype(np.float32)
        if self.augment:
            points = self._augment(points).astype(np.float32)
        label = int(self.labels[i])
        return points, label


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.utils.config import OOD_CLASSES, SEED, NUM_POINTS, VAL_FRACTION

    train_ds = ModelNet40Dataset(split="train", num_points=NUM_POINTS, val_fraction=VAL_FRACTION,
                                  seed=SEED, excluded_classes=OOD_CLASSES)
    val_ds = ModelNet40Dataset(split="val", num_points=NUM_POINTS, val_fraction=VAL_FRACTION,
                                seed=SEED, excluded_classes=OOD_CLASSES)
    test_ds = ModelNet40Dataset(split="test", num_points=NUM_POINTS, excluded_classes=OOD_CLASSES)
    ood_ds = ModelNet40Dataset(split="ood", num_points=NUM_POINTS, excluded_classes=OOD_CLASSES)

    print(f"OOD classes: {OOD_CLASSES}")
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}  ood: {len(ood_ds)}")
    print(f"num_classes: {train_ds.num_classes}")
    full_pts = test_ds.get_full_points(0)
    print(f"full-resolution point cloud shape (for multi-view use): {full_pts.shape}")
