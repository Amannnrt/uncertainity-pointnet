"""
Point cloud corruption functions — noise and occlusion — applied to already-sampled,
fixed-size (N, 3) point clouds. Each function returns a point cloud of the SAME size N,
so corrupted output stays compatible with normal batched model inference:

- add_gaussian_noise: jitters coordinates, does not change point count.
- occlude: removes points beyond a random cutting plane (simulating a single-viewpoint scan
  losing the far side of an object), then resamples back up to N points (with replacement
  if needed) so downstream batching still works. This means "occlusion" here means the
  points are duplicated/redundant, not that N shrinks — that redundancy itself is realistic
  (a real occluded scan just has denser sampling of the visible surface).
"""

import numpy as np


def add_gaussian_noise(points: np.ndarray, sigma: float) -> np.ndarray:
    """points: (N, 3). sigma: std dev of Gaussian noise added to each coordinate."""
    if sigma <= 0:
        return points
    noise = np.random.normal(0, sigma, size=points.shape).astype(points.dtype)
    return points + noise


def occlude(points: np.ndarray, fraction: float) -> np.ndarray:
    """
    points: (N, 3). fraction: fraction of points to remove (0 = no occlusion, 0.9 = severe).
    Simulates a single-viewpoint scan by picking a random direction and dropping the points
    farthest along it, then resampling (with replacement) back to N points.
    """
    if fraction <= 0:
        return points
    N = points.shape[0]

    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)
    projections = points @ direction  # (N,) how far each point is along the random direction

    n_keep = max(int(N * (1 - fraction)), 10)  # always keep a minimum handful of points
    keep_idx = np.argsort(projections)[:n_keep]  # keep the "near" side, drop the "far" side
    visible_points = points[keep_idx]

    # resample back up to N points so the tensor shape stays fixed for batching
    resample_idx = np.random.choice(len(visible_points), N, replace=True)
    return visible_points[resample_idx]


def apply_corruption(points: np.ndarray, noise_sigma: float = 0.0, occlusion_fraction: float = 0.0) -> np.ndarray:
    """Apply occlusion first (structural), then noise (sensor-level), matching a realistic
    LiDAR pipeline order. Either can be 0 to isolate one corruption type."""
    if occlusion_fraction > 0:
        points = occlude(points, occlusion_fraction)
    if noise_sigma > 0:
        points = add_gaussian_noise(points, noise_sigma)
    return points
