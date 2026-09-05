"""
Point cloud corruption and multi-view utilities, operating on (N, 3) point clouds.

Original functions (used by corruption_robustness_eval.py):
    - add_gaussian_noise, occlude, apply_corruption

New functions (used by multiview_rescan_experiment.py):
    - to_fixed_size: resample any variable-length point set to exactly N points
    - visible_mask: boolean mask of which points are "visible" from a given viewing
      direction (the near side of a random cutting plane) — returns the MASK, not a
      resampled array, so multiple views' visible sets can be properly unioned at the
      index level without padding artifacts contaminating the fusion.
"""

import numpy as np


def to_fixed_size(points: np.ndarray, n: int) -> np.ndarray:
    """Resamples points (variable length) to exactly n points: random subsample without
    replacement if there are enough points, or resample with replacement to pad if not."""
    if len(points) == 0:
        raise ValueError("Cannot resample an empty point set.")
    if len(points) >= n:
        idx = np.random.choice(len(points), n, replace=False)
    else:
        idx = np.random.choice(len(points), n, replace=True)
    return points[idx]


def visible_mask(points: np.ndarray, direction: np.ndarray = None, visible_fraction: float = 0.5):
    """
    Returns a boolean mask marking which points are "visible" from a simulated viewpoint —
    the points nearest along a given (or random) direction, i.e. the near side of a cutting
    plane, same idea as a real single-viewpoint scan only seeing the near surface of an object.

    Returns (mask, direction) so the caller can reuse or record which direction was used.
    """
    if direction is None:
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)

    projections = points @ direction
    n_visible = max(int(len(points) * visible_fraction), 10)
    visible_idx = np.argsort(projections)[:n_visible]  # nearest side = "visible" from this direction

    mask = np.zeros(len(points), dtype=bool)
    mask[visible_idx] = True
    return mask, direction


def add_gaussian_noise(points: np.ndarray, sigma: float) -> np.ndarray:
    """points: (N, 3). sigma: std dev of Gaussian noise added to each coordinate."""
    if sigma <= 0:
        return points
    noise = np.random.normal(0, sigma, size=points.shape).astype(points.dtype)
    return points + noise


def occlude(points: np.ndarray, fraction: float) -> np.ndarray:
    """
    points: (N, 3). fraction: fraction of points to remove (0 = no occlusion, 0.9 = severe).
    Simulates a single-viewpoint scan, then resamples back up to N points so the output stays
    a fixed size for batched inference. NOTE: for multi-view fusion, use visible_mask +
    to_fixed_size directly instead — this function's resample-to-N-with-duplicates behavior
    is appropriate for single-view robustness testing but NOT for fusing multiple views,
    since duplicated points would masquerade as new information during a union.
    """
    if fraction <= 0:
        return points
    N = points.shape[0]
    mask, _ = visible_mask(points, visible_fraction=1 - fraction)
    visible_points = points[mask]
    return to_fixed_size(visible_points, N)


def apply_corruption(points: np.ndarray, noise_sigma: float = 0.0, occlusion_fraction: float = 0.0) -> np.ndarray:
    """Apply occlusion first (structural), then noise (sensor-level), matching a realistic
    LiDAR pipeline order. Either can be 0 to isolate one corruption type."""
    if occlusion_fraction > 0:
        points = occlude(points, occlusion_fraction)
    if noise_sigma > 0:
        points = add_gaussian_noise(points, noise_sigma)
    return points
