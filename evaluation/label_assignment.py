"""
Per-point vertebra label assignment for whole-spine point clouds.

For methods that output a whole-spine point cloud (OctFusion, PDR),
this assigns each predicted point a vertebra level label (L1-L5 / verLev20-24)
by nearest-neighbour lookup against GT per-vertebra point clouds in the
same coordinate space.

If per-vertebra GT point clouds are not available, falls back to
axis-based K-means clustering (less accurate but parameter-free).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


LEVELS        = [20, 21, 22, 23, 24]          # verLev codes: L1=20 … L5=24
LEVEL_NAMES   = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5"}


# ─────────────────────────────────────────────────────────────────────────────
# Primary method: KD-tree from GT per-vertebra point clouds
# ─────────────────────────────────────────────────────────────────────────────

def assign_labels_kdtree(pred_pcd: np.ndarray,
                         gt_vertebra_pcds: dict) -> np.ndarray:
    """
    Assign per-point vertebra labels by nearest-neighbour in GT per-vertebra PCDs.

    Args:
        pred_pcd          : [N, 3]  whole-spine predicted point cloud
        gt_vertebra_pcds  : {level (int): np.ndarray [M, 3]}
                            GT per-vertebra point clouds in the SAME coordinate
                            space as pred_pcd.  Keys are verLev codes 20-24.

    Returns:
        labels : [N] int32, element ∈ {20, 21, 22, 23, 24}
    """
    all_pts, all_lbl = [], []
    for level in LEVELS:
        if level not in gt_vertebra_pcds:
            continue
        pts = gt_vertebra_pcds[level]
        all_pts.append(pts)
        all_lbl.append(np.full(len(pts), level, dtype=np.int32))

    if not all_pts:
        raise ValueError("gt_vertebra_pcds is empty — cannot assign labels.")

    all_pts = np.concatenate(all_pts, axis=0)
    all_lbl = np.concatenate(all_lbl, axis=0)

    tree = cKDTree(all_pts)
    _, indices = tree.query(pred_pcd, workers=-1)
    return all_lbl[indices]


# ─────────────────────────────────────────────────────────────────────────────
# Fallback method: axis-based K-means
# ─────────────────────────────────────────────────────────────────────────────

def assign_labels_kmeans(pred_pcd: np.ndarray,
                         n_vertebrae: int = 5,
                         random_state: int = 42) -> np.ndarray:
    """
    Assign labels by K-means clustering along the axis of maximal variance.
    Clusters are sorted so that label 0 (L1) is at the cranial end and
    label 4 (L5) at the caudal end.

    Args:
        pred_pcd    : [N, 3] whole-spine point cloud (any coordinate space)
        n_vertebrae : number of vertebra clusters (default 5 for L1-L5)
        random_state: reproducibility seed

    Returns:
        labels : [N] int, mapped to verLev codes 20-24 in cranial→caudal order
    """
    from sklearn.cluster import KMeans

    # Project onto axis of maximal variance
    cov = np.cov(pred_pcd.T)
    _, vecs = np.linalg.eigh(cov)
    main_axis = vecs[:, -1]               # largest eigenvector
    proj = pred_pcd @ main_axis           # [N] scalar projection

    km = KMeans(n_clusters=n_vertebrae, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(proj.reshape(-1, 1))  # [N] 0..4

    # Sort cluster centres → map to verLev codes in ascending order
    centres = km.cluster_centers_.flatten()            # [5]
    sorted_idx = np.argsort(centres)                   # ascending projection
    remap = {old: new for new, old in enumerate(sorted_idx)}
    ordered = np.array([remap[c] for c in cluster_ids], dtype=np.int32)

    return np.array([LEVELS[o] for o in ordered], dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: split a point cloud by label dict
# ─────────────────────────────────────────────────────────────────────────────

def split_by_label(pcd: np.ndarray,
                   labels: np.ndarray) -> dict:
    """
    Split pcd into per-vertebra subsets.

    Returns:
        {level (int): np.ndarray [n_level, 3]}
    """
    return {level: pcd[labels == level] for level in LEVELS}
