"""
Unified shape-completion metrics: Chamfer Distance (L1/L2), F1, EMD,
Hausdorff-95 (H95), and Normal Consistency (NC).

All metrics are computed after normalising both pred and gt by the GT
bounding-box diagonal, making them scale-invariant and directly comparable
across methods and vertebra levels.

EMD is approximated on a 2048-point subsample (tractable on CPU; GPU
implementations from the existing codebases are tried first).

NC requires surface normals.  For mesh-based methods (OctFusion) exact face
normals are supplied by the loader; for point-cloud methods they are estimated
via PCA using Open3D (less reliable near artifact regions — which is exactly
what the metric is designed to reveal).  NC is only computed when
``compute_nc=True`` is passed to ``compute_all_metrics``.

H95 is always computed; it is derived from the same NN distances used for CD
so there is zero additional cost.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

EMD_SUBSAMPLE = 2048   # point count used for EMD approximation


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_by_gt_bbox(pred: np.ndarray,
                         gt: np.ndarray):
    """
    Centre both clouds on the GT centroid and scale by the GT bounding-box
    diagonal.  After this transform the GT fits in a unit ball.

    Returns:
        pred_n  : [N, 3] normalised prediction
        gt_n    : [M, 3] normalised GT
        D       : float  GT bounding-box diagonal (for reporting in original units)
        center  : [3]    GT centroid used for centering
    """
    gt_min  = gt.min(axis=0)
    gt_max  = gt.max(axis=0)
    center  = (gt_min + gt_max) / 2.0
    D       = float(np.linalg.norm(gt_max - gt_min))
    if D < 1e-8:
        D = 1.0
    pred_n = (pred - center) / D
    gt_n   = (gt   - center) / D
    return pred_n, gt_n, D, center


# ─────────────────────────────────────────────────────────────────────────────
# Chamfer Distance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nn_distances(pred: np.ndarray,
                  gt:   np.ndarray):
    """
    Return nearest-neighbour Euclidean distances in both directions.
    d_p2g : [N]  distance from each pred point to its closest GT point
    d_g2p : [M]  distance from each GT   point to its closest pred point
    """
    d_p2g, _ = cKDTree(gt).query(pred, workers=-1)
    d_g2p, _ = cKDTree(pred).query(gt,  workers=-1)
    return d_p2g, d_g2p


def compute_cd_l1(d_p2g: np.ndarray, d_g2p: np.ndarray) -> float:
    """
    CD-L1: symmetric mean of Euclidean (not squared) distances.
    CD-L1 = 0.5 * (mean(d_p2g) + mean(d_g2p))
    Matches cd_p in MVP / SITD benchmarks.
    """
    return float(0.5 * (d_p2g.mean() + d_g2p.mean()))


def compute_cd_l2(d_p2g: np.ndarray, d_g2p: np.ndarray) -> float:
    """
    CD-L2: sum of mean SQUARED Euclidean distances (no /2).
    CD-L2 = mean(d_p2g²) + mean(d_g2p²)
    Matches cd_t in MVP / SITD benchmarks exactly.
    """
    return float(np.mean(d_p2g ** 2) + np.mean(d_g2p ** 2))


def hausdorff_95(d_p2g: np.ndarray, d_g2p: np.ndarray) -> float:
    """
    Hausdorff distance at the 95th percentile.
    H95 = max(p95(d_pred→gt), p95(d_gt→pred))

    Less sensitive to single outlier spikes than the true Hausdorff (max),
    while still exposing systematic artifact regions that CD/F1 average away.
    Lower is better.  Computed for free from the NN distances already used
    for CD and F1.
    """
    return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))


# ─────────────────────────────────────────────────────────────────────────────
# Normal Consistency
# ─────────────────────────────────────────────────────────────────────────────

def estimate_normals_o3d(pts: np.ndarray,
                         radius: float = 0.05,
                         max_nn: int = 30) -> np.ndarray:
    """
    Estimate surface normals using Open3D PCA-based normal estimation.

    pts    : [N, 3] point cloud in a consistent coordinate frame
    radius : neighbourhood radius (same units as pts; 0.05 ≈ 5 % of a
             unit-diagonal bbox — appropriate for normalised spine clouds)
    max_nn : max neighbours used for PCA

    Returns [N, 3] unit normal vectors.  Falls back to zero vectors if
    Open3D is unavailable or estimation fails.
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        k_orient = min(20, len(pts) - 1)
        if k_orient >= 1:
            pcd.orient_normals_consistent_tangent_plane(k=k_orient)
        normals = np.asarray(pcd.normals, dtype=np.float32)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return normals / norms
    except Exception:
        return np.zeros_like(pts, dtype=np.float32)


def normal_consistency(pred_pts:     np.ndarray,
                       pred_normals: np.ndarray,
                       gt_pts:       np.ndarray,
                       gt_normals:   np.ndarray) -> float:
    """
    Normal Consistency (NC): mean absolute cosine similarity between each
    predicted surface normal and the GT surface normal at the nearest GT point.

    NC = (1/N) Σ_i  |n_pred_i · n_gt_nn(i)|

    Range [0, 1]; higher is better (1.0 = perfect normal alignment).

    For mesh-based methods (OctFusion) normals are exact face normals —
    reliable everywhere.  For point-cloud methods they are PCA-estimated and
    less reliable near artifact regions, which is precisely what this metric
    is designed to reveal.
    """
    _, idx = cKDTree(gt_pts).query(pred_pts)
    gt_nn_normals = gt_normals[idx]                          # (N, 3)
    dot = np.abs((pred_normals * gt_nn_normals).sum(axis=1)) # (N,)
    return float(dot.mean())


# ─────────────────────────────────────────────────────────────────────────────
# F1 Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_f1(pred:      np.ndarray,
               gt:        np.ndarray,
               threshold: float,
               d_p2g:     np.ndarray | None = None,
               d_g2p:     np.ndarray | None = None) -> float:
    """
    F1 score at Euclidean distance `threshold` (normalised space).

    precision = fraction of predicted points within threshold of any GT point
    recall    = fraction of GT points within threshold of any predicted point
    F1        = harmonic mean of precision and recall

    Pre-computed distances (d_p2g, d_g2p) can be passed to avoid recomputing
    KD-trees when already computed for CD.
    """
    if d_p2g is None or d_g2p is None:
        d_p2g, d_g2p = _nn_distances(pred, gt)
    precision = float((d_p2g < threshold).mean())
    recall    = float((d_g2p < threshold).mean())
    if precision + recall > 0:
        return 2.0 * precision * recall / (precision + recall)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Earth Mover's Distance  (approximate, with GPU fallback)
# ─────────────────────────────────────────────────────────────────────────────

def compute_emd(pred: np.ndarray,
                gt:   np.ndarray,
                subsample: int = EMD_SUBSAMPLE,
                seed: int = 42) -> float:
    """
    Approximate Earth Mover's Distance.

    1. Subsample both clouds to `subsample` points (default 2048).
    2. scipy linear_sum_assignment on CPU.

    Returns EMD in the same units as pred / gt.

    NOTE: EMD is not a reported Table-1 metric; it is retained only for
    completeness and is skipped by ``compute_all_metrics(..., with_emd=False)``.
    """
    rng = np.random.default_rng(seed)

    def _sub(arr):
        if len(arr) <= subsample:
            return arr
        idx = rng.choice(len(arr), subsample, replace=False)
        return arr[idx]

    pred_s = _sub(pred)
    gt_s   = _sub(gt)

    # CPU assignment (slow for large subsample — keep ≤ 512 if very slow)
    from scipy.optimize import linear_sum_assignment
    diff = pred_s[:, None, :] - gt_s[None, :, :]   # [N, M, 3]
    D_mat = np.sqrt((diff ** 2).sum(-1))            # [N, M]
    r, c  = linear_sum_assignment(D_mat)
    return float(D_mat[r, c].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute all metrics for one sample
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(pred: np.ndarray,
                        gt:   np.ndarray,
                        f1_threshold_ratio: float = 0.01,
                        emd_subsample: int = EMD_SUBSAMPLE,
                        pred_normals: np.ndarray | None = None,
                        gt_normals:   np.ndarray | None = None,
                        compute_nc: bool = False,
                        with_emd: bool = True) -> dict:
    """
    Compute CD-L1, CD-L2, F1, EMD, H95, and optionally NC for one pair.

    Args:
        pred               : [N, 3] predicted point cloud
        gt                 : [M, 3] GT point cloud
        f1_threshold_ratio : F1 threshold as fraction of GT bbox diagonal
                             (default 0.01 → 1 % of spine/vertebra extent)
        emd_subsample      : number of points used for EMD approximation
        pred_normals       : [N, 3] unit normals for pred (optional).
                             For OctFusion, pass exact mesh face normals.
                             If None and compute_nc=True, normals are
                             estimated from pred via PCA (Open3D).
        gt_normals         : [M, 3] unit normals for GT (optional).
                             If None and compute_nc=True, estimated via PCA.
        compute_nc         : if True, compute Normal Consistency (NC).
                             Adds Open3D normal-estimation overhead when
                             normals are not pre-supplied.

    Returns:
        dict with keys: cd_l1, cd_l2, f1, emd, bbox_diag, h95, nc
        nc is NaN when compute_nc=False.
    """
    _nan = {"cd_l1": np.nan, "cd_l2": np.nan, "f1": np.nan,
            "emd": np.nan, "bbox_diag": np.nan, "h95": np.nan, "nc": np.nan}
    if len(pred) == 0 or len(gt) == 0:
        return _nan

    pred_n, gt_n, D, _ = normalise_by_gt_bbox(pred, gt)
    threshold = f1_threshold_ratio          # threshold in normalised space (D=1)

    d_p2g, d_g2p = _nn_distances(pred_n, gt_n)
    cd_l1 = compute_cd_l1(d_p2g, d_g2p)
    cd_l2 = compute_cd_l2(d_p2g, d_g2p)
    f1    = compute_f1(pred_n, gt_n, threshold, d_p2g=d_p2g, d_g2p=d_g2p)
    emd   = compute_emd(pred_n, gt_n, subsample=emd_subsample) if with_emd else np.nan
    h95   = hausdorff_95(d_p2g, d_g2p)

    # NC: use supplied normals or estimate from point clouds
    nc = np.nan
    if compute_nc:
        pn = pred_normals if pred_normals is not None else estimate_normals_o3d(pred_n)
        gn = gt_normals   if gt_normals   is not None else estimate_normals_o3d(gt_n)
        nc = normal_consistency(pred_n, pn, gt_n, gn)

    return {"cd_l1": float(cd_l1), "cd_l2": float(cd_l2),
            "f1": float(f1), "emd": float(emd), "bbox_diag": D,
            "h95": float(h95), "nc": float(nc)}


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(values: list) -> dict:
    """mean / std / median for a list of floats (NaN values excluded)."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "n": 0}
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "median": float(np.median(arr)),
        "n":      int(len(arr)),
    }
