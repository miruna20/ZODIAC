"""
Method-specific data loaders.

Each loader yields dicts with:
    spine_id          : str   e.g. "sub-verse001"
    pred_pcd          : np.ndarray [n_points, 3]   normalised prediction
    gt_pcd            : np.ndarray [n_points, 3]   normalised GT (same space)
    gt_vertebra_pcds  : dict {level (int): np.ndarray [M, 3]}
                        Per-vertebra GT in the SAME coordinate space as
                        pred_pcd / gt_pcd — used for KD-tree label assignment.
                        Keys are verLev codes {20,21,22,23,24}.
    pred_pcd_raw      : np.ndarray [n_points, 3]   prediction in ORIGINAL space
                        (mm for SITD after stitching; own normalised space for
                        OctFusion / PDR) — used only for the pose metric.
    gt_pcd_raw        : np.ndarray [n_points, 3]   GT in original space
    pred_vertebra_raw : dict {level: np.ndarray} per-vertebra prediction in
                        original space (for pose-metric PLY export)
    gt_vertebra_raw   : dict {level: np.ndarray} per-vertebra GT in original
                        space (for pose-metric PLY export)

Normalisation convention
------------------------
pred_pcd / gt_pcd are centred on the GT whole-spine centroid and divided by
the GT whole-spine bounding-box diagonal — making all metrics unitless and
directly comparable across methods.
"""

from __future__ import annotations

import os
import re
import sys
import numpy as np
import trimesh
from pathlib import Path
from collections import defaultdict

LEVELS      = [20, 21, 22, 23, 24]
LEVEL_NAMES = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5"}
_LEVEL_NAME_TO_CODE = {v: k for k, v in LEVEL_NAMES.items()}  # "L1"→20 … "L5"→24


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def resample(pcd: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Random resample to exactly n points."""
    if len(pcd) == 0:
        return pcd
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pcd), n, replace=(len(pcd) < n))
    return pcd[idx]


def normalise_pair(pred: np.ndarray, gt: np.ndarray):
    """
    Normalise by GT bounding-box diagonal. Returns (pred_n, gt_n, D, center).
    All vertebra GTs must be passed through the same transform for label
    assignment to be consistent.
    """
    gt_min  = gt.min(axis=0)
    gt_max  = gt.max(axis=0)
    center  = (gt_min + gt_max) / 2.0
    D       = float(np.linalg.norm(gt_max - gt_min))
    if D < 1e-8:
        D = 1.0
    return (pred - center) / D, (gt - center) / D, D, center


def extract_spine_id(name: str) -> str | None:
    """Return 'sub-versexxx' from a filename or any string."""
    m = re.search(r"(sub-verse\d+)", name)
    return m.group(1) if m else None


def find_file(directory: Path, spine_id: str, *extensions) -> Path | None:
    """Return the first file in directory whose name contains spine_id."""
    for ext in extensions:
        for f in sorted(directory.glob(f"*{spine_id}*{ext}")):
            return f
    return None


def sample_mesh(path: Path, n: int) -> np.ndarray | None:
    """Load mesh and uniformly sample n surface points."""
    try:
        mesh = trimesh.load(str(path), force="mesh")
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            return None
        return np.array(mesh.sample(n), dtype=np.float32)
    except Exception as e:
        print(f"  Warning: failed to load mesh {path}: {e}")
        return None


def sample_mesh_with_normals(path: Path,
                             n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load mesh and uniformly sample n surface points together with their
    face normals.

    Returns (pts [N,3], normals [N,3]) both float32, or None on failure.
    Normals are unit-length face normals at each sampled point — exact for
    mesh-based methods, no estimation required.  Uniform scaling of the
    coordinate space (as applied by normalise_pair) does not change normal
    directions, so the returned normals are valid after normalisation.
    """
    try:
        mesh = trimesh.load(str(path), force="mesh")
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            return None
        pts, face_idx = trimesh.sample.sample_surface(mesh, n)
        normals = mesh.face_normals[face_idx]
        return np.array(pts, dtype=np.float32), np.array(normals, dtype=np.float32)
    except Exception as e:
        print(f"  Warning: failed to load mesh with normals {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GT vertebra helper — path A: nested directory + bbox transform
# (used for OctFusion when bbox_dir is supplied)
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt_vertebra_pcds_with_bbox(vert_base_dir: Path,
                                     bbox_dir: Path,
                                     spine_id: str,
                                     forcefield_n: int,
                                     center: np.ndarray,
                                     D: float,
                                     n_per_vert: int = 4096) -> dict:
    """
    Load per-vertebra GT meshes for one spine and bring them into OctFusion
    space using the same bbox transform that preprocess_from_mesh.py applied.

    Directory layout expected
    -------------------------
    vert_base_dir/
        {spine_id}_verLev20/
            *forces{forcefield_n}*_scaled.obj
        {spine_id}_verLev21/  ...
    bbox_dir/
        {spine_id}_forcefield{forcefield_n}_*.npz   (any shift variant)

    The transform applied is:
        v_oct = (v_scaled - center_bbox) * scale_bbox * 0.5
    where center_bbox and scale_bbox are derived from bbmin/bbmax/mul in the
    bbox .npz file — exactly as in preprocess_from_mesh.py.

    Returned points are then normalised with the whole-spine (center, D) so
    they are in the same space as pred_pcd / gt_pcd.
    """
    shape_scale = 0.5  # must match preprocess_from_mesh.py

    # Load bbox transform (any shift variant is fine — geometry is shift-independent)
    bbox_files = sorted(bbox_dir.glob(
        f"{spine_id}_forcefield{forcefield_n}_*.npz"))
    if not bbox_files:
        print(f"  Warning: no bbox.npz found for "
              f"{spine_id}_forcefield{forcefield_n} in {bbox_dir}")
        return {}

    print(f"  [paths] bbox       : {bbox_files[0]}")
    bbox      = np.load(str(bbox_files[0]))
    bbmin     = bbox["bbmin"].astype(np.float64)
    bbmax     = bbox["bbmax"].astype(np.float64)
    mul       = float(bbox["mul"])                     # = mesh_scale = 0.8
    center_bbox = (bbmin + bbmax) * 0.5
    max_extent  = float((bbmax - bbmin).max())
    scale_bbox  = 2.0 * mul / max_extent               # same as preprocess_from_mesh.py

    vert_pcds = {}
    vert_paths = {"bbox": str(bbox_files[0])}
    for level in LEVELS:
        vert_dir = vert_base_dir / f"{spine_id}_verLev{level}"
        if not vert_dir.exists():
            print(f"  Warning: vertebra dir not found: {vert_dir}")
            vert_paths[f"verLev{level}"] = "NOT FOUND"
            continue

        candidates = sorted(vert_dir.glob(
            f"*forces{forcefield_n}*_scaled.obj"))
        if not candidates:
            print(f"  Warning: no *forces{forcefield_n}*_scaled.obj in {vert_dir}")
            vert_paths[f"verLev{level}"] = "NOT FOUND"
            continue

        print(f"  [paths] verLev{level} (L{level-19})  : {candidates[0]}")
        vert_paths[f"verLev{level}"] = str(candidates[0])
        pts = sample_mesh(candidates[0], n_per_vert)
        if pts is None:
            continue

        # Bring into OctFusion space (same transform as preprocess_from_mesh.py)
        pts_oct = (pts.astype(np.float64) - center_bbox) * scale_bbox * shape_scale

        # Apply whole-spine normalisation (same as pred_pcd / gt_pcd)
        vert_pcds[level] = ((pts_oct - center) / D).astype(np.float32)

    return vert_pcds, vert_paths


# ─────────────────────────────────────────────────────────────────────────────
# GT vertebra helper — path B: flat directory
# (legacy / PDR fallback; filenames must encode spine ID + verLev code)
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_vertebra_pcds_from_dir(gt_vert_dir: Path,
                                   spine_id: str,
                                   center: np.ndarray,
                                   D: float,
                                   n_per_vert: int = 4096) -> dict:
    """
    Load per-vertebra GT meshes/arrays for one spine from gt_vert_dir.
    Files are expected to contain both the spine ID and the level token:
        <anything>sub-verse001<anything>verLev20<anything>.{obj,ply,npy}
    Returned points are normalised with the SAME (center, D) as the whole spine.
    """
    vert_pcds = {}
    for level in LEVELS:
        pattern_candidates = [
            f"*{spine_id}*verLev{level}*.obj",
            f"*{spine_id}*verLev{level}*.ply",
            f"*{spine_id}*verLev{level}*.npy",
            f"*verLev{level}*{spine_id}*.obj",
            f"*verLev{level}*{spine_id}*.ply",
        ]
        pts = None
        for pat in pattern_candidates:
            matches = list(sorted(gt_vert_dir.glob(pat)))
            if matches:
                f = matches[0]
                if f.suffix == ".npy":
                    pts = np.load(str(f)).astype(np.float32)
                else:
                    pts = sample_mesh(f, n_per_vert)
                break

        if pts is not None:
            vert_pcds[level] = (pts - center) / D   # same normalisation as spine

    return vert_pcds


# ─────────────────────────────────────────────────────────────────────────────
# OctFusion loader
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# GT vertebra helper — path C: phantom (no forcefield, flat dir, bbox transform)
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt_vertebra_pcds_phantom(gt_vert_dir: Path,
                                   bbox_dir: Path,
                                   spine_id: str,
                                   center: np.ndarray,
                                   D: float,
                                   n_per_vert: int = 4096) -> dict:
    """
    Load per-vertebra GT meshes for a phantom spine (Phantom1 / Phantom2).

    GT vertebra meshes are in mm space.  The bbox .npz provides the same
    transform that preprocess_from_mesh.py applied when building the octree dataset,
    bringing them into OctFusion space.  The whole-spine (center, D) is then
    applied for cross-method normalisation.

    File pattern searched: any .obj (or .ply) in gt_vert_dir whose name
    contains 'verLev{N}' — works for both
        'rotated_verLev20_clean.obj'   (Phantom1)
        'sub-verse818_verLev20.obj'    (Phantom2)

    bbox_dir/{spine_id}.npz is required (e.g. Phantom1.npz).
    """
    shape_scale = 0.5  # must match preprocess_from_mesh.py

    bbox_file = bbox_dir / f"{spine_id}.npz"
    if not bbox_file.exists():
        print(f"  Warning: bbox file not found: {bbox_file}")
        return {}

    bbox       = np.load(str(bbox_file))
    bbmin      = bbox["bbmin"].astype(np.float64)
    bbmax      = bbox["bbmax"].astype(np.float64)
    mul        = float(bbox["mul"])
    center_bbox = (bbmin + bbmax) * 0.5
    max_extent  = float((bbmax - bbmin).max())
    scale_bbox  = 2.0 * mul / max_extent

    # Derive patient_id prefix for multi-patient flat directories (e.g. Balgrist).
    # e.g. spine_id="URS08_H1" → patient_id="URS08"
    patient_id = spine_id.split("_")[0] if "_" in spine_id else spine_id

    vert_pcds = {}
    for level in LEVELS:
        # Try patient-specific prefix first (needed for multi-patient flat dirs).
        # If found, use only those to avoid cross-patient contamination.
        candidates = sorted(gt_vert_dir.glob(f"{patient_id}_verLev{level}*.obj"))
        if not candidates:
            candidates = sorted(gt_vert_dir.glob(f"{patient_id}_verLev{level}*.ply"))
        if not candidates:
            # No patient-specific file: try generic only if the pattern clearly
            # does NOT look like a multi-patient directory (i.e. no other patient
            # prefix files are present in the dir at all).
            generic_obj = sorted(gt_vert_dir.glob(f"*verLev{level}*.obj"))
            generic_ply = sorted(gt_vert_dir.glob(f"*verLev{level}*.ply"))
            all_generic = generic_obj + generic_ply
            if len(all_generic) == 1:
                # Single-patient directory (e.g. a phantom dir): the single file
                # for this level is unambiguous regardless of its filename prefix
                # (e.g. 'sub-verse818_verLev20.obj' for Phantom2, which would
                # otherwise be wrongly dropped by the multi-patient guard below).
                candidates = all_generic
            else:
                # Multi-patient flat dir (e.g. Balgrist): only accept files for
                # this patient to avoid cross-patient contamination.
                candidates = [p for p in all_generic
                              if p.stem.startswith(patient_id) or
                                 not any(p.stem.startswith(pid)
                                         for pid in ("URS", "sub-verse", "Phantom"))]
        if not candidates:
            print(f"  Warning: no verLev{level} file for patient {patient_id} in {gt_vert_dir}")
            continue

        print(f"  [paths] verLev{level} (L{level-19})  : {candidates[0]}")
        pts = sample_mesh(candidates[0], n_per_vert)
        if pts is None:
            continue

        # mm → OctFusion space (same transform as preprocess_from_mesh.py)
        pts_oct = (pts.astype(np.float64) - center_bbox) * scale_bbox * shape_scale

        # Apply whole-spine normalisation
        vert_pcds[level] = ((pts_oct - center) / D).astype(np.float32)

    return vert_pcds


def load_octfusion(pred_dir:      str | Path,
                   gt_mesh_dir:   str | Path,
                   gt_vert_dir:   str | Path | None = None,
                   bbox_dir:      str | Path | None = None,
                   n_points:      int = 8192,
                   n_pts_large:   int = 30000,
                   n_per_vert:    int = 4096,
                   spine_list:    str | Path | None = None):
    """
    Load OctFusion completion results.

    Directory layout expected
    -------------------------
    pred_dir/    ← one .obj per whole-spine completion; filename must contain
                   spine ID (e.g. "sub-verse502_forcefield0_shiftx0.05_...obj")
    gt_mesh_dir/ ← GT whole-spine .obj files (same naming convention)

    Per-vertebra GT (two mutually exclusive modes, selected by bbox_dir):

    Mode A — bbox transform (recommended for OctFusion):
        gt_vert_dir  = top-level dir with sub-folders {spine_id}_verLev{N}/
                       containing *forces{N}*_scaled.obj files.
        bbox_dir     = dir with bbox .npz files saved by preprocess_from_mesh.py;
                       used to analytically place each vertebra into OctFusion
                       space without any ICP registration.
        The forcefield index is extracted from the pred filename automatically.

    Mode B — flat directory (legacy / PDR):
        gt_vert_dir  = flat dir whose files encode both spine ID and verLev.
        bbox_dir     = None  (no bbox transform applied).

    Two-stage sampling
    ------------------
    n_pts_large points are sampled from the mesh upfront.  The yielded dict
    contains both:
      pred_pcd / gt_pcd       — resampled to n_points (for whole-spine metrics)
      pred_pcd_large / gt_pcd_large — the full n_pts_large cloud (for splitting)

    Yields
    ------
    One dict per spine (same keys as other loaders).
    """
    pred_dir    = Path(pred_dir)
    gt_mesh_dir = Path(gt_mesh_dir)
    gt_vert_dir = Path(gt_vert_dir) if gt_vert_dir else None
    bbox_dir    = Path(bbox_dir)    if bbox_dir    else None

    pred_files = sorted(pred_dir.glob("*.obj"))
    if not pred_files:
        raise FileNotFoundError(f"No .obj files found in {pred_dir}")

    if spine_list is not None:
        with open(spine_list) as fh:
            wanted = [ln.strip() for ln in fh if ln.strip()]
        # Build a stem→Path lookup for fast exact matching
        stem_to_file = {f.stem: f for f in pred_files}
        filtered = []
        for entry in wanted:
            stem = entry[:-4] if entry.endswith(".obj") else entry  # strip .obj if included
            if stem in stem_to_file:
                filtered.append(stem_to_file[stem])
            else:
                print(f"  Warning: no file found for '{stem}' in {pred_dir}")
        pred_files = filtered
        print(f"  spine_list filter: {len(wanted)} requested → {len(pred_files)} files selected")

    for pred_file in pred_files:
        spine_id = extract_spine_id(pred_file.name)
        if spine_id is None:
            # Fallback for non-VerSe IDs (e.g. phantom files: Phantom1.obj)
            spine_id = pred_file.stem
            print(f"  [info] no sub-versexxx in filename — using stem as spine_id: {spine_id}")

        # Extract forcefield index from filename (e.g. "forcefield0" → 0)
        ff_match     = re.search(r"forcefield(\d+)", pred_file.name)
        forcefield_n = int(ff_match.group(1)) if ff_match else None

        paths_log = {"completion": str(pred_file)}
        print(f"  [paths] completion : {pred_file}")

        # --- prediction (points + face normals from mesh) ---
        pred_result = sample_mesh_with_normals(pred_file, n_pts_large)
        if pred_result is None:
            continue
        pred_pts, pred_normals_raw = pred_result

        # --- GT whole spine ---
        # Match spine_id AND forcefield so forcefield1 completion gets forcefield1 GT
        # (shift does not affect GT geometry, so any shift variant is fine)
        ff_tag  = f"forcefield{forcefield_n}" if forcefield_n is not None else None
        gt_file = find_file(gt_mesh_dir,
                            f"{spine_id}_{ff_tag}" if ff_tag else spine_id,
                            ".obj", ".ply")
        if gt_file is None:
            # Fallback: spine_id only (for datasets without forcefield in filename)
            gt_file = find_file(gt_mesh_dir, spine_id, ".obj", ".ply")
        if gt_file is None:
            print(f"  Skipping {spine_id}: GT mesh not found in {gt_mesh_dir}")
            continue
        paths_log["gt_mesh"] = str(gt_file)
        print(f"  [paths] gt mesh    : {gt_file}")
        gt_result = sample_mesh_with_normals(gt_file, n_pts_large)
        if gt_result is None:
            continue
        gt_pts, gt_normals_raw = gt_result

        # --- normalise large clouds (used for splitting) ---
        # Normals are direction vectors: uniform scaling (1/D) does not change
        # their direction, so no additional transform is needed.
        pred_n_large, gt_n_large, D, center = normalise_pair(pred_pts, gt_pts)

        # --- resample to n_points for whole-spine metrics ---
        # resample() uses seed=42; applying it to normals gives indices aligned
        # with the resampled point cloud.
        pred_n = resample(pred_n_large, n_points)
        gt_n   = resample(gt_n_large,   n_points)

        # --- per-vertebra GT ---
        gt_vert_pcds = {}
        if gt_vert_dir is not None:
            if bbox_dir is not None and forcefield_n is not None:
                # Mode A: nested dirs + bbox transform (analytically aligned)
                gt_vert_pcds, vert_paths = _load_gt_vertebra_pcds_with_bbox(
                    vert_base_dir = gt_vert_dir,
                    bbox_dir      = bbox_dir,
                    spine_id      = spine_id,
                    forcefield_n  = forcefield_n,
                    center        = center,
                    D             = D,
                    n_per_vert    = n_per_vert,
                )
                paths_log.update(vert_paths)
            elif bbox_dir is not None and forcefield_n is None:
                # Mode C: phantom — flat dir + bbox transform (no forcefield)
                gt_vert_pcds = _load_gt_vertebra_pcds_phantom(
                    gt_vert_dir = Path(gt_vert_dir),
                    bbox_dir    = bbox_dir,
                    spine_id    = spine_id,
                    center      = center,
                    D           = D,
                    n_per_vert  = n_per_vert,
                )
            else:
                # Mode B: flat directory (legacy)
                gt_vert_pcds = load_gt_vertebra_pcds_from_dir(
                    gt_vert_dir, spine_id, center, D, n_per_vert=n_per_vert
                )

        yield {
            "spine_id":          spine_id,
            "shift":             pred_file.stem.split("_forcefield")[1]
                                 if "_forcefield" in pred_file.stem else "",
            # whole-spine metrics (n_points)
            "pred_pcd":          pred_n,
            "gt_pcd":            gt_n,
            # pred normals: exact mesh face normals (OctFusion outputs a mesh).
            # gt normals: None → PCA-estimated on demand, consistent with
            # SITD/PDR which also estimate GT normals via PCA.  Using PCA for
            # GT ensures a fair cross-method comparison of NC; the pred-normal
            # source (exact vs PCA) is what legitimately differs between methods.
            "pred_normals":      resample(pred_normals_raw, n_points),
            "gt_normals":        None,
            # large clouds for splitting (n_pts_large)
            "pred_pcd_large":    pred_n_large,
            "gt_pcd_large":      gt_n_large,
            "gt_vertebra_pcds":  gt_vert_pcds,
            "pred_pcd_raw":      resample(pred_pts, n_points),
            "gt_pcd_raw":        resample(gt_pts,   n_points),
            "pred_vertebra_raw": None,
            "gt_vertebra_raw":   None,
            "_center":           center,
            "_D":                D,
            "_paths":            paths_log,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GT vertebra helper — mesh_vertebrae_256 layout (per-vertebra L1..L5 meshes)
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt_vert_mesh_vertebrae_256(
    stem:          str,
    vert_base_dir: Path,
    center:        np.ndarray,
    D:             float,
    n_per_vert:    int = 4096,
) -> dict:
    """
    Load per-vertebra GT meshes from the mesh_vertebrae_256 layout:
        vert_base_dir/{stem}/L1.obj  …  L5.obj
    Meshes are already in OctFusion space [-0.5, 0.5].
    Returned points are normalised with the same (center, D) as the whole spine.
    """
    stem_dir = vert_base_dir / stem
    if not stem_dir.exists():
        print(f"  Warning: vertebra dir not found: {stem_dir}")
        return {}
    vert_pcds = {}
    for name, code in _LEVEL_NAME_TO_CODE.items():
        obj = stem_dir / f"{name}.obj"
        if not obj.exists():
            print(f"  Warning: {obj.name} not found in {stem_dir}")
            continue
        pts = sample_mesh(obj, n_per_vert)
        if pts is None:
            continue
        vert_pcds[code] = ((pts.astype(np.float64) - center) / D).astype(np.float32)
    return vert_pcds
