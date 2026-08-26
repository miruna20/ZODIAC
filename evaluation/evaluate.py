"""
Single entry point for the ZODIAC Table-1 evaluation.

Replaces the old driver sprawl (ws_10seed_table.py, vw_10seed.py,
ws_hd95_10seed.py, add_h95_mm.py, build_results_table.py, run_phantom_evaluation.sh)
with one script that, for the three octfusion-based methods (ZODIAC, TP-ODIAC,
ZODIAC-clean), evaluates both Phantom and Balgrist at whole-spine (WS) and
vertebra-wise (VW) level and writes one tidy CSV.

Metrics (paper Table 1 only): CD (cd_l2 x1e4), F1, HD95 (mm). No EMD, no CD-L1.

Every metric is recomputed fresh from the seed meshes with a fixed point-sampling
seed (no dependency on any pre-computed per_scan_summary.csv). Averaged over
`--num_seeds` stochastic samples (seeds 42 .. 42+num_seeds-1) per scan, then
aggregated to group mean +/- population std (ddof=0):
  WS : over per-scan values within the group.
  VW : over per-(scan, level) values within the group.

Output CSV (one row per dataset-group x method x eval-level):
  dataset, method, level, cd, cd_std, f1, f1_std, hd95mm, hd95mm_std

Input roots default to the in-repo layout (data/, results/inference/); override via
CLI (--data_root / --pred_base) or env (ZODIAC_DATA_ROOT / ZODIAC_RESULTS_DIR).

Run:  conda activate octfusion
      python evaluation/evaluate.py --num_seeds 10 --output evaluation/results/table.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from loaders import normalise_pair, resample, sample_mesh, _load_gt_vert_mesh_vertebrae_256  # noqa: E402
from label_assignment import assign_labels_kdtree, split_by_label                            # noqa: E402
from metrics_utils import compute_all_metrics                                                # noqa: E402

import trimesh  # noqa: E402


# ── fixed evaluation constants (match the original drivers) ──────────────────
N_POINTS = 8192        # whole-spine sampling
N_PTS_LARGE = 30000    # dense sampling before per-vertebra splitting
N_PER_VERT = 4096      # points per vertebra for VW metrics
F1_THRESH = 0.01       # F1 threshold as fraction of GT bbox diagonal
SAMPLING_SEED = 0      # numpy seed for reproducible surface sampling

LEVELS = [20, 21, 22, 23, 24]   # verLev codes L1..L5

# Scans / groups (hardcoded, as the original scripts do).
PHANTOM_SCANS = ["Phantom1", "Phantom2"]
BALGRIST_SCANS = ["URS08_H1", "URS08_R2", "URS36_H2", "URS36_D2", "URS45_H2", "URS45_R2"]
DIFFICULT = {"URS36_D2", "URS45_H2", "URS45_R2"}

METHODS = {"zodiac": "ZODIAC", "tpodiac": "TP-ODIAC", "zodiac_clean": "ZODIAC-clean"}

# Output group -> (dataset, difficult-only?)
GROUPS = [("Phantom", "phantom", False),
          ("Balgrist_all", "balgrist", False),
          ("Balgrist_difficult", "balgrist", True)]


# ── input roots (repo-relative defaults) ─────────────────────────────────────
def default_roots() -> dict:
    """Repo-relative input roots. Override via CLI or env
    (ZODIAC_DATA_ROOT / ZODIAC_RESULTS_DIR).

    data_root/  → spines_<ds>/{mesh_256, bbox_256, mesh_vertebrae_256}
    pred_base/  → <method>/<dataset>/<scan>/seed_<n>.obj  (multi-seed reorg layout)
    """
    data_root = Path(os.environ.get("ZODIAC_DATA_ROOT", REPO_ROOT / "data"))
    pred_base = Path(os.environ.get("ZODIAC_RESULTS_DIR",
                                    REPO_ROOT / "results" / "inference"))
    return {"data_root": data_root, "pred_base": pred_base}


def dataset_scans(dataset: str) -> list[str]:
    return PHANTOM_SCANS if dataset == "phantom" else BALGRIST_SCANS


def sample_surface(path: Path, n: int) -> np.ndarray | None:
    """Deterministic uniform surface sampling of a mesh to n points."""
    np.random.seed(SAMPLING_SEED)
    try:
        mesh = trimesh.load(str(path), force="mesh")
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            return None
        return np.asarray(mesh.sample(n), dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"  Warning: failed to load mesh {path}: {e}")
        return None


def oct_scale(bbox_dir: Path, scan: str) -> float:
    """mm per OctFusion unit for a scan = max_extent / mul, from its bbox .npz."""
    b = np.load(str(bbox_dir / f"{scan}.npz"))
    return float((b["bbmax"].astype(np.float64)
                  - b["bbmin"].astype(np.float64)).max()) / float(b["mul"])


def pred_mesh_path(roots: dict, method: str, dataset: str, scan: str, seed: int) -> Path:
    return roots["pred_base"] / method / dataset / scan / f"seed_{seed}.obj"


def method_available(roots: dict, method: str, seeds: list[int]) -> bool:
    """True if at least one prediction mesh exists for this method (any
    dataset/scan/seed). Lets evaluation run over whatever the user has
    produced, skipping methods whose inference hasn't been run yet."""
    for dataset in ("phantom", "balgrist"):
        for scan in dataset_scans(dataset):
            for seed in seeds:
                if pred_mesh_path(roots, method, dataset, scan, seed).exists():
                    return True
    return False


# ── whole-spine (WS) ─────────────────────────────────────────────────────────
def ws_per_scan(roots: dict, method: str, dataset: str, scan: str,
                seeds: list[int]) -> dict | None:
    """WS metrics for one scan = mean over available seeds. None if no seeds."""
    gt_mesh = roots["data_root"] / f"spines_{'phantoms' if dataset == 'phantom' else 'balgrist'}" / "mesh_256" / f"{scan}.obj"
    bbox_dir = roots["data_root"] / f"spines_{'phantoms' if dataset == 'phantom' else 'balgrist'}" / "bbox_256"
    gt = sample_surface(gt_mesh, N_POINTS)
    if gt is None:
        print(f"  [skip] {method}/{scan}: GT mesh missing ({gt_mesh})")
        return None
    s = oct_scale(bbox_dir, scan)
    cds, f1s, h95s = [], [], []
    for seed in seeds:
        p = pred_mesh_path(roots, method, dataset, scan, seed)
        if not p.exists():
            continue
        pred = sample_surface(p, N_POINTS)
        if pred is None:
            continue
        m = compute_all_metrics(pred, gt, f1_threshold_ratio=F1_THRESH, with_emd=False)
        if not np.isfinite(m["cd_l2"]):
            continue
        cds.append(m["cd_l2"]); f1s.append(m["f1"])
        h95s.append(m["h95"] * m["bbox_diag"] * s)
    if not cds:
        print(f"  [skip] {method}/{scan}: no seed meshes")
        return None
    print(f"  {method:12s} WS {scan:10s} seeds={len(cds):2d}  "
          f"CD={np.mean(cds) * 1e4:7.3f} F1={np.mean(f1s):.3f} "
          f"HD95={np.mean(h95s):6.3f}mm")
    return {"cd_l2": float(np.mean(cds)), "f1": float(np.mean(f1s)),
            "h95_mm": float(np.mean(h95s))}


# ── vertebra-wise (VW) ───────────────────────────────────────────────────────
def vw_gt_cache(roots: dict, dataset: str, scan: str):
    """(gt_n_large, D, center, gt_vert_pcds) — fixed across seeds; None on failure."""
    ds_dir = roots["data_root"] / f"spines_{'phantoms' if dataset == 'phantom' else 'balgrist'}"
    gt_mesh = ds_dir / "mesh_256" / f"{scan}.obj"
    vert_dir = ds_dir / "mesh_vertebrae_256"
    np.random.seed(SAMPLING_SEED)
    gt_pts = sample_mesh(gt_mesh, N_PTS_LARGE)
    if gt_pts is None:
        return None
    # center / D depend only on the GT; a dummy pred fetches the same transform.
    _, gt_n_large, D, center = normalise_pair(gt_pts.copy(), gt_pts)
    # Form-2 GT: per-stem mesh_vertebrae_256/<scan>/{L1..L5}.obj, already in octree space.
    gt_vert = _load_gt_vert_mesh_vertebrae_256(scan, vert_dir, center, D, n_per_vert=N_PER_VERT)
    return gt_n_large, D, center, gt_vert


def vw_seed_levels(pred_pts, gt_n_large, D, center, gt_vert, s) -> dict:
    """Per-level WS-space metrics for one seed → {level: {cd_l2, f1, h95_mm}}."""
    pred_n_large = (resample(pred_pts, N_PTS_LARGE).astype(np.float64) - center) / D
    pred_labels = assign_labels_kdtree(pred_n_large, gt_vert)
    gt_labels = assign_labels_kdtree(gt_n_large, gt_vert)
    pred_split = split_by_label(pred_n_large, pred_labels)
    gt_split = split_by_label(gt_n_large, gt_labels)
    out = {}
    for lv in LEVELS:
        pv = pred_split.get(lv, np.empty((0, 3)))
        gv = gt_split.get(lv, np.empty((0, 3)))
        if len(pv) < 10 or len(gv) < 10:
            continue
        m = compute_all_metrics(resample(pv, N_PER_VERT), resample(gv, N_PER_VERT),
                                f1_threshold_ratio=F1_THRESH, with_emd=False)
        if not np.isfinite(m["cd_l2"]):
            continue
        out[lv] = {"cd_l2": m["cd_l2"], "f1": m["f1"],
                   "h95_mm": m["h95"] * m["bbox_diag"] * s}
    return out


def vw_per_scan(roots: dict, method: str, dataset: str, scan: str,
                seeds: list[int]) -> list[dict]:
    """Per-(scan, level) seed-mean rows for one scan (possibly empty)."""
    cache = vw_gt_cache(roots, dataset, scan)
    if cache is None:
        print(f"  [skip] {method}/{scan}: GT cache failed")
        return []
    gt_n_large, D, center, gt_vert = cache
    if not gt_vert:
        print(f"  [skip] {method}/{scan}: no GT vertebrae")
        return []
    bbox_dir = roots["data_root"] / f"spines_{'phantoms' if dataset == 'phantom' else 'balgrist'}" / "bbox_256"
    s = oct_scale(bbox_dir, scan)

    acc = {lv: {"cd_l2": [], "f1": [], "h95_mm": []} for lv in LEVELS}
    n_seed_ok = 0
    for seed in seeds:
        p = pred_mesh_path(roots, method, dataset, scan, seed)
        if not p.exists():
            continue
        np.random.seed(SAMPLING_SEED)
        pred = sample_mesh(p, N_PTS_LARGE)
        if pred is None:
            continue
        per_lv = vw_seed_levels(pred, gt_n_large, D, center, gt_vert, s)
        if per_lv:
            n_seed_ok += 1
        for lv, m in per_lv.items():
            for k in ("cd_l2", "f1", "h95_mm"):
                acc[lv][k].append(m[k])

    rows = []
    for lv in LEVELS:
        if not acc[lv]["cd_l2"]:
            continue
        rows.append({"scan": scan, "level": lv,
                     "cd_l2": float(np.mean(acc[lv]["cd_l2"])),
                     "f1": float(np.mean(acc[lv]["f1"])),
                     "h95_mm": float(np.mean(acc[lv]["h95_mm"]))})
    levs = sorted(lv for lv in LEVELS if acc[lv]["cd_l2"])
    print(f"  {method:12s} VW {scan:10s} seeds_ok={n_seed_ok:2d} "
          f"levels={[f'L{l - 19}' for l in levs]}")
    return rows


# ── aggregation ──────────────────────────────────────────────────────────────
def group_cell(values: list[float]) -> tuple[float, float]:
    """(mean, population std) over group members; std=0 for a single member."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(v.std(ddof=0))


def aggregate(per_scan: dict, level: str, method: str) -> list[dict]:
    """per_scan: {(method, dataset, scan): unit or [units]} → group rows.

    WS units are per-scan dicts; VW units are per-(scan, level) dicts.  Group
    std for both is taken over the units falling inside the group.
    """
    out = []
    for group_name, dataset, diff_only in GROUPS:
        units = []
        for scan in dataset_scans(dataset):
            if diff_only and scan not in DIFFICULT:
                continue
            key = (method, dataset, scan)
            if key not in per_scan:
                continue
            u = per_scan[key]
            units.extend(u if isinstance(u, list) else [u])
        if not units:
            continue
        cd_mean, cd_std = group_cell([u["cd_l2"] * 1e4 for u in units])
        f1_mean, f1_std = group_cell([u["f1"] for u in units])
        hd_mean, hd_std = group_cell([u["h95_mm"] for u in units])
        out.append({"dataset": group_name, "method": METHODS[method], "level": level,
                    "cd": cd_mean, "cd_std": cd_std, "f1": f1_mean, "f1_std": f1_std,
                    "hd95mm": hd_mean, "hd95mm_std": hd_std})
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_seeds", type=int, default=10,
                    help="average over seeds 42 .. 42+num_seeds-1 (default 10)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=list(METHODS),
                    help="methods to evaluate (default: all three)")
    ap.add_argument("--output", type=Path, default=HERE / "results" / "table.csv",
                    help="output CSV path")
    # Input-root overrides (defaults are repo-relative; see default_roots()).
    r = default_roots()
    ap.add_argument("--data_root", type=Path, default=r["data_root"],
                    help="repo data/ dir: spines_<ds>/{mesh_256,bbox_256,mesh_vertebrae_256}")
    ap.add_argument("--pred_base", type=Path, default=r["pred_base"],
                    help="prediction root: <pred_base>/<method>/<dataset>/<scan>/seed_<n>.obj")
    args = ap.parse_args()

    roots = {"pred_base": args.pred_base, "data_root": args.data_root}
    seeds = list(range(42, 42 + args.num_seeds))
    datasets = ["phantom", "balgrist"]

    # Only evaluate methods that actually have predictions on disk, so the run
    # succeeds even if the user hasn't run inference for every method yet.
    available = [m for m in args.methods if method_available(roots, m, seeds)]
    missing = [m for m in args.methods if m not in available]
    if available:
        print("Evaluating results for: "
              + ", ".join(METHODS[m] for m in available))
    if missing:
        print("Missing results (skipped — run inference for these first): "
              + ", ".join(METHODS[m] for m in missing))
        print(f"  (looked under {roots['pred_base']}/<method>/<dataset>/<scan>/seed_<n>.obj)")
    if not available:
        print(f"\nNo predictions found for any requested method under "
              f"{roots['pred_base']}. Nothing to evaluate.")
        return

    all_rows = []
    for method in available:
        print(f"\n=== {METHODS[method]} — WS ===")
        ws = {}
        for dataset in datasets:
            for scan in dataset_scans(dataset):
                cell = ws_per_scan(roots, method, dataset, scan, seeds)
                if cell is not None:
                    ws[(method, dataset, scan)] = cell
        all_rows += aggregate(ws, "WS", method)

        print(f"\n=== {METHODS[method]} — VW ===")
        vw = {}
        for dataset in datasets:
            for scan in dataset_scans(dataset):
                rows = vw_per_scan(roots, method, dataset, scan, seeds)
                if rows:
                    vw[(method, dataset, scan)] = rows
        all_rows += aggregate(vw, "VW", method)

    # ── write tidy CSV ──
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "level", "cd", "cd_std", "f1", "f1_std",
              "hd95mm", "hd95mm_std"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                        for k in fields})
    print(f"\nWrote {len(all_rows)} rows -> {args.output}")

    # ── compact human-readable summary ──
    print("\n=== Table 1 (CD x1e4, F1, HD95 mm) — mean +/- std(ddof=0) ===")
    print(f"{'dataset':20s} {'method':13s} {'lvl':3s}  "
          f"{'CD':>15s} {'F1':>13s} {'HD95':>15s}")
    for row in all_rows:
        print(f"{row['dataset']:20s} {row['method']:13s} {row['level']:3s}  "
              f"{row['cd']:6.2f}+/-{row['cd_std']:5.2f}  "
              f"{row['f1']:.3f}+/-{row['f1_std']:.3f}  "
              f"{row['hd95mm']:6.2f}+/-{row['hd95mm_std']:5.2f}")


if __name__ == "__main__":
    main()
