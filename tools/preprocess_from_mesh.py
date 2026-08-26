#!/usr/bin/env python3
"""Example: preprocess your own spine meshes into the OctFusion `dataset_<size>` format.

This is a **reference/example** script (it replaces the old per-dataset
`prepare_spines*.py` scripts, which were near-identical and differed only in how the raw
files were read). It turns a folder of raw meshes into the exact on-disk layout the ZODIAC
data loader and evaluation expect:

    <data_root>/<dataset>/
    ├── dataset_<size>/<sample>/{pointcloud.npz, sdf.npz, points.npz[, partial.npz]}
    ├── bbox_<size>/<sample>.npz
    ├── sdf_<size>/<sample>.npy
    ├── mesh_<size>/<sample>.obj            (+ <sample>_partial.ply if a partial pcd exists)
    └── filelist/{train,test,validation,all}.txt   (you provide these — see below)

You do NOT need this script if you downloaded the released data: it already ships in this
format. Use it only to prepare a NEW dataset from your own raw meshes.

--------------------------------------------------------------------------------------------
Expected raw input layout (the ONE part you adapt to your own data — see `load_raw()` below):

    <raw_data_root>/<sample>/<sample>.obj          # the raw mesh (required)
    <raw_data_root>/<sample>/<sample>_partial.pcd  # a partial point cloud (optional)

The samples to process come from a filelist (default `<data_root>/<dataset>/filelist/all.txt`,
one sample id per line); if it is absent, every sub-directory of `<raw_data_root>` is used.

--------------------------------------------------------------------------------------------
Usage:
    conda activate octfusion
    # one shot (mesh -> sdf, then sample point cloud / sdf / occupancy):
    python tools/preprocess_from_mesh.py --run all \
        --raw_data_root /path/to/raw/spines --dataset spines_mine --sdf_size 256

    # or run the two stages separately:
    python tools/preprocess_from_mesh.py --run convert_mesh_to_sdf --raw_data_root ... --dataset spines_mine
    python tools/preprocess_from_mesh.py --run generate_dataset    --dataset spines_mine

NOTE: `--dataset` must contain the substring "spines" — the data loader
(`datasets/dataloader.py`) selects the spine dataset by that keyword.

The scaling constants below (shape_scale / mesh_scale / level) match the released data;
keep them unchanged so your data is compatible with the shipped checkpoints and configs.
"""

import argparse
import logging
import os

import numpy as np
import ocnn
import torch
import trimesh
from tqdm import tqdm

try:
    import mesh2sdf
except ImportError as e:  # pragma: no cover
    raise SystemExit("mesh2sdf is required: pip install mesh2sdf") from e

try:
    import open3d as o3d
except ImportError:
    o3d = None  # only needed if you preprocess partial point clouds

from plyfile import PlyData, PlyElement

logging.getLogger("trimesh").setLevel(logging.ERROR)

# ── fixed geometry constants (MUST match the released data / configs) ───────────────────
SHAPE_SCALE = 0.5   # final shapes live in [-0.5, 0.5] (== point_scale in the .yaml configs)
MESH_SCALE = 0.8    # see microsoft/DualOctreeGNN issue #2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# ── raw I/O — ADAPT THIS to your own raw data format ────────────────────────────────────
def load_raw(raw_data_root, sample):
    """Return (mesh, partial_points_or_None) for one sample.

    This is the only function that knows your raw layout. To support a different raw
    format, change the paths / readers here; the rest of the pipeline is format-agnostic.
    """
    mesh_path = os.path.join(raw_data_root, sample, f"{sample}.obj")
    mesh = trimesh.load(mesh_path, force="mesh")

    partial = None
    pcd_path = os.path.join(raw_data_root, sample, f"{sample}_partial.pcd")
    if os.path.exists(pcd_path):
        if o3d is None:
            raise SystemExit("open3d is required to read partial .pcd files")
        partial = o3d.io.read_point_cloud(pcd_path)
    return mesh, partial


# ── paths / filelist helpers ────────────────────────────────────────────────────────────
def dataset_dirs(data_root, dataset, size):
    root = os.path.join(data_root, dataset)
    return {
        "root": root,
        "dataset": os.path.join(root, f"dataset_{size}"),
        "mesh": os.path.join(root, f"mesh_{size}"),
        "bbox": os.path.join(root, f"bbox_{size}"),
        "sdf": os.path.join(root, f"sdf_{size}"),
        "filelist": os.path.join(root, "filelist"),
    }


def ensure_parent(*paths):
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)


def get_sample_ids(args, dirs):
    """Samples come from filelist/all.txt if present, else raw_data_root sub-dirs."""
    all_txt = args.filelist or os.path.join(dirs["filelist"], "all.txt")
    if os.path.exists(all_txt):
        with open(all_txt) as fid:
            ids = [line.split()[0] for line in fid if line.strip()]
    elif args.raw_data_root and os.path.isdir(args.raw_data_root):
        ids = sorted(d for d in os.listdir(args.raw_data_root)
                     if os.path.isdir(os.path.join(args.raw_data_root, d)))
    else:
        raise SystemExit(f"No filelist at {all_txt} and no --raw_data_root to enumerate.")
    end = args.end if args.end is not None else len(ids)
    return ids[args.start:end]


# ── stage 1: mesh -> SDF (+ bbox, manifold mesh, optional partial pcd) ──────────────────
def convert_mesh_to_sdf(args, dirs):
    size = args.sdf_size
    level = 2.0 / size
    print("-> mesh2sdf")
    for sample in tqdm(get_sample_ids(args, dirs), ncols=80):
        mesh, partial = load_raw(args.raw_data_root, sample)

        out_mesh = os.path.join(dirs["mesh"], f"{sample}.obj")
        out_bbox = os.path.join(dirs["bbox"], f"{sample}.npz")
        out_sdf = os.path.join(dirs["sdf"], f"{sample}.npy")
        ensure_parent(out_mesh, out_bbox, out_sdf)

        # normalise mesh to [-mesh_scale, mesh_scale] for mesh2sdf
        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * MESH_SCALE / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale

        sdf, mesh_new = mesh2sdf.compute(
            vertices, mesh.faces, size, fix=False, level=level, return_mesh=True)
        mesh_new.vertices = mesh_new.vertices * SHAPE_SCALE

        np.savez(out_bbox, bbmax=bbmax, bbmin=bbmin, mul=MESH_SCALE)
        np.save(out_sdf, sdf)
        mesh_new.export(out_mesh)

        # optional: partial point cloud (needed for inference conditioning)
        if partial is not None:
            points = (np.asarray(partial.points) - center) * scale * SHAPE_SCALE
            partial.points = o3d.utility.Vector3dVector(points)
            partial.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=30))
            normals = np.asarray(partial.normals)
            out_partial = os.path.join(dirs["dataset"], sample, "partial.npz")
            out_ply = os.path.join(dirs["mesh"], f"{sample}_partial.ply")
            ensure_parent(out_partial, out_ply)
            np.savez(out_partial, points=points.astype(np.float16),
                     normals=normals.astype(np.float16))
            pcd_out = o3d.geometry.PointCloud()
            pcd_out.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(out_ply, pcd_out)


# ── stage 2a: sample a dense surface point cloud from the manifold mesh ──────────────────
def sample_pts_from_mesh(args, dirs):
    num_samples = 100000
    print("-> sample_pts_from_mesh")
    for sample in tqdm(get_sample_ids(args, dirs), ncols=80):
        mesh = trimesh.load(os.path.join(dirs["mesh"], f"{sample}.obj"), force="mesh")
        points, idx = trimesh.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[idx]
        out = os.path.join(dirs["dataset"], sample, "pointcloud.npz")
        ensure_parent(out)
        np.savez(out, points=points.astype(np.float16), normals=normals.astype(np.float16))


# ── stage 2b: sample octree-node SDF values + gradients for training ─────────────────────
def sample_sdf(args, dirs):
    size = args.sdf_size
    depth, full_depth = 6, 4          # octree resolution 8^6 == 64^3 at full_depth 4
    sample_num = 4                    # points sampled per octree node
    grid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    print("-> sample_sdf")
    for sample in tqdm(get_sample_ids(args, dirs), ncols=80):
        pts = np.load(os.path.join(dirs["dataset"], sample, "pointcloud.npz"))
        sdf = torch.from_numpy(np.load(os.path.join(dirs["sdf"], f"{sample}.npy")))
        points = (pts["points"].astype(np.float32)) / SHAPE_SCALE  # -> [-1, 1]
        normals = pts["normals"].astype(np.float32)

        octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
        octree.build_octree(ocnn.octree.Points(torch.from_numpy(points),
                                               torch.from_numpy(normals)))

        xyzs, grads, sdfs = [], [], []
        for d in range(full_depth, depth + 1):
            x, y, z, _ = octree.xyzb(d)
            xyz = torch.stack((x, y, z), dim=1).float().unsqueeze(1)
            xyz = (xyz + torch.rand(xyz.shape[0], sample_num, 3)).view(-1, 3)
            xyz = xyz * (size / 2 ** d)
            xyz = xyz[(xyz < size - 1).all(dim=1)]
            xyzs.append(xyz)

            xyzi = torch.floor(xyz)
            corners = (xyzi.unsqueeze(1) + grid)
            coordsf = xyz.unsqueeze(1) - corners
            weights = (1 - coordsf.abs()).prod(dim=-1)
            corners = corners.long().view(-1, 3)
            cx, cy, cz = corners[:, 0], corners[:, 1], corners[:, 2]
            s = sdf[cx, cy, cz].view(-1, 8)
            sdfs.append(torch.sum(s * weights, dim=1))

            gx = s[:, 4] - s[:, 0] + s[:, 5] - s[:, 1] + s[:, 6] - s[:, 2] + s[:, 7] - s[:, 3]
            gy = s[:, 2] - s[:, 0] + s[:, 3] - s[:, 1] + s[:, 6] - s[:, 4] + s[:, 7] - s[:, 5]
            gz = s[:, 1] - s[:, 0] + s[:, 3] - s[:, 2] + s[:, 5] - s[:, 4] + s[:, 7] - s[:, 6]
            grad = torch.stack([gx, gy, gz], dim=-1)
            grad = grad / (torch.sqrt((grad ** 2).sum(-1, keepdim=True)) + 1e-8)
            grads.append(grad)

        xyzs = torch.cat(xyzs, dim=0).numpy()
        points = (2 * xyzs / size - 1).astype(np.float16) * SHAPE_SCALE  # -> [-0.5, 0.5]
        out = os.path.join(dirs["dataset"], sample, "sdf.npz")
        ensure_parent(out)
        np.savez(out, points=points,
                 grad=torch.cat(grads, dim=0).numpy().astype(np.float16),
                 sdf=torch.cat(sdfs, dim=0).numpy().astype(np.float16))


# ── stage 2c: sample occupancy (for IoU-style eval) -> points.npz ───────────────────────
def sample_occu(args, dirs):
    size = args.sdf_size
    num_samples = 100000
    grid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    print("-> sample_occu")
    for sample in tqdm(get_sample_ids(args, dirs), ncols=80):
        sdf_path = os.path.join(dirs["sdf"], f"{sample}.npy")
        if not os.path.exists(sdf_path):
            continue
        sdf = np.load(sdf_path)
        factor = float(size - 1) / float(size)
        points_uniform = np.random.rand(num_samples, 3) * factor
        points = ((points_uniform - 0.5) * (2 * SHAPE_SCALE)).astype(np.float16)

        xyz = points_uniform * size
        corners = np.expand_dims(np.floor(xyz), 1) + grid
        coordsf = np.expand_dims(xyz, 1) - corners
        weights = np.prod(1 - np.abs(coordsf), axis=-1)
        corners = np.reshape(corners.astype(np.int64), (-1, 3))
        cx, cy, cz = corners[:, 0], corners[:, 1], corners[:, 2]
        values = np.reshape(sdf[cx, cy, cz], (-1, 8))
        occu = np.packbits((np.sum(values * weights, axis=1) < 0))

        out = os.path.join(dirs["dataset"], sample, "points")  # np.savez -> points.npz
        ensure_parent(out + ".npz")
        np.savez(out, points=points, occupancies=occu)


def generate_dataset(args, dirs):
    sample_pts_from_mesh(args, dirs)
    sample_sdf(args, dirs)
    sample_occu(args, dirs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="all",
                    choices=["all", "convert_mesh_to_sdf", "generate_dataset"])
    ap.add_argument("--raw_data_root", default=None,
                    help="folder of raw meshes (see load_raw); required for convert_mesh_to_sdf")
    ap.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"),
                    help="output data root (default: <repo>/data)")
    ap.add_argument("--dataset", required=True,
                    help="dataset sub-dir name; MUST contain 'spines'")
    ap.add_argument("--sdf_size", type=int, default=256)
    ap.add_argument("--filelist", default=None,
                    help="sample list (default: <data_root>/<dataset>/filelist/all.txt)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    if "spines" not in args.dataset:
        raise SystemExit("--dataset must contain 'spines' (the data loader keys off it).")
    dirs = dataset_dirs(args.data_root, args.dataset, args.sdf_size)

    if args.run in ("all", "convert_mesh_to_sdf"):
        if not args.raw_data_root:
            raise SystemExit("--raw_data_root is required for convert_mesh_to_sdf / all")
        convert_mesh_to_sdf(args, dirs)
    if args.run in ("all", "generate_dataset"):
        generate_dataset(args, dirs)
    print("Done.")


if __name__ == "__main__":
    main()
