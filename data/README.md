# ZODIAC data

This folder holds the prepared datasets used to train and evaluate ZODIAC. The data is
**not** stored in the repository. Download it
from LRZ Sync+Share and unzip it here.

## Download & unzip

Data archive: https://syncandshare.lrz.de/getlink/fiK577UYP9iYWQiLwm5hBh/


## Layout

```
data/
├── spines_deformed+total_segmentator/   # TRAINING (complete shapes, 504 spines)
│   ├── dataset_256/                      #   504 samples = 182 deformed VerSe20 + 322 TotalSegmentator
│   └── filelist/                         #   train.txt = 504  → ZODIAC / ZODIAC-clean prior
│
├── spines_balgrist/                      # TEST only (6 volunteer scans)
│   ├── dataset_256/
│   ├── bbox_256/                         #   mm-scale bounding boxes (needed for HD95)
│   ├── mesh_256/                         #   whole-spine ground-truth meshes
│   ├── mesh_vertebrae_256/               #   per-vertebra ground-truth meshes (vertebra-wise eval)
│   └── filelist/                         #   test.txt = 6
│
└── spines_phantoms/                      # TEST only (2 anthropomorphic phantoms)
    ├── dataset_256/
    ├── bbox_256/
    ├── mesh_256/
    ├── mesh_vertebrae_256/
    └── filelist/                         #   test.txt = 2
```

Each sample directory `dataset_256/<sample>/` contains a subset of the files below,
depending on the dataset's role:

| File | Used for | Present in |
|---|---|---|
| `pointcloud.npz`| surface point cloud sampled from the complete shape | all samples |
| `sdf.npz`       | signed distance field samples (training target) | all samples |
| `partial.npz`   | partial point cloud (observed / US-visible surface) — inference conditioning | all test samples |
| `points.npz`    | occupancy / query points | phantoms only |

Training samples therefore carry `pointcloud.npz` + `sdf.npz`; the test sets carry `partial.npz` + `pointcloud.npz` + `sdf.npz`.

`mesh_256/` and `mesh_vertebrae_256/` are ground-truth meshes used **only for evaluation**
(whole-spine and vertebra-wise, respectively); `bbox_256/` provides the mm scale so HD95 is
reported in millimetres. These three exist only for the test sets (`spines_balgrist`,
`spines_phantoms`).

## Datasets & splits

| Directory | Role | Split | Contents |
|---|---|---|---|
| `spines_deformed+total_segmentator` | **train** | `filelist/train.txt` = 504 | 182 deformed VerSe20 (91 IDs × 2 deformations) + 322 TotalSegmentator lumbar spines |
| `spines_balgrist` | **test** | `filelist/test.txt` = 6 | 6 volunteer scans (Balgrist paired US–CT) |
| `spines_phantoms` | **test** | `filelist/test.txt` = 2 | 2 lumbar-spine phantoms |

- **ZODIAC / ZODIAC-clean** (zero-shot prior) train on the full **504** complete spines
  (`filelist/train.txt`), using complete shapes only.
- All methods are evaluated on `spines_balgrist` + `spines_phantoms` (partial point clouds
  from `partial.npz`; ground truth from `mesh_256/` + `mesh_vertebrae_256/`).

## Public sources

- **VerSe20** — https://github.com/anjany/verse
- **TotalSegmentator** — https://github.com/wasserth/TotalSegmentator
- **Balgrist** paired US–CT dataset — Cavalcanti et al., *Scientific Data* 2025
- **Phantoms** — Anthropomorphic Phantoms
