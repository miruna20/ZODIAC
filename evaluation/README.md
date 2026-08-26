# ZODIAC evaluation

Reproduces the paper's **Table 1**: Chamfer Distance (CD), F1, and HD95 at both
whole-spine (WS) and vertebra-wise (VW) level, for the three octfusion-based methods
(ZODIAC, TP-ODIAC, ZODIAC-clean) on the Phantom and Balgrist test sets.

A single entry point, `evaluate.py`, computes every metric fresh from the completion
meshes and writes one tidy CSV. All inputs resolve **relative to the repository** — no
absolute-path edits are needed once the data is downloaded and inference has been run.

## Prerequisites

1. **Environment:** `conda activate octfusion` (pure numpy / scipy / trimesh).
2. **Data** under `../data/` (downloaded from LRZ — see [`../data/README.md`](../data/README.md)).
   Evaluation reads, per test set `spines_<ds>` (`ds` = `phantoms` | `balgrist`):
   - `mesh_256/<scan>.obj` — whole-spine GT (WS)
   - `mesh_vertebrae_256/<scan>/{L1..L5}.obj` — per-vertebra GT (VW)
   - `bbox_256/<scan>.npz` — mm scale, so HD95 is reported in millimetres
3. **Predictions** under `../results/inference/` in the multi-seed layout
   `‹method›/‹dataset›/‹scan›/seed_‹n›.obj`, produced by the inference scripts:

   ```bash
   bash scripts/inference_ZODIAC/run_multiseed.sh        # → results/inference/zodiac/...
   bash scripts/inference_ZODIAC_clean/run_multiseed.sh  # → results/inference/zodiac_clean/...
   bash scripts/inference_TPODIAC/run_multiseed.sh       # → results/inference/tpodiac/...
   ```

   Each `run_multiseed.sh` runs 10 seeds (42–51) × {balgrist, phantom} and reorganises the
   completions into the per-scan `‹scan›/seed_‹n›.obj` layout that `evaluate.py` expects.
   The `run_singleseed.sh` scripts produce the same layout for seed 42 only, so a
   single-seed run can be evaluated directly with `--num_seeds 1`.

   You do **not** need predictions for every method: `evaluate.py` reports which methods have
   results on disk and which are missing, then evaluates only the available ones. So you can
   evaluate, say, ZODIAC and ZODIAC-clean before TP-ODIAC inference has been run.

## Run

From the repository root:

```bash
python evaluation/evaluate.py                       # all defaults → 10 seeds, all 3 methods
```

Writes `evaluation/results/table.csv` — one row per `dataset-group × method × level`
(18 rows: 3 groups × 3 methods × {WS, VW}) with columns
`dataset, method, level, cd, cd_std, f1, f1_std, hd95mm, hd95mm_std`
(CD ×1e4; F1; HD95 in mm; `_std` = population std over the group), and prints a compact
summary. The three dataset groups are **Phantom**, **Balgrist_all** (6 subjects), and
**Balgrist_difficult** (URS36_D2, URS45_H2, URS45_R2).

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--num_seeds` | `10` | average over seeds `42 … 42+num_seeds-1` (also works with `1`) |
| `--methods` | `zodiac tpodiac zodiac_clean` | subset of methods to evaluate |
| `--output` | `evaluation/results/table.csv` | output CSV path |
| `--data_root` | `data/` | override the data root (or set `ZODIAC_DATA_ROOT`) |
| `--pred_base` | `results/inference/` | override the prediction root (or set `ZODIAC_RESULTS_DIR`) |

**Single-seed** runs (`--num_seeds 1`, or `run_singleseed.sh`) are a quick sanity check;
the paper table needs the full multi-seed set. Metrics are recomputed from the meshes with
a fixed point-sampling seed, so results are reproducible run-to-run.

## Files

| File | Role |
|---|---|
| `evaluate.py` | single entry point (WS + VW, all metrics, tidy CSV) |
| `metrics_utils.py` | CD / F1 / HD95 (`compute_all_metrics`) |
| `loaders.py` | mesh sampling, GT-normalisation, per-vertebra GT loaders |
| `label_assignment.py` | assigns predicted points to vertebra levels for VW |
| `results/` | output directory (CSVs are git-ignored) |
