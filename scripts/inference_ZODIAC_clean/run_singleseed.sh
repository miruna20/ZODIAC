#!/bin/bash
# ZODIAC-clean inference — single seed (42), quick run on Balgrist + phantom.
# Shares the ZODIAC prior; differs only by the config flag use_partial_pcd_zero_shot_clean.
# Output: results/inference/zodiac_clean/<dataset>/<scan>/seed_42.obj (directly evaluatable)
#
# Run from anywhere: bash scripts/inference_ZODIAC_clean/run_singleseed.sh [GPU_ID]
# Environment: conda activate octfusion
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
GPU_ID=${1:-0}

RESULTS_OUT="${RESULTS_OUT:-${REPO_ROOT}/results/inference}"
export RESULTS_OUT
VARIANT="zodiac_clean"; export VARIANT

CKPT="checkpoints/zodiac/diffusion/df_steps-latest.pth"
VAE_CKPT="checkpoints/zodiac/vae/vae_steps-latest.pth"
DF_CFG="configs/inference/octfusion_zodiac_clean.yaml"
BALGRIST_VAE="configs/inference/vae_spines_balgrist.yaml"
PHANTOM_VAE="configs/inference/vae_spines_phantom.yaml"

for DS in balgrist phantom; do
    [ "${DS}" = "balgrist" ] && VQ="${BALGRIST_VAE}" || VQ="${PHANTOM_VAE}"
    echo "=== ${VARIANT}: ${DS} (seed 42) ==="
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
        --name "${VARIANT}/${DS}" --logs_dir "${RESULTS_OUT}" --seed 42 \
        --mode generate --stage_flag hr --model union_2t \
        --df_cfg "${DF_CFG}" --ckpt "${CKPT}" \
        --vq_model GraphVAE --vq_cfg "${VQ}" --vq_ckpt "${VAE_CKPT}" \
        --gpu_ids ${GPU_ID} --batch_size 1 --ddim_steps 100 --ddim_eta 0.0 --lr 0.0002 --ema_rate 0.999
done

echo "=== Reorganising completions per scan (seed 42)... ==="
python3 - <<'PYEOF'
import os, shutil, glob
OUT = os.environ["RESULTS_OUT"]; VARIANT = os.environ["VARIANT"]; SEED = 42
for dataset in ["balgrist", "phantom"]:
    completion = os.path.join(OUT, VARIANT, dataset, "completion")
    if not os.path.isdir(completion):
        print(f"[WARN] missing: {completion}"); continue
    for f in sorted(glob.glob(os.path.join(completion, "*.obj"))):
        scan = os.path.basename(f).replace(".obj", "")
        dest = os.path.join(OUT, VARIANT, dataset, scan)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(f, os.path.join(dest, f"seed_{SEED}.obj"))
        print(f"  {VARIANT}/{dataset}/{scan}/seed_{SEED}.obj")
print("Done.")
PYEOF

echo "=== ZODIAC-clean single-seed complete: results/inference/${VARIANT}/<dataset>/<scan>/seed_42.obj ==="
