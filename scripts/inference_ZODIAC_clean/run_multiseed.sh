#!/bin/bash
# ZODIAC-clean inference — multi-seed (paper reproduction): 10 seeds (42-51) x {Balgrist, phantom},
# then reorganise the completions per scan for mean +/- std reporting.
# ZODIAC-clean shares the ZODIAC prior; it differs only by the config flag
# use_partial_pcd_zero_shot_clean (clean partial pinned at every DDIM step).
# Output: results/inference/zodiac_clean/<dataset>/<scan>/seed_<n>.obj
#
# Run from anywhere: bash scripts/inference_ZODIAC_clean/run_multiseed.sh [GPU_ID]
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

for SEED in $(seq 42 51); do
    for DS in balgrist phantom; do
        [ "${DS}" = "balgrist" ] && VQ="${BALGRIST_VAE}" || VQ="${PHANTOM_VAE}"
        echo "=== ${VARIANT}: ${DS} seed ${SEED} ==="
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
            --name "${VARIANT}/${DS}/seed_${SEED}" --logs_dir "${RESULTS_OUT}" --seed ${SEED} \
            --mode generate --stage_flag hr --model union_2t \
            --df_cfg "${DF_CFG}" --ckpt "${CKPT}" \
            --vq_model GraphVAE --vq_cfg "${VQ}" --vq_ckpt "${VAE_CKPT}" \
            --gpu_ids ${GPU_ID} --batch_size 1 --ddim_steps 100 --ddim_eta 0.0 --lr 0.0002 --ema_rate 0.999
    done
done

echo "=== Reorganising completions per scan... ==="
python3 - <<'PYEOF'
import os, shutil, glob
OUT = os.environ["RESULTS_OUT"]; VARIANT = os.environ["VARIANT"]
for dataset in ["balgrist", "phantom"]:
    for seed in range(42, 52):
        completion = os.path.join(OUT, VARIANT, dataset, f"seed_{seed}", "completion")
        if not os.path.isdir(completion):
            print(f"[WARN] missing: {completion}"); continue
        for f in sorted(glob.glob(os.path.join(completion, "*.obj"))):
            scan = os.path.basename(f).replace(".obj", "")
            dest = os.path.join(OUT, VARIANT, dataset, scan)
            os.makedirs(dest, exist_ok=True)
            shutil.copy(f, os.path.join(dest, f"seed_{seed}.obj"))
            print(f"  {VARIANT}/{dataset}/{scan}/seed_{seed}.obj")
print("Done.")
PYEOF

echo "=== ZODIAC-clean multi-seed complete: results/inference/${VARIANT}/ ==="
