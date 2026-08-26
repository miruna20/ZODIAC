#!/bin/bash
# ZODIAC / ZODIAC-clean training — Stage 2: low-resolution (LR) diffusion, from scratch.
# Denoises the base-depth octree split signal (global spine topology). Reads the Stage-1 VAE.
#
# Attaches to the run created by train_vae.sh (logs/.zodiac_latest_run), or set
# ZODIAC_RUN=training_ZODIAC_<ts> to target a specific run. Output:
#   logs/training_ZODIAC_<timestamp>/diffusion-lr/ckpt/df_steps-latest.pth
#
# Run AFTER train_vae.sh, from anywhere (single GPU):
#   bash scripts/training_ZODIAC/train_uncond_lr.sh [GPU_ID]
# Environment: conda activate octfusion
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
GPU_ID=${1:-0}

RUN="${ZODIAC_RUN:-$(cat logs/.zodiac_latest_run 2>/dev/null || true)}"
if [ -z "${RUN}" ]; then
    echo "[!] No training run found. Run train_vae.sh first, or set ZODIAC_RUN=training_ZODIAC_<timestamp>." >&2
    exit 1
fi
echo "[*] Training run: logs/${RUN}"

VAE_CKPT="logs/${RUN}/vae/ckpt/vae_steps-latest.pth"
if [ ! -f "${VAE_CKPT}" ]; then
    echo "[!] VAE checkpoint not found: ${VAE_CKPT}. Run train_vae.sh for this run first." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
    --name "${RUN}/diffusion-lr" --logs_dir logs --gpu_ids ${GPU_ID} --mode train \
    --model union_2t --stage_flag lr --vq_model GraphVAE \
    --df_cfg configs/training_ZODIAC/octfusion_spines_uncond.yaml \
    --vq_cfg configs/training_ZODIAC/vae_spines_train.yaml \
    --vq_ckpt "${VAE_CKPT}" \
    --lr 2e-4 --min_lr 1e-6 --epochs 1500 --warmup_epochs 40 --ema_rate 0.999 --seed 42 \
    --ckpt_num 3 --display_freq 1000 --print_freq 25 \
    --save_steps_freq 3000 --save_latest_freq 500 --debug 0
