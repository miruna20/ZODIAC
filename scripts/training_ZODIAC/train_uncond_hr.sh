#!/bin/bash
# ZODIAC / ZODIAC-clean training — Stage 3: high-resolution (HR) diffusion.
# Denoises the per-node latent features (fine surface detail). Pre-trains from the Stage-2
# LR checkpoint and reads the Stage-1 VAE.
#
# Attaches to the run created by train_vae.sh (logs/.zodiac_latest_run), or set
# ZODIAC_RUN=training_ZODIAC_<ts> to target a specific run. Output:
#   logs/training_ZODIAC_<timestamp>/diffusion-hr/ckpt/df_steps-latest.pth
#
# Run AFTER train_uncond_lr.sh, from anywhere (single GPU):
#   bash scripts/training_ZODIAC/train_uncond_hr.sh [GPU_ID]
# Environment: conda activate octfusion
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
GPU_ID=${1:-0}

RUN="${ZODIAC_RUN:-$(cat logs/.zodiac_latest_run 2>/dev/null || true)}"
if [ -z "${RUN}" ]; then
    echo "[!] No training run found. Run train_vae.sh + train_uncond_lr.sh first, or set ZODIAC_RUN=training_ZODIAC_<timestamp>." >&2
    exit 1
fi
echo "[*] Training run: logs/${RUN}"

VAE_CKPT="logs/${RUN}/vae/ckpt/vae_steps-latest.pth"
PRETRAIN_CKPT="logs/${RUN}/diffusion-lr/ckpt/df_steps-latest.pth"
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "[!] LR checkpoint not found: ${PRETRAIN_CKPT}. Run train_uncond_lr.sh for this run first." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
    --name "${RUN}/diffusion-hr" --logs_dir logs --gpu_ids ${GPU_ID} --mode train \
    --model union_2t --stage_flag hr --vq_model GraphVAE \
    --df_cfg configs/training_ZODIAC/octfusion_spines_uncond.yaml \
    --vq_cfg configs/training_ZODIAC/vae_spines_train.yaml \
    --vq_ckpt "${VAE_CKPT}" --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --lr 2e-4 --min_lr 1e-6 --epochs 300 --warmup_epochs 40 --ema_rate 0.999 --seed 42 \
    --ckpt_num 3 --display_freq 1000 --print_freq 25 \
    --save_steps_freq 3000 --save_latest_freq 500 --debug 0
