#!/bin/bash
# ZODIAC / ZODIAC-clean training — Stage 1: VAE.
# Trains the graph VAE on the 504 complete spines (deformed + TotalSegmentator).
#
# Each launch starts a fresh, timestamped run:
#   logs/training_ZODIAC_<timestamp>/vae/ckpt/vae_steps-latest.pth
# The run name is recorded in logs/.zodiac_latest_run so Stages 2-3 attach to it
# automatically. To target a specific existing run, set ZODIAC_RUN=training_ZODIAC_<ts>.
# (Hyperparameters are saved per run in logs/<run>/vae/opt.txt + the copied configs.)
#
# Run from anywhere (single GPU): bash scripts/training_ZODIAC/train_vae.sh [GPU_ID]
# Environment: conda activate octfusion
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
GPU_ID=${1:-0}

RUN="${ZODIAC_RUN:-training_ZODIAC_$(date +%Y-%m-%dT%H-%M-%S)}"
mkdir -p logs
echo "${RUN}" > logs/.zodiac_latest_run
echo "[*] Training run: logs/${RUN}  (Stages 2-3 will use this run automatically)"

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
    --name "${RUN}/vae" --logs_dir logs --gpu_ids ${GPU_ID} --mode train \
    --model vae --vq_model GraphVAE \
    --df_cfg configs/training_ZODIAC/octfusion_spines_uncond.yaml \
    --vq_cfg configs/training_ZODIAC/vae_spines_train.yaml \
    --lr 1e-3 --min_lr 1e-6 --epochs 900 --warmup_epochs 350 --ema_rate 0.999 --seed 42 \
    --ckpt_num 30 --display_freq 3000 --print_freq 25 \
    --save_steps_freq 5000 --save_latest_freq 1000 --debug 0
