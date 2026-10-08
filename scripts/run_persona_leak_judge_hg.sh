#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -N 0380_leak_retry
#PBS -j oe

# 少数のjudge_error候補を1 GPUで再判定する。
set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
set -u

CANDIDATES="${CANDIDATES:?CANDIDATES is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/judge_persona_leaks.py \
    --candidates "${CANDIDATES}" --output "${OUTPUT}" \
    --model-dir models/instruct31b --nshard 1 --shard 0
