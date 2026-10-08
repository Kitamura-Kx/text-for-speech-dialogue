#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=2:00:00
#PBS -N 0380_dialgen_general
#PBS -j oe

set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"

: "${OUT:?qsub -v OUT=... で出力先を指定してください}"
COUNT_PER_TOPIC="${COUNT_PER_TOPIC:-15000}"
TOPICS="${TOPICS:-}"
NSHARD="${NSHARD:-1}"
SHARD="${SHARD:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1792}"
MAX_RETRIES="${MAX_RETRIES:-3}"

uv run --no-sync python gen_dataset_general.py \
    --model-dir models/instruct31b \
    --out "${OUT}" \
    --count-per-topic "${COUNT_PER_TOPIC}" \
    --topics "${TOPICS}" \
    --nshard "${NSHARD}" \
    --shard "${SHARD}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-retries "${MAX_RETRIES}"
