#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N 0380_t01_leak_retry
#PBS -j oe

set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
set -u

DATASET_ROOT="${DATASET_ROOT:-datasets/general_t01_t15}"
AUDIT_DIR="${AUDIT_DIR:-${DATASET_ROOT}/provenance/persona_leak_audit_20260802}"
TARGETS="${TARGETS:-${AUDIT_DIR}/regeneration_targets.jsonl}"
STAGE_DIR="${STAGE_DIR:-${AUDIT_DIR}/regeneration_staging}"
SEED_ROUND="${SEED_ROUND:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
mkdir -p "${STAGE_DIR}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"
uv run --no-sync python scripts/regenerate_persona_leaks.py \
    --dataset-root "${DATASET_ROOT}" \
    --targets "${TARGETS}" \
    --stage-dir "${STAGE_DIR}" \
    --model-dir models/instruct31b --nshard 1 --shard 0 --only-missing \
    --seed-round "${SEED_ROUND}" --max-regeneration-attempts "${MAX_ATTEMPTS}" \
    >"${STAGE_DIR}/single_retry.log" 2>&1
