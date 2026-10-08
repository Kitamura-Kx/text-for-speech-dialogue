#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -N 0380_t01_uncertain
#PBS -j oe
set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"
AUDIT="${AUDIT:-datasets/general_t01_t15/provenance/persona_leak_audit_20260802}"
INPUT="${INPUT:-${AUDIT}/remaining_uncertain_26.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${AUDIT}/contextual_final_check}"
LOG_FILE="${LOG_FILE:-${AUDIT}/contextual_final_check.log}"
mkdir -p "${OUTPUT_DIR}"
uv run --no-sync python scripts/resolve_uncertain_by_context.py \
    --input "${INPUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-dir models/instruct31b \
    >"${LOG_FILE}" 2>&1
