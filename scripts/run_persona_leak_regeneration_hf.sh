#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -N 0380_t01_leak_regen
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
MODEL_DIR="${MODEL_DIR:-models/instruct31b}"
AUDIT_REFERENCE="${AUDIT_REFERENCE:-${AUDIT_DIR#${DATASET_ROOT}/}}"
ONLY_MISSING="${ONLY_MISSING:-0}"
SEED_ROUND="${SEED_ROUND:-0}"
NSHARD=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"
mkdir -p "${STAGE_DIR}/logs"

pids=()
terminate_children() { if ((${#pids[@]})); then kill "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true; fi; }
trap terminate_children INT TERM
for gpu in 0 1 2 3 4 5 6 7; do
    extra_args=(--seed-round "${SEED_ROUND}")
    if [[ "${ONLY_MISSING}" == "1" ]]; then extra_args+=(--only-missing); fi
    CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python scripts/regenerate_persona_leaks.py \
        --dataset-root "${DATASET_ROOT}" --targets "${TARGETS}" --stage-dir "${STAGE_DIR}" \
        --model-dir "${MODEL_DIR}" --nshard "${NSHARD}" --shard "${gpu}" \
        --audit-reference "${AUDIT_REFERENCE}" \
        "${extra_args[@]}" \
        >"${STAGE_DIR}/logs/shard_${gpu}.log" 2>&1 &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
exit "${rc}"
