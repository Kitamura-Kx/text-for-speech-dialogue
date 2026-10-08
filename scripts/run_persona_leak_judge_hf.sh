#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N 0380_t01_leak_judge
#PBS -j oe

set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
set -u

CANDIDATES="${CANDIDATES:-datasets/general_t01_t15/provenance/persona_leak_audit_20260802/candidates.jsonl}"
RESULT_DIR="${RESULT_DIR:-datasets/general_t01_t15/provenance/persona_leak_audit_20260802/judgements}"
MODEL_DIR="${MODEL_DIR:-models/instruct31b}"
ONLY_ERRORS_FROM="${ONLY_ERRORS_FROM:-}"
NSHARD=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"
mkdir -p "${RESULT_DIR}"

pids=()
terminate_children() { if ((${#pids[@]})); then kill "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true; fi; }
trap terminate_children INT TERM
for gpu in 0 1 2 3 4 5 6 7; do
    extra_args=()
    if [[ -n "${ONLY_ERRORS_FROM}" ]]; then
        extra_args+=(--only-errors-from "${ONLY_ERRORS_FROM}")
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python scripts/judge_persona_leaks.py \
        --candidates "${CANDIDATES}" --output "${RESULT_DIR}/shard_${gpu}.jsonl" \
        --model-dir "${MODEL_DIR}" --nshard "${NSHARD}" --shard "${gpu}" \
        "${extra_args[@]}" \
        >"${RESULT_DIR}/shard_${gpu}.log" 2>&1 &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
exit "${rc}"
