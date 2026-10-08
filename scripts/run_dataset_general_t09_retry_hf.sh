#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N 0380_dialgen_t09_x8
#PBS -j oe

# T09の旧8-shard中shard 6に残ったIDを、64-shard表現の8本へ分割して8 GPUで補完する。
set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
set -u

out="datasets/general_t01_t15"
topic="T09"
nshard=64
shards=(6 14 22 30 38 46 54 62)
job_tag="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
log_dir="${out}/logs/generation/${topic}/${job_tag}"
mkdir -p "${log_dir}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"

for kind in dialogues metadata raw; do
    count=$(find "${out}/${kind}/${topic}" -type f | wc -l)
    if ((count > 6500)); then
        echo "ERROR: ${kind}/${topic} count=${count}, expected at most 6500" >&2
        exit 2
    fi
done

pids=()
terminate_children() {
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap terminate_children INT TERM

for gpu in 0 1 2 3 4 5 6 7; do
    shard="${shards[$gpu]}"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python gen_dataset_general.py \
        --model-dir models/instruct31b --out "${out}" \
        --count-per-topic 6500 --topics "${topic}" \
        --nshard "${nshard}" --shard "${shard}" \
        --max-new-tokens 1792 --max-retries 3 \
        --artifact-tag t09_retry64 >"${log_dir}/shard_$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "complete shard=${shards[$i]}"
    else
        status=$?
        echo "FAILED shard=${shards[$i]} status=${status}" >&2
        rc=1
    fi
done

for kind in dialogues metadata raw; do
    count=$(find "${out}/${kind}/${topic}" -type f | wc -l)
    if [[ "${count}" != "6500" ]]; then
        echo "ERROR: final ${kind}/${topic} count=${count}, expected=6500" >&2
        rc=1
    fi
done
exit "${rc}"
