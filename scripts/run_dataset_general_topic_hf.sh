#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -N 0380_dialgen_topic
#PBS -j oe

# 単一Topic 6500件をrt_HF 1ノード8 GPUで生成する。
# manifest/config/logはTopic別に分離し、複数Topicジョブの並行実行を安全にする。
set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
set -u

OUT="${OUT:-datasets/general_t01_t15}"
TOPIC="${TOPIC:?TOPIC is required (T02-T15)}"
COUNT_PER_TOPIC="${COUNT_PER_TOPIC:-6500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1792}"
MAX_RETRIES="${MAX_RETRIES:-3}"
LOAD_ONLY="${LOAD_ONLY:-0}"
NSHARD=8

if [[ ! "${TOPIC}" =~ ^T(0[2-9]|1[0-5])$ ]]; then
    echo "ERROR: TOPIC must be one of T02-T15: ${TOPIC}" >&2
    exit 2
fi
if [[ "${COUNT_PER_TOPIC}" != "6500" ]]; then
    echo "ERROR: production COUNT_PER_TOPIC must be 6500: ${COUNT_PER_TOPIC}" >&2
    exit 2
fi
if [[ ! "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ || ! "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid generation limits" >&2
    exit 2
fi
if [[ ! -f models/instruct31b/model.safetensors.index.json || ! -x .venv/bin/python ]]; then
    echo "ERROR: model or .venv is not prepared" >&2
    exit 2
fi
if [[ "${LOAD_ONLY}" != "0" && "${LOAD_ONLY}" != "1" ]]; then
    echo "ERROR: LOAD_ONLY must be 0 or 1" >&2
    exit 2
fi

if [[ "${LOAD_ONLY}" == "1" ]]; then
    job_tag="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
    log_dir="logs/model_load_smoke/${job_tag}"
    mkdir -p "${log_dir}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
    export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"
    pids=()
    for gpu in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python scripts/load_instruct31b_smoke.py \
            --model-dir models/instruct31b >"${log_dir}/gpu_${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
    echo "LOAD_ONLY END: $(date -Is) rc=${rc} logs=${log_dir}"
    exit "${rc}"
fi

for kind in dialogues metadata raw; do
    existing=0
    if [[ -d "${OUT}/${kind}/${TOPIC}" ]]; then
        existing=$(find "${OUT}/${kind}/${TOPIC}" -type f | wc -l)
    fi
    if ((existing > COUNT_PER_TOPIC)); then
        echo "ERROR: unexpected existing ${kind}/${TOPIC} count=${existing}" >&2
        exit 2
    fi
done

topic_tag="${TOPIC,,}"
job_tag="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
log_dir="${OUT}/logs/generation/${TOPIC}/${job_tag}"
mkdir -p "${log_dir}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"

echo "START: $(date -Is) job=${job_tag} topic=${TOPIC} count=${COUNT_PER_TOPIC} out=${OUT}"
pids=()
shards=()
terminate_children() {
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap terminate_children INT TERM

for gpu in 0 1 2 3 4 5 6 7; do
    log_file="${log_dir}/shard_$(printf '%02d' "${gpu}").log"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python gen_dataset_general.py \
        --model-dir models/instruct31b --out "${OUT}" \
        --count-per-topic "${COUNT_PER_TOPIC}" --topics "${TOPIC}" \
        --nshard "${NSHARD}" --shard "${gpu}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" --max-retries "${MAX_RETRIES}" \
        --artifact-tag "${topic_tag}" >"${log_file}" 2>&1 &
    pids+=("$!")
    shards+=("${gpu}")
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "complete topic=${TOPIC} shard=${shards[$i]}"
    else
        status=$?
        echo "FAILED topic=${TOPIC} shard=${shards[$i]} status=${status}" >&2
        rc=1
    fi
done
echo "END: $(date -Is) topic=${TOPIC} rc=${rc}"
exit "${rc}"
