#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -N 0380_dialgen_t01_6500
#PBS -j oe

# 汎用対話データを rt_HF 1ノード（8 GPU）で8シャード並列生成する。
# 既存の対話ファイルは gen_dataset_general.py 側でskipされるため、同じOUTへ再投入可能。
#
# T01を6500件生成する標準投入例:
#   qsub scripts/run_dataset_general_hf.sh
# 出力先を変える場合:
#   qsub -v OUT=datasets/my_t01_t15 scripts/run_dataset_general_hf.sh

set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
set -u

OUT="${OUT:-datasets/general_t01_t15}"
COUNT_PER_TOPIC="${COUNT_PER_TOPIC:-6500}"
TOPICS="${TOPICS:-T01}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1792}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NSHARD=8

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PBS_O_WORKDIR}/.uv-cache}"

if [[ ! "${COUNT_PER_TOPIC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: COUNT_PER_TOPIC must be a positive integer: ${COUNT_PER_TOPIC}" >&2
    exit 2
fi
if [[ ! "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_NEW_TOKENS must be a positive integer: ${MAX_NEW_TOKENS}" >&2
    exit 2
fi
if [[ ! "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MAX_RETRIES must be a non-negative integer: ${MAX_RETRIES}" >&2
    exit 2
fi
if [[ ! -f models/instruct31b/model.safetensors.index.json ]]; then
    echo "ERROR: models/instruct31b is not prepared" >&2
    exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv is not prepared; run uv sync --frozen first" >&2
    exit 2
fi

log_dir="${OUT}/logs"
mkdir -p "${log_dir}"

pids=()
shards=()
terminate_children() {
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap terminate_children INT TERM

echo "START: $(date -Is)"
echo "OUT=${OUT} TOPICS=${TOPICS} COUNT_PER_TOPIC=${COUNT_PER_TOPIC} NSHARD=${NSHARD}"

for gpu in 0 1 2 3 4 5 6 7; do
    shard="${gpu}"
    log_file="${log_dir}/shard_$(printf '%02d' "${shard}").log"
    echo "launch gpu=${gpu} shard=${shard} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync python gen_dataset_general.py \
        --model-dir models/instruct31b \
        --out "${OUT}" \
        --count-per-topic "${COUNT_PER_TOPIC}" \
        --topics "${TOPICS}" \
        --nshard "${NSHARD}" \
        --shard "${shard}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-retries "${MAX_RETRIES}" \
        >"${log_file}" 2>&1 &
    pids+=("$!")
    shards+=("${shard}")
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

echo "END: $(date -Is) rc=${rc}"
exit "${rc}"
