#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=1:00:00
#PBS -N 0380_t09_latin_norm
#PBS -j oe

# T09不足分の生成成功後に、許可英字の表記と品質情報を同期する。
set -eo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
set -u

dataset_root="datasets/general_t01_t15"
for kind in dialogues metadata raw; do
    count=$(find "${dataset_root}/${kind}/T09" -type f | wc -l)
    if [[ "${count}" != "6500" ]]; then
        echo "ERROR: ${kind}/T09 count=${count}, expected=6500" >&2
        exit 2
    fi
done

.venv/bin/python scripts/normalize_allowed_latin.py "${dataset_root}" \
    --topics T09 \
    --report-name allowed_latin_normalization_t09_20260804.json

UV_CACHE_DIR=.uv-cache uv run --no-sync python -m unittest discover -s tests -v
echo "T09 allowed-Latin normalization complete: $(date -Is)"
