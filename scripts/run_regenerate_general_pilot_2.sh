#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=2:00:00
#PBS -N 0380_regenerate_general_pilot_2
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR}"
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_OFFLINE=1

uv run --no-sync python scripts/regenerate_dialogues_from_metadata.py \
    --out datasets/general_pilot_2 \
    --indices 0,1 \
    --model-dir models/instruct31b
