#!/bin/bash -x
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=2:00:00
#PBS -N 0380_wkv4_shard
#PBS -j oe
# weekend_v4 テキスト生成(31B) 1シャード分。設計: docs/weekend_v4_design.md
# -v COUNT,NSHARD,SHARD,OUT で指定。冪等 (既存skip)・削除なし。
cd $PBS_O_WORKDIR
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
export CUDA_VISIBLE_DEVICES="0"
export HF_HUB_OFFLINE=1
uv run --no-sync python gen_dataset_v4.py \
    --model-dir models/instruct31b \
    --out "${OUT}" --count "${COUNT}" --nshard "${NSHARD}" --shard "${SHARD}"
