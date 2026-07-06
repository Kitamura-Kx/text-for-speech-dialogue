#!/bin/bash -x
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -N 0378_wkv4_hf
#PBS -j oe
# weekend_v4 テキスト生成: rt_HF 1ノード = 8GPU で 8シャード並列。
# -v COUNT,NSHARD,SHARD_BASE,OUT  (このノードは SHARD_BASE..SHARD_BASE+7 を担当)
# 例: 2ノードで16シャード -> 各ノード NSHARD=16, SHARD_BASE=0/8
cd $PBS_O_WORKDIR
source ~/.bashrc
source /etc/profile.d/modules.sh
module load cuda/12.4
module load cudnn nccl
module load hpcx/2.20
export HF_HUB_OFFLINE=1

pids=()
for g in 0 1 2 3 4 5 6 7; do
  s=$((SHARD_BASE + g))
  if [ "$s" -lt "$NSHARD" ]; then
    CUDA_VISIBLE_DEVICES=$g uv run --no-sync python gen_dataset_v4.py \
        --model-dir models/instruct31b \
        --out "${OUT}" --count "${COUNT}" --nshard "${NSHARD}" --shard "$s" \
        > "logs_v4hf_${s}.log" 2>&1 &
    pids+=($!)
  fi
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "=== NODE DONE (SHARD_BASE=$SHARD_BASE rc=$rc) ==="
exit $rc
