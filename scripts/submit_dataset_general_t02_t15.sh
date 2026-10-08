#!/bin/bash
# T02〜T15を、各Topic 1本のrt_HFジョブとして投入する。
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/check_pbs_job_names.sh

submit=0
if [[ "${1:-}" == "--submit" ]]; then submit=1; shift; fi
if (($#)); then echo "usage: $0 [--submit]" >&2; exit 2; fi

out="datasets/general_t01_t15"
topics=(T02 T03 T04 T05 T06 T07 T08 T09 T10 T11 T12 T13 T14 T15)
for topic in "${topics[@]}"; do
    for kind in dialogues metadata raw; do
        count=0
        if [[ -d "${out}/${kind}/${topic}" ]]; then
            count=$(find "${out}/${kind}/${topic}" -type f | wc -l)
        fi
        if ((count != 0)); then
            echo "ERROR: ${kind}/${topic} is not empty: ${count}" >&2
            exit 2
        fi
    done
done

if ((submit == 0)); then
    for topic in "${topics[@]}"; do
        echo "qsub -N 0380_dialgen_${topic,,} -v TOPIC=${topic},OUT=${out},COUNT_PER_TOPIC=6500 scripts/run_dataset_general_topic_hf.sh"
    done
    exit 0
fi

record="${out}/provenance/t02_t15_generation_jobs_$(date +%Y%m%d_%H%M%S).tsv"
mkdir -p "$(dirname "${record}")"
printf 'topic\tjob_id\tcount\tout\n' >"${record}"
for topic in "${topics[@]}"; do
    job_id=$(qsub -N "0380_dialgen_${topic,,}" \
        -v "TOPIC=${topic},OUT=${out},COUNT_PER_TOPIC=6500" \
        scripts/run_dataset_general_topic_hf.sh)
    printf '%s\t%s\t6500\t%s\n' "${topic}" "${job_id}" "${out}" | tee -a "${record}"
done
echo "job record: ${record}"
