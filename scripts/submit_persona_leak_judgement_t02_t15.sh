#!/bin/bash
# T02〜T15のペルソナ先取り候補をTopic別rt_HFジョブ（各8 GPU）で判定する。
set -euo pipefail
cd "$(dirname "$0")/.."

submit=0
sequential=0
if [[ "${1:-}" == "--submit" ]]; then
    submit=1
elif [[ "${1:-}" == "--submit-sequential" ]]; then
    submit=1
    sequential=1
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--submit|--submit-sequential]" >&2
    exit 2
fi

audit_root="datasets/general_t01_t15/provenance/persona_leak_audit_20260805"
IFS=',' read -r -a topics <<<"${TOPICS:-T02,T03,T04,T05,T06,T07,T08,T09,T10,T11,T12,T13,T14,T15}"
stamp=$(date +%Y%m%d_%H%M%S)
record="${audit_root}/judgement_jobs_${stamp}.tsv"

scripts/check_pbs_job_names.sh
for topic in "${topics[@]}"; do
    if [[ ! "${topic}" =~ ^T(0[2-9]|1[0-5])$ ]]; then
        echo "ERROR: invalid TOPICS entry: ${topic}" >&2
        exit 2
    fi
done

if ((submit)); then
    printf 'topic\tjob_id\tdependency\tcandidates\tcandidate_file\tresult_dir\n' >"${record}"
fi

previous_job=""
for topic in "${topics[@]}"; do
    candidates="${audit_root}/${topic}/candidates.jsonl"
    result_dir="${audit_root}/${topic}/judgements"
    if [[ ! -s "${candidates}" ]]; then
        echo "ERROR: missing candidates: ${candidates}" >&2
        exit 2
    fi
    count=$(wc -l <"${candidates}")
    if find "${result_dir}" -maxdepth 1 -name 'shard_*.jsonl' -type f 2>/dev/null | grep -q .; then
        echo "ERROR: judgement output already exists: ${result_dir}" >&2
        exit 2
    fi
    command=(qsub -N "0380_leak_${topic,,}")
    dependency=""
    if ((sequential)) && [[ -n "${previous_job}" ]]; then
        dependency="afterok:${previous_job}"
        command+=(-W "depend=${dependency}")
    fi
    command+=(-v "CANDIDATES=${candidates},RESULT_DIR=${result_dir}"
              scripts/run_persona_leak_judge_hf.sh)
    if ((submit)); then
        job_id=$("${command[@]}")
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${topic}" "${job_id}" "${dependency}" "${count}" \
            "${candidates}" "${result_dir}" | tee -a "${record}"
        previous_job="${job_id}"
    else
        printf '%s candidates=%s: ' "${topic}" "${count}"
        printf '%q ' "${command[@]}"
        printf '\n'
    fi
done

if ((submit)); then
    echo "job record: ${record}"
fi
