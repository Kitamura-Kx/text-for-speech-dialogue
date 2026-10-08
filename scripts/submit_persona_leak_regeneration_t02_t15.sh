#!/bin/bash
# T02〜T15の確定leak対話をTopic別rt_HF通常ジョブで再生成する。
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/check_pbs_job_names.sh

submit=0
resume=0
if [[ "${1:-}" == "--submit" ]]; then
    submit=1
elif [[ "${1:-}" == "--resume" ]]; then
    submit=1
    resume=1
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--submit|--resume]" >&2
    exit 2
fi

audit="datasets/general_t01_t15/provenance/persona_leak_audit_20260805"
regen="${audit}/regeneration"
IFS=',' read -r -a topics <<<"${TOPIC_LIST:-T02,T03,T04,T05,T06,T07,T08,T09,T10,T11,T12,T13,T14,T15}"
for topic in "${topics[@]}"; do
    if [[ ! "${topic}" =~ ^T(0[2-9]|1[0-5])$ ]]; then
        echo "ERROR: invalid topic in TOPIC_LIST: ${topic}" >&2
        exit 2
    fi
done
record="${audit}/regeneration_jobs_$(date +%Y%m%d_%H%M%S).tsv"
if ((submit)); then printf 'topic\tjob_id\ttargets\tstage_dir\n' >"${record}"; fi

for topic in "${topics[@]}"; do
    targets="${regen}/${topic}/targets.jsonl"
    stage="${regen}/${topic}/staging"
    if [[ ! -s "${targets}" ]]; then
        echo "ERROR: missing targets: ${targets}" >&2
        exit 2
    fi
    if ((resume)) && [[ ! -d "${stage}" ]]; then
        echo "ERROR: resume stage is missing: ${stage}" >&2
        exit 2
    elif ((!resume)) && [[ -e "${stage}" ]]; then
        echo "ERROR: stage already exists: ${stage}" >&2
        exit 2
    fi
    count=$(wc -l <"${targets}")
    variables="DATASET_ROOT=datasets/general_t01_t15,AUDIT_DIR=${audit},TARGETS=${targets},STAGE_DIR=${stage},AUDIT_REFERENCE=provenance/persona_leak_audit_20260805"
    if ((resume)); then
        variables+=",ONLY_MISSING=1,SEED_ROUND=1"
    fi
    command=(qsub -N "0380_regen_${topic,,}" -v "${variables}"
        scripts/run_persona_leak_regeneration_hf.sh)
    if ((submit)); then
        job_id=$("${command[@]}")
        printf '%s\t%s\t%s\t%s\n' "${topic}" "${job_id}" "${count}" "${stage}" | tee -a "${record}"
    else
        printf '%s targets=%s: ' "${topic}" "${count}"
        printf '%q ' "${command[@]}"
        printf '\n'
    fi
done
if ((submit)); then echo "job record: ${record}"; fi
