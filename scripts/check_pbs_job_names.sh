#!/bin/bash
# 全PBSジョブ名と投入時の名前上書きが0380_で始まることを検査する。
set -euo pipefail
cd "$(dirname "$0")/.."

bad=0
if command -v rg >/dev/null 2>&1; then
    pbs_name_matches() { rg -n '^#PBS -N ' scripts; }
    qsub_name_matches() {
        rg -n --glob '!check_pbs_job_names.sh' \
            'qsub .* -N |qsub -N |command=\(qsub.*-N' scripts
    }
else
    pbs_name_matches() { grep -RIn --include='*.sh' '^#PBS -N ' scripts; }
    qsub_name_matches() {
        grep -RInE --include='*.sh' --exclude='check_pbs_job_names.sh' \
            'qsub .* -N |qsub -N |command=\(qsub.*-N' scripts
    }
fi

while IFS=: read -r file line value; do
    if [[ "${value}" != '#PBS -N 0380_'* ]]; then
        echo "ERROR: PBS job name must start with 0380_: ${file}:${line}: ${value}" >&2
        bad=1
    fi
done < <(pbs_name_matches)

while IFS= read -r match; do
    if [[ "${match}" =~ qsub[^#]*-N[[:space:]]+\"?([^\"[:space:]]+) ]]; then
        name="${BASH_REMATCH[1]}"
        if [[ "${name}" != 0380_* ]]; then
            echo "ERROR: qsub -N name must start with 0380_: ${match}" >&2
            bad=1
        fi
    fi
done < <(qsub_name_matches || true)

exit "${bad}"
