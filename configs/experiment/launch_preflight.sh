#!/bin/bash
# Paper preflight for the 650M/index_t33 path.
#
# Usage:
#   ./configs/experiment/launch_preflight.sh --all
#   ./configs/experiment/launch_preflight.sh --only verify
#   ./configs/experiment/launch_preflight.sh --all --dry-run

set -euo pipefail

DRY_RUN=false
MODE=""
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            MODE="all"
            shift
            ;;
        --only)
            MODE="only"
            ONLY="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            sed -n '2,9p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Choose --all or --only {verify|coverage|ceiling|correlation}." >&2
    exit 2
fi

source "$(dirname "$0")/_launch_common.sh"
prepare_output_dirs
require_clean_worktree

submit_preflight() {
    local key="$1"
    local script="$2"
    local job_name="$3"
    local time_limit="$4"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ${script} experiment=diagnostics/ceiling 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "${time_limit}" 16 "256G" >/dev/null
}

run_selected() {
    local key="$1"
    local script="$2"
    local job_name="$3"
    local time_limit="$4"
    if [[ "${MODE}" == "all" || "${ONLY}" == "${key}" ]]; then
        submit_preflight "${key}" "${script}" "${job_name}" "${time_limit}"
    fi
}

if [[ "${MODE}" == "only" ]]; then
    case "${ONLY}" in
        verify|coverage|ceiling|correlation) ;;
        *)
            echo "Invalid --only value: ${ONLY}" >&2
            exit 2
            ;;
    esac
fi

echo "650M preflight, commit=$(git_commit), dry_run=${DRY_RUN}"
run_selected "verify" "scripts/verify_no_leak.py" "paper_verify_no_leak_t33" "12:00:00"
run_selected "coverage" "scripts/template_coverage.py" "paper_template_coverage_t33" "08:00:00"
run_selected "ceiling" "scripts/ceiling.py" "paper_ceiling_t33" "02:00:00"
run_selected "correlation" "scripts/feature_correlation.py" "paper_feature_corr_t33" "02:00:00"
