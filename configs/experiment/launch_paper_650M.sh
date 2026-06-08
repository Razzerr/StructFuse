#!/bin/bash
# Final 650M headline and baseline runs.
#
# Usage:
#   ./configs/experiment/launch_paper_650M.sh --seed42-only
#   ./configs/experiment/launch_paper_650M.sh --baselines
#   ./configs/experiment/launch_paper_650M.sh --remaining-seeds
#   ./configs/experiment/launch_paper_650M.sh --all
# Add --dry-run to inspect commands without submitting.

set -euo pipefail

DRY_RUN=false
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed42-only|--baselines|--remaining-seeds|--all)
            if [[ -n "${MODE}" ]]; then
                echo "Choose exactly one launch mode." >&2
                exit 2
            fi
            MODE="${1#--}"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Choose --seed42-only, --baselines, --remaining-seeds, or --all." >&2
    exit 2
fi

source "$(dirname "$0")/_launch_common.sh"
prepare_output_dirs
require_clean_worktree

submit_training() {
    local experiment="$1"
    local job_name="$2"
    local time_limit="$3"
    local seed_override="${4:-}"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=${experiment} test=true task_name=${job_name} ${seed_override} 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "${time_limit}" 32 "512G" >/dev/null
}

submit_seed42() {
    submit_training "frontier" "paper_650m_frontier_s42" "72:00:00" "seed=42"
}

submit_baselines() {
    submit_training "baseline/esm2_650m_trained" "paper_650m_trained_no_templates" "72:00:00"
    submit_training "baseline/esm2_650m_only" "paper_650m_esm2_raw" "16:00:00"
    submit_training "baseline/template_only" "paper_650m_template_only" "36:00:00"
}

submit_remaining_seeds() {
    submit_training "frontier" "paper_650m_frontier_s0" "72:00:00" "seed=0"
    submit_training "frontier" "paper_650m_frontier_s1337" "72:00:00" "seed=1337"
}

echo "650M paper launcher, mode=${MODE}, commit=$(git_commit), dry_run=${DRY_RUN}"
case "${MODE}" in
    seed42-only)
        submit_seed42
        ;;
    baselines)
        submit_baselines
        ;;
    remaining-seeds)
        submit_remaining_seeds
        ;;
    all)
        submit_seed42
        submit_baselines
        submit_remaining_seeds
        ;;
esac

