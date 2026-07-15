#!/bin/bash
# Launch the two remaining ESM2-650M TruFor+distance-prior seeds.
#
# Seed 42 has already been run as fvxhx6ib:
#   paper_650m_trufor_fusion_dist_s42
#
# This launcher submits only:
#   paper_650m_trufor_fusion_dist_s0
#   paper_650m_trufor_fusion_dist_s1337
#
# Usage:
#   ./configs/experiment/launch_trufor_650M_remaining.sh --dry-run
#   SLURM_EXCLUDE_NODES=gpu28 ./configs/experiment/launch_trufor_650M_remaining.sh

set -euo pipefail

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

source "$(dirname "$0")/_launch_common.sh"
prepare_output_dirs
require_clean_worktree

submit_training() {
    local seed="$1"
    local job_name="paper_650m_trufor_fusion_dist_s${seed}"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=trufor_fusion_with_dist_650M test=true task_name=${job_name} seed=${seed} 2>&1 | tee ${TEMP_DIR}/${job_name}.log"

    submit_job "${command}" "${job_name}" "72:00:00" 32 "512G" >/dev/null
}

echo "TruFor+dist 650M remaining-seeds launcher, commit=$(git_commit), dry_run=${DRY_RUN}, exclude=${SLURM_EXCLUDE_NODES:-none}"
submit_training 0
submit_training 1337
