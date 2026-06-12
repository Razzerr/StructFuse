#!/bin/bash
# Rerun the no-template controls after disabling the zero-input template branch.
#
# Required sequence:
#   1. ./configs/experiment/launch_no_template_reruns.sh --pilot
#   2. Inspect the pilot: stable validation and
#      train/grad_norm/fusion_tpl_contact=0.
#   3. ./configs/experiment/launch_no_template_reruns.sh --full
#
# Individual final controls can be launched with --8m or --650m.
# Add --dry-run to inspect commands without submitting.

set -euo pipefail

DRY_RUN=false
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot|--8m|--650m|--full)
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
    echo "Choose --pilot, --8m, --650m, or --full." >&2
    exit 2
fi

source "$(dirname "$0")/_launch_common.sh"
prepare_output_dirs
require_clean_worktree

submit_pilot() {
    local job_name="paper_pilot_650m_no_templates_fix"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=baseline/esm2_650m_trained seed=42 test=false task_name=${job_name} tags=[pilot,no-template,650M,template-branch-fix] trainer.max_epochs=8 callbacks.early_stopping.patience=8 callbacks.model_checkpoint.save_top_k=0 callbacks.model_checkpoint.save_last=false 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "16:00:00" 32 "512G" >/dev/null
}

submit_8m() {
    local job_name="paper_8m_no_templates_fixed"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=ablation/no_templates seed=42 test=true task_name=${job_name} tags=[paper,ablation,no-template,8M,template-branch-fix] 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "24:00:00" 16 "256G" >/dev/null
}

submit_650m() {
    local job_name="paper_650m_trained_no_templates_fixed"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=baseline/esm2_650m_trained seed=42 test=true task_name=${job_name} tags=[paper,baseline,no-template,650M,template-branch-fix] 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "72:00:00" 32 "512G" >/dev/null
}

echo "No-template rerun launcher, mode=${MODE}, commit=$(git_commit), dry_run=${DRY_RUN}"
case "${MODE}" in
    pilot)
        submit_pilot
        if [[ "${DRY_RUN}" == "true" ]]; then
            echo "Pilot dry-run validated." >&2
        else
            echo "Pilot submitted. Do not launch --full until validation is stable and train/grad_norm/fusion_tpl_contact is zero." >&2
        fi
        ;;
    8m)
        submit_8m
        ;;
    650m)
        submit_650m
        ;;
    full)
        echo "Submitting final 8M and 650M no-template controls independently (parallel SLURM jobs)." >&2
        submit_8m
        submit_650m
        ;;
esac
