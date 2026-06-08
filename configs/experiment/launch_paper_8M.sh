#!/bin/bash
# Final 8M mechanism/ablation runs.
#
# Usage:
#   ./configs/experiment/launch_paper_8M.sh --smoke
#   ./configs/experiment/launch_paper_8M.sh --core
#   ./configs/experiment/launch_paper_8M.sh --supplementary
#   ./configs/experiment/launch_paper_8M.sh --all
# Add --dry-run to print the launch matrix without submitting jobs.

set -euo pipefail

DRY_RUN=false
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke|--core|--supplementary|--all)
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
    echo "Choose --smoke, --core, --supplementary, or --all." >&2
    exit 2
fi

source "$(dirname "$0")/_launch_common.sh"
prepare_output_dirs
require_clean_worktree

submit_training() {
    local experiment="$1"
    local job_name="$2"
    local time_limit="${3:-24:00:00}"
    local extra_args="${4:-}"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=${experiment} test=true task_name=${job_name} ${extra_args} 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "${time_limit}" 16 "256G" >/dev/null
}

submit_smoke() {
    local repro_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/smoke_repro_test.py 2>&1 | tee ${TEMP_DIR}/paper_smoke_repro.log"
    local repro_id
    repro_id="$(submit_job "${repro_cmd}" "paper_smoke_repro" "02:00:00" 16 "256G")"

    local eval_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=frontier_8M task_name=paper_smoke_eval test=true ++trainer.max_epochs=1 ++trainer.limit_train_batches=0.01 ++trainer.limit_val_batches=10 ++trainer.limit_test_batches=10 data.num_workers=0 2>&1 | tee ${TEMP_DIR}/paper_smoke_eval.log"
    local eval_id
    eval_id="$(submit_job "${eval_cmd}" "paper_smoke_eval" "03:00:00" 16 "256G" "afterok:${repro_id}")"

    # Separate eval-only canary: no preceding fit(), so this exercises
    # DataModule.setup("validate") and val-threshold calibration for B1/B3.
    local eval_only_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=baseline/esm2_only task_name=paper_smoke_eval_only test=true ++trainer.limit_val_batches=2 ++trainer.limit_test_batches=2 data.num_workers=0 2>&1 | tee ${TEMP_DIR}/paper_smoke_eval_only.log"
    submit_job "${eval_only_cmd}" "paper_smoke_eval_only" "02:00:00" 16 "256G" "afterok:${eval_id}" >/dev/null
}

submit_core() {
    submit_training "frontier_8M" "paper_8m_frontier_k4"
    submit_training "ablation/tpl_contact_only" "paper_8m_stage1_tpl_contact"
    submit_training "ablation/no_triangle" "paper_8m_stage2_no_triangle"
    submit_training "ablation/no_dist" "paper_8m_no_dist"
    submit_training "ablation/no_templates" "paper_8m_no_templates"
    submit_training "ablation/random_retrieval" "paper_8m_random_retrieval"
    submit_training "ablation/bce_only" "paper_8m_bce_only"
    submit_training "baseline/esm2_only" "paper_8m_esm2_raw" "08:00:00"
}

submit_supplementary() {
    submit_training "ablation/standard_fusion" "paper_8m_standard_fusion"
    submit_training "ablation/trufor_fusion" "paper_8m_trufor_fusion"
    submit_training "ablation/dilated_head" "paper_8m_dilated_head"
    submit_training "ablation/k1_templates" "paper_8m_k1"
    submit_training "ablation/k8_templates" "paper_8m_k8"
    submit_training "ablation/k16_templates" "paper_8m_k16"
}

echo "8M paper launcher, mode=${MODE}, commit=$(git_commit), dry_run=${DRY_RUN}"
case "${MODE}" in
    smoke)
        submit_smoke
        ;;
    core)
        submit_core
        ;;
    supplementary)
        submit_supplementary
        ;;
    all)
        submit_core
        submit_supplementary
        ;;
esac
