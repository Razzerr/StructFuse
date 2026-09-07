#!/bin/bash
# Final TruFor+dist 8M mechanism/ablation reruns.
#
# Existing reusable references:
#   - ablation/trufor_fusion_with_dist: full TruFor+dist K=4 reference
#   - ablation/trufor_fusion: TruFor without distance bins
#
# Usage:
#   ./configs/experiment/launch_trufor_ablation_8M.sh --fusion-decision  # 1 job, for stage E
#   ./configs/experiment/launch_trufor_ablation_8M.sh --smoke
#   ./configs/experiment/launch_trufor_ablation_8M.sh --retrieval
#   ./configs/experiment/launch_trufor_ablation_8M.sh --k-sweep
#   ./configs/experiment/launch_trufor_ablation_8M.sh --architecture
#   ./configs/experiment/launch_trufor_ablation_8M.sh --all
# Add --include-references to rerun the two already-completed references.
# Add --dry-run to inspect commands without submitting jobs.

set -euo pipefail

DRY_RUN=false
INCLUDE_REFERENCES=false
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fusion-decision|--smoke|--retrieval|--k-sweep|--architecture|--all)
            if [[ -n "${MODE}" ]]; then
                echo "Choose exactly one launch mode." >&2
                exit 2
            fi
            MODE="${1#--}"
            shift
            ;;
        --include-references)
            INCLUDE_REFERENCES=true
            shift
            ;;
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

if [[ -z "${MODE}" ]]; then
    echo "Choose --smoke, --retrieval, --k-sweep, --architecture, or --all." >&2
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

# Stage E needs exactly one cell: TruFor+dist at k=4, to pair against the grouped
# panel's paper_8m_frontier_k4. One variable (fusion strategy), everything else
# matched. Running the whole TruFor panel before the decision would pay for a
# second full panel to answer a question one run settles.
submit_fusion_decision_cell() {
    submit_training "ablation/trufor_fusion_with_dist" "paper_8m_trufor_full_k4"
}

submit_reference_runs() {
    submit_training "ablation/trufor_fusion_with_dist" "paper_8m_trufor_full_k4"
    submit_training "ablation/trufor_fusion" "paper_8m_trufor_no_dist"
}

submit_retrieval() {
    submit_training "ablation/trufor_random_retrieval" "paper_8m_trufor_random_retrieval"
}

submit_k_sweep() {
    submit_training "ablation/trufor_k1_templates" "paper_8m_trufor_k1"
    submit_training "ablation/trufor_k8_templates" "paper_8m_trufor_k8"
    submit_training "ablation/trufor_k16_templates" "paper_8m_trufor_k16"
}

submit_architecture() {
    submit_training "ablation/trufor_no_triangle_with_dist" "paper_8m_trufor_no_triangle_dist"
    submit_training "ablation/trufor_no_dist_no_triangle" "paper_8m_trufor_no_dist_no_triangle"
    submit_training "ablation/trufor_dilated_head_with_dist" "paper_8m_trufor_dilated_head_dist"
    submit_training "ablation/trufor_bce_only_with_dist" "paper_8m_trufor_bce_only_dist"
}

submit_smoke() {
    local smoke_args="++trainer.max_epochs=1 ++trainer.limit_train_batches=0.01 ++trainer.limit_val_batches=10 ++trainer.limit_test_batches=10 data.num_workers=0 callbacks.early_stopping.patience=1"
    submit_training \
        "ablation/trufor_random_retrieval" \
        "trufor_ablation_smoke_random_retrieval" \
        "04:00:00" \
        "${smoke_args}"
    submit_training \
        "ablation/trufor_no_triangle_with_dist" \
        "trufor_ablation_smoke_no_triangle_dist" \
        "04:00:00" \
        "${smoke_args}"
}

echo "TruFor 8M ablation launcher, mode=${MODE}, commit=$(git_commit), dry_run=${DRY_RUN}, include_references=${INCLUDE_REFERENCES}, exclude=${SLURM_EXCLUDE_NODES:-none}"

if [[ "${INCLUDE_REFERENCES}" == "true" && "${MODE}" != "smoke" ]]; then
    submit_reference_runs
elif [[ "${INCLUDE_REFERENCES}" == "false" && "${MODE}" == "all" ]]; then
    echo "Skipping reference runs by default: use completed full K=4 and no-dist references, or pass --include-references." >&2
fi

case "${MODE}" in
    fusion-decision)
        submit_fusion_decision_cell
        ;;
    smoke)
        submit_smoke
        ;;
    retrieval)
        submit_retrieval
        ;;
    k-sweep)
        submit_k_sweep
        ;;
    architecture)
        submit_architecture
        ;;
    all)
        submit_retrieval
        submit_k_sweep
        submit_architecture
        ;;
esac
