#!/bin/bash
# Equal-input comparison of grouped, standard, and TruFor-style fusion.
#
# Required sequence:
#   1. ./configs/experiment/launch_fusion_parity.sh --smoke
#   2. ./configs/experiment/launch_fusion_parity.sh --8m
#   3. Review both 8M runs and select a candidate, if either is promising.
#   4. Launch exactly one seed-42 650M candidate.
#   5. Only after promotion, launch that candidate's remaining seeds.
#
# Add --dry-run to inspect commands without submitting.

set -euo pipefail

DRY_RUN=false
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke|--8m|--650m-standard|--650m-trufor|--remaining-seeds-standard|--remaining-seeds-trufor)
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
            sed -n '2,11p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Choose --smoke, --8m, --650m-standard, --650m-trufor, --remaining-seeds-standard, or --remaining-seeds-trufor." >&2
    exit 2
fi

source "$(dirname "$0")/_launch_common.sh"
prepare_output_dirs
require_clean_worktree

submit_training() {
    local experiment="$1"
    local job_name="$2"
    local time_limit="$3"
    local cpus="$4"
    local memory="$5"
    local extra_args="${6:-}"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=${experiment} test=true task_name=${job_name} ${extra_args} 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "${time_limit}" "${cpus}" "${memory}" >/dev/null
}

submit_smoke() {
    local smoke_args="++trainer.max_epochs=1 ++trainer.limit_train_batches=0.01 ++trainer.limit_val_batches=10 ++trainer.limit_test_batches=10 data.num_workers=0 callbacks.early_stopping.patience=1"
    submit_training \
        "ablation/standard_fusion_with_dist" \
        "fusion_parity_smoke_standard" \
        "04:00:00" 16 "256G" \
        "${smoke_args}"
    submit_training \
        "ablation/trufor_fusion_with_dist" \
        "fusion_parity_smoke_trufor" \
        "04:00:00" 16 "256G" \
        "${smoke_args}"
}

submit_8m() {
    submit_training \
        "ablation/standard_fusion_with_dist" \
        "paper_8m_standard_fusion_dist" \
        "24:00:00" 16 "256G"
    submit_training \
        "ablation/trufor_fusion_with_dist" \
        "paper_8m_trufor_fusion_dist" \
        "24:00:00" 16 "256G"
}

submit_650m_candidate() {
    local strategy="$1"
    submit_training \
        "${strategy}_fusion_with_dist_650M" \
        "paper_650m_${strategy}_fusion_dist_s42" \
        "72:00:00" 32 "512G" \
        "seed=42"
}

submit_remaining_seeds() {
    local strategy="$1"
    submit_training \
        "${strategy}_fusion_with_dist_650M" \
        "paper_650m_${strategy}_fusion_dist_s0" \
        "72:00:00" 32 "512G" \
        "seed=0"
    submit_training \
        "${strategy}_fusion_with_dist_650M" \
        "paper_650m_${strategy}_fusion_dist_s1337" \
        "72:00:00" 32 "512G" \
        "seed=1337"
}

echo "Fusion parity launcher, mode=${MODE}, commit=$(git_commit), dry_run=${DRY_RUN}"
case "${MODE}" in
    smoke)
        submit_smoke
        ;;
    8m)
        submit_8m
        ;;
    650m-standard)
        submit_650m_candidate "standard"
        ;;
    650m-trufor)
        submit_650m_candidate "trufor"
        ;;
    remaining-seeds-standard)
        submit_remaining_seeds "standard"
        ;;
    remaining-seeds-trufor)
        submit_remaining_seeds "trufor"
        ;;
esac
