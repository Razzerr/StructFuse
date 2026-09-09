#!/bin/bash
# Final 650M headline and baseline runs, on the Stage-E selected stack.
#
# Stage G is scoped to the 650M HEADLINE, not to the fusion claim (decision
# 2026-09-09). Six runs, no grouped cells:
#   TruFor+dist seeds 42/0/1337  3 trainings   result and seed variability
#   TruFor no-templates seed 42  1 training    matched retrieval control
#   raw ESM2-650M                1 evaluation  reference without a trained head
#   template-only                1 training    template-based reference
#
# Three seeds of TruFor alone CANNOT establish TruFor-over-grouped multi-seed
# (that needs a matched grouped panel) and CANNOT establish the fusion x distance
# interaction (that needs four cells repeated across seeds). Both remain
# single-seed 8M results. Do not add grouped cells here to "complete" them.
#
# Usage:
#   ./configs/experiment/launch_paper_650M.sh --headline-s42     # run FIRST, then verify
#   ./configs/experiment/launch_paper_650M.sh --remaining-seeds
#   ./configs/experiment/launch_paper_650M.sh --baselines
#   ./configs/experiment/launch_paper_650M.sh --final-eval       # bs=1 reported numbers
#   ./configs/experiment/launch_paper_650M.sh --all              # every training, no final-eval
# Add --dry-run to inspect commands without submitting.

set -euo pipefail

DRY_RUN=false
# logs/ still holds the 2025-generation 650M runs, and "newest .ckpt" would
# silently pick one of those. Refuse anything older than the rebuild.
MIN_CKPT_DATE="${MIN_CKPT_DATE:-2026-08-22}"
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --headline-s42|--remaining-seeds|--baselines|--final-eval|--all)
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
        --min-ckpt-date)
            MIN_CKPT_DATE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Choose a launch mode; see --help." >&2
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

# Reported numbers come from a padding-free bs=1 pass, never from the bs=8 test
# that closes a training run (2026-09-06/09-07). ESM embeddings are cached for
# these cells, so bs=1 costs the head only.
submit_final_eval() {
    local experiment="$1"
    local train_task="$2"
    local time_limit="${3:-08:00:00}"
    local ckpt
    # `|| true`: with `set -e -o pipefail` a missing log dir makes find exit
    # non-zero, which killed the whole launcher instead of reaching the SKIP
    # branch below. LOGS_DIR is absolute; ${REPO_ROOT} was never defined in
    # _launch_common.sh, so the old ${REPO_ROOT:-.} silently meant "./logs" and
    # only worked when invoked from the repo root.
    ckpt="$(find "${LOGS_DIR}/${train_task}" -name '*.ckpt' ! -name 'last.ckpt' \
            -newermt "${MIN_CKPT_DATE}" -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -1 | cut -d' ' -f2- || true)"
    if [[ -z "${ckpt}" ]]; then
        local stale
        stale="$(find "${LOGS_DIR}/${train_task}" -name '*.ckpt' ! -name 'last.ckpt' \
                 2>/dev/null | head -1 || true)"
        if [[ -n "${stale}" ]]; then
            echo "SKIP ${train_task}: only checkpoints older than ${MIN_CKPT_DATE} exist" \
                 "(e.g. ${stale}) — that is a previous data generation, not this panel." >&2
        else
            echo "SKIP ${train_task}: no checkpoint found (run not finished?)" >&2
        fi
        return
    fi
    local job_name="${train_task}_bs1"
    local command="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/eval.py \
experiment=${experiment} task_name=${job_name} ckpt_path=${ckpt} \
validate_before_test=true data.eval_batch_size=1 logger=wandb 2>&1 | tee ${TEMP_DIR}/${job_name}.log"
    submit_job "${command}" "${job_name}" "${time_limit}" 16 "256G" >/dev/null
}

# Run these two first and check them before committing the rest of the budget.
submit_headline_s42() {
    submit_training "trufor_fusion_with_dist_650M" "paper_650m_trufor_s42"              "72:00:00" "seed=42"
    submit_training "trufor_no_templates_650M"     "paper_650m_trufor_no_templates_s42" "72:00:00" "seed=42"
}

submit_remaining_seeds() {
    submit_training "trufor_fusion_with_dist_650M" "paper_650m_trufor_s0"    "72:00:00" "seed=0"
    submit_training "trufor_fusion_with_dist_650M" "paper_650m_trufor_s1337" "72:00:00" "seed=1337"
}

submit_baselines() {
    submit_training "baseline/esm2_650m_only" "paper_650m_esm2_raw"     "16:00:00"
    submit_training "baseline/template_only"  "paper_650m_template_only" "36:00:00"
}

submit_final_evals() {
    submit_final_eval "trufor_fusion_with_dist_650M" "paper_650m_trufor_s42"
    submit_final_eval "trufor_fusion_with_dist_650M" "paper_650m_trufor_s0"
    submit_final_eval "trufor_fusion_with_dist_650M" "paper_650m_trufor_s1337"
    submit_final_eval "trufor_no_templates_650M"     "paper_650m_trufor_no_templates_s42"
    submit_final_eval "baseline/template_only"       "paper_650m_template_only"
    # B1 is attention-only: no trained weights, so no ckpt_path. It runs through
    # train.py here as it does in --baselines, just at bs=1. The 650M forward is
    # computed online (no ESM cache for this module), hence the longer limit.
    local b1="paper_650m_esm2_raw_bs1"
    submit_job "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py \
experiment=baseline/esm2_650m_only test=true task_name=${b1} data.eval_batch_size=1 \
2>&1 | tee ${TEMP_DIR}/${b1}.log" "${b1}" "24:00:00" 16 "256G" >/dev/null
}

echo "650M paper launcher, mode=${MODE}, commit=$(git_commit), dry_run=${DRY_RUN}"
case "${MODE}" in
    headline-s42)
        submit_headline_s42
        ;;
    remaining-seeds)
        submit_remaining_seeds
        ;;
    baselines)
        submit_baselines
        ;;
    final-eval)
        submit_final_evals
        ;;
    all)
        submit_headline_s42
        submit_remaining_seeds
        submit_baselines
        ;;
esac
